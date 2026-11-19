# -*- coding: utf-8 -*-
"""分步作画流水线（Painter Pipeline）—— 用时间换有效分辨率。

核心思想：
  扩散模型不是"在画布上添笔"，而是每次采样都重新想象整张图。
  所以我们之前实测 Hires.fix 会把半身推成特写、把手重画坏 —— 那是"重新想象"的铁证。
  因此分步的成败不在"分几步"，而在"每一步怎么锁住上一步的成果"。

本脚本实现最小可验证闭环（不依赖 ControlNet，只用已有模型）：
  stage 1  构图    低分辨率出姿势，只求布局对
  stage 2  超分    纯像素放大，不重采样，结构不被改写
  stage 3  部位    脸/手各自裁出、放大到高分辨率重绘、羽化贴回
  stage 4  光影    全图低强度重绘，统一光向

与"一次性出图"的差别就是 stage 3：让原本只占 200px 的脸，被单独画到 1024px。
这就是 6GB 显存能出 4K 级局部细节的机制 —— 任何时刻显存里只有一块工作集。

用法：
  python painter_pipeline.py run                    # 跑完整流程
  python painter_pipeline.py run --no-parts         # 跳过部位精修（对照组）
  python painter_pipeline.py run --parts face,hand  # 只精修指定部位
"""
import argparse
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_lib as SL

INPUT_DIR = SL.INPUT_DIR
OUTPUT_DIR = SL.OUTPUT_DIR

# 半身立绘的默认部位框（相对坐标 x, y, w, h）。
# 真实使用要按每张图的实际位置改 —— 没有人体检测器就只能手填，这是本方案的真实缺点。
PART_BOXES = {
    "face": (0.34, 0.06, 0.32, 0.30),
    "hand": (0.02, 0.42, 0.34, 0.34),
}
PART_PROMPTS = {
    "face": ("1girl, close-up face, detailed eyes, clean iris, natural highlights, "
             "well-defined eyelashes, soft skin, correct facial features, "
             "soft painterly illustration, masterpiece, best quality"),
    "hand": ("1girl, hand with five fingers, correct anatomy, clean joints, natural pose, "
             "soft painterly illustration, masterpiece, best quality"),
}


# ---------------------------------------------------------------- 图构建
def wf_t2i(p, w, h, steps, seed, prefix, prompt=None):
    return SL.build(p, seed=seed, size=(w, h), prefix=prefix,
                    upscale=False, ref_image=None)


def wf_img2img(p, image_name, denoise, prefix, prompt=None, steps=None, seed=None):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=prompt or p["positive"])
    neg = g.node("CLIPTextEncode", clip=clip, text=p.get("negative", ""))
    im = g.node("LoadImage", image=image_name)
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    ks = g.node("KSampler", model=model,
                seed=int(seed if seed is not None else p.get("seed", 0)),
                steps=int(steps or p.get("steps", 28)), cfg=float(p.get("cfg", 5.0)),
                sampler_name=p.get("sampler", "euler"),
                scheduler=p.get("scheduler", "normal"), denoise=float(denoise),
                positive=g.ref(pos, 0), negative=g.ref(neg, 0),
                latent_image=g.ref(enc, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


def wf_upscale(image_name, prefix):
    g = SL.Graph()
    im = g.node("LoadImage", image=image_name)
    um = g.node("UpscaleModelLoader", model_name="4x_NMKD-Superscale-SP_178000_G.pth")
    ui = g.node("ImageUpscaleWithModel", upscale_model=g.ref(um, 0), image=g.ref(im, 0))
    g.node("SaveImage", images=g.ref(ui, 0), filename_prefix=prefix)
    return g.nodes


# ---------------------------------------------------------------- 执行工具
def run_stage(wf, tag):
    before = SL.snapshot()
    if not SL.submit(wf, tag=tag):
        raise SystemExit("阶段失败：" + tag)
    files = SL.new_files(before)
    if not files:
        raise SystemExit("阶段无输出：" + tag)
    return files[-1]


def to_input(img_path, name):
    os.makedirs(INPUT_DIR, exist_ok=True)
    dst = os.path.join(INPUT_DIR, name)
    if img_path.lower().endswith((".png", ".jpg")):
        Image.open(img_path).convert("RGB").save(dst)
    return dst


def paste_back(base, patch_path, box, feather=28):
    """把重绘后的部位缩放回原尺寸，用羽化 mask 贴回，避免矩形接缝。"""
    x, y, w, h = [int(v) for v in box]
    patch = Image.open(patch_path).convert("RGB").resize((w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, w - 1, h - 1], radius=max(8, min(w, h) // 6), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather))
    base.paste(patch, (x, y), mask)
    return base


def post_light(im, saturation=0.94, brightness=1.02, contrast=1.03):
    """光影收尾：纯像素调色，不经过模型、不重新采样。

    实测结论：这一步如果用模型低 denoise 重绘（哪怕 0.25），也会把前面积累的
    局部细节抹平 —— 因为"重采样"本质就是重新想象一遍。能不用就不用。
    """
    from PIL import ImageEnhance
    im = ImageEnhance.Color(im).enhance(saturation)
    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    return im


# ---------------------------------------------------------------- 流水线
def paint(profile="soft", seed=None, compose_size=(640, 832), target_w=1664,
          parts=("face", "hand"), part_denoise=0.42, light_denoise=0.0,
          compose_steps=24, tag="painter"):
    p = SL.load(profile)
    seed = int(seed if seed is not None else p.get("seed", 0))
    out = {}
    print("=== 画风 %s（%s）seed=%s ===" % (profile, p.get("label", ""), seed))

    # stage 1 构图：低分辨率，只求姿势对
    print("[1/4] 构图 %sx%s / %d 步" % (compose_size[0], compose_size[1], compose_steps))
    f1 = run_stage(wf_t2i(p, compose_size[0], compose_size[1], compose_steps,
                          seed, tag + "/01_compose"), "构图")
    out["compose"] = f1

    # stage 2 纯超分：不重采样，结构完全保留
    print("[2/4] 超分（纯像素放大，不重新想象）")
    # LoadImage 只认 ComfyUI/input 下的文件，先把构图结果搬过去
    to_input(f1, tag + "_compose.png")
    f2 = run_stage(wf_upscale(tag + "_compose.png", tag + "/02_upscale"), "超分")
    im = Image.open(f2).convert("RGB")
    if im.width > target_w:
        ratio = target_w / float(im.width)
        im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
    base_path = os.path.join(INPUT_DIR, tag + "_base.png")
    im.save(base_path)
    print("      %dx%d -> %dx%d" % (Image.open(f1).width, Image.open(f1).height,
                                    im.width, im.height))
    out["upscaled"] = f2

    # stage 3 部位精修：把脸/手单独放大重画再贴回
    W, H = im.size
    for name in parts:
        if name not in PART_BOXES:
            continue
        rx, ry, rw, rh = PART_BOXES[name]
        x, y = int(W * rx), int(H * ry)
        w, h = int(W * rw), int(H * rh)
        # 裁出并放大到 1024 长边，让模型"专心画这一块"
        crop = im.crop((x, y, x + w, y + h))
        scale = 1024.0 / max(crop.width, crop.height)
        big = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
        src_name = tag + "_%s_in.png" % name
        big.save(os.path.join(INPUT_DIR, src_name))
        print("[3/4] 部位 %s: %dx%d -> %dx%d 重绘 denoise %.2f"
              % (name, w, h, big.width, big.height, part_denoise))
        f3 = run_stage(wf_img2img(p, src_name, part_denoise,
                                  tag + "/03_" + name, prompt=PART_PROMPTS.get(name),
                                  seed=seed + 7),
                       "部位-" + name)
        paste_back(im, f3, (x, y, w, h))
        out[name] = f3

    parts_done = os.path.join(INPUT_DIR, tag + "_parts.png")
    im.save(parts_done)
    out["after_parts"] = parts_done

    # stage 4 光影收尾
    if light_denoise > 0:
        # 老做法：用模型低强度重绘 —— 实测会抹平细节，保留仅作对照实验
        print("[4/4] 光影重绘 denoise %.2f（对照实验，通常负收益）" % light_denoise)
        f4 = run_stage(wf_img2img(p, tag + "_parts.png", light_denoise,
                                  tag + "/04_light",
                                  prompt=p["positive"] + ", unified lighting, coherent shadows",
                                  seed=seed), "光影")
        out["final"] = f4
    else:
        # 推荐做法：纯调色，不碰模型
        print("[4/4] 光影收尾：纯调色（不重采样）")
        final_path = os.path.join(INPUT_DIR, tag + "_final.png")
        post_light(im).save(final_path)
        out["final"] = final_path

    print("\n完成：")
    for k, v in out.items():
        print("  %-12s %s" % (k, v))
    return out


def main():
    ap = argparse.ArgumentParser(description="分步作画流水线")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    ap.add_argument("--profile", default="soft")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--parts", default="face,hand", help="face,hand 或 none")
    ap.add_argument("--part-denoise", type=float, default=0.42)
    ap.add_argument("--light-denoise", type=float, default=0.0,
                    help="0=纯调色（推荐）；>0 则用模型重绘，实测会抹平细节")
    ap.add_argument("--tag", default="painter")
    a = ap.parse_args()

    parts = () if a.parts.strip().lower() in ("", "none") else tuple(
        s.strip() for s in a.parts.split(",") if s.strip())
    paint(profile=a.profile, seed=a.seed, parts=parts,
          part_denoise=a.part_denoise, light_denoise=a.light_denoise, tag=a.tag)


if __name__ == "__main__":
    main()
