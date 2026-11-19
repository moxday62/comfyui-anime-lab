# -*- coding: utf-8 -*-
"""质感流水线：出图 -> 细节 LoRA（可选）-> 4x 超分 -> 高光/质感后期。

为什么要单独一条流水线：
  「质感」和「构图/角色」是两层东西。调参器改不动质感，
  IPAdapter 只能整张贴。真正能把它单独拧出来的手段是：
    1 细节 LoRA（现成的 add-detail / Detail Tweaker，叠在采样阶段）
    2 超分（把已有的高频细节还原出来，不是无中生有）
    3 后期（高光提亮 + 局部对比 + 轻锐化，纯 PIL，0.5 秒）
  这三步各管一段，串起来才是"质感增强"，而不是单点调一个参数。

用法：
  python texture_pipeline.py                      用 3dcg 档，默认参数
  python texture_pipeline.py --profile soft
  python texture_pipeline.py --detail add-detail-xl.safetensors:0.5
  python texture_pipeline.py --no-upscale          跳过超分（快速看构图）
  python texture_pipeline.py -p "提示词" --random
"""
import argparse
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL

LORA_DIR = os.path.join(SL.MODELS, "loras")
OUT_DIR = os.path.join(BASE, "output", "texture")


def have_lora(name):
    return os.path.isfile(os.path.join(LORA_DIR, name))


def lora_candidates():
    """扫一遍本地有没有现成的细节增强 LoRA。"""
    if not os.path.isdir(LORA_DIR):
        return []
    keys = ("detail", "tweaker", "texture", "film", "cinematic", "hdr")
    return [f for f in os.listdir(LORA_DIR)
            if f.endswith(".safetensors") and any(k in f.lower() for k in keys)]


def post_process(src, dst, highlight=1.06, contrast=1.05, sharpen=55,
                 radius=1.2, threshold=2):
    """高光提亮 + 局部对比 + USM 锐化。参数都刻意调得很轻，避免油光/halo。"""
    from PIL import Image, ImageFilter, ImageEnhance

    im = Image.open(src).convert("RGB")
    w, h = im.size

    # 1 局部对比：LAB 的 L 通道做轻微 S 曲线，暗部不动、中间调拉开
    lab = im.convert("LAB")
    L, A, B = lab.split()
    L = L.point(lambda v: min(255, int(v + (v - 128) * (contrast - 1.0) * 0.6)))
    im = Image.merge("LAB", (L, A, B)).convert("RGB")

    # 2 高光提亮：只抬亮度前段的像素，模拟"受光面更亮"
    px = im.load()
    img = im.split()
    L2 = img[0].point(lambda v: min(255, int(v * highlight)) if v > 170 else v)
    im = Image.merge("RGB", (L2, img[1], img[2]))

    # 3 USM 锐化（阈值保护，避免噪点被放大）
    im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen,
                                           threshold=threshold))
    im = ImageEnhance.Color(im).enhance(1.02)
    im.save(dst, "PNG")
    return im.size


def compare(base_path, post_path, out_path):
    """左右拼一张对比图，方便一眼看差异。"""
    from PIL import Image, ImageDraw
    a = Image.open(base_path).convert("RGB")
    b = Image.open(post_path).convert("RGB")
    tw = 900
    def fit(im):
        s = tw / float(im.width)
        return im.resize((tw, int(im.height * s)), Image.LANCZOS)
    a, b = fit(a), fit(b)
    h = min(a.height, b.height)
    a, b = a.crop((0, 0, tw, h)), b.crop((0, 0, tw, h))
    canvas = Image.new("RGB", (tw * 2 + 12, h), (24, 24, 24))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (tw + 12, 0))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), "BEFORE", fill=(255, 255, 255))
    d.text((tw + 22, 8), "AFTER", fill=(255, 255, 255))
    canvas.save(out_path, "JPEG", quality=90)
    return canvas.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("-p", "--prompt")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--detail", help="细节 LoRA，写成 文件名:权重，如 add-detail-xl.safetensors:0.5")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", default=True)
    ap.add_argument("--ref", help="参考图（喂图锁角色）")
    a = ap.parse_args()

    p = SL.load(a.profile)
    if a.prompt:
        p["positive"] = a.prompt

    # 细节 LoRA：显式给了就用，没给就自动找本地现成的
    detail = a.detail
    if not detail:
        cand = lora_candidates()
        if cand:
            detail = "%s:0.5" % cand[0]
            print("发现本地细节 LoRA，自动启用：%s（权重 0.5）" % cand[0])
        else:
            print("没找到细节 LoRA。想加这一层，把 add-detail-xl.safetensors 放进：")
            print("  %s" % LORA_DIR)
            print("（也可以现在跳过，只走超分 + 后期）")

    if detail:
        name, _, w = detail.partition(":")
        w = float(w or 0.5)
        if not have_lora(name):
            print("找不到 %s，跳过细节 LoRA" % name)
        else:
            p.setdefault("loras", [])
            p["loras"].append({"file": name, "strength_model": w, "strength_clip": w})
            print("细节 LoRA：%s @ %.2f" % (name, w))

    ref = SL.prepare_ref(p, a.ref)
    seed = a.seed if a.seed is not None else (
        __import__("random").randint(0, 2 ** 48) if a.random else p.get("seed", 0))

    print("\n画风 %s | %s | %d 步 | CFG %s"
          % (a.profile, "x".join(map(str, p.get("size", []))), p.get("steps"), p.get("cfg")))
    print("种子 %s%s" % (seed, " | 喂图 %s" % ref if ref else ""))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M")

    print("\n[1/3] 出图%s ..." % ("+超分" if a.upscale else ""))
    before = SL.snapshot()
    t0 = time.time()
    wf = SL.build(p, ref_image=ref, seed=seed,
                  prefix="texture/base_%s" % stamp, upscale=a.upscale)
    if not SL.submit(wf, tag="出图"):
        return 1
    files = SL.new_files(before)
    if not files:
        print("没有产出")
        return 1
    base_path = files[-1]
    print("  用时 %.1f 秒 -> %s" % (time.time() - t0, os.path.basename(base_path)))

    print("[2/3] 高光/质感后期 ...")
    t1 = time.time()
    post_path = os.path.join(OUT_DIR, "post_%s.png" % stamp)
    size = post_process(base_path, post_path)
    print("  用时 %.1f 秒，输出 %dx%d" % (time.time() - t1, size[0], size[1]))

    print("[3/3] 生成对比图 ...")
    cmp_path = os.path.join(OUT_DIR, "compare_%s.jpg" % stamp)
    compare(base_path, post_path, cmp_path)

    print("\n成品：")
    print("  原图   %s" % base_path)
    print("  后期   %s" % post_path)
    print("  对比   %s" % cmp_path)
    try:
        os.startfile(OUT_DIR)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
