# -*- coding: utf-8 -*-
"""分步提示词：每层一句专属提示词，而不是全程一句话。

你提的"粗图一个 / 细图一个 / 眼睛一个 / 手部一个"，在扩散模型里能落地，
但有一个必须先说清楚的前提 ——

  低 denoise 时，提示词几乎不起作用。
  denoise 0.3 时，latent 里原图信息占 70%，模型只是"微调"，
  你写 detailed eyes 它也只会在原有眼形上加一点点；
  denoise 0.6 时提示词才真正接管，但那时它会顺便把表情、脸型一起改掉。

  所以"每层一句提示词"能生效，靠的不是提示词本身，
  而是 **denoise 与 mask 粒度** 的配合。这三者是一套的，不能单拆。

对应的正确分层（也就是你说的描色→填色→局部P图）：

  S1 粗图层   denoise 1.0，512x832，18 步
              提示词只写"是什么 + 什么风格"，不写细节
              -> 定构图、比例、大色块、渲染倾向
  S2 细化层   denoise 0.50，latent 放大 1.5 倍后二遍采样
              提示词只管"加高频"（织物纹理、发丝、次表面散射）
              -> 这是唯一能真正"从粗到细"的一步：在已有结构上加信息
  S3 眼周层   denoise 0.42，只框双眼横带（不是整张脸）
              提示词必须写明当前睁闭状态，否则它会把眼睛弄开
  S4 边缘层   纯算法（USM），不给模型 —— 模型做边缘只会画脏边
  S5 全局滤镜 纯 PIL 调色，不给模型 —— 全图重绘会抹平所有细节

用法：
  python staged_prompt.py --profile 3dcg --random      完整五层
  python staged_prompt.py --compare                    同一粗图，A/B/C 三组对照
  python staged_prompt.py --no-eye                     跳过眼周层
"""
import argparse
import os
import random
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL
from texture_pipeline import post_process

OUT_DIR = os.path.join(BASE, "output", "staged")
RTMLIB = os.path.join(BASE, "rtmlib_src", "rtmlib-main")

# ---- 主体：全程不变，这是"提示词拆分"的前提 ---------------------------
# 每层换掉整句提示词，模型就会当成"画另一个东西"，语义必然漂移。
# 正确做法是 base 固定，只在后面追加该层的职责词。
BASE_POS = (
    "1girl, solo, petite, slender, full body, standing, from head to toe, "
    "feet visible, white shoes, arms at sides, "
    "pale skin, porcelain skin, soft matte skin, "
    "pink hair, short hair, blue eyes, half-closed eyes, sleepy expression, "
    "white sailor uniform, navy collar, blue bow, cropped top, "
    "white pleated skirt, white thighhighs, "
    "plain wall background, soft studio lighting, indoor"
)

BASE_NEG = (
    "worst quality, low quality, blurry, bad anatomy, extra limbs, "
    "bad hands, missing fingers, extra fingers, watermark, text, signature, "
    "speech bubble, text bubble, thought bubble, speech balloon, comic bubble, "
    "empty circle, floating dots, sparkle doodle, "
    "cropped, out of frame, close-up, cowboy shot, upper body, "
    "knees cut off, feet out of frame, sitting, lying, "
    "flat color, cel shading, glossy plastic, shiny oily skin"
)

# ---- 每层专属提示词（只写该层职责，不重复主体） -----------------------
STAGE = {
    "rough": {
        "pos": "3d, cgi, octane render, subsurface scattering, soft matte skin, "
               "smooth shading, simple background, smooth gradients, low detail, "
               "clean silhouette, painterly base",
        "neg": "highly detailed, sharp focus, fine texture, intricate details, "
               "noise, high frequency",
        "steps": 18, "cfg": 5.0, "denoise": 1.0,
    },
    "detail": {
        "pos": "highly detailed, fine texture, fabric weave, individual hair strands, "
               "skin pores, sharp focus, cinematic lighting, soft global illumination, "
               "ambient occlusion, intricate details, subsurface scattering, "
               "octane render, photorealistic rendering",
        "neg": "blurry, smooth plastic surface, flat shading, low detail, "
               "featureless skin, melted wax",
        "steps": 26, "cfg": 4.5, "denoise": 0.50,
    },
    "eye": {
        # 睁闭状态必须写死，否则模型会按"精致眼睛"的先验把它强行打开。
        # denoise 不能高：0.42 实测把眼周重采样成了青黑暗沉（黑眼圈），
        # 因为小区域重新采样时，噪声比信号多，暗部会被无中生有地加深。
        "pos": "same half-closed eyes, keep the same eyelids, same gaze direction, "
               "same expression, delicate eyelashes, soft eyelid crease, "
               "clean bright skin around eyes, natural under-eye",
        "neg": "wide open eyes, staring, different expression, blank eyes, "
               "anime sparkle eyes, glow, blue sclera, mismatched eyes, "
               "dark circles, bruised under-eye, heavy shadow, hollow eye socket, "
               "sunken eyes, tired dark rings",
        "steps": 20, "cfg": 4.5, "denoise": 0.32,
    },
    "hand": {
        "pos": "natural relaxed hand, exactly five fingers, correct knuckles, "
               "soft matte skin, subsurface scattering, delicate nails",
        "neg": "six fingers, extra fingers, fused fingers, missing fingers, "
               "deformed hand, claw hand, mangled fingers",
        "steps": 22, "cfg": 4.5, "denoise": 0.50,
    },
}


PLAIN_STAGE = {
    "rough": {
        "pos": "simple shading, plain colors, plain background, clean silhouette",
        "neg": "highly detailed, sharp focus, fine texture, intricate details, noise",
        "steps": 18, "cfg": 5.0, "denoise": 1.0,
    },
    "detail": {
        "pos": "natural soft lighting, clean simple shading, smooth clean rendering",
        "neg": "blurry, noisy, over-sharpened, plastic, burnt highlights",
        "steps": 26, "cfg": 4.5, "denoise": 0.50,
    },
}


def rough_and_detail(p, seed, w, h, wf_tag):
    """S1 粗图 + S2 latent 升采样细化，一条图走完。返回 vae 输出文件名前的 Graph。"""
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)

    rp = g.node("CLIPTextEncode", clip=clip, text=BASE_POS + ", " + STAGE["rough"]["pos"])
    rn = g.node("CLIPTextEncode", clip=clip, text=BASE_NEG + ", " + STAGE["rough"]["neg"])
    lat = g.node("EmptyLatentImage", width=w, height=h, batch_size=1)
    ks1 = g.node("KSampler", model=model, seed=seed,
                 steps=STAGE["rough"]["steps"], cfg=STAGE["rough"]["cfg"],
                 sampler_name="euler_ancestral", scheduler="normal", denoise=1.0,
                 positive=g.ref(rp, 0), negative=g.ref(rn, 0),
                 latent_image=g.ref(lat, 0))

    up = g.node("LatentUpscaleBy", samples=g.ref(ks1, 0),
                upscale_method="nearest-exact", scale_by=1.5)

    dp = g.node("CLIPTextEncode", clip=clip, text=BASE_POS + ", " + STAGE["detail"]["pos"])
    dn = g.node("CLIPTextEncode", clip=clip, text=BASE_NEG + ", " + STAGE["detail"]["neg"])
    ks2 = g.node("KSampler", model=model, seed=seed + 1,
                 steps=STAGE["detail"]["steps"], cfg=STAGE["detail"]["cfg"],
                 sampler_name="euler_ancestral", scheduler="normal",
                 denoise=STAGE["detail"]["denoise"],
                 positive=g.ref(dp, 0), negative=g.ref(dn, 0),
                 latent_image=g.ref(up, 0))

    dec = g.node("VAEDecode", samples=g.ref(ks2, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=wf_tag)
    return g.nodes


def rough_only(p, seed, w, h, tag):
    """A 组：只有粗图，不做细化，直接交后续超分。用来隔离"细化层到底加了多少"。"""
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    rp = g.node("CLIPTextEncode", clip=clip, text=BASE_POS + ", " + STAGE["rough"]["pos"])
    rn = g.node("CLIPTextEncode", clip=clip, text=BASE_NEG + ", " + STAGE["rough"]["neg"])
    lat = g.node("EmptyLatentImage", width=w, height=h, batch_size=1)
    ks = g.node("KSampler", model=model, seed=seed,
                steps=STAGE["rough"]["steps"], cfg=STAGE["rough"]["cfg"],
                sampler_name="euler_ancestral", scheduler="normal", denoise=1.0,
                positive=g.ref(rp, 0), negative=g.ref(rn, 0),
                latent_image=g.ref(lat, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=tag)
    return g.nodes


def upscale(p, base_name, tag):
    g = SL.Graph()
    im = g.node("LoadImage", image=base_name)
    um = g.node("UpscaleModelLoader",
                model_name=(p.get("upscale") or {}).get(
                    "model", "4x_NMKD-Superscale-SP_178000_G.pth"))
    ui = g.node("ImageUpscaleWithModel", upscale_model=g.ref(um, 0), image=g.ref(im, 0))
    g.node("SaveImage", images=g.ref(ui, 0), filename_prefix=tag)
    return g.nodes


def inpaint(p, base_name, mask_name, prompt, neg, denoise, steps, cfg, tag):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=prompt)
    ng = g.node("CLIPTextEncode", clip=clip, text=neg)
    im = g.node("LoadImage", image=base_name)
    mk = g.node("LoadImageMask", image=mask_name, channel="red")
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    noise = g.node("SetLatentNoiseMask", samples=g.ref(enc, 0), mask=g.ref(mk, 0))
    ks = g.node("KSampler", model=model, seed=int(p.get("seed", 0) + 7),
                steps=steps, cfg=cfg, sampler_name=p.get("sampler", "euler_ancestral"),
                scheduler="normal", denoise=denoise,
                positive=g.ref(pos, 0), negative=g.ref(ng, 0),
                latent_image=g.ref(noise, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=tag)
    return g.nodes


def eye_box(face, ratio=1.0):
    """从脸框切出双眼横带。整脸框会改表情，横带只覆盖眼+眉。"""
    x0, y0, x1, y1 = [v * ratio for v in face]
    w, h = x1 - x0, y1 - y0
    return (int(x0 + w * 0.04), int(y0 + h * 0.20),
            int(x1 - w * 0.04), int(y0 + h * 0.52))


def detect_face(path):
    sys.path.insert(0, RTMLIB)
    import cv2
    from rtmlib import Wholebody
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    img = cv2.imread(path)
    if img is None:
        return None
    kpts, scores = model(img)
    if kpts is None or len(kpts) == 0:
        return None
    kp, sc = kpts[0], scores[0]
    pts = [(kp[i], sc[i]) for i in range(23, 91) if i < len(kp) and sc[i] > 0.4]
    if len(pts) < 12:
        return None
    xs = [p[0][0] for p in pts]
    ys = [p[0][1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    return (int(min(xs) - bw * 0.10), int(min(ys) - bh * 0.10),
            int(max(xs) + bw * 0.10), int(max(ys) + bh * 0.10))


def run_one(wf, tag, expect=True):
    before = SL.snapshot()
    t = time.time()
    if not SL.submit(wf, tag=tag):
        return None
    files = SL.new_files(before)
    if not files:
        return None
    print("  %-6s %.1fs -> %s" % (tag, time.time() - t, os.path.basename(files[-1])))
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("--size", default="512x832", help="粗图尺寸，默认 512x832")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("-p", "--prompt", help="覆盖 BASE_POS 主体描述")
    ap.add_argument("--no-eye", action="store_true")
    ap.add_argument("--plain", action="store_true",
                    help="朴素模式：每层提示词都不带 masterpiece/absurdres/detailed 等华丽词")
    ap.add_argument("--compare", action="store_true",
                    help="同粗图跑 A(无细化) / B(细化) / C(细化+眼周) 三组")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", default=True)
    a = ap.parse_args()

    if a.plain:
        STAGE.update(PLAIN_STAGE)
        print("[朴素模式] 提示词已去掉华丽词")
    p = SL.load(a.profile)
    if a.prompt:
        global BASE_POS
        BASE_POS = a.prompt
    w, h = [int(v) for v in a.size.lower().replace("*", "x").split("x")]
    p["size"] = [w, h]

    ref = SL.prepare_ref(p, a.ref)
    seed = a.seed if a.seed is not None else (
        random.randint(0, 2 ** 48) if a.random else p.get("seed", 0))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    print("画风 %s | 粗图 %dx%d | 种子 %s" % (a.profile, w, h, seed))
    print("分层：S1 粗图(18步) -> S2 细化(latent x1.5, denoise 0.50) "
          "-> S3 眼周(denoise 0.42) -> S4 超分 -> S5 调色\n")

    results = {}

    if a.compare:
        print("[A 组] 只粗图，不细化")
        fa = run_one(rough_only(p, seed, w, h, "staged/A_%s" % stamp), "A")
        if fa:
            results["A"] = fa
    else:
        fa = None

    print("[B 组] 粗图 + 细化")
    fb = run_one(rough_and_detail(p, seed, w, h, "staged/B_%s" % stamp), "B")
    if fb:
        results["B"] = fb

    cur = fb
    if cur and not a.no_eye:
        print("\n[S3 眼周] 定位双眼 ...")
        face = detect_face(cur)
        if face:
            img_name = os.path.basename(cur)
            src = os.path.join(SL.INPUT_DIR, "sp_eye_%s.png" % stamp)
            shutil.copy2(cur, src)
            mk = os.path.join(SL.INPUT_DIR, "sp_eyemask_%s.png" % stamp)
            from PIL import Image as _Im
            W0, H0 = _Im.open(src).size
            box = eye_box(face, ratio=1.0)
            print("  脸框 %s -> 眼带 %s" % (face, box))
            import style_workflows_api as SW
            SW.make_hand_mask(src, mk, box, feather=10)
            fp = BASE_POS + ", " + STAGE["eye"]["pos"]
            fn = BASE_NEG + ", " + STAGE["eye"]["neg"]
            fe = run_one(inpaint(p, os.path.basename(src), os.path.basename(mk),
                                 fp, fn, STAGE["eye"]["denoise"],
                                 STAGE["eye"]["steps"], STAGE["eye"]["cfg"],
                                 "staged/C_%s" % stamp), "C")
            if fe:
                results["C"] = fe
                cur = fe
        else:
            print("  没定位到脸，跳过眼周层")

    # 收尾：超分 + 调色
    finals = {}
    for k, f in results.items():
        if not a.upscale:
            finals[k] = f
            continue
        print("\n[%s 超分] 4x ..." % k)
        src = os.path.join(SL.INPUT_DIR, "sp_up_%s_%s.png" % (k, stamp))
        shutil.copy2(f, src)
        u = run_one(upscale(p, os.path.basename(src), "staged/up_%s_%s" % (k, stamp)), k + "up")
        if not u:
            finals[k] = f
            continue
        dst = os.path.join(OUT_DIR, "final_%s_%s.png" % (k, stamp))
        inp = os.path.join(SL.INPUT_DIR, "sp_ps_%s_%s.png" % (k, stamp))
        shutil.copy2(u, inp)
        post_process(inp, dst)
        finals[k] = dst
        print("  调色 -> %s" % os.path.basename(dst))

    from PIL import Image
    for k, f in finals.items():
        im = Image.open(f)
        print("%s: %s (%dx%d)" % (k, f, im.width, im.height))

    # 对比图：同一粗图，A/B/C 横排
    if a.compare and len(finals) >= 2:
        order = [k for k in ("A", "B", "C") if k in finals]
        ims = [Image.open(finals[k]).convert("RGB") for k in order]
        th = 1400
        tiles = [im.resize((int(im.width * th / im.height), th), Image.LANCZOS) for im in ims]
        from PIL import ImageDraw
        W = sum(t.width for t in tiles) + 12 * (len(tiles) - 1)
        canvas = Image.new("RGB", (W, th + 34), (18, 18, 18))
        x = 0
        d = ImageDraw.Draw(canvas)
        labels = {"A": "A 只粗图 + 超分", "B": "B 粗图+细化(latent 1.5x, dn0.50)",
                  "C": "C B+眼周层(dn0.42)"}
        for k, t in zip(order, tiles):
            canvas.paste(t, (x, 34))
            d.text((x + 8, 10), labels[k], fill=(255, 255, 255))
            x += t.width + 12
        cp = os.path.join(OUT_DIR, "compare_%s.jpg" % stamp)
        canvas.save(cp, "JPEG", quality=92)
        print("\n对比图：%s" % cp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
