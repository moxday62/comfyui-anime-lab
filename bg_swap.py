# -*- coding: utf-8 -*-
"""背景层：人物一动不动，只把背景换掉。

这是四层拆解里之前卡住的一层，卡在"没有人物分割"。
现在用 person_mask.py 的 u2net 打通了，剩下就是一次反相 mask 的局部重绘。

做法：
  1 u2net 出人物 alpha
  2 反相 -> 背景为白
  3 背景 mask **向内膨胀** 若干像素（宁可少换一点背景，
    也不能吃到人物边缘 —— 裙子飘起的地方最容易少一块）
  4 denoise 0.90 重绘背景，提示词换成新场景
  5 输出

为什么 denoise 敢开到 0.9：
  背景和人物是完全不同的两套语义，不需要保留任何原背景信息。
  而且 mask 已经把人物挡在外面，重绘不会碰到人。
  这和"人物区域重绘"完全相反 —— 那边一开高就崩，这边不开高就换不掉。

用法：
  python bg_swap.py --src 图.png --scene cherry
  python bg_swap.py --src 图.png -p "自定义背景提示词"
  python bg_swap.py --src 图.png --scene ruin --denoise 0.95
  python bg_swap.py --src 图.png --compare    原图 / 换背景 并排
"""
import argparse
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL
import person_mask as PM

OUT_DIR = os.path.join(BASE, "output", "bgswap")

SCENES = {
    "cherry": ("japanese garden in spring, cherry blossom trees, falling petals, "
               "soft daylight, wooden shrine in the background, shallow depth of field, "
               "warm sunlight, photorealistic environment, highly detailed background"),
    "ruin": ("abandoned cathedral interior, broken stained glass windows, "
             "dust in the air, shafts of light through the roof, ivy on stone pillars, "
             "gloomy atmosphere, cinematic environment, highly detailed background"),
    "night": ("neon lit alley at night, rain, wet reflective ground, hanging signs, "
              "steam from vents, purple and cyan lighting, cinematic environment, "
              "highly detailed background"),
    "void": ("minimal dark studio backdrop, single soft rim light, subtle vignette, "
             "clean seamless background, no props"),
}


def bg_mask(src, out_png, shrink_background=10, feather=14):
    """人物 mask 反相 -> 背景 mask，并向人物内侧收缩，避免吃边。"""
    import cv2
    import numpy as np

    import numpy as np
    m = PM.alpha(src)
    # 阈值降到 40：薄纱裙边、飘发这类半透明附属物，u2net 只给 0.15~0.4 的置信度，
    # 用 0.5 当阈值会把它们归进背景，重绘时就给你补出两只"透明翅膀"。
    # 宁可多包一点人物，也不要让背景层去猜这些东西是什么。
    # 三段分区（关键）：
    #   >=127  人物核心，绝不碰
    #   40~127 灰区（薄纱裙边、飘发、原场景烘进人物的边缘光），保留像素但重新染色
    #   <40    纯背景，交给重绘
    # 只用二分法会在两件事之间二选一：要么把薄纱补成"透明翅膀"，
    # 要么把青光一起保留成人物的青色描边。分区才能两头都不丢。
    bg = np.where(m < 40, 255, 0).astype(np.uint8)
    if shrink_background > 0:
        k = np.ones((shrink_background * 2 + 1,) * 2, np.uint8)
        bg = cv2.erode(bg, k)          # 背景区域向内收缩 = 给人物留一圈安全边
    if feather > 0:
        bg = cv2.GaussianBlur(bg, (feather * 2 + 1, feather * 2 + 1), 0)
    cv2.imwrite(out_png, bg)
    return out_png


def edge_light_blend(src, dst, width=26, strength=0.55, tint=None):
    """把人物边缘的环境光换成新背景的色温。

    换背景最难的不是换，是人物身上还留着原场景的光。
    原图是青色霓虹，人物边缘被青光照过；换到暖粉樱花场景后，
    那圈青光就变成一圈"描边"，一眼假。
    这里的办法是算"离人物边缘的距离"，越靠边权重越高，
    用新背景的主色做一次渗透 —— 相当于给人物补一圈环境反射光。
    """
    import cv2
    import numpy as np

    img = cv2.imread(dst)
    m = PM.alpha(src)
    core = (m >= 127).astype(np.uint8)
    gray = ((m >= 40) & (m < 127)).astype(np.float32)
    core_edge = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    near = np.exp(-core_edge / max(1.0, width * 0.5)) * core.astype(np.float32)
    # 灰区按 strength 全额染，核心只有最外一圈轻染
    w = (gray * strength + near * strength * 0.55)[:, :, None]

    if tint is None:
        bgsel = (m < 40)
        if bgsel.sum() < 100:
            return dst
        tint = img[bgsel].reshape(-1, 3).mean(axis=0)

    out = img.astype(np.float32) * (1 - w) + tint.reshape(1, 1, 3).astype(np.float32) * w
    cv2.imwrite(dst, np.clip(out, 0, 255).astype(np.uint8))
    return dst


def build(p, img_name, mask_name, prompt, denoise, prefix):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip,
                 text="people, human, character, figure, blurry, low quality, "
                      "lowres, jpeg artifacts, watermark, text")
    im = g.node("LoadImage", image=img_name)
    mk = g.node("LoadImageMask", image=mask_name, channel="red")
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    nz = g.node("SetLatentNoiseMask", samples=g.ref(enc, 0), mask=g.ref(mk, 0))
    ks = g.node("KSampler", model=model, seed=int(p.get("seed", 0) + 29),
                steps=28, cfg=5.0, sampler_name="euler_ancestral",
                scheduler="normal", denoise=denoise,
                positive=g.ref(pos, 0), negative=g.ref(neg, 0),
                latent_image=g.ref(nz, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("--scene", default="cherry", choices=list(SCENES))
    ap.add_argument("-p", "--prompt", help="自定义背景提示词")
    ap.add_argument("--denoise", type=float, default=0.90)
    ap.add_argument("--shrink", type=int, default=6, help="背景 mask 向人物方向收缩像素")
    ap.add_argument("--no-grade", dest="grade", action="store_false", default=True,
                    help="不做边缘环境光融合")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(a.src))[0]

    img_in = os.path.join(SL.INPUT_DIR, "bg_%s.png" % stamp)
    shutil.copy2(a.src, img_in)
    mk_in = os.path.join(SL.INPUT_DIR, "bgmask_%s.png" % stamp)
    bg_mask(a.src, mk_in, a.shrink, feather=14)
    print("背景 mask -> 已生成（收缩 %dpx）" % a.shrink)

    prompt = a.prompt or SCENES[a.scene]
    print("场景 %s | denoise %.2f" % (a.scene if not a.prompt else "自定义", a.denoise))

    before = SL.snapshot()
    t = time.time()
    wf = build(SL.load(a.profile), os.path.basename(img_in), os.path.basename(mk_in),
               prompt, a.denoise, "bgswap/%s_%s" % (base, stamp))
    if not SL.submit(wf, tag="换背景"):
        return 1
    got = SL.new_files(before)
    if not got:
        print("没产出")
        return 1
    dst = os.path.join(OUT_DIR, "swap_%s_%s.png" % (base, a.scene))
    shutil.copy2(got[-1], dst)
    print("%.1fs -> %s" % (time.time() - t, dst))
    if a.grade:
        edge_light_blend(a.src, dst, width=18, strength=0.80)
        print("边缘环境光融合完成")

    if a.compare:
        from PIL import Image, ImageDraw
        tiles = [("原背景", Image.open(a.src).convert("RGB")),
                 ("换背景：%s" % a.scene, Image.open(dst).convert("RGB"))]
        ims = [(t, im.resize((560, int(im.height * 560 / im.width)), Image.LANCZOS))
               for t, im in tiles]
        h = min(im.height for _, im in ims)
        W = sum(im.width for _, im in ims) + 12
        cv = Image.new("RGB", (W, h + 32), (18, 18, 18))
        d = ImageDraw.Draw(cv)
        x = 0
        for tag, im in ims:
            cv.paste(im.crop((0, 0, im.width, h)), (x, 32))
            d.text((x + 6, 10), tag, fill=(255, 255, 255))
            x += im.width + 12
        cp = os.path.join(OUT_DIR, "cmp_%s_%s.jpg" % (base, a.scene))
        cv.save(cp, "JPEG", quality=93)
        print("对比图 %s" % cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
