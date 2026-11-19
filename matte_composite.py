# -*- coding: utf-8 -*-
"""合成：把人放进新背景，带去色溢、接触阴影、环境光。

用户的原话："像抠图贴在另一张画上，重点是还没扣干净"。
这个判断是对的，问题出在流程顺序，不在抠图算法本身：

  错的流程：复杂背景图 -> 抠 -> 贴
    1 原图的墙、霓虹、雾气会留在头发丝和裙边里，抠不干净
    2 人物身上带着原场景的光（青的/白的），贴到新场景就不匹配
    3 没有接触阴影，人就"浮"在地上

  对的流程：中性底图 -> 软 alpha 抠 -> 去色溢 -> 合成 -> 加阴影 -> 匹配环境光
    1 出图时就用中性灰底（不要绿幕 —— 实测绿幕权重 1.3 会把白上衣染绿，
      白底更糟，会把白裙子糊成一片）
    2 用 u2net 的软 alpha，不要二值化，头发丝的过渡全靠它
    3 despill：把边缘像素里混进去的背景色成分减掉
    4 接触阴影：人物最低点画模糊椭圆，人立刻"落地"
    5 环境光：边缘朝新背景色偏移一点

用法：
  python matte_composite.py --person 绿幕或灰底图.png --bg 背景图.png
  python matte_composite.py --person A.png --bg-scene cherry
  python matte_composite.py --person A.png --bg B.png --shadow 0.5 --env 0.6
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

OUT_DIR = os.path.join(BASE, "output", "composite")

# 只用于"没有现成背景图"时，让 ComfyUI 生成一张
SCENE_PROMPTS = {
    "cherry": "japanese garden in spring, cherry blossom trees, falling petals, "
              "soft daylight, wooden shrine, shallow depth of field, photorealistic",
    "ruin": "abandoned cathedral interior, broken stained glass windows, "
            "dust in the air, shafts of light, ivy on stone pillars, cinematic",
    "night": "neon lit alley at night, rain, wet reflective ground, hanging signs, "
             "steam from vents, purple and cyan lighting, cinematic",
    "studio": "seamless studio backdrop, soft gradient, subtle vignette, "
              "single soft rim light, no props",
}


def despill(img, alpha, strength=1.0):
    """把边缘像素里的背景色溢出去掉（绿幕就是去绿边）。"""
    import numpy as np
    out = img.astype(np.float32).copy()
    a = alpha.astype(np.float32) / 255.0
    edge = ((a > 0.03) & (a < 0.97)).astype(np.float32)
    if edge.sum() < 10:
        return out
    # 通用 despill：如果边缘像素的某个通道明显高于另两个，
    # 多出来的就是背景渗进来的，减掉
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    for ch in range(3):
        others = [c for i, c in enumerate([r, g, b]) if i != ch]
        excess = np.maximum(0.0, out[:, :, ch] - np.maximum(others[0], others[1]))
        out[:, :, ch] -= excess * edge * strength
    return np.clip(out, 0, 255)


def imread_u(path):
    """Unicode 安全的读图（cv2.imread 在 Windows 上读不了中文路径）。"""
    import numpy as np
    import cv2
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    """Unicode 安全的写图。"""
    import os
    import numpy as np
    import cv2
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def contact_shadow(bg, alpha, strength=0.55, spread=1.0):
    """在人物脚下压一块模糊阴影，让她落地。

    两处修正（都是实测踩出来的）：
      1 中心要往上抬一点。原来直接画在 ys.max()（脚底那一行），
         结果椭圆大半在画面外，合成出来人像悬空。
      2 加一层更大更淡的环境遮蔽（AO）。单靠一块深椭圆会显得像"贴纸压了个圆"，
         两层叠起来才有"站在那儿"的感觉。
    """
    import cv2
    import numpy as np
    H, W = alpha.shape
    ys, xs = np.where(alpha > 128)
    if len(ys) == 0:
        return bg
    y_bottom = int(ys.max()) - int(H * 0.008)
    x_center = int(xs.mean()) + int((xs.max() - xs.min()) * 0.06)   # 光从左上，影子略偏右下
    body_w = int((xs.max() - xs.min()))

    sh = np.zeros((H, W), np.float32)
    # 内圈：紧贴脚底，深
    rx = max(20, int(body_w * 0.40 * spread))
    ry = max(7, int(rx * 0.15))
    cv2.ellipse(sh, (x_center, y_bottom), (rx, ry), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(5.0, rx * 0.18))
    # 外圈：更大更淡的落影
    sh2 = np.zeros((H, W), np.float32)
    cv2.ellipse(sh2, (x_center, y_bottom), (int(rx * 2.1), int(ry * 3.2)), 0, 0, 360, 1.0, -1)
    sh2 = cv2.GaussianBlur(sh2, (0, 0), max(10.0, rx * 0.45)) * 0.45

    sh = np.clip(sh + sh2, 0, 1.0) * strength
    out = bg.astype(np.float32) * (1 - sh[:, :, None])
    return np.clip(out, 0, 255)


def env_light(person, alpha, bg, width=26, strength=0.55):
    """人物边缘朝新背景的色调偏一点，模拟环境反射光。"""
    import cv2
    import numpy as np
    a = (alpha > 128).astype(np.uint8)
    bgmask = (alpha < 40)
    if bgmask.sum() < 100:
        return person
    tint = bg[bgmask].reshape(-1, 3).mean(axis=0)
    dist = cv2.distanceTransform(a, cv2.DIST_L2, 5)
    w = np.exp(-dist / max(1.0, width * 0.5)) * strength * a.astype(np.float32)
    w = w[:, :, None]
    out = person.astype(np.float32) * (1 - w) + tint.reshape(1, 1, 3) * w
    return np.clip(out, 0, 255)


def global_grade(person, alpha, bg, strength=0.35):
    """整体色温匹配：只动 R/B 比例，不动亮度分布。

    边缘染色只能解决"接缝"，解决不了"人物比场景亮好几档"。
    这一步看的是人物和背景的平均色差，把人往背景色温推一点。
    刻意只调色温不调亮度 —— 调亮度会破坏人物的明暗结构。
    """
    import numpy as np
    pmask = alpha > 128
    bmask = alpha < 40
    if pmask.sum() < 100 or bmask.sum() < 100:
        return person
    p_mean = person[pmask].reshape(-1, 3).mean(axis=0)
    b_mean = bg[bmask].reshape(-1, 3).mean(axis=0)
    ratio = np.clip((b_mean + 1.0) / (p_mean + 1.0), 0.85, 1.15)
    adj = 1.0 + (ratio - 1.0) * strength
    out = person.astype(np.float32).copy()
    out[pmask] = np.clip(out[pmask] * adj, 0, 255)
    return out

def trim_shadow_halo(person, alpha, level=0.15, dark_ratio=0.55):
    """去掉源图自带的那圈投影。

    为什么必须做：模型出图时常常自己在人物周围画一圈投影（尤其参考图有强
    侧光时）。这圈影子在 u2net 眼里是"半透明的人物边缘"，alpha 给到
    40~200，于是被当成人物一起搬进新场景 —— 合成后看起来就是一圈黑边，
    也就是"抠图贴上"的来源。

    做法分两步：
      1 抬阈值：把低置信的过渡带整体削掉（level）
      2 判暗：过渡带里明显比人物核心暗的像素，判为投影，alpha 直接清零
         （只对过渡带生效，人物内部的暗部——深色头发、藏青水手领——不受影响）
    """
    import numpy as np
    a = alpha.astype(np.float32) / 255.0
    a = np.clip((a - level) / max(1e-3, 1 - level), 0, 1)

    lum = person.mean(axis=2) / 255.0
    core = a > 0.88
    if core.sum() < 100:
        return (a * 255).astype(np.uint8)
    core_lum = float(np.median(lum[core]))

    band = (a > 0.02) & (a < 0.88)
    dark = lum < core_lum * dark_ratio
    a[band & dark] = 0.0
    return (a * 255).astype(np.uint8)


def match_luma(person, alpha, bg, strength=0.25):
    """轻微把人物亮度往背景靠，避免"人比场景暗一档"的贴图感。

    只做整体亮度的温和位移，不动对比度 —— 动对比会把脸部明暗结构压平。
    """
    import numpy as np
    pmask = alpha > 200
    bmask = alpha < 30
    if pmask.sum() < 100 or bmask.sum() < 100:
        return person
    p_lum = float(person[pmask].mean())
    b_lum = float(bg[bmask].mean())
    ratio = b_lum / max(1.0, p_lum)
    ratio = float(np.clip(ratio, 0.85, 1.25))
    k = 1.0 + (ratio - 1.0) * strength
    return np.clip(person.astype(np.float32) * k, 0, 255)

def cut_uniform_bg(person, alpha, tol=0.085, lum_cap=1.05, dark_lum=26):
    """单色底抠图：剔除背景和"画在背景上的投影"，同时保住黑色线稿。

    为什么不能暴力删暗像素：实测直接按亮度删，那圈黑影是没了，
    但人物脸上的五官线、裙褶线一起消失，整张图糊成一片。
    线稿和投影都是黑的，区别不在颜色，在**拓扑**：
        线稿被人物包住（是个洞）
        投影贴着外轮廓，一侧是背景

    所以顺序必须是：
      1 solid = 人物 且 非黑 且 非背景色  ← 只剩实心色块
      2 solid 取最大连通块                ← 孤立碎片（投影、反光）自动丢弃
      3 填洞                              ← 被包住的黑色线稿重新归位
      4 用 solid 乘原始 alpha             ← 细发丝、软边保持原样
    """
    import cv2
    import numpy as np
    a = alpha.astype(np.float32) / 255.0
    px = person.astype(np.float32)

    bgsel = a < 0.08
    if bgsel.sum() < 200:
        return alpha
    bg_mean = px[bgsel].reshape(-1, 3).mean(axis=0)
    bg_chroma = bg_mean / (bg_mean.sum() + 1e-6)
    bg_lum = float(bg_mean.mean())

    ssum = px.sum(axis=2) + 1e-6
    chroma = px / ssum[:, :, None]
    dist = np.sqrt(((chroma - bg_chroma[None, None, :]) ** 2).sum(axis=2))
    lum = px.mean(axis=2)

    bgish = (dist < tol) & (lum < bg_lum * lum_cap)
    solid = (a > 0.12) & (~bgish) & (lum >= dark_lum)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    if n > 2:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid = (lab == biggest)

    m = solid.astype(np.uint8)
    ff = m.copy()
    mask = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    solid = solid | (ff == 0)                      # 填洞：把被包住的线稿找回来
    solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8), 1).astype(bool)

    out = (a * solid.astype(np.float32))
    out = cv2.GaussianBlur(out, (0, 0), 0.8)       # 轻微羽化，避免硬边
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def keep_largest(alpha, min_ratio=0.02):
    """只保留最大连通域。

    抠图后常留一些孤立的碎片（投影、反光、飘落的背景元素），
    它们和人物不相连，直接按连通域删掉最省事也最不容易误伤。
    """
    import cv2
    import numpy as np
    a = (alpha > 40).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
    if n <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    keep = (lab == biggest)
    if keep.sum() < (alpha > 40).sum() * (1 - 0.9):
        pass
    out = alpha.copy()
    # 离群的碎片直接清零，主体的软边保留
    out[~keep & (alpha > 0)] = 0
    return out


def run(person_path, bg_path, dst, shadow=0.55, env=0.55, despill_s=1.0, debug=False,
        a_level=0.15, dark_ratio=0.55, luma=0.25, chroma=False):
    import cv2
    import numpy as np
    import person_mask as PM

    person = imread_u(person_path)
    bg = imread_u(bg_path)
    if person is None or bg is None:
        raise SystemExit("读不到图")
    H, W = person.shape[:2]
    if bg.shape[:2] != (H, W):
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LANCZOS4)

    alpha = PM.alpha(person_path)                 # 软 alpha，不做二值化
    print("  alpha 完成，人物占比 %.1f%%" % ((alpha > 127).mean() * 100))

    person = despill(person, alpha, despill_s)
    alpha = trim_shadow_halo(person, alpha, level=a_level, dark_ratio=dark_ratio)
    if chroma:
        before = (alpha > 40).mean()
        alpha = cut_uniform_bg(person, alpha)
        print('  单色底去投影：面积 %.1f%% -> %.1f%%' % (before * 100, (alpha > 40).mean() * 100))
    if debug:
        imwrite_u(os.path.join(OUT_DIR, "_despill.png"), person)
        imwrite_u(os.path.join(OUT_DIR, "_alpha.png"), alpha)

    bg = contact_shadow(bg, alpha, shadow)
    person = global_grade(person, alpha, bg, strength=0.35)
    person = env_light(person, alpha, bg, strength=env)
    person = match_luma(person, alpha, bg, strength=luma)

    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    out = person.astype(np.float32) * a + bg.astype(np.float32) * (1 - a)
    imwrite_u(dst, np.clip(out, 0, 255).astype(np.uint8))
    print("  合成 -> %s (%dx%d)" % (dst, W, H))
    return dst


def gen_bg(scene, w, h, stamp):
    """没有现成背景图时，用 ComfyUI 空跑一张。"""
    import shutil
    import style_lib as SL
    p = SL.load("3dcg")
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=SCENE_PROMPTS[scene] + ", no people, empty scene")
    neg = g.node("CLIPTextEncode", clip=clip,
                 text="people, human, character, figure, person, text, watermark, blurry")
    lat = g.node("EmptyLatentImage", width=w, height=h, batch_size=1)
    ks = g.node("KSampler", model=model, seed=20261120, steps=26, cfg=5.0,
                sampler_name="euler_ancestral", scheduler="normal", denoise=1.0,
                positive=g.ref(pos, 0), negative=g.ref(neg, 0),
                latent_image=g.ref(lat, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix="composite/bg_%s" % stamp)
    before = SL.snapshot()
    if not SL.submit(g.nodes, tag="背景"):
        return None
    got = SL.new_files(before)
    if not got:
        return None
    dst = os.path.join(OUT_DIR, "bg_%s_%s.png" % (scene, stamp))
    shutil.copy2(got[-1], dst)
    print("  背景生成 -> %s" % dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--bg", help="现成背景图")
    ap.add_argument("--bg-scene", choices=list(SCENE_PROMPTS))
    ap.add_argument("--out")
    ap.add_argument("--shadow", type=float, default=0.55)
    ap.add_argument("--env", type=float, default=0.55)
    ap.add_argument("--despill", type=float, default=1.0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--a-level", type=float, default=0.15,
                    help="alpha 阈值抬升量，越大越能削掉源图投影暗边")
    ap.add_argument("--dark-ratio", type=float, default=0.55,
                    help="过渡带里亮度低于人物核心多少倍判为投影")
    ap.add_argument("--luma", type=float, default=0.25,
                    help="人物亮度向背景靠拢的强度，0 关闭")
    ap.add_argument("--chroma", action="store_true",
                    help="按色度剔除单色背景及其投影（源图是均匀底色时开）")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    import time
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")

    bg = a.bg
    if not bg:
        if not a.bg_scene:
            raise SystemExit("给 --bg 或 --bg-scene")
        from PIL import Image
        W, H = Image.open(a.person).size
        bg = gen_bg(a.bg_scene, W, H, stamp)
        if not bg:
            return 1

    dst = a.out or os.path.join(OUT_DIR, "comp_%s.png" % stamp)
    run(a.person, bg, dst, a.shadow, a.env, a.despill, a.debug,
        a.a_level, a.dark_ratio, a.luma, a.chroma)

    if a.compare:
        from PIL import Image, ImageDraw
        tiles = [("原图（待抠）", Image.open(a.person).convert("RGB")),
                 ("合成结果", Image.open(dst).convert("RGB"))]
        tw = 520
        ims = [(t, im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS))
               for t, im in tiles]
        h = min(im.height for _, im in ims)
        Wd = sum(im.width for _, im in ims) + 12
        cv = Image.new("RGB", (Wd, h + 32), (18, 18, 18))
        d = ImageDraw.Draw(cv)
        x = 0
        for t, im in ims:
            cv.paste(im.crop((0, 0, im.width, h)), (x, 32))
            d.text((x + 6, 10), t, fill=(255, 255, 255))
            x += im.width + 12
        cp = os.path.join(OUT_DIR, "cmp_%s.jpg" % stamp)
        cv.save(cp, "JPEG", quality=93)
        print("对比图 %s" % cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
