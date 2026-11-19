# -*- coding: utf-8 -*-
"""眼部细节合成：眼白 / 虹膜 / 瞳孔 / 高光，只画在眼裂里面。

为什么这一步必须"画"而不是"生成"：

  实测原图眼裂的像素分布（30x26 和 35x29）：
      眼0  亮度 min 27  max 130  mean 89
      眼1  亮度 min 26  max 208  mean 124
  眼白正常应该有 220+，第一只眼最高只有 130 ——
  也就是说模型根本没画眼白，整个眼裂是一团灰蓝色。
  所以"分不清眼白和瞳孔"不是对比度不够，是**这两个东西不存在**。
  再怎么拉对比，也只能把一团灰拉成明暗两团灰。

  而重绘又被证明不可行（形状必变、先验必开源，见 eye_fix.py 的五次失败）。
  剩下唯一的路：按画师的做法直接画上去。

  画师画眼是四笔：眼白打底 -> 虹膜圆 -> 瞳孔圆 -> 高光点。
  这里就照这四笔画，只是用代码：
      1 用 rtmw 的 6 个眼裂关键点做凸包遮罩，绝不越界（眼眶一动不动）
      2 遮罩内先铺眼白（取原图最亮的中位数，提到 205 左右）
      3 画虹膜：正圆，半径 = 眼裂高 x 0.62，上下被遮罩自然切掉 ——
        这正好就是半闭眼的解剖形态，不用额外做遮挡
      4 虹膜色从原图采样（亮度 30~70 百分位的中位数），保留原来的色相
      5 画瞳孔（虹膜 x 0.42）+ 左上高光点（虹膜 x 0.20）
      6 上眼睑投影：遮罩顶部加一道渐变压暗，让眼球"陷"在眼窝里
      7 边缘羽化，按 alpha 混回原图，保留一点原有纹理

  合成强度可调，默认 0.82：全 1.0 会像贴图，0.6 以下又回到糊。

用法：
  python eye_detail.py --src 图.png
  python eye_detail.py --src 图.png --strength 0.9 --iris-gain 1.15
  python eye_detail.py --src 图.png --compare      出修复前后眼部对比
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

RTMLIB = os.path.join(BASE, "rtmlib_src", "rtmlib-main")
OUT_DIR = os.path.join(BASE, "output", "eyefix")

# 全域索引：左眼 59-64，右眼 65-70（68 点脸段从 23 起）
EYE_IDX = {"eye_l": range(59, 65), "eye_r": range(65, 71)}


def keypoints(img_path):
    sys.path.insert(0, RTMLIB)
    import cv2
    from rtmlib import Wholebody
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit("读不到图：" + img_path)
    kpts, scores = model(img)
    if kpts is None or len(kpts) == 0:
        return None, None
    return kpts[0], scores[0]


def draw_eye(canvas, kp, sc, idx, strength, iris_gain, sclera_level):
    """在 canvas(BGR) 上合成一只眼，返回该眼的遮罩（用于后续统计）。"""
    import cv2
    import numpy as np

    pts = [kp[i] for i in idx if i < len(kp) and sc[i] > 0.3]
    if len(pts) < 4:
        return None
    P = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(P.astype(np.int32))
    x, y, w, h = cv2.boundingRect(hull)
    if w < 4 or h < 3:
        return None

    # 眼裂遮罩：凸包稍微外扩一点点再羽化，避免把睫毛切掉
    m = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [hull], 255)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    m = cv2.GaussianBlur(m, (5, 5), 0)

    # 只要 convex hull 内部的样本，用来取色
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    sel = inner > 128
    if sel.sum() < 20:
        return None
    pix = canvas[sel].astype(np.float32)
    lum = pix.mean(axis=1)
    # 先判断这只眼是不是本来就画清楚了。
    # 判断依据：眼裂内的亮度动态范围。睁眼/结构完整的眼睛，
    # 眼白 (200+) 和虹膜/瞳孔 (30-60) 会拉开 150 以上的差；
    # 糊成一团的只有 70-100。已经清楚的别动 —— 合成只会盖掉它。
    spread = float(np.percentile(lum, 95) - np.percentile(lum, 5))
    print("    眼裂亮度动态范围 %.0f（<110 通常糊成一片，>150 说明已有结构）" % spread)
    lo, hi = np.percentile(lum, 30), np.percentile(lum, 70)
    mid = pix[(lum >= lo) & (lum <= hi)]
    if len(mid) == 0:
        mid = pix
    iris = np.median(mid, axis=0)                     # 虹膜基色（BGR）
    # 底模画出来的虹膜常常是灰的，提一档饱和度，让它和眼影的色调接上
    hsv = cv2.cvtColor(np.uint8([[iris]]), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 1] = min(255.0, hsv[0, 0, 1] * 1.45)
    hsv[0, 0, 2] = min(255.0, hsv[0, 0, 2] * 0.92)
    iris = cv2.cvtColor(np.uint8([[hsv[0, 0]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    bright = np.percentile(lum, 92)
    sclera = np.clip(iris * 0.35 + bright * 0.65 + 30, 0, 255)   # 眼白：偏亮、留色相

    # 半闭眼时上眼睑压下来，可见的只是虹膜的下半部。
    # 所以虹膜圆心要比眼裂中心偏上、半径比眼裂高更大，
    # 让遮罩自然把上半切掉 —— 切出来的就是"半月形"，这才是半闭眼的样子。
    cx = int(x + w * 0.5)
    cy_eye = int(y + h * 0.5)
    cy = int(y + h * 0.56)
    r_iris = min(max(3.0, h * 0.74), w * 0.30)
    r_pupil = max(1.6, r_iris * 0.40)

    layer = canvas.copy()

    # 1 眼白打底（整个眼裂铺一层）
    sclera_bgr = tuple(int(v) for v in np.clip(sclera * (sclera_level / 205.0), 0, 255))
    cv2.fillPoly(layer, [hull], sclera_bgr)

    # 2 虹膜：径向渐变 + 外缘角膜环（真人眼球的深色轮廓）
    for rr in range(int(r_iris), 0, -1):
        t = 1.0 - rr / r_iris
        edge = 1.0
        if rr > r_iris * 0.90:          # 角膜缘环：外圈明显加深
            edge = 0.55
        col = np.clip(iris * (0.72 + 0.55 * t) * iris_gain * edge, 0, 255)
        cv2.circle(layer, (cx, cy), rr, tuple(int(v) for v in col), -1, lineType=cv2.LINE_AA)

    # 2b 放射纹理：极坐标细纹，让虹膜不像一块纯色
    import math
    for k in range(64):
        ang = k * (2 * math.pi / 64)
        r0, r1 = r_iris * 0.48, r_iris * 0.97
        p0 = (int(cx + r0 * math.cos(ang)), int(cy + r0 * math.sin(ang)))
        p1 = (int(cx + r1 * math.cos(ang)), int(cy + r1 * math.sin(ang)))
        shade = 0.86 if k % 2 == 0 else 1.08
        col = tuple(int(v) for v in np.clip(iris * shade * iris_gain, 0, 255))
        cv2.line(layer, p0, p1, col, 1, cv2.LINE_AA)

    # 3 瞳孔
    cv2.circle(layer, (cx, cy), int(r_pupil), (14, 14, 18), -1, lineType=cv2.LINE_AA)
    cv2.circle(layer, (cx, cy), max(1, int(r_pupil * 0.6)), (8, 8, 10), -1, lineType=cv2.LINE_AA)

    # 4 高光：左上一个小点 + 右下一点微弱反光
    hx, hy = int(cx - r_iris * 0.30), int(cy - r_iris * 0.26)
    cv2.circle(layer, (hx, hy), max(1, int(r_iris * 0.072)), (250, 251, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(layer, (int(cx + r_iris * 0.30), int(cy + r_iris * 0.36)),
               max(1, int(r_iris * 0.06)), (170, 190, 210), -1, lineType=cv2.LINE_AA)

    # 5 上眼睑投影：顶部压暗，让眼球陷进眼窝
    sh = layer.copy()
    cv2.ellipse(sh, (cx, int(cy_eye - h * 0.72)), (int(w * 0.80), int(h * 1.15)),
                0, 0, 360, (0, 0, 0), -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, h * 0.22))
    layer = cv2.addWeighted(layer, 1.0, sh, -0.45, 0)

    alpha = (m.astype(np.float32) / 255.0)[:, :, None] * strength
    out = canvas.astype(np.float32) * (1 - alpha) + layer.astype(np.float32) * alpha
    canvas[:] = np.clip(out, 0, 255).astype(np.uint8)
    return m


def run(src, dst, strength=0.82, iris_gain=1.0, sclera_level=205, debug=False):
    import cv2
    img = cv2.imread(src)
    if img is None:
        raise SystemExit("读不到图：" + src)
    kp, sc = keypoints(src)
    if kp is None:
        print("rtmw 没找到人")
        return None
    H, W = img.shape[:2]
    print("图 %dx%d" % (W, H))
    for name, idx in EYE_IDX.items():
        pts = [kp[i] for i in idx if i < len(kp) and sc[i] > 0.3]
        if len(pts) < 4:
            print("  %s 关键点不足，跳过" % name)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print("  %s 眼裂 %dx%d" % (name, int(max(xs) - min(xs)), int(max(ys) - min(ys))))
        m = draw_eye(img, kp, sc, idx, strength, iris_gain, sclera_level)
        if m is not None and debug:
            cv2.imwrite(os.path.join(OUT_DIR, "_mask_%s.png" % name), m)
    cv2.imwrite(dst, img)
    print("写出 %s" % dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--strength", type=float, default=0.65)
    ap.add_argument("--iris-gain", type=float, default=1.0)
    ap.add_argument("--sclera-level", type=int, default=178)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.src))[0]
    dst = a.out or os.path.join(OUT_DIR, "detail_%s.png" % base)

    run(a.src, dst, a.strength, a.iris_gain, a.sclera_level, a.debug)

    if a.compare:
        from PIL import Image, ImageDraw
        import eye_fix as EF
        tiles = [("修复前", EF.crop_face(a.src, 620)), ("眼白+虹膜+瞳孔+高光", EF.crop_face(dst, 620))]
        h = min(t.height for _, t in tiles)
        Wd = sum(t.width for _, t in tiles) + 12
        cv = Image.new("RGB", (Wd, h + 32), (18, 18, 18))
        d = ImageDraw.Draw(cv)
        x = 0
        for tag, im in tiles:
            cv.paste(im.crop((0, 0, im.width, h)), (x, 32))
            d.text((x + 6, 10), tag, fill=(255, 255, 255))
            x += im.width + 12
        cp = os.path.join(OUT_DIR, "detail_cmp_%s.jpg" % base)
        cv.save(cp, "JPEG", quality=95)
        print("对比图 %s" % cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
