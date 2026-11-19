# -*- coding: utf-8 -*-
"""眼部加固 v2：ROI 局部放大后再重绘，修完羽化贴回。

v1 失败的原因（必须记下来，这是关键认知）：
  在 768x1248 的全身图上，眼睛只有 45x21 像素。
  除以 8 换到 latent 空间 = 5.6 x 2.6 格。
  模型在 2.6 格的高度里，只能决定"这里是暗的还是亮的"，
  物理上不可能画出瞳孔、虹膜分层、睫毛、高光。
  所以 v1 无论怎么调 denoise、加不加线稿锁定，结果都是
  "睁眼 + 粉色虹膜"这种自由发挥 —— 不是提示词的问题，是像素不够。

  这也解释了之前所有眼部层的失败：
  整脸框 / 眼带 / 线稿锁，全都在 768 尺度上做，全都缺像素。

v2 的做法：
  1 rtmw 在原图上定位双眼（68 点，精确到眼裂）
  2 裁出含双眼的 ROI（往上带到眉毛、往下带到面中，给足上下文）
  3 ROI 放大 3~4 倍 -> 此时眼睛有 180+ 像素宽，够画细节
  4 在放大的 ROI 上做 inpaint（只重绘眼椭圆，其他保留）
  5 缩回原尺寸
  6 用羽化 alpha 贴回原图，接缝被过渡吃掉

  本质上就是"局部分辨率优先"：把有限的像素预算，
  从"整张画布平均分配"改成"眼睛这里多给几十倍"。

用法：
  python eye_fix.py --src 图.png                  ROI 模式，自动 3x
  python eye_fix.py --src 图.png --scale 4        ROI 放大 4 倍
  python eye_fix.py --src 图.png --denoise 0.45   重绘强度
  python eye_fix.py --src 图.png --compare        原图 / 低分辨率直修 / ROI 三组对比
  python eye_fix.py --src 图.png --debug          额外存出 ROI 与 mask 便于排查
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

RTMLIB = os.path.join(BASE, "rtmlib_src", "rtmlib-main")
OUT_DIR = os.path.join(BASE, "output", "eyefix")

EYE_POS = (
    "same eyes, keep the exact same eye shape and eyelid position, "
    "same openness, same gaze direction, same expression, "
    "detailed iris pattern, distinct pupil, small natural catchlight, "
    "clean eyelid crease, individual eyelashes, bright sclera, "
    "soft skin around eyes, subtle under-eye shading"
)
EYE_NEG = (
    "wide open eyes, staring, surprised, different expression, different gaze, "
    "glow, glowing iris, anime sparkle eyes, lens flare, "
    "blue sclera, black sclera, red eyes, pink eyes, mismatched eyes, "
    "dark circles, bruised under-eye, heavy shadow, sunken eyes, "
    "melted eyes, distorted pupil, multiple pupils, extra eyelashes"
)


def _kp(img_path):
    sys.path.insert(0, RTMLIB)
    import cv2
    from rtmlib import Wholebody
    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    kpts, scores = model(img)
    if kpts is None or len(kpts) == 0:
        return None, None
    return kpts[0], scores[0]


def eye_boxes(img_path, pad_x=0.32, pad_y=0.42):
    import cv2
    kp, sc = _kp(img_path)
    if kp is None:
        return None
    H, W = cv2.imread(img_path).shape[:2]

    def one(idx, thr=0.35):
        pts = [(kp[i], sc[i]) for i in idx if i < len(kp) and sc[i] > thr]
        if len(pts) < 3:
            return None
        xs = [p[0][0] for p in pts]
        ys = [p[0][1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bw < 5:
            return None
        px = max(bw * pad_x, 6)
        py = max(bh * pad_y, bw * 0.55)
        return (max(0, int(min(xs) - px)), max(0, int(min(ys) - py)),
                min(W, int(max(xs) + px)), min(H, int(max(ys) + py)))

    return [b for b in (one(range(59, 65)), one(range(65, 71))) if b]


def roi_from_boxes(boxes, img_w, img_h, up=0.55, down=1.05, side=0.18):
    """把双眼框扩成一块 ROI：上带眉毛、下带面中、左右留脸颊。对齐到 8。"""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    w, h = x1 - x0, y1 - y0
    rx0 = int(max(0, x0 - w * side))
    rx1 = int(min(img_w, x1 + w * side))
    ry0 = int(max(0, y0 - h * up))
    ry1 = int(min(img_h, y1 + h * down))
    rx0 = (rx0 // 8) * 8
    ry0 = (ry0 // 8) * 8
    rx1 = min(img_w, ((rx1 + 7) // 8) * 8)
    ry1 = min(img_h, ((ry1 + 7) // 8) * 8)
    return (rx0, ry0, rx1, ry1)


def make_roi(src, roi, scale, out_png, target_w=1024):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    c = im.crop(roi)
    if target_w:
        scale = target_w / float(c.width)
    tw = (int(c.width * scale) // 8) * 8
    th = (int(c.height * scale) // 8) * 8
    m = max(tw, th)
    if m > 1280:                      # 6GB 显存上限
        k = 1280.0 / m
        tw = (int(tw * k) // 8) * 8
        th = (int(th * k) // 8) * 8
    tw, th = max(256, tw), max(256, th)
    c.resize((tw, th), Image.LANCZOS).save(out_png)
    return (tw, th)


def roi_mask(roi_img_path, boxes, roi, out_mask, feather=12):
    """在 ROI 坐标系里画双眼椭圆。"""
    import cv2
    import numpy as np
    im = cv2.imread(roi_img_path)
    H, W = im.shape[:2]
    sx = W / float(roi[2] - roi[0])
    sy = H / float(roi[3] - roi[1])
    m = np.zeros((H, W), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        cx = int((((x0 + x1) / 2.0) - roi[0]) * sx)
        cy = int((((y0 + y1) / 2.0) - roi[1]) * sy)
        ax = max(6, int(((x1 - x0) / 2.0) * sx))
        ay = max(6, int(((y1 - y0) / 2.0) * sy))
        cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    cv2.imwrite(out_mask, m)
    return out_mask


def build_inpaint(p, roi_name, mask_name, denoise, prefix):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=EYE_POS)
    neg = g.node("CLIPTextEncode", clip=clip, text=EYE_NEG)
    im = g.node("LoadImage", image=roi_name)
    mk = g.node("LoadImageMask", image=mask_name, channel="red")
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    nz = g.node("SetLatentNoiseMask", samples=g.ref(enc, 0), mask=g.ref(mk, 0))
    ks = g.node("KSampler", model=model, seed=int(p.get("seed", 0) + 13),
                steps=26, cfg=7.0, sampler_name="euler_ancestral",
                scheduler="normal", denoise=denoise,
                positive=g.ref(pos, 0), negative=g.ref(neg, 0),
                latent_image=g.ref(nz, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


def paste_back(base_path, patched_roi_path, roi, dst, feather=40):
    """把修好的 ROI 缩回原尺寸，羽化矩形 alpha 贴回，接缝被过渡吃掉。"""
    from PIL import Image, ImageDraw, ImageFilter
    base = Image.open(base_path).convert("RGB")
    roi_im = Image.open(patched_roi_path).convert("RGB")
    rw, rh = roi[2] - roi[0], roi[3] - roi[1]
    roi_im = roi_im.resize((rw, rh), Image.LANCZOS)
    mask = Image.new("L", (rw, rh), 0)
    pad = max(3, feather // 2)
    ImageDraw.Draw(mask).rectangle([pad, pad, rw - pad, rh - pad], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather / 2.0))
    out = base.copy()
    out.paste(roi_im, (roi[0], roi[1]), mask)
    out.save(dst)
    return dst


def process(src, denoise, scale, profile, stamp, tag, debug=False):
    from PIL import Image
    W0, H0 = Image.open(src).size
    boxes = eye_boxes(src)
    if not boxes:
        print("  rtmw 没定位到眼睛")
        return None
    print("  眼框 %s（原图 %dx%d）" % (boxes, W0, H0))
    roi = roi_from_boxes(boxes, W0, H0)
    print("  ROI %s = %dx%d" % (roi, roi[2] - roi[0], roi[3] - roi[1]))

    roi_in = os.path.join(SL.INPUT_DIR, "ef2_roi_%s_%s.png" % (tag, stamp))
    tw, th = make_roi(src, roi, scale, roi_in)
    eye_px = int((boxes[0][2] - boxes[0][0]) * tw / max(1, roi[2] - roi[0]))
    print("  ROI 放大到 %dx%d（眼睛约 %d 像素宽，原图只有 %d）"
          % (tw, th, eye_px, boxes[0][2] - boxes[0][0]))

    mk_in = os.path.join(SL.INPUT_DIR, "ef2_mask_%s_%s.png" % (tag, stamp))
    roi_mask(roi_in, boxes, roi, mk_in)

    if debug:
        shutil.copy2(roi_in, os.path.join(OUT_DIR, "_roi_%s.png" % tag))
        shutil.copy2(mk_in, os.path.join(OUT_DIR, "_mask_%s.png" % tag))

    before = SL.snapshot()
    t = time.time()
    wf = build_inpaint(SL.load(profile), os.path.basename(roi_in),
                       os.path.basename(mk_in), denoise,
                       "eyefix/roi_%s_%s" % (tag, stamp))
    if not SL.submit(wf, tag=tag):
        return None
    got = SL.new_files(before)
    if not got:
        return None
    print("  %-6s %.1fs -> %s" % (tag, time.time() - t, os.path.basename(got[-1])))

    dst = os.path.join(OUT_DIR, "fixed_%s_%s.png" % (tag, stamp))
    paste_back(src, got[-1], roi, dst)
    return dst


def lowres_process(src, denoise, profile, stamp, tag):
    """对照组：不放大，直接在低分辨率整图上修。"""
    from PIL import Image
    boxes = eye_boxes(src)
    if not boxes:
        return None
    in_src = os.path.join(SL.INPUT_DIR, "ef2_low_%s_%s.png" % (tag, stamp))
    shutil.copy2(src, in_src)
    W0, H0 = Image.open(in_src).size
    mk = os.path.join(SL.INPUT_DIR, "ef2_lowmask_%s_%s.png" % (tag, stamp))
    roi_mask(in_src, boxes, (0, 0, W0, H0), mk, feather=8)
    before = SL.snapshot()
    wf = build_inpaint(SL.load(profile), os.path.basename(in_src),
                       os.path.basename(mk), denoise,
                       "eyefix/low_%s_%s" % (tag, stamp))
    if not SL.submit(wf, tag=tag):
        return None
    got = SL.new_files(before)
    return got[-1] if got else None


def crop_face(p, tw=520):
    from PIL import Image
    import staged_prompt as SP
    im = Image.open(p).convert("RGB")
    f = SP.detect_face(p)
    if not f:
        return im.crop((0, 0, im.width, im.height // 3))
    x0, y0, x1, y1 = f
    h = y1 - y0
    c = im.crop((x0, int(y0 - h * 0.10), x1, int(y0 + h * 0.62)))
    return c.resize((tw, int(c.height * tw / c.width)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("--denoise", type=float, default=0.50)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    from PIL import Image
    W, H = Image.open(a.src).size
    print("源图 %s (%dx%d)" % (os.path.basename(a.src), W, H))

    if a.compare:
        print("\n[对照] 低分辨率直修 dn %.2f" % a.denoise)
        r1 = lowres_process(a.src, a.denoise, a.profile, stamp, "lowres")
        print("\n[ROI] 放大 %.1fx 后再修 dn %.2f" % (a.scale, a.denoise))
        r2 = process(a.src, a.denoise, a.scale, a.profile, stamp, "roi", a.debug)
        outs = [("原图", a.src)]
        if r1:
            outs.append(("低分辨率直修", r1))
        if r2:
            outs.append(("ROI %.1fx" % a.scale, r2))
        tiles = [(t, crop_face(p)) for t, p in outs]
        h = min(im.height for _, im in tiles)
        Wd = sum(im.width for _, im in tiles) + 12 * (len(tiles) - 1)
        from PIL import ImageDraw
        cv = Image.new("RGB", (Wd, h + 32), (18, 18, 18))
        d = ImageDraw.Draw(cv)
        x = 0
        for t, im in tiles:
            cv.paste(im.crop((0, 0, im.width, h)), (x, 32))
            d.text((x + 6, 10), t, fill=(255, 255, 255))
            x += im.width + 12
        cp = os.path.join(OUT_DIR, "cmp_%s.jpg" % stamp)
        cv.save(cp, "JPEG", quality=94)
        print("\n对比图：%s" % cp)
    else:
        r = process(a.src, a.denoise, a.scale, a.profile, stamp, "roi", a.debug)
        if r:
            print("\n成品：%s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
