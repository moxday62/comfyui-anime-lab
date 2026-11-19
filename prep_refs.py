"""准备 IP-Adapter 参考图：抠白底 + 紧贴脸/上半身的正方形裁切。

为什么要专门做这一步（实测）：
  同一句提示词、同一个种子，只差挂不挂 IP-Adapter：
    不挂 -> 背景纯白 std=0.3，轮廓外一圈亮度比 1.000（完全没有投影）
    挂了 -> 背景变橄榄绿墙，身后多一圈纯黑投影（RGB 1,0,1 / alpha 254）
  也就是 IP-Adapter 不止搬"长相"，它把参考图的**背景、光向、投影**一起搬。
  参考图背景越花，搬过来的就越多。所以先把参考图变成"白底人像"。

为什么用 isnetis 而不是 u2net：
  u2net 是通用模型，在二次元图（床/沙发 + 柔和肤色）上会把背景一起判成人物，
  抠出来是一整块矩形。isnetis（skytnt/anime-seg）是动漫专用，1024 输入，
  发丝和轮廓都准。实测同一张图：u2net 抠成一整块，isnetis 轮廓贴合。

裁切为什么要紧：
  CLIP 会先中心裁剪再编码。竖版全身图中心裁出来是躯干，脸完全丢失，
  IP-Adapter 拿到的就是"一块躯干"，喂了粉发也照样给你蓝发。
  所以按脸定位、裁成正方形，让脸和上半身占满画面。

用法：
  python prep_refs.py                 处理 lora/inbox 全部
  python prep_refs.py --src 某张.jpg   只处理一张
  python prep_refs.py --tight 2.6     裁切边长 = 脸高 x 倍数（3.2 约头到胸）
  python prep_refs.py --no-cutout     只裁切、不抠图
输出：D:\\AI-Video-Setup\\refs\\char\\*.png
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

import person_mask as PM

INBOX = r"D:\AI-Video-Setup\lora\inbox"
OUT_DIR = r"D:\AI-Video-Setup\refs\char"
TMP = os.path.join(os.environ.get("TEMP", "."), "_prep_ref_tmp.png")


def cut_to_white(src, bg_level=250, margin=0.90):
    """isnetis 抠人 -> 贴白底，人物占画幅 margin。返回临时文件路径。"""
    a = PM.alpha(src, model="isnet-anime")
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    al = (a.astype(np.float32) / 255.0)[:, :, None]

    W, H = im.size
    ys, xs = np.where(a > 60)
    if len(ys) < 500:
        return None, im
    y0, y1 = max(0, ys.min()), min(H, ys.max())
    x0, x1 = max(0, xs.min()), min(W, xs.max())
    # 只保留人物包围盒，避免原图大片空白参与构图
    pad = int(max(x1 - x0, y1 - y0) * 0.01)
    y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(W, x1 + pad)

    crop = arr[y0:y1, x0:x1]
    acrop = al[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = int(max(h, w) / max(0.2, margin))
    canvas = np.full((side, side, 3), float(bg_level), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    roi = canvas[oy:oy + h, ox:ox + w].copy()
    canvas[oy:oy + h, ox:ox + w] = crop * acrop + roi * (1 - acrop)
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out.save(TMP)
    return TMP, out


def face_box(path):
    try:
        import staged_prompt as SP
        return SP.detect_face(path)
    except Exception:
        return None


def crop_square(im, cx, cy, side, size=1024):
    W, H = im.size
    side = int(min(side, W, H))
    x0 = int(max(0, min(W - side, cx - side / 2)))
    y0 = int(max(0, min(H - side, cy - side / 2)))
    return (im.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.LANCZOS),
            (x0, y0, x0 + side, y0 + side))


def prepare(src, tight=3.2, size=1024, cutout=True):
    im = Image.open(src).convert("RGB")
    if cutout:
        tmp, cleaned = cut_to_white(src)
        if tmp:
            im = cleaned
            face_src = tmp
        else:
            face_src = src
    else:
        face_src = src

    W, H = im.size
    fb = face_box(face_src)
    if fb:
        x0, y0, x1, y1 = fb
        fh = max(1, y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 + fh * 0.75          # 往下一头，带进胸肩
        crop, rect = crop_square(im, cx, cy, fh * tight, size)

        # 贴边时 clamp 会把方框滑走，脸可能整个被挤出框外（实测有图裁成纯躯干）。
        # 用"脸框面积有多少落在裁切框内"判断，不到一半就退回首部居中。
        ix = max(0, min(rect[2], x1) - max(rect[0], x0))
        iy = max(0, min(rect[3], y1) - max(rect[1], y0))
        if (ix * iy) / max(1, (x1 - x0) * (y1 - y0)) < 0.5:
            crop, _ = crop_square(im, W / 2.0, H * 0.30, min(W, H) * 0.55, size)
            return crop, False
        return crop, True

    crop, _ = crop_square(im, W / 2.0, H * 0.30, min(W, H) * 0.55, size)
    return crop, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--tight", type=float, default=3.2)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--no-cutout", dest="cutout", action="store_false", default=True)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if a.src:
        files = [a.src]
    else:
        files = sorted(glob.glob(os.path.join(INBOX, "*.*")))
        files = [f for f in files if os.path.splitext(f)[1].lower()
                 in (".jpg", ".jpeg", ".png", ".webp")]

    ok = noface = 0
    for f in files:
        im, has_face = prepare(f, a.tight, a.size, a.cutout)
        name = os.path.splitext(os.path.basename(f))[0]
        im.save(os.path.join(OUT_DIR, "ref_%s.png" % name[:12]))
        ok += 1
        if not has_face:
            noface += 1
        if ok <= 3 or a.src:
            print("  %s  脸=%s" % (os.path.basename(f), "有" if has_face else "无(退回首部居中)"))
    print("\n完成 %d 张 -> %s（%d 张没检出脸）" % (ok, OUT_DIR, noface))


if __name__ == "__main__":
    main()
