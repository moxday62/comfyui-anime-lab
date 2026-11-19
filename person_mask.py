# -*- coding: utf-8 -*-
"""人物分割：直接用 onnxruntime 跑 u2net，不依赖 rembg。

为什么自己跑：rembg 的依赖链在国内 pip 上卡了 9 分钟没装完，
但真正需要的只有一个 176MB 的 onnx 模型 + onnxruntime（ComfyUI 里已经有了）。
所以这里跳过 rembg，直接做推理。

用的 u2net.onnx 来自 danielgatis/rembg 的 release，输入 320x320，
输出显著性图 d0（0-1），resize 回原尺寸就是人物 alpha。

用法：
  python person_mask.py --src 图.png                      出黑白 mask
  python person_mask.py --src 图.png --cut                顺带出透明底 PNG
  python person_mask.py --src 图.png --bg                  顺带出纯色底，看抠图质量
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "u2net.onnx")
# 动漫专用分割模型（skytnt/anime-seg 的 isnetis.onnx，1024x1024 输入）
# 为什么加它：u2net 是通用模型，遇到二次元图里"床/沙发 + 柔和肤色"的背景，
# 会把背景一起判成人物（实测占比 86%，抠出来是一整块矩形）。
# isnetis 是在动漫图上训练的，边缘和发丝处理明显好一截。
MODEL_ISNET = os.path.join(BASE, "models", "isnetis.onnx")
OUT_DIR = os.path.join(BASE, "output", "bgswap")

_sess = {}


def session(model=None):
    """按模型路径缓存 session（u2net 和 isnetis 可以同时存在）。"""
    global _sess
    path = model or MODEL
    if path not in _sess:
        import onnxruntime as ort
        _sess[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sess[path]


def alpha(src, size=320, model="u2net"):
    """返回原尺寸的人物 alpha（uint8，0-255）。

    model="u2net"       通用显著性模型，320x320。适合"人物 + 简单/中性背景"
    model="isnet-anime" 动漫专用，1024x1024。适合二次元立绘、CG
                        （u2net 在二次元图上常把柔和背景一起判成人物）

    关于缩放：这里**故意**把图直接压成正方形，不保持长宽比。
      两个模型训练时都是压扁喂进去的（rembg 的 naive_cutout 就是
      resize((320,320))）。实测改成"等比缩放 + 黑边 padding"之后，
      原本抠得很准的图（人物占比 16.8%）会退化成整屏 mask（93%）。

    关于归一化：min-max 必须留。u2net 的 d0 原始输出并不铺满 0~1，
      去掉之后同样会退化成整屏 mask。rembg 的处理是
      (pred - pred.min()) / (pred.max() - pred.min())。
    """
    import cv2
    import numpy as np
    from PIL import Image

    img = Image.open(src).convert("RGB")
    W, H = img.size
    if model == "isnet-anime" and size == 320:        # 调用方没指定时用模型原尺寸
        size = 1024
    small = img.resize((size, size), Image.BILINEAR)
    x = np.asarray(small).astype(np.float32) / 255.0

    if model == "isnet-anime":
        # isnetis 的预处理是 (x-0.5)/0.5，不是 ImageNet 均值方差
        path = MODEL_ISNET
        x = (x - 0.5) / 0.5
    else:
        path = MODEL
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
            np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None]

    s = session(path)
    out = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    if out.max() > 1.5 or out.min() < -0.5:          # 未过 sigmoid 的情况
        out = 1.0 / (1.0 + np.exp(-out))
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    m = cv2.resize((out * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, W / 900.0))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--cut", action="store_true", help="输出透明底 PNG")
    ap.add_argument("--bg", action="store_true", help="输出纯绿底，方便看抠图边缘")
    a = ap.parse_args()

    import cv2
    import numpy as np
    from PIL import Image

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.src))[0]
    m = alpha(a.src)
    dst = a.out or os.path.join(OUT_DIR, "mask_%s.png" % base)
    cv2.imwrite(dst, m)
    cover = float((m > 127).mean()) * 100
    print("mask -> %s  人物占比 %.1f%%" % (dst, cover))

    if a.cut or a.bg:
        im = Image.open(a.src).convert("RGB")
        arr = np.array(im)
        al = m
        if a.cut:
            rgba = np.dstack([arr, al])
            p = os.path.join(OUT_DIR, "cut_%s.png" % base)
            Image.fromarray(rgba, "RGBA").save(p)
            print("透明底 -> %s" % p)
        if a.bg:
            green = np.zeros_like(arr)
            green[:, :] = (0, 177, 64)
            k = (al.astype(np.float32) / 255.0)[:, :, None]
            out = (arr * k + green * (1 - k)).astype(np.uint8)
            p = os.path.join(OUT_DIR, "green_%s.png" % base)
            Image.fromarray(out).save(p)
            print("绿底 -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
