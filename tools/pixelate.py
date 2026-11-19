# -*- coding: utf-8 -*-
"""批量把图变成像素画（无界面，给批量用）。

调参用 tools/像素小人.html，滑条拖到满意，把参数抄过来跑批：
    python pixelate.py output/make/1119_202528/01_shot.png -s 6 -c 16
    python pixelate.py output/make -o output/pixel --scale 6 --colors 16 -r 4

参数含义和网页里一一对应：
    -s/--scale    像素块大小（原图宽度 ÷ scale = 像素网格宽度）
    -c/--colors   减到几种颜色，0 = 不减
    --no-dither   关掉 Floyd–Steinberg 抖动
    -r/--ratio    导出放大倍数（用最近邻，边缘保持硬）

注意：缩小用 LANCZOS（面积平均），放大必须用 NEAREST。
放大用插值会让像素块之间出现过渡色，一眼就不像像素画了。
"""
import argparse
import os
import sys

from PIL import Image

OK_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def pixelate(src, dst, scale=6, colors=16, dither=True, ratio=1):
    img = Image.open(src)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
    else:
        img = img.convert("RGB")
        alpha = None

    # 目标像素网格。至少 8px，再小就画不出五官了。
    w = max(8, int(round(img.width / float(max(1, scale)))))
    h = max(8, int(round(img.height * w / float(img.width))))

    small = img.resize((w, h), Image.LANCZOS)

    if colors and colors > 0:
        if small.mode == "RGBA":
            rgb = small.convert("RGB")
            q = rgb.quantize(
                colors=colors,
                method=Image.MEDIANCUT,
                dither=Image.FLOYDSTEINBERG if dither else Image.NONE,
            ).convert("RGB")
            q = q.convert("RGBA")
            q.putalpha(small.getchannel("A"))
            small = q
        else:
            small = small.quantize(
                colors=colors,
                method=Image.MEDIANCUT,
                dither=Image.FLOYDSTEINBERG if dither else Image.NONE,
            ).convert("RGB")

    if ratio and ratio > 1:
        small = small.resize((w * ratio, h * ratio), Image.NEAREST)

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    small.save(dst)
    return w, h


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="批量像素化",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="图片或目录")
    ap.add_argument("-o", "--out", default="", help="输出目录，默认 <src 同级>/pixel")
    ap.add_argument("-s", "--scale", type=int, default=6, help="像素块大小，默认 6")
    ap.add_argument("-c", "--colors", type=int, default=16, help="颜色数，0=不减，默认 16")
    ap.add_argument("--no-dither", action="store_true", help="关抖动")
    ap.add_argument("-r", "--ratio", type=int, default=1, help="放大倍数，默认 1")
    a = ap.parse_args(argv)

    if os.path.isdir(a.src):
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.src)), "pixel")
        files = [os.path.join(a.src, f) for f in sorted(os.listdir(a.src))
                 if f.lower().endswith(OK_EXT)]
        if not files:
            print("目录里没有图：%s" % a.src)
            return 2
    else:
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.src)), "pixel")
        files = [a.src]

    for i, f in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(out_dir, base + "_px.png")
        try:
            w, h = pixelate(f, dst, a.scale, a.colors, not a.no_dither, a.ratio)
            size = "%d×%d" % (w * max(1, a.ratio), h * max(1, a.ratio))
            print("  [%d/%d] %s -> %s  (%s)" % (i, len(files), base, os.path.basename(dst), size))
        except Exception as e:
            print("  [%d/%d] %s 失败：%s" % (i, len(files), base, e))

    print("\n输出目录：%s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
