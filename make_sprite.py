r"""
把 ComfyUI 生成的逐帧 PNG 打包成 sprite sheet / GIF / MP4，
并导出一帧给 TripoSG 做单图转 3D。

用法示例（在 ComfyUI 的 python 环境下）：
  python make_sprite.py --input "D:\ComfyUI\output\anime_sprite" --name hero_run --cols 4 --fps 8
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

from PIL import Image


def natural_key(path):
    name = os.path.basename(path)
    nums = "".join(ch if ch.isdigit() else " " for ch in name).split()
    return [int(x) for x in nums] if nums else [0]


def collect_frames(directory, count):
    files = [p for p in glob.glob(os.path.join(directory, "*.png"))
             if not os.path.basename(p).startswith("sprite_")]
    if not files:
        raise SystemExit("没有找到 PNG 帧，请检查 --input 目录")
    files.sort(key=natural_key)
    return files[-count:] if count else files


def make_sheet(frames, out_path, cols):
    imgs = [Image.open(p).convert("RGBA") for p in frames]
    w, h = imgs[0].size
    cols = min(cols, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGBA", (w * cols, h * rows), (0, 0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.alpha_composite(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)
    for im in imgs:
        im.close()
    return sheet.size


def make_gif(frames, out_path, fps, pingpong):
    seq = [Image.open(p).convert("RGBA") for p in frames]
    if pingpong:
        seq = seq + seq[-2:0:-1]
    paletted = []
    for im in seq:
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        paletted.append(bg.convert("P", palette=Image.ADAPTIVE, colors=256))
    paletted[0].save(out_path, save_all=True, append_images=paletted[1:],
                     duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    for im in seq:
        im.close()


def make_mp4(frames, out_path, fps, pingpong):
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return False
    patterns = list(frames)
    if pingpong:
        patterns = patterns + patterns[-2:0:-1]
    tmp_dir = os.path.join(os.path.dirname(out_path), "_mp4frames")
    os.makedirs(tmp_dir, exist_ok=True)
    for i, p in enumerate(patterns):
        shutil.copy2(p, os.path.join(tmp_dir, f"f_{i:05d}.png"))
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i",
           os.path.join(tmp_dir, "f_%05d.png"), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"D:\ComfyUI\output\anime_sprite")
    ap.add_argument("--outdir", default=r"D:\AI-Video-Setup\output")
    ap.add_argument("--name", default="sprite")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--pingpong", action="store_true", help="正放+倒放，循环更自然")
    args = ap.parse_args()

    frames = collect_frames(args.input, args.count)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"使用 {len(frames)} 帧，第一帧: {os.path.basename(frames[0])}")

    sheet_path = os.path.join(args.outdir, f"{args.name}_sheet.png")
    size = make_sheet(frames, sheet_path, args.cols)
    print(f"sprite sheet: {sheet_path}  {size[0]}x{size[1]}")

    gif_path = os.path.join(args.outdir, f"{args.name}.gif")
    make_gif(frames, gif_path, args.fps, args.pingpong)
    print(f"GIF: {gif_path}")

    mp4_path = os.path.join(args.outdir, f"{args.name}.mp4")
    if make_mp4(frames, mp4_path, args.fps, args.pingpong):
        print(f"MP4: {mp4_path}")
    else:
        print("MP4 跳过（未找到 ffmpeg，GIF 已可用）")

    tripo_path = os.path.join(args.outdir, f"{args.name}_triposg.png")
    shutil.copy2(frames[0], tripo_path)
    print(f"TripoSG 单帧: {tripo_path}")


if __name__ == "__main__":
    main()
