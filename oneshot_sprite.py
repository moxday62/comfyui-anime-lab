# -*- coding: utf-8 -*-
"""一键精灵图：双击就出 16 帧 + sprite sheet + GIF + MP4 + TripoSG 单帧。

省掉的全是手工步骤：
  1 不用先去网页里把帧数从 1 改成 16（配置档里已经是 16，代码里还有硬保护）
  2 不用自己等图、自己找输出目录
  3 不用再单独跑一遍打包脚本
  4 ComfyUI 没开的话会自动拉起来

用法：
  双击 6_一键精灵图.bat
  或命令行：python oneshot_sprite.py [--frames 16] [--cols 4] [--fps 8] [--no-pingpong]
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

COMFY_ROOT = r"D:\ComfyUI\ComfyUI_windows_portable"
COMFY_PY = os.path.join(COMFY_ROOT, "python_embeded", "python.exe")
SERVER = "http://127.0.0.1:8188"
FRAME_DIR = os.path.join(COMFY_ROOT, "ComfyUI", "output", "sprite_run")
PACK_DIR = os.path.join(BASE, "output", "sprite_run")


def comfy_up(timeout=2):
    try:
        urllib.request.urlopen(SERVER + "/system_stats", timeout=timeout)
        return True
    except Exception:
        return False


def start_comfy():
    print("ComfyUI 没在跑，正在拉起（首次加载约 30-60 秒）...")
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        [COMFY_PY, "-s", r"ComfyUI\main.py", "--windows-standalone-build"],
        cwd=COMFY_ROOT, creationflags=flags,
    )
    t0 = time.time()
    while time.time() - t0 < 180:
        if comfy_up():
            print("  ComfyUI 就绪，用时 %.0f 秒" % (time.time() - t0))
            return True
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(3)
    print("\n  ComfyUI 启动超时，请手动双击 4_启动_高画质模式.bat 后再试")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16, help="帧数，低于 16 会被抬到 16")
    ap.add_argument("--cols", type=int, default=4, help="sprite sheet 列数")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--no-pingpong", action="store_true", help="关掉正放+倒放")
    args = ap.parse_args()

    frames = max(16, args.frames)
    if frames != args.frames:
        print("[提示] 帧数 %d 会出噪点，已按 16 帧跑" % args.frames)

    import style_lib as S

    p = S.load("sprite")
    if not comfy_up():
        if not start_comfy():
            return 1

    print("画风档：sprite（%s）" % p.get("label", ""))
    print("帧数 %d | %dx%d | %d 步 | CFG %s"
          % (frames, p["size"][0], p["size"][1], p["steps"], p["cfg"]))
    print("开始出图，预计 40-50 秒 ...")

    before = S.snapshot()
    wf = S.build(p, frames=frames, prefix="sprite_run/frame")
    t0 = time.time()
    if not S.submit(wf, tag="sprite"):
        print("\n出图失败。多半是 ComfyUI 窗口里有报错，看一眼黑窗口最后几行。")
        return 1
    made = S.new_files(before)
    if not made:
        print("没有产出图片")
        return 1
    print("  出图 %d 张，%.1f 秒" % (len(made), time.time() - t0))

    os.makedirs(PACK_DIR, exist_ok=True)
    cmd = [sys.executable, os.path.join(BASE, "make_sprite.py"),
           "--input", FRAME_DIR, "--outdir", PACK_DIR, "--name", "sprite",
           "--count", str(frames), "--cols", str(args.cols), "--fps", str(args.fps)]
    if not args.no_pingpong:
        cmd.append("--pingpong")
    print("\n打包 sprite sheet / GIF / MP4 ...")
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        print("打包失败，帧已生成在 %s" % FRAME_DIR)
        return 1

    print("\n成品目录：%s" % PACK_DIR)
    for f in ("sprite_sheet.png", "sprite.gif", "sprite.mp4", "sprite_triposg.png"):
        fp = os.path.join(PACK_DIR, f)
        if os.path.isfile(fp):
            print("  %-20s %6.1f KB" % (f, os.path.getsize(fp) / 1024.0))
    try:
        os.startfile(PACK_DIR)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
