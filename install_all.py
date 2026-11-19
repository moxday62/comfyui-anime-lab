"""把下载好的模型/节点/工作流安装到 ComfyUI 目录（同盘优先硬链接，不重复占空间）。"""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE, COMFY
import shutil
import sys

SRC = BASE
COMFY_ROOT = COMFY


def place(src, dst_dir, new_name=None):
    if not os.path.isfile(src):
        print(f"  [跳过] 不存在: {src}")
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, new_name or os.path.basename(src))
    if os.path.exists(dst):
        print(f"  [已存在] {dst}")
        return True
    try:
        os.link(src, dst)
        print(f"  [硬链接] {os.path.basename(dst)}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  [复制] {os.path.basename(dst)}")
    return True


def main():
    if not os.path.isdir(COMFY_ROOT):
        print("ComfyUI 尚未解压，请先解压 7z 安装包")
        sys.exit(1)

    print("安装模型...")
    place(os.path.join(SRC, "models", "MeinaV7-pruned.safetensors"),
          os.path.join(COMFY_ROOT, "models", "checkpoints"))
    place(os.path.join(SRC, "models", "anything-v5-vae.safetensors"),
          os.path.join(COMFY_ROOT, "models", "vae"))
    place(os.path.join(SRC, "models", "lcm-lora-sdv1-5.safetensors"),
          os.path.join(COMFY_ROOT, "models", "loras"))
    place(os.path.join(SRC, "models", "v3_sd15_mm.ckpt"),
          os.path.join(COMFY_ROOT, "models", "animatediff_models"))

    print("安装 AnimateDiff 节点...")
    node_src = os.path.join(SRC, "ComfyUI-AnimateDiff-Evolved-main")
    node_dst = os.path.join(COMFY_ROOT, "custom_nodes", "ComfyUI-AnimateDiff-Evolved")
    if os.path.isdir(node_src):
        if os.path.exists(node_dst):
            print("  [已存在] ComfyUI-AnimateDiff-Evolved")
        else:
            shutil.copytree(node_src, node_dst)
            print("  [已安装] ComfyUI-AnimateDiff-Evolved")

    print("安装工作流...")
    wf_src = os.path.join(SRC, "workflows")
    wf_dst = os.path.join(COMFY_ROOT, "user", "default", "workflows")
    os.makedirs(wf_dst, exist_ok=True)
    for name in os.listdir(wf_src):
        if name.endswith(".json"):
            place(os.path.join(wf_src, name), wf_dst)
    shutil.copy2(os.path.join(SRC, "make_sprite.py"), os.path.join(COMFY_ROOT, "make_sprite.py"))
    print("完成。")


if __name__ == "__main__":
    main()
