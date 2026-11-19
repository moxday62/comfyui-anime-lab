"""把高质量模型（SDXL / FLUX / 超分）安装到 ComfyUI，同盘硬链接不重复占空间。
关键 FLUX 文件会校验官方 SHA256，防止下载到错误/损坏文件导致提示词失效。"""
import hashlib
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE, COMFY
import shutil
import sys

SRC = os.path.join(BASE, "hq_models")
COMFY_ROOT = COMFY

SHA256 = {
    "flux1-dev-fp8.safetensors": "8e91b68084b53a7fc44ed2a3756d821e355ac1a7b6fe29be760c1db532f3d88a",
    "clip_l.safetensors": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    "t5xxl_fp8_e4m3fn.safetensors": "7d330da4816157540d6bb7838bf63a0f02f573fc48ca4d8de34bb0cbfd514f09",
    "ae.safetensors": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path, name):
    expected = SHA256.get(name)
    if not expected:
        return True
    actual = sha256_file(path)
    if actual != expected:
        print(f"  [哈希不符] {name}: {actual} != {expected}")
        return False
    return True


def place(src, dst_dir, new_name=None):
    if not os.path.isfile(src):
        print(f"  [缺失] {os.path.basename(src)}")
        return False
    name = new_name or os.path.basename(src)
    if not check_hash(src, name):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        if not check_hash(dst, name):
            print(f"  [目标文件哈希不符，请先删除] {dst}")
            return False
        print(f"  [已存在] {name}")
        return True
    try:
        os.link(src, dst)
        print(f"  [硬链接] {name}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  [复制] {name}")
    return check_hash(dst, name)


def main():
    if not os.path.isdir(COMFY_ROOT):
        print("ComfyUI 未找到")
        sys.exit(1)

    print("=== SDXL 高质量 ===")
    place(os.path.join(SRC, "Illustrious-XL-v2.0.safetensors"),
          os.path.join(COMFY_ROOT, "models", "checkpoints"))
    place(os.path.join(SRC, "mm_sdxl_v10_beta.ckpt"),
          os.path.join(COMFY_ROOT, "models", "animatediff_models"))
    place(os.path.join(SRC, "sdxl_vae.safetensors"),
          os.path.join(COMFY_ROOT, "models", "vae"))

    print("=== FLUX 极致单帧 ===")
    place(os.path.join(SRC, "flux1-dev-fp8.safetensors"),
          os.path.join(COMFY_ROOT, "models", "unet"))
    place(os.path.join(SRC, "flux1-dev-Q4_K_S.gguf"),
          os.path.join(COMFY_ROOT, "models", "unet"))
    place(os.path.join(SRC, "t5xxl_fp8_e4m3fn.safetensors"),
          os.path.join(COMFY_ROOT, "models", "text_encoders"))
    place(os.path.join(SRC, "clip_l.safetensors"),
          os.path.join(COMFY_ROOT, "models", "text_encoders"))
    place(os.path.join(SRC, "ae.safetensors"),
          os.path.join(COMFY_ROOT, "models", "vae"))

    print("=== 二次元超分 ===")
    place(os.path.join(SRC, "4x_NMKD-Superscale-SP_178000_G.pth"),
          os.path.join(COMFY_ROOT, "models", "upscale_models"))

    print("完成。")


if __name__ == "__main__":
    main()
