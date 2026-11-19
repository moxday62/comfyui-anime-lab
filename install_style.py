"""把新下载的底模/IPAdapter 组件硬链接进 ComfyUI，同盘不重复占空间。
另外校验新底模的 SHA256（WAI v17 官方哈希）。"""
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE, COMFY
import hashlib
import shutil

SRC = os.path.join(BASE, "style_models")
COMFY_ROOT = COMFY
MODELS = os.path.join(COMFY_ROOT, "models")

# (源文件名, ComfyUI 子目录, 目标文件名或 None)
ITEMS = [
    ("waiIllustriousSDXL_v170.safetensors", "checkpoints", None),
    ("hassakuXLIllustrious_v22.safetensors", "checkpoints", None),
    ("ip-adapter_sdxl.safetensors", "ipadapter", None),
    ("clip_vision_h.safetensors", "clip_vision", None),
]

# WAI v17 官方 SHA256（CivitAI / HF LyliaEngine 镜像一致）
WAI_SHA256 = "f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04"


def sha256(path, chunk=64 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def place(src, dst_dir, new_name=None):
    if not os.path.isfile(src):
        print(f"  [缺失] {os.path.basename(src)}")
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, new_name or os.path.basename(src))
    if os.path.exists(dst):
        print(f"  [已存在] {os.path.basename(dst)}")
        return True
    try:
        os.link(src, dst)
        print(f"  [硬链接] {os.path.basename(dst)}")
    except OSError:
        shutil.copy2(src, dst)
        print(f"  [复制] {os.path.basename(dst)}")
    return True


def main():
    print("== 校验 WAI v17 SHA256 ==")
    wai = os.path.join(SRC, "waiIllustriousSDXL_v170.safetensors")
    if os.path.isfile(wai):
        actual = sha256(wai)
        ok = actual.lower() == WAI_SHA256.lower()
        print(f"  actual={actual}")
        print(f"  expect={WAI_SHA256}")
        print(f"  [{'OK' if ok else 'MISMATCH!'}] WAI v17 官方哈希校验")
        if not ok:
            print("  !! 哈希不符，停止安装 !!")
            return
    else:
        print("  [缺失] WAI v17 尚未下载完")
        return

    print("== 安装到 ComfyUI ==")
    for name, subdir, rename in ITEMS:
        place(os.path.join(SRC, name), os.path.join(MODELS, subdir), rename)

    print("== IPAdapter 节点 ==")
    nodes = os.path.join(COMFY_ROOT, "custom_nodes", "ComfyUI_IPAdapter_plus")
    print(f"  [{'OK' if os.path.isdir(nodes) else '缺失'}] {nodes}")
    print("DONE - 请重启 ComfyUI 让新模型与节点生效")


if __name__ == "__main__":
    main()
