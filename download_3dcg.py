"""下载 Illustrious 底座的 3DCG 模型（diffusers 格式）到 ComfyUI/models/diffusers/"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

REPO = "John6666/nova-3dcg-xl-illustrious-v40-sdxl"
DEST = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\diffusers\nova3dcg_v40"

PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/*",
    "vae/*",
]

print("REPO", REPO)
print("DEST", DEST)
path = snapshot_download(
    repo_id=REPO,
    local_dir=DEST,
    allow_patterns=PATTERNS,
    max_workers=4,
)
print("DOWNLOADED", path)
for root, dirs, files in os.walk(DEST):
    for f in files:
        p = os.path.join(root, f)
        if os.path.getsize(p) > 10 * 1024 * 1024:
            print(f"  {os.path.relpath(p, DEST)}  {os.path.getsize(p)/1e9:.2f}GB")
