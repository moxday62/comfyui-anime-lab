#!/usr/bin/env bash
# 串行下载：两个新底模 + IP-Adapter + CLIP Vision image encoder
# 单进程、每文件 8 连接，避免 hf-mirror 对多进程并发限速
set -u
ARIA="D:/AI-Video-Setup/aria2-1.37.0-win-64bit-build1/aria2c.exe"
DIR="D:/AI-Video-Setup/style_models"
mkdir -p "$DIR"

download() {
  local name="$1" url="$2"
  echo "===== START $name ====="
  "$ARIA" -x 8 -s 8 -k 1M -c --max-tries=10 --retry-wait=3 \
    --console-log-level=warn --summary-interval=10 \
    -d "$DIR" -o "$name" "$url"
  local rc=$?
  echo "===== DONE $name (rc=$rc) ====="
  return $rc
}

download "waiIllustriousSDXL_v170.safetensors" \
  "https://hf-mirror.com/LyliaEngine/waiIllustriousSDXL_v170/resolve/main/waiIllustriousSDXL_v170.safetensors"

download "hassakuXLIllustrious_v22.safetensors" \
  "https://hf-mirror.com/LyliaEngine/hassakuXLIllustrious_v22/resolve/main/hassakuXLIllustrious_v22.safetensors"

download "ip-adapter_sdxl.safetensors" \
  "https://hf-mirror.com/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"

download "clip_vision_h.safetensors" \
  "https://hf-mirror.com/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"

echo "ALL DOWNLOADS FINISHED"
