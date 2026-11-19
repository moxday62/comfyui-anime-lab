#!/bin/bash
ARIA="/d/AI-Video-Setup/aria2-1.37.0-win-64bit-build1/aria2c.exe"
DIR="/d/AI-Video-Setup/hq_models"
cd "$DIR"
declare -A URLS
URLS[Illustrious-XL-v2.0.safetensors]='https://hf-mirror.com/OnomaAIResearch/Illustrious-XL-v2.0/resolve/main/Illustrious-XL-v2.0.safetensors'
URLS[mm_sdxl_v10_beta.ckpt]='https://hf-mirror.com/guoyww/animatediff/resolve/main/mm_sdxl_v10_beta.ckpt'
URLS[flux1-dev-Q4_K_S.gguf]='https://hf-mirror.com/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q4_K_S.gguf'
URLS[t5xxl_fp8_e4m3fn.safetensors]='https://hf-mirror.com/fmoraes2k/t5xxl_fp8_e4m3fn.safetensors/resolve/main/t5xxl_fp8_e4m3fn.safetensors'
URLS[clip_l.safetensors]='https://hf-mirror.com/DanielMarius/clip_l.safetensors/resolve/main/clip_l.safetensors'
URLS[ae.safetensors]='https://hf-mirror.com/Osrivers/ae.safetensors/resolve/main/ae.safetensors'
URLS[sdxl_vae.safetensors]='https://hf-mirror.com/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors'
for f in "${!URLS[@]}"; do
  if [ -f "$f" ] && [ ! -f "$f.aria2" ]; then echo "SKIP (done): $f"; continue; fi
  echo "RESUME: $f"
  "$ARIA" -x 8 -s 8 -k 1M --user-agent='Mozilla/5.0' --file-allocation=none --continue=true --console-log-level=warn --summary-interval=30 -d "$DIR" -o "$f" "${URLS[$f]}"
done
echo "ALL RESUMED"
