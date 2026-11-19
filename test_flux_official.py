# -*- coding: utf-8 -*-
"""临时诊断：按 ComfyUI 官方 Flux.1 Dev 蓝图结构跑一张简单提示词图。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_hq import run

PROMPT = "A single bright red cube centered on a flat blue background, clean studio lighting, high detail."

workflow = {
    "1": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn_fast"}},
    "2": {"class_type": "DualCLIPLoader", "inputs": {
        "clip_name1": "clip_l.safetensors",
        "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": PROMPT}},
    "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
    "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "7": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "seed": 777, "steps": 16, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {
        "images": ["8", 0], "filename_prefix": "flux_test/official_style"}},
}

if __name__ == "__main__":
    ok = run("flux_official_style", workflow)
    sys.exit(0 if ok else 1)
