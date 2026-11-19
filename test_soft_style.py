# -*- coding: utf-8 -*-
"""画风测试：euler + 干净提示词，小图快速验证。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_hq import run

POS = ("1girl, solo, silver hair, red eyes, heart-shaped beauty mark under eye, "
       "red and white japanese outfit, black choker, frills, peace sign, tongue out, "
       "smiling, upper body, clean finished illustration, soft painterly anime, "
       "muted warm grey palette, cream skin, soft diffuse lighting, gentle gradient "
       "shading, detailed, polished, masterpiece, best quality")
NEG = ("sketch, rough, pencil lines, lineart, unfinished, monochrome, worst quality, "
       "low quality, blurry, bad anatomy, bad hands, oversaturated, neon, high contrast, "
       "harsh lighting, 3d, realistic, text, watermark")

workflow = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Illustrious-XL-v2.0.safetensors"}},
    "2": {"class_type": "VAELoader", "inputs": {"vae_name": "sdxl_vae.safetensors"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 640, "height": 832, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "seed": 20261118, "steps": 32, "cfg": 5.5,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
    "8": {"class_type": "SaveImage", "inputs": {
        "images": ["7", 0], "filename_prefix": "hq_soft/test_euler"}},
}

if __name__ == "__main__":
    ok = run("soft_test_euler", workflow)
    sys.exit(0 if ok else 1)
