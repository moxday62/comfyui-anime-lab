# -*- coding: utf-8 -*-
"""诊断：对比不同参数下的小图，定位崩坏发生在采样还是超分阶段。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_hq import run

POS = ("1girl, solo, silver hair, red eyes, heart-shaped beauty mark under eye, "
       "red and white japanese outfit, black choker, frills, peace sign, tongue out, "
       "smiling, upper body, clean finished illustration, soft painterly anime, "
       "muted warm grey palette, cream skin, soft diffuse lighting, gentle gradient "
       "shading, detailed, polished, masterpiece, best quality")
NEG = ("sketch, rough, pencil lines, lineart, unfinished, worst quality, "
       "low quality, blurry, bad anatomy, bad hands, oversaturated, neon, high contrast, "
       "harsh lighting, 3d, realistic, text, watermark")


def wf(sampler, cfg, steps=32):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Illustrious-XL-v2.0.safetensors"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "sdxl_vae.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1088, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": 20261118, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0], "filename_prefix": f"hq_soft/diag_{sampler}_cfg{cfg}"}},
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "euler55"
    configs = {
        "euler55": ("euler", 5.5),
        "euler45": ("euler", 4.5),
        "dpm45": ("dpmpp_2m_sde", 4.5),
        "dpm50": ("dpmpp_3m_sde", 5.0),
    }
    sampler, cfg = configs[which]
    ok = run(f"diag_{which}", wf(sampler, cfg))
    sys.exit(0 if ok else 1)
