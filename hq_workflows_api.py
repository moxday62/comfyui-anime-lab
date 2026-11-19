# -*- coding: utf-8 -*-
"""高质量工作流（ComfyUI API 格式）—— 可直接提交，也用于端到端验证。"""
import json

ILXL = "Illustrious-XL-v2.0.safetensors"
SDXL_VAE = "sdxl_vae.safetensors"
AD_SDXL = "mm_sdxl_v10_beta.ckpt"
UPSCALER = "4x_NMKD-Superscale-SP_178000_G.pth"
FLUX_GGUF = "flux1-dev-Q4_K_S.gguf"
T5 = "t5xxl_fp8_e4m3fn.safetensors"
CLIP_L = "clip_l.safetensors"
FLUX_VAE = "ae.safetensors"

POS = ("masterpiece, best quality, ultra-detailed, anime style, 1girl, solo, "
       "cute, beautiful detailed eyes, long hair, cat ears, maid outfit, frills, lace, "
       "ribbon, holding tray, smiling, waving, soft lighting, vibrant colors, "
       "sharp focus, simple green background")
NEG = ("worst quality, low quality, blurry, jpeg artifacts, deformed, bad anatomy, "
       "extra limbs, bad hands, missing fingers, ugly, watermark, text, signature, "
       "realistic, 3d, nsfw")

# 柔光半厚涂画风（对齐参考图：低饱和暖灰、柔光无硬阴影、线融于面、半厚涂）
SOFT_POS = (
    "1girl, solo, silver hair, red eyes, heart-shaped beauty mark under eye, "
    "red and white japanese outfit, black choker, frills, peace sign, tongue out, "
    "smiling, upper body, clean finished illustration, soft painterly anime, "
    "muted warm grey palette, cream skin, soft diffuse lighting, gentle gradient "
    "shading, detailed, polished, masterpiece, best quality"
)
SOFT_NEG = (
    "sketch, rough, pencil lines, lineart, unfinished, monochrome, worst quality, "
    "low quality, blurry, bad anatomy, bad hands, oversaturated, neon, high contrast, "
    "harsh lighting, 3d, realistic, text, watermark"
)


def sdxl_portrait(w=832, h=1216, upscale=1.5, steps=32, ref_steps=20, seed=987654321):
    """SDXL 两段式 + Hires.fix + 4x 超分 立绘"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ILXL}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": SDXL_VAE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": 7.0,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "LatentUpscaleBy", "inputs": {
            "samples": ["6", 0], "scale_by": upscale, "upscale_method": "nearest-exact"}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": ref_steps, "cfg": 7.0,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 0.5,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
        "10": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
        "11": {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["10", 0], "image": ["9", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {
            "images": ["11", 0], "filename_prefix": "hq_portrait/final"}},
    }


def soft_painterly(w=832, h=1088, steps=32, cfg=4.5, upscale=1, seed=20261118):
    """柔光半厚涂画风 · 质量优先（不追速度）。
    画风锚点：低饱和暖灰、柔光漫射、软边缘无硬描边、半厚涂细腻质感。

    实测结论（重要）：
    - er_sde 会带成粗线稿，必须用 euler
    - Hires.fix 二段采样会重绘构图，导致眼睛崩坏、手指错乱、半身变特写
    - 正确做法：一步出图 + 纯超分（不重采样），结构最稳
    - 默认 upscale=1 输出 832x1088；设 upscale=4 接 4x-NMKD 出 3328x4352
    """
    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ILXL}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": SDXL_VAE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": SOFT_POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": SOFT_NEG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
    }
    if upscale == 1:
        nodes["8"] = {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0], "filename_prefix": "hq_soft/final"}}
    else:
        nodes["9"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
        nodes["10"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["9", 0], "image": ["7", 0]}}
        nodes["8"] = {"class_type": "SaveImage", "inputs": {
            "images": ["10", 0], "filename_prefix": "hq_soft/final"}}
    return nodes


def sdxl_anim(w=768, h=768, frames=16, steps=28, seed=42):
    """SDXL + AnimateDiff 高质量动画"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ILXL}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": SDXL_VAE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": NEG}},
        "5": {"class_type": "ADE_AnimateDiffLoaderGen1", "inputs": {
            "model": ["1", 0], "model_name": AD_SDXL, "beta_schedule": "autoselect"}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {
            "width": w, "height": h, "batch_size": frames}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "seed": seed, "steps": steps, "cfg": 7.0,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": "hq_anim/frame"}},
    }


def flux_portrait(w=1024, h=1024, steps=28, cfg=1.0, guidance=3.5, seed=1234, prefix="flux_portrait/final", prompt=None):
    """FLUX.1-dev 官方 fp8 极致单帧（原生格式，不用 GGUF）。
    ComfyUI 0.45 明确建议用原生 fp8 而非 GGUF —— GGUF 在新版会输出纯噪声。
    FLUX.1-dev 的 KSampler CFG 必须 1.0、负向提示词必须为空；
    提示词引导强度由 FluxGuidance 节点提供。ComfyUI 默认已是 3.5，
    想明显改变提示词遵从度可提到 5~8（过高会过饱和/变形）。
    提示词用完整自然语言句子，不要堆 tag。"""
    FLUX_POS = ("A masterpiece, ultra-detailed anime illustration of a cute girl with "
                "long dark hair and black cat ears, wearing a frilled black and white maid "
                "outfit with lace and ribbons, holding a silver tray, smiling and waving, "
                "soft studio lighting, sharp focus, simple solid green background, "
                "beautiful detailed eyes.")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "flux1-dev-fp8.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": CLIP_L, "clip_name2": T5, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt or FLUX_POS}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": ["4", 0], "guidance": guidance}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "positive": ["10", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


WORKFLOWS = {
    "sdxl_portrait": sdxl_portrait,
    "soft_painterly": soft_painterly,
    "sdxl_anim": sdxl_anim,
    "flux_portrait": flux_portrait,
}

if __name__ == "__main__":
    for name, fn in WORKFLOWS.items():
        print(name, "->", json.dumps(fn(), ensure_ascii=False)[:80], "...")
