# -*- coding: utf-8 -*-
"""角色一致性 + 手部修复工作流（ComfyUI API 格式）。

三条能力：
1. A/B 底模对比：wai / hassaku 两个微调模型共用同一提示词
2. IPAdapter 喂图锁角色：用参考图锁住长相/服装（参考图需本地路径 -> 通过 LoadImage）
3. 手部局部重绘：先出全身，再把手部矩形区域 crop -> img2img 重画 -> 贴回

用法见 verify_style.py。
"""
import json

WAI = "waiIllustriousSDXL_v170.safetensors"
HASSAKU = "hassakuXLIllustrious_v22.safetensors"
ILXL = "Illustrious-XL-v2.0.safetensors"
SDXL_VAE = "sdxl_vae.safetensors"
UPSCALER = "4x_NMKD-Superscale-SP_178000_G.pth"
IPADAPTER_SDXL = "ip-adapter_sdxl_vit-h.safetensors"
CLIP_VISION_H = "clip_vision_h.safetensors"

# 柔光半厚涂画风（沿用已验证的 SOFT 词，微调为微调底模优化）
STYLE_POS = (
    "masterpiece, best quality, ultra-detailed, 1girl, solo, "
    "silver hair, red eyes, heart-shaped beauty mark under eye, "
    "red and white japanese outfit, black choker, frills, "
    "peace sign, smiling, upper body, "
    "soft painterly illustration, muted warm grey palette, cream skin, "
    "soft diffuse lighting, gentle gradient shading, clean finished illustration"
)
STYLE_NEG = (
    "sketch, rough, pencil lines, lineart, unfinished, worst quality, low quality, "
    "blurry, bad anatomy, extra limbs, bad hands, missing fingers, extra fingers, "
    "oversaturated, neon, high contrast, harsh lighting, 3d, realistic, text, watermark"
)
# 手部重绘专用词：强调正确的手指结构
HAND_POS = (
    "1girl, hand doing peace sign, two fingers extended, clean fingers, "
    "five fingers, natural hand, correct anatomy, "
    "soft painterly illustration, cream skin, soft diffuse lighting, best quality"
)
HAND_NEG = (
    "bad hands, extra fingers, missing fingers, fused fingers, deformed hand, "
    "extra limbs, mutated fingers, worst quality, low quality, blurry, sketch"
)


def base_portrait(ckpt=WAI, w=832, h=1088, steps=32, cfg=4.5,
                  seed=20261119, ref_image=None, ip_weight=0.6,
                  prefix="style_test/final", use_upscale=False):
    """单张立绘：可选 IPAdapter 喂图锁角色、可选 4x 超分。

    ref_image: 参考图文件名（放到 ComfyUI/output 下，或 input 目录），None 则不喂图。
    ip_weight: IPAdapter 权重，0.5~0.7 锁角色又保留提示词；过高会锁构图。
    """
    nodes = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": SDXL_VAE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": STYLE_POS}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": STYLE_NEG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
    }
    next_id = 7
    model_ref = ["6", 0]  # KSampler 输出 latent，模型引用仍是 checkpoint

    # IPAdapter 分支：在采样前把模型包装
    if ref_image:
        nodes["10"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IPADAPTER_SDXL}}
        nodes["11"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION_H}}
        nodes["12"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        nodes["14"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["10", 0], "image": ["12", 0],
            "weight": ip_weight, "weight_type": "linear",
            "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
            "embeds_scaling": "V only", "clip_vision": ["11", 0]}}
        # 让采样器用被 IPAdapter 包装过的模型
        nodes["6"]["inputs"]["model"] = ["14", 0]

    nodes["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}}
    last_image = ["7", 0]

    if use_upscale:
        nodes["8"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALER}}
        nodes["9"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["8", 0], "image": last_image}}
        last_image = ["9", 0]

    nodes["99"] = {"class_type": "SaveImage", "inputs": {
        "images": last_image, "filename_prefix": prefix}}
    return nodes


def make_hand_mask(image_path, out_path, box, feather=12):
    """生成全图尺寸的手部 mask 图片：手部区域白(重绘)，其余黑(保留)。

    box: (x, y, w, h) 像素坐标。feather: 边缘羽化像素，避免接缝。
    用 PIL 画圆角矩形 + 高斯模糊，保存到 ComfyUI/input 供 LoadImageMask 载入。
    """
    from PIL import Image, ImageDraw, ImageFilter
    im = Image.open(image_path).convert("L")
    W_, H_ = im.size
    mask = Image.new("L", (W_, H_), 0)
    d = ImageDraw.Draw(mask)
    x, y, w, h = box
    r = min(w, h) // 5
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    mask.convert("RGB").save(out_path)
    return out_path


def hand_inpaint(init_image, mask_image, ckpt=WAI, steps=28, cfg=4.5,
                 seed=20261120, denoise=0.55, prompt=None, prefix="style_hand/fixed"):
    """latent 局部重绘（正确方案）：VAEEncode 完整原图 + SetLatentNoiseMask 只加噪 mask 区域。

    不用 VAEEncodeForInpaint —— 它会用灰色填充 mask 区域，denoise 不足时残留灰块。
    init_image/mask_image: ComfyUI/input 目录下的文件名。
    prompt: 可选，手部重绘专用提示词；None 用 STYLE_POS。
    """
    pos_text = prompt or STYLE_POS
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": SDXL_VAE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos_text}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": STYLE_NEG}},
        "10": {"class_type": "LoadImage", "inputs": {"image": init_image}},
        "11": {"class_type": "LoadImageMask", "inputs": {"image": mask_image, "channel": "red"}},
        # 完整编码原图，保留全部上下文
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}},
        # 只对 mask 区域加噪，其余区域无噪 → 不会被改动
        "13": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        "14": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "normal", "denoise": denoise,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["13", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["2", 0]}},
        "99": {"class_type": "SaveImage", "inputs": {
            "images": ["15", 0], "filename_prefix": prefix}},
    }


# ============ 3DCG 手办/渲染质感（Illustrious 底座 diffusers 模型） ============
NOVA_3DCG = "nova3dcg_v40"

POS_3DCG = (
    "1girl, solo, petite, slender, slim waist, pale skin, luminous skin, porcelain skin, "
    "soft matte skin, delicate skin, "
    "pink hair, short hair, blue eyes, half-closed eyes, sleepy expression, "
    "white sailor uniform, short sleeves, navy collar, blue bow, cropped top, "
    "white pleated skirt, white thighhighs, "
    "sitting on bed, dim room, soft warm lighting, indoor, "
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
    "subsurface scattering, 3d, cgi, octane render, "
    "cinematic lighting, soft global illumination, depth of field, highly detailed"
)
NEG_3DCG = (
    "bad quality, worst quality, worst detail, sketch, flat color, cel shading, "
    "monochrome, 2d, anime screencap, dark skin, tan skin, muscular, mature female, "
    "oversaturated, glossy plastic, shiny oily skin, doll joint, "
    "speech bubble, text bubble, speech balloon, comic bubble, watermark, logo, "
    "bad anatomy, bad hands, malformed hands, "
    "extra digits, fewer digits, missing fingers, blurry, lowres, "
    "signature, jpeg artifacts, text"
)


def nova_3dcg(w=832, h=1216, steps=28, cfg=5.0, seed=20261130,
              prompt=None, prefix="style_3dcg/final",
              ref_image=None, ip_weight=0.65):
    """3DCG 手办质感：用 DiffusersLoader 加载 NOVA 3DCG XL（Illustrious 底座）。

    3DCG 与 2D 半厚涂是两个赛道：这里靠 glossy skin / subsurface scattering /
    octane render 等关键词拉出 CG 渲染感，负向必须压制 flat color / cel shading。
    ref_image: 可选，IPAdapter 喂参考图锁角色与画风。
    """
    nodes = {
        "1": {"class_type": "DiffusersLoader", "inputs": {"model_path": NOVA_3DCG}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["1", 1], "text": prompt or POS_3DCG}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["1", 1], "text": NEG_3DCG}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {
            "width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
            "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "99": {"class_type": "SaveImage", "inputs": {
            "images": ["7", 0], "filename_prefix": prefix}},
    }
    if ref_image:
        nodes["10"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": IPADAPTER_SDXL}}
        nodes["11"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION_H}}
        nodes["12"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        nodes["14"] = {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["1", 0], "ipadapter": ["10", 0], "image": ["12", 0],
            "weight": ip_weight, "weight_type": "linear",
            "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
            "embeds_scaling": "V only", "clip_vision": ["11", 0]}}
        nodes["6"]["inputs"]["model"] = ["14", 0]
    return nodes
