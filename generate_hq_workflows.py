# -*- coding: utf-8 -*-
"""生成三套高质量工作流：SDXL立绘(超分) / SDXL动画 / FLUX极致单帧"""
import json
import os

OUT_DIR = r"D:\AI-Video-Setup\workflows_hq"
os.makedirs(OUT_DIR, exist_ok=True)

ILXL = "Illustrious-XL-v2.0.safetensors"
SDXL_VAE = "sdxl_vae.safetensors"
AD_SDXL = "mm_sdxl_v10_beta.ckpt"
UPSCALER = "4x_NMKD-Superscale-SP_178000_G.pth"
FLUX_FP8 = "flux1-dev-fp8.safetensors"
T5 = "t5xxl_fp8_e4m3fn.safetensors"
CLIP_L = "clip_l.safetensors"
FLUX_VAE = "ae.safetensors"

POS = ("masterpiece, best quality, ultra-detailed, anime style, 1girl, solo, "
       "cute, beautiful detailed eyes, long hair, cat ears, maid outfit, "
       "frills, lace, ribbon, holding tray, smiling, waving, "
       "soft lighting, vibrant colors, sharp focus, simple green background")
NEG = ("worst quality, low quality, blurry, jpeg artifacts, deformed, bad anatomy, "
       "extra limbs, bad hands, missing fingers, ugly, watermark, text, signature, "
       "realistic, 3d, nsfw")


def node(nid, typ, pos, size, widgets, inputs, outputs, title=None, mode=0):
    d = {
        "id": nid, "type": typ, "pos": pos, "size": size,
        "flags": {}, "order": 0, "mode": mode,
        "inputs": [{"name": n, "type": t, "link": link} for n, t, link in inputs],
        "outputs": [{"name": n, "type": t, "links": links} for n, t, links in outputs],
        "properties": {"Node name for S&R": typ},
        "widgets_values": widgets,
    }
    if title:
        d["title"] = title
    return d


def finish(nodes, links, name, info=""):
    for i, n in enumerate(nodes):
        n["order"] = i
    wf = {"id": name, "revision": 0, "last_node_id": max(n["id"] for n in nodes),
          "last_link_id": max(l[0] for l in links), "nodes": nodes, "links": links,
          "groups": [], "config": {}, "extra": {}, "version": 0.4}
    path = os.path.join(OUT_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print("written:", path, f"({info})")


# ---------- 1. SDXL 高质量立绘（Hires.fix 两段式 + 超分） ----------
def wf_sdxl_portrait():
    nodes = [
        node(1, "CheckpointLoaderSimple", [40, 60], [280, 110], [ILXL], [],
             [("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [2]), ("VAE", "VAE", [])],
             "1  Illustrious-XL 二次元底模"),
        node(2, "VAELoader", [40, 220], [280, 80], [SDXL_VAE], [],
             [("VAE", "VAE", [10, 13])], "SDXL 专用 VAE (fp16-fix)"),
        node(3, "CLIPTextEncode", [360, 60], [340, 140], [POS],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [3])],
             "正向提示词 (可写角色/服装/动作)"),
        node(4, "CLIPTextEncode", [360, 230], [340, 140], [NEG],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [4])],
             "负向提示词"),
        node(5, "EmptyLatentImage", [360, 410], [300, 110], [832, 1216, 1], [],
             [("LATENT", "LATENT", [5])], "2  基础分辨率 (二次元竖版)"),
        node(6, "KSampler", [740, 60], [300, 260],
             [987654321, "randomize", 32, 7.0, "euler", "normal", 1.0],
             [("model", "MODEL", 1), ("positive", "CONDITIONING", 3),
              ("negative", "CONDITIONING", 4), ("latent_image", "LATENT", 5)],
             [("LATENT", "LATENT", [6])], "3  一段采样 (32步,满画质)"),
        node(7, "LatentUpscaleBy", [740, 380], [300, 90], [1.5, "nearest-exact"],
             [("samples", "LATENT", 6)], [("LATENT", "LATENT", [7])],
             "Hires.fix 放大 1.5x"),
        node(8, "KSampler", [1080, 60], [300, 260],
             [987654321, "fixed", 20, 7.0, "euler", "normal", 0.5],
             [("model", "MODEL", 1), ("positive", "CONDITIONING", 3),
              ("negative", "CONDITIONING", 4), ("latent_image", "LATENT", 7)],
             [("LATENT", "LATENT", [8])], "4  二段精修 (20步,denoise 0.5)"),
        node(9, "VAEDecode", [1420, 60], [220, 80], [],
             [("samples", "LATENT", 8), ("vae", "VAE", 10)],
             [("IMAGE", "IMAGE", [9])], "解码"),
        node(10, "UpscaleModelLoader", [1420, 200], [300, 80], [UPSCALER], [],
             [("UPSCALE_MODEL", "UPSCALE_MODEL", [11])], "4x NMKD 二次元超分"),
        node(11, "ImageUpscaleWithModel", [1760, 60], [260, 80], [],
             [("upscale_model", "UPSCALE_MODEL", 11), ("image", "IMAGE", 9)],
             [("IMAGE", "IMAGE", [12])], "5  超分到 ~5K"),
        node(12, "SaveImage", [2060, 60], [300, 330], ["hq_portrait/final"],
             [("images", "IMAGE", 12)], [], "6  输出 hq_portrait/"),
    ]
    links = [
        [1, 1, 0, 6, 0, "MODEL"], [2, 1, 1, 3, 0, "CLIP"], [3, 3, 0, 6, 1, "CONDITIONING"],
        [4, 4, 0, 6, 2, "CONDITIONING"], [5, 5, 0, 6, 3, "LATENT"],
        [6, 6, 0, 7, 0, "LATENT"], [7, 7, 0, 8, 3, "LATENT"],
        [8, 1, 0, 8, 0, "MODEL"], [9, 3, 0, 8, 1, "CONDITIONING"],
        [10, 4, 0, 8, 2, "CONDITIONING"],
        [11, 8, 0, 9, 0, "LATENT"], [12, 2, 0, 9, 1, "VAE"],
        [13, 9, 0, 11, 1, "IMAGE"], [14, 10, 0, 11, 0, "UPSCALE_MODEL"],
        [15, 11, 0, 12, 0, "IMAGE"],
    ]
    # 修正节点输入链接号（inputs 里的 link 值需对应 links 的 id）
    nodes[5]["inputs"] = [("model", "MODEL", 8), ("positive", "CONDITIONING", 9),
                          ("negative", "CONDITIONING", 10), ("latent_image", "LATENT", 7)]
    finish(nodes, links, "03_SDXL高质量立绘_超分", "832x1216 -> 5K")


# ---------- 2. SDXL 动画 ----------
def wf_sdxl_anim():
    nodes = [
        node(1, "CheckpointLoaderSimple", [40, 60], [280, 110], [ILXL], [],
             [("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [2]), ("VAE", "VAE", [11])],
             "1  Illustrious-XL 底模"),
        node(2, "VAELoader", [40, 220], [280, 80], [SDXL_VAE], [],
             [("VAE", "VAE", [11])], "SDXL VAE"),
        node(3, "CLIPTextEncode", [360, 60], [340, 140], [POS],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [3])], "正向(写动作)"),
        node(4, "CLIPTextEncode", [360, 230], [340, 140], [NEG],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [4])], "负向"),
        node(5, "ADE_AnimateDiffLoaderGen1", [360, 410], [340, 110], [AD_SDXL, "autoselect"],
             [("model", "MODEL", 1)], [("MODEL", "MODEL", [5])], "2  AnimateDiff SDXL 运动模块"),
        node(6, "EmptyLatentImage", [740, 410], [300, 110], [768, 768, 16], [],
             [("LATENT", "LATENT", [7])], "3  分辨率/帧数 (显存吃紧调小)"),
        node(7, "KSampler", [740, 60], [300, 260],
             [42, "randomize", 28, 7.0, "euler", "normal", 1.0],
             [("model", "MODEL", 5), ("positive", "CONDITIONING", 3),
              ("negative", "CONDITIONING", 4), ("latent_image", "LATENT", 7)],
             [("LATENT", "LATENT", [8])], "4  采样 (28步)"),
        node(8, "VAEDecode", [1080, 60], [220, 80], [],
             [("samples", "LATENT", 8), ("vae", "VAE", 11)],
             [("IMAGE", "IMAGE", [9])], "解码"),
        node(9, "SaveImage", [1340, 60], [300, 330], ["hq_anim/frame"],
             [("images", "IMAGE", 9)], [], "5  输出 hq_anim/"),
    ]
    links = [
        [1, 1, 0, 5, 0, "MODEL"], [2, 1, 1, 3, 0, "CLIP"], [3, 3, 0, 7, 1, "CONDITIONING"],
        [4, 4, 0, 7, 2, "CONDITIONING"], [5, 5, 0, 7, 0, "MODEL"],
        [6, 6, 0, 7, 3, "LATENT"], [7, 7, 0, 8, 0, "LATENT"],
        [8, 2, 0, 8, 1, "VAE"], [9, 8, 0, 9, 0, "IMAGE"],
    ]
    finish(nodes, links, "04_SDXL高质量动画", "768x768 x16帧")


# ---------- 3. FLUX 极致单帧 ----------
FLUX_POS = ("A masterpiece, ultra-detailed anime illustration of a cute girl with long "
            "dark hair and black cat ears, wearing a frilled black and white maid outfit "
            "with lace and ribbons, holding a silver tray, smiling and waving, soft studio "
            "lighting, sharp focus, simple solid green background, beautiful detailed eyes.")


def wf_flux():
    nodes = [
        node(1, "UNETLoader", [40, 60], [320, 80], [FLUX_FP8, "default"], [],
             [("MODEL", "MODEL", [1])], "1  FLUX.1-dev fp8 官方版"),
        node(2, "DualCLIPLoader", [40, 180], [320, 110], [CLIP_L, T5, "flux"], [],
             [("CLIP", "CLIP", [2, 3])], "2  CLIP-L + T5xxl"),
        node(3, "VAELoader", [40, 330], [320, 80], [FLUX_VAE], [],
             [("VAE", "VAE", [9])], "FLUX VAE"),
        node(4, "CLIPTextEncode", [400, 60], [360, 160], [FLUX_POS],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [4])],
             "正向提示词 (FLUX 用自然语言整句)"),
        node(5, "CLIPTextEncode", [400, 250], [360, 140], [""],
             [("clip", "CLIP", 3)], [("CONDITIONING", "CONDITIONING", [5])],
             "负向提示词 (FLUX 必须留空)"),
        node(6, "EmptySD3LatentImage", [400, 430], [320, 110], [1024, 1024, 1], [],
             [("LATENT", "LATENT", [6])], "3  1024x1024 (16通道)"),
        node(10, "FluxGuidance", [800, 360], [260, 80], [3.5],
             [("conditioning", "CONDITIONING", 4)],
             [("CONDITIONING", "CONDITIONING", [7])], "提示词引导 3.5"),
        node(7, "KSampler", [800, 60], [320, 260],
             [1234, "fixed", 28, 1.0, "euler", "simple", 1.0],
             [("model", "MODEL", 1), ("positive", "CONDITIONING", 7),
              ("negative", "CONDITIONING", 5), ("latent_image", "LATENT", 6)],
             [("LATENT", "LATENT", [8])], "4  FLUX 采样 (28步,CFG 1.0)"),
        node(8, "VAEDecode", [1160, 60], [220, 80], [],
             [("samples", "LATENT", 8), ("vae", "VAE", 9)],
             [("IMAGE", "IMAGE", [10])], "解码"),
        node(9, "SaveImage", [1420, 60], [320, 330], ["flux_portrait/final"],
             [("images", "IMAGE", 10)], [], "5  输出 flux_portrait/"),
    ]
    links = [
        [1, 1, 0, 7, 0, "MODEL"], [2, 2, 0, 4, 0, "CLIP"], [3, 2, 0, 5, 0, "CLIP"],
        [4, 4, 0, 10, 0, "CONDITIONING"], [5, 5, 0, 7, 2, "CONDITIONING"],
        [6, 6, 0, 7, 3, "LATENT"], [7, 10, 0, 7, 1, "CONDITIONING"],
        [8, 7, 0, 8, 0, "LATENT"], [9, 3, 0, 8, 1, "VAE"],
        [10, 8, 0, 9, 0, "IMAGE"],
    ]
    finish(nodes, links, "05_FLUX极致单帧", "1024x1024")


# ---------- 4. 柔光半厚涂画风（画风对齐，质量优先） ----------
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


def wf_soft_painterly():
    """实测：Hires.fix 二段采样会重绘构图导致崩坏，已改为一步出图 + 纯超分可选。"""
    nodes = [
        node(1, "CheckpointLoaderSimple", [40, 60], [280, 110], [ILXL], [],
             [("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [2]), ("VAE", "VAE", [])],
             "1  Illustrious-XL 二次元底模"),
        node(2, "VAELoader", [40, 220], [280, 80], [SDXL_VAE], [],
             [("VAE", "VAE", [5])], "SDXL 专用 VAE"),
        node(3, "CLIPTextEncode", [360, 60], [340, 140], [SOFT_POS],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [3])],
             "正向 (柔光半厚涂风格词固定)"),
        node(4, "CLIPTextEncode", [360, 230], [340, 140], [SOFT_NEG],
             [("clip", "CLIP", 2)], [("CONDITIONING", "CONDITIONING", [4])],
             "负向 (禁 sketch/lineart)"),
        node(5, "EmptyLatentImage", [360, 410], [300, 110], [832, 1088, 1], [],
             [("LATENT", "LATENT", [5])], "2  832x1088 (实测稳)"),
        node(6, "KSampler", [740, 60], [300, 260],
             [20261118, "fixed", 32, 4.5, "euler", "normal", 1.0],
             [("model", "MODEL", 1), ("positive", "CONDITIONING", 3),
              ("negative", "CONDITIONING", 4), ("latent_image", "LATENT", 5)],
             [("LATENT", "LATENT", [6])], "3  一步采样 (32步, CFG 4.5)"),
        node(7, "VAEDecode", [1060, 60], [220, 80], [],
             [("samples", "LATENT", 6), ("vae", "VAE", 5)],
             [("IMAGE", "IMAGE", [7])], "解码"),
        node(8, "SaveImage", [1300, 60], [300, 330], ["hq_soft/final"],
             [("images", "IMAGE", 7)], [], "4  输出 hq_soft/"),
    ]
    links = [
        [1, 1, 0, 6, 0, "MODEL"], [2, 2, 0, 7, 1, "VAE"],
        [3, 1, 1, 3, 0, "CLIP"], [4, 1, 1, 4, 0, "CLIP"],
        [5, 3, 0, 6, 1, "CONDITIONING"], [6, 4, 0, 6, 2, "CONDITIONING"],
        [7, 5, 0, 6, 3, "LATENT"], [8, 6, 0, 7, 0, "LATENT"],
        [9, 7, 0, 8, 0, "IMAGE"],
    ]
    # 修正节点输入链接号
    nodes[5]["inputs"] = [("model", "MODEL", 1), ("positive", "CONDITIONING", 5),
                          ("negative", "CONDITIONING", 6), ("latent_image", "LATENT", 7)]
    nodes[6]["inputs"] = [("samples", "LATENT", 8), ("vae", "VAE", 2)]
    nodes[7]["inputs"] = [("images", "IMAGE", 9)]
    finish(nodes, links, "06_柔光半厚涂_画风对齐", "832x1088, 一步出图, 无 Hires.fix")


wf_sdxl_portrait()
wf_sdxl_anim()
wf_flux()
wf_soft_painterly()
