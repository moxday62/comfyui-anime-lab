import json
import os

OUT_DIR = r"D:\AI-Video-Setup\workflows"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = "MeinaV7-pruned.safetensors"
LCM = "lcm-lora-sdv1-5.safetensors"
MM = "v3_sd15_mm.ckpt"

POS = ("masterpiece, best quality, anime style, 1girl, solo, cute, detailed eyes, "
       "vibrant colors, clean lineart, full body, gentle smile, hair flowing, soft lighting, "
       "(plain solid color background:1.4), (simple background:1.3), centered character")
NEG = ("worst quality, low quality, blurry, deformed, bad anatomy, extra limbs, ugly, "
       "watermark, text, signature, realistic, 3d, malformed hands, "
       "complex background, scenery, detailed background")


def node(nid, typ, pos, widgets, inputs, outputs, title=None):
    d = {
        "id": nid,
        "type": typ,
        "pos": pos,
        "size": [260, 120],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": n, "type": t, "link": link} for n, t, link in inputs],
        "outputs": [{"name": n, "type": t, "links": links} for n, t, links in outputs],
        "properties": {"Node name for S&R": typ},
        "widgets_values": widgets,
    }
    if title:
        d["title"] = title
    return d


def note(nid, pos, text):
    """前端提示板：只在 UI 里显示，不参与执行。"""
    return {
        "id": nid, "type": "Note", "pos": pos, "size": [320, 150],
        "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
        "properties": {"text": ""}, "widgets_values": [text],
        "color": "#432", "bgcolor": "#653",
    }


# 重要：ComfyUI 的 link id 必须全局唯一，且每条 link 只能有一个目标口。
# 早期版本把同一个 id 同时给了 VAEEncode 和 VAEDecode 的 vae 口，
# 结果导入 UI 后 VAEDecode 的 vae 掉了 → 报 "Required input is missing: vae"。
NOTE_FRAMES = ("帧数必须 ≥ 16！\n\nAnimateDiff 运动模块没有时序上下文就会崩：\n"
               "1 帧 = 纯彩色噪点\n8 帧 = 霓虹条纹怪图\n16 帧起才正常。\n\n"
               "单帧出图请用 06_柔光半厚涂 或 5_画风切换.bat。")


def workflow_txt2anime():
    nodes = [
        node(1, "CheckpointLoaderSimple", [40, 100], [CKPT], [],
             [("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [2]), ("VAE", "VAE", [11])],
             "1  二次元大模型"),
        node(2, "LoraLoader", [360, 100], [LCM, 1.0, 1.0],
             [("model", "MODEL", 1), ("clip", "CLIP", 2)],
             [("MODEL", "MODEL", [3]), ("CLIP", "CLIP", [4, 5])],
             "2  LCM 蒸馏加速 LoRA (4-8步)"),
        node(3, "ADE_AnimateDiffLoaderGen1", [680, 100], [MM, "autoselect"],
             [("model", "MODEL", 3)],
             [("MODEL", "MODEL", [6])],
             "3  AnimateDiff 运动模块"),
        node(4, "CLIPTextEncode", [360, 330], [POS],
             [("clip", "CLIP", 4)], [("CONDITIONING", "CONDITIONING", [7])],
             "正向提示词 (在这里写动作)"),
        node(5, "CLIPTextEncode", [360, 530], [NEG],
             [("clip", "CLIP", 5)], [("CONDITIONING", "CONDITIONING", [8])],
             "负向提示词"),
        node(6, "EmptyLatentImage", [680, 330], [512, 512, 16], [],
             [("LATENT", "LATENT", [9])],
             "4  分辨率 / 帧数 (必须≥16，调小会崩)"),
        node(7, "KSampler", [1000, 100], [123456789, "randomize", 8, 1.0,
             "euler", "normal", 1.0],
             [("model", "MODEL", 6), ("positive", "CONDITIONING", 7),
              ("negative", "CONDITIONING", 8), ("latent_image", "LATENT", 9)],
             [("LATENT", "LATENT", [10])],
             "5  采样器 (LCM: 步数8 / CFG 1)"),
        node(8, "VAEDecode", [1320, 100], [],
             [("samples", "LATENT", 10), ("vae", "VAE", 11)],
             [("IMAGE", "IMAGE", [12])], "解码"),
        node(9, "SaveImage", [1600, 100], ["anime_sprite/frame"],
             [("images", "IMAGE", 12)], [],
             "6  输出帧 → ComfyUI/output/anime_sprite/"),
        note(10, [40, 620], NOTE_FRAMES),
    ]
    links = [
        [1, 1, 0, 2, 0, "MODEL"],
        [2, 1, 1, 2, 1, "CLIP"],
        [3, 2, 0, 3, 0, "MODEL"],
        [4, 2, 1, 4, 0, "CLIP"],
        [5, 2, 1, 5, 0, "CLIP"],
        [6, 3, 0, 7, 0, "MODEL"],
        [7, 4, 0, 7, 1, "CONDITIONING"],
        [8, 5, 0, 7, 2, "CONDITIONING"],
        [9, 6, 0, 7, 3, "LATENT"],
        [10, 7, 0, 8, 0, "LATENT"],
        [11, 1, 2, 8, 1, "VAE"],
        [12, 8, 0, 9, 0, "IMAGE"],
    ]
    for i, n in enumerate(nodes):
        n["order"] = i
    return {
        "id": "txt2anime-lcm-ad", "revision": 0,
        "last_node_id": 10, "last_link_id": 12,
        "nodes": nodes, "links": links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
    }


def workflow_img2anime():
    nodes = [
        node(1, "CheckpointLoaderSimple", [40, 100], [CKPT], [],
             [("MODEL", "MODEL", [1]), ("CLIP", "CLIP", [2]), ("VAE", "VAE", [12, 14])],
             "1  二次元大模型"),
        node(2, "LoraLoader", [360, 100], [LCM, 1.0, 1.0],
             [("model", "MODEL", 1), ("clip", "CLIP", 2)],
             [("MODEL", "MODEL", [3]), ("CLIP", "CLIP", [4, 5])],
             "2  LCM 蒸馏加速 LoRA"),
        node(3, "ADE_AnimateDiffLoaderGen1", [680, 100], [MM, "autoselect"],
             [("model", "MODEL", 3)], [("MODEL", "MODEL", [6])],
             "3  AnimateDiff 运动模块"),
        node(4, "CLIPTextEncode", [360, 330], [POS],
             [("clip", "CLIP", 4)], [("CONDITIONING", "CONDITIONING", [7])],
             "正向提示词 (描述想要的动作)"),
        node(5, "CLIPTextEncode", [360, 530], [NEG],
             [("clip", "CLIP", 5)], [("CONDITIONING", "CONDITIONING", [8])],
             "负向提示词"),
        node(6, "LoadImage", [680, 330], ["example.png", "input"], [],
             [("IMAGE", "IMAGE", [9]), ("MASK", "MASK", [])],
             "4  上传你的角色立绘/单帧 sprite"),
        node(7, "VAEEncode", [980, 330], [],
             [("pixels", "IMAGE", 9), ("vae", "VAE", 12)],
             [("LATENT", "LATENT", [10])], "编码参考图"),
        node(8, "KSampler", [1000, 100], [123456789, "randomize", 8, 1.0,
             "euler", "normal", 1.0],
             [("model", "MODEL", 6), ("positive", "CONDITIONING", 7),
              ("negative", "CONDITIONING", 8), ("latent_image", "LATENT", 10)],
             [("LATENT", "LATENT", [11])],
             "5  采样器 (步数8 / CFG 1)"),
        node(9, "VAEDecode", [1320, 100], [],
             [("samples", "LATENT", 11), ("vae", "VAE", 14)],
             [("IMAGE", "IMAGE", [13])], "解码"),
        node(10, "SaveImage", [1600, 100], ["anime_i2a/frame"],
             [("images", "IMAGE", 13)], [],
             "6  输出帧 → ComfyUI/output/anime_i2a/"),
        note(11, [40, 620], NOTE_FRAMES),
    ]
    links = [
        [1, 1, 0, 2, 0, "MODEL"],
        [2, 1, 1, 2, 1, "CLIP"],
        [3, 2, 0, 3, 0, "MODEL"],
        [4, 2, 1, 4, 0, "CLIP"],
        [5, 2, 1, 5, 0, "CLIP"],
        [6, 3, 0, 8, 0, "MODEL"],
        [7, 4, 0, 8, 1, "CONDITIONING"],
        [8, 5, 0, 8, 2, "CONDITIONING"],
        [9, 6, 0, 7, 0, "IMAGE"],
        [10, 7, 0, 8, 3, "LATENT"],
        [11, 8, 0, 9, 0, "LATENT"],
        [12, 1, 2, 7, 1, "VAE"],
        [13, 9, 0, 10, 0, "IMAGE"],
        [14, 1, 2, 9, 1, "VAE"],
    ]
    for i, n in enumerate(nodes):
        n["order"] = i
    return {
        "id": "img2anime-lcm-ad", "revision": 0,
        "last_node_id": 11, "last_link_id": 14,
        "nodes": nodes, "links": links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
    }


for name, wf in [
    ("01_文生二次元动画.json", workflow_txt2anime()),
    ("02_图生二次元动画.json", workflow_img2anime()),
]:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print("written:", path)
