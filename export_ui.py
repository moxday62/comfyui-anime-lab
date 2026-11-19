# -*- coding: utf-8 -*-
"""把「API 格式」工作流导出成「ComfyUI 界面格式（UI JSON）」。

为什么要这个脚本：
  API 格式（{"1":{"class_type":...,"inputs":...}}）只给 /prompt 用，
  丢进 ComfyUI 网页是打不开的；UI 格式又容易手写错连线
  （旧生成脚本就踩过：同一个 link id 给了两个输入口，导入后 vae 线断了，
   跑到一半报 "Required input is missing: vae"；还有把 inputs 写成旧版列表
   格式的，界面直接不认）。

  这里用「API 语义图 → UI 图」自动转换：link id 自动分配保证唯一，
  输入口顺序按 ComfyUI 节点定义排，控件值顺序按 WIDGET_ORDER 排，
  导出的 JSON 再交给 check_workflows.py 体检。

用法：
  python export_ui.py            重建 workflows_hq/03、04、06
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)   # python_embeded 的 ._pth 不自动加脚本目录

import hq_workflows_api as HQ

OUT = os.path.join(BASE, "workflows_hq")

# ComfyUI 节点定义里的输入口顺序（错序会让连线挂到错误槽位）
INPUT_ORDER = {
    "CheckpointLoaderSimple": [],
    "DiffusersLoader": [],
    "VAELoader": [],
    "CLIPTextEncode": ["clip"],
    "EmptyLatentImage": [],
    "EmptySD3LatentImage": [],
    "KSampler": ["model", "positive", "negative", "latent_image"],
    "LatentUpscaleBy": ["samples"],
    "VAEDecode": ["samples", "vae"],
    "VAEEncode": ["pixels", "vae"],
    "SaveImage": ["images"],
    "UpscaleModelLoader": [],
    "ImageUpscaleWithModel": ["upscale_model", "image"],
    "UNETLoader": [],
    "DualCLIPLoader": [],
    "FluxGuidance": ["conditioning"],
    "ConditioningZeroOut": ["conditioning"],
    "ADE_AnimateDiffLoaderGen1": ["model"],
    "LoraLoader": ["model", "clip"],
    "LoadImage": [],
    "SetLatentNoiseMask": ["samples", "mask"],
    "LoadImageMask": [],
    "CLIPVisionLoader": [],
    "IPAdapterModelLoader": [],
    "IPAdapterAdvanced": ["model", "ipadapter", "image", "clip_vision"],
}

# 控件（widgets_values）的顺序 —— 必须是 ComfyUI 里的显示顺序
WIDGET_ORDER = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "DiffusersLoader": ["model_path"],
    "VAELoader": ["vae_name"],
    "CLIPTextEncode": ["text"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
    "LatentUpscaleBy": ["upscale_method", "scale_by"],
    "VAEDecode": [],
    "VAEEncode": [],
    "SaveImage": ["filename_prefix"],
    "UpscaleModelLoader": ["model_name"],
    "ImageUpscaleWithModel": [],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type"],
    "FluxGuidance": ["guidance"],
    "ConditioningZeroOut": [],
    "ADE_AnimateDiffLoaderGen1": ["model_name", "beta_schedule"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LoadImage": ["image", "upload"],
    "SetLatentNoiseMask": [],
    "LoadImageMask": ["image", "channel"],
    "CLIPVisionLoader": ["clip_name"],
    "IPAdapterModelLoader": ["ipadapter_file"],
    "IPAdapterAdvanced": ["weight", "weight_type", "combine_embeds",
                          "start_at", "end_at", "embeds_scaling"],
}

OUTPUT_TYPES = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "DiffusersLoader": ["MODEL", "CLIP", "VAE"],
    "VAELoader": ["VAE"],
    "CLIPTextEncode": ["CONDITIONING"],
    "EmptyLatentImage": ["LATENT"],
    "EmptySD3LatentImage": ["LATENT"],
    "KSampler": ["LATENT"],
    "LatentUpscaleBy": ["LATENT"],
    "VAEDecode": ["IMAGE"],
    "VAEEncode": ["LATENT"],
    "SaveImage": [],
    "UpscaleModelLoader": ["UPSCALE_MODEL"],
    "ImageUpscaleWithModel": ["IMAGE"],
    "UNETLoader": ["MODEL"],
    "DualCLIPLoader": ["CLIP"],
    "FluxGuidance": ["CONDITIONING"],
    "ConditioningZeroOut": ["CONDITIONING"],
    "ADE_AnimateDiffLoaderGen1": ["MODEL"],
    "LoraLoader": ["MODEL", "CLIP"],
    "LoadImage": ["IMAGE", "MASK"],
    "SetLatentNoiseMask": ["LATENT"],
    "LoadImageMask": ["MASK"],
    "CLIPVisionLoader": ["CLIP_VISION"],
    "IPAdapterModelLoader": ["IPADAPTER"],
    "IPAdapterAdvanced": ["MODEL"],
}

TITLES = {
    "CheckpointLoaderSimple": "底模",
    "DiffusersLoader": "3DCG 底模（diffusers 目录）",
    "VAELoader": "VAE",
    "CLIPTextEncode": "提示词",
    "EmptyLatentImage": "分辨率 / 帧数（帧数必须≥16）",
    "EmptySD3LatentImage": "分辨率",
    "KSampler": "采样器",
    "LatentUpscaleBy": "潜空间放大",
    "VAEDecode": "解码",
    "SaveImage": "保存",
    "UpscaleModelLoader": "超分模型",
    "ImageUpscaleWithModel": "超分",
    "UNETLoader": "UNET",
    "DualCLIPLoader": "双 CLIP",
    "FluxGuidance": "FLUX 引导强度",
    "ADE_AnimateDiffLoaderGen1": "AnimateDiff 运动模块",
    "LoraLoader": "LoRA",
    "LoadImage": "参考图（换成自己的图）",
    "IPAdapterModelLoader": "IPAdapter 模型",
    "CLIPVisionLoader": "CLIP Vision",
    "IPAdapterAdvanced": "IPAdapter 权重",
}


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def convert(api):
    ids = sorted(api, key=lambda k: int(k))
    nodes = {}
    for nid in ids:
        typ = api[nid]["class_type"]
        widgets = []
        for name in WIDGET_ORDER.get(typ, []):
            v = api[nid]["inputs"].get(name)
            if v is None and name == "upload":
                v = "image"
            widgets.append(v)
            if typ == "KSampler" and name == "seed":
                widgets.append("randomize")   # 前端专用：种子控制项
        nodes[nid] = {
            "id": int(nid),
            "type": typ,
            "pos": [0, 0],
            "size": [300, 130],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [{"name": n, "type": "*", "link": None}
                       for n in INPUT_ORDER.get(typ, [])
                       if n in api[nid]["inputs"]],
            "outputs": [{"name": t, "type": t, "links": []}
                        for t in OUTPUT_TYPES.get(typ, [])],
            "properties": {"Node name for S&R": typ},
            "widgets_values": widgets,
            "title": TITLES.get(typ, typ),
        }

    links = []
    out_links = {}
    link_id = 0
    for nid in ids:
        typ = api[nid]["class_type"]
        names = [i["name"] for i in nodes[nid]["inputs"]]
        for name in names:
            v = api[nid]["inputs"][name]
            if not _is_link(v):
                continue
            src, sslot = int(v[0]), int(v[1])
            link_id += 1
            dst_slot = names.index(name)
            ltype = OUTPUT_TYPES[api[str(src)]["class_type"]][sslot]
            links.append([link_id, src, sslot, int(nid), dst_slot, ltype])
            nodes[nid]["inputs"][dst_slot]["type"] = ltype
            nodes[nid]["inputs"][dst_slot]["link"] = link_id
            out_links.setdefault((src, sslot), []).append(link_id)

    for (src, sslot), ls in out_links.items():
        node = nodes[str(src)]
        if sslot < len(node["outputs"]):
            node["outputs"][sslot]["links"] = ls

    # 分层布局：按依赖深度排成列
    depth = {}

    def depth_of(nid):
        if nid in depth:
            return depth[nid]
        p = 0
        for name in INPUT_ORDER.get(api[nid]["class_type"], []):
            v = api[nid]["inputs"].get(name)
            if _is_link(v):
                p = max(p, depth_of(v[0]) + 1)
        depth[nid] = p
        return p

    col = {}
    for nid in ids:
        c = depth_of(nid)
        r = col.get(c, 0)
        col[c] = r + 1
        nodes[nid]["pos"] = [40 + c * 340, 80 + r * 210]
    for i, nid in enumerate(ids):
        nodes[nid]["order"] = i

    return nodes, links, link_id


def note(nid, pos, text):
    return {
        "id": nid, "type": "Note", "pos": pos, "size": [380, 180],
        "flags": {}, "order": 99, "mode": 0, "inputs": [], "outputs": [],
        "properties": {"text": ""}, "widgets_values": [text],
        "color": "#432", "bgcolor": "#653",
    }


def export(api, filename, note_text, wf_id):
    nodes, links, last_link = convert(api)
    last_node = max(int(k) for k in api)
    nd = note(last_node + 1, [40, 80 + max(1, len(nodes) // 3) * 210], note_text)
    all_nodes = [nodes[k] for k in sorted(nodes, key=lambda x: int(x))] + [nd]
    wf = {
        "id": wf_id, "revision": 0,
        "last_node_id": nd["id"], "last_link_id": last_link,
        "nodes": all_nodes, "links": links, "groups": [],
        "config": {}, "extra": {}, "version": 0.4,
    }
    path = os.path.join(OUT, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2)
    print("已生成 %s（%d 节点 / %d 连线）" % (os.path.basename(path), len(all_nodes), len(links)))
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    export(
        HQ.sdxl_portrait(), "03_SDXL高质量立绘_超分.json",
        "SDXL 立绘：832x1216 → Hires.fix(0.5) → 4x 超分，约 95 秒。\n"
        "注意：这条链的二段重采样会重画构图，\n"
        "追求稳定的立绘请用 06 或 5_画风切换.bat。",
        "sdxl-portrait-hires")
    export(
        HQ.sdxl_anim(), "04_SDXL高质量动画.json",
        "SDXL + AnimateDiff：768x768 / 16 帧 / 28 步，约 150 秒。\n\n"
        "帧数必须 ≥ 16！\n1 帧 = 纯彩色噪点\n8 帧 = 霓虹条纹怪图\n"
        "输出在 output/hq_anim/，用 3_打包精灵图.bat 合成 sprite/GIF。",
        "sdxl-anim-ad")
    export(
        HQ.soft_painterly(), "06_柔光半厚涂_画风对齐.json",
        "柔光半厚涂定稿版：一步出图 + 纯超分，刻意不做 Hires.fix。\n"
        "832x1088 / 32 步 / CFG 4.5 / euler，约 50 秒。\n\n"
        "喂图锁角色、修手请用 5_画风切换.bat（IPAdapter + latent 局部重绘）。",
        "soft-painterly-final")

    import style_workflows_api as SW

    export(
        SW.nova_3dcg(), "07_3DCG手办质感.json",
        "3DCG 手办质感：NOVA 3DCG XL v40（Illustrious 底座，diffusers 目录加载）。\n"
        "832x1216 / 28 步 / CFG 5.0 / euler_ancestral，约 45-60 秒。\n\n"
        "质感靠这几个词：subsurface scattering、octane render、soft matte skin、\n"
        "cinematic lighting。负向必须压 flat color / cel shading / speech bubble。\n"
        "坑：glossy 类词会让皮肤变硅胶，已从提示词里拿掉。",
        "3dcg-nova-v40")

    export(
        SW.nova_3dcg(ref_image="example.png"),
        "08_3DCG_喂图锁角色.json",
        "3DCG + 喂图锁角色：比 07 多一条 IPAdapter 链。\n\n"
        "把「参考图」节点里的 example.png 换成自己的图\n"
        "（图要先放进 ComfyUI\\input\\ 目录），权重 0.65 是验证过的平衡点：\n"
        "  0.5 以下 几乎看不出效果\n"
        "  0.65    锁住长相和服装，姿势仍自由\n"
        "  0.85 以上 连构图一起抄，手也更容易崩",
        "3dcg-nova-ref")


if __name__ == "__main__":
    main()
