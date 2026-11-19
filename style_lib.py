# -*- coding: utf-8 -*-
"""画风权重库（style profile）—— 一个画风一份配置，切换即换档。

设计原则（重要）：
  「权重库」不等于给每个画风复制一份 GB 级模型文件。
  同一个底模可以派生出多个画风档，真正决定画风的是：
      底模 + 提示词模板 + 采样参数 + LoRA 叠加 + IPAdapter 权重 + 后期策略
  所以这里做的是「配置档」：切画风 = 换一个 JSON，不搬文件、不额外占盘。
  将来训了风格 LoRA，只要在档里 loras 数组加一条就能叠加，不用改架构。

用法：
  python style_lib.py list                     列出所有画风档
  python style_lib.py show soft                看某档完整参数
  python style_lib.py use soft                 切换到某画风（写入当前档）
  python style_lib.py run                      用当前画风出图
  python style_lib.py run 3dcg --ref xxx.jpg   指定画风 + 喂图锁角色
  python style_lib.py run soft --hand --upscale 出图后修手 + 4x 超分
  python style_lib.py check                    检查所有档引用的模型文件是否存在
  python style_lib.py export --all             导出 API 格式工作流到 profiles/api/
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request

BASE = r"D:\AI-Video-Setup"
PROFILE_DIR = os.path.join(BASE, "profiles")
CURRENT_FILE = os.path.join(PROFILE_DIR, "_current.txt")
COMFY = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI"
MODELS = os.path.join(COMFY, "models")
INPUT_DIR = os.path.join(COMFY, "input")
OUTPUT_DIR = os.path.join(COMFY, "output")
SERVER = "http://127.0.0.1:8188"

MODEL_DIRS = {
    "checkpoint": "checkpoints",
    "vae": "vae",
    "lora": "loras",
    "animatediff": "animatediff_models",
    "upscale": "upscale_models",
    "ipadapter": "ipadapter",
    "clip_vision": "clip_vision",
}


# ---------------------------------------------------------------- 图构建
class Graph(object):
    """极简 ComfyUI 图构建器：自动分配节点 id。"""

    def __init__(self):
        self.nodes = {}
        self._n = 0

    def node(self, class_type, **inputs):
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    def ref(self, nid, index=0):
        return [nid, index]


def _load_model(g, p):
    """加载底模并串好 LoRA / AnimateDiff，返回 (model, clip, vae) 三个引用。"""
    loader = p.get("loader", "checkpoint")
    if loader == "diffusers":
        nid = g.node("DiffusersLoader", model_path=p["model"])
        model, clip, vae = g.ref(nid, 0), g.ref(nid, 1), g.ref(nid, 2)
    else:
        nid = g.node("CheckpointLoaderSimple", ckpt_name=p["model"])
        model, clip, vae = g.ref(nid, 0), g.ref(nid, 1), g.ref(nid, 2)
        if p.get("vae"):
            vid = g.node("VAELoader", vae_name=p["vae"])
            vae = g.ref(vid, 0)

    for lora in p.get("loras") or []:
        nid = g.node(
            "LoraLoader", model=model, clip=clip, lora_name=lora["file"],
            strength_model=lora.get("strength_model", 0.8),
            strength_clip=lora.get("strength_clip", lora.get("strength_model", 0.8)),
        )
        model, clip = g.ref(nid, 0), g.ref(nid, 1)

    ad = p.get("animatediff")
    if ad:
        nid = g.node(
            "ADE_AnimateDiffLoaderGen1", model=model, model_name=ad["model"],
            beta_schedule=ad.get("beta_schedule", "autoselect"),
        )
        model = g.ref(nid, 0)

    return model, clip, vae


def _apply_ipadapter(g, p, model, ref_image, ip_weight=None):
    """可选：把 IPAdapter 串到模型上，用参考图锁角色/画风。"""
    ip = p.get("ipadapter") or {}
    if not ref_image or not ip.get("enabled", True):
        return model
    weight = ip_weight if ip_weight is not None else ip.get("weight", 0.6)
    m = g.node("IPAdapterModelLoader", ipadapter_file=ip.get(
        "file", "ip-adapter_sdxl_vit-h.safetensors"))
    cv = g.node("CLIPVisionLoader", clip_name=ip.get(
        "clip_vision", "clip_vision_h.safetensors"))
    img = g.node("LoadImage", image=ref_image)
    nid = g.node(
        "IPAdapterAdvanced", model=model, ipadapter=g.ref(m, 0), image=g.ref(img, 0),
        weight=weight, weight_type="linear", combine_embeds="concat",
        start_at=0.0, end_at=1.0, embeds_scaling="V only", clip_vision=g.ref(cv, 0),
    )
    return g.ref(nid, 0)


def build(p, ref_image=None, seed=None, size=None, prefix=None,
          ip_weight=None, upscale=None, frames=None):
    """按画风档构建一整套文生图（或动画）工作流。"""
    g = Graph()
    model, clip, vae = _load_model(g, p)
    model = _apply_ipadapter(g, p, model, ref_image, ip_weight)

    pos = g.node("CLIPTextEncode", clip=clip, text=p["positive"])
    neg = g.node("CLIPTextEncode", clip=clip, text=p.get("negative", ""))

    ad = p.get("animatediff")
    n_frames = frames or (ad.get("frames") if ad else 1) or 1
    if ad and int(n_frames) < 16:
        # 实测：AnimateDiff 运动模块没有 16 帧时序上下文会直接崩
        #   1 帧 → 纯彩色噪点；8 帧 → 霓虹条纹怪图
        print("[警告] AnimateDiff 至少要 16 帧，%s 帧会出噪点/怪图，已自动抬到 16 帧。"
              % n_frames)
        n_frames = 16
    w, h = size or p.get("size", [832, 1088])
    lat = g.node("EmptyLatentImage", width=int(w), height=int(h), batch_size=int(n_frames))

    ks = g.node(
        "KSampler", model=model,
        seed=int(seed if seed is not None else p.get("seed", 0)),
        steps=int(p.get("steps", 28)), cfg=float(p.get("cfg", 5.0)),
        sampler_name=p.get("sampler", "euler"), scheduler=p.get("scheduler", "normal"),
        denoise=float(p.get("denoise", 1.0)),
        positive=g.ref(pos, 0), negative=g.ref(neg, 0), latent_image=g.ref(lat, 0),
    )
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    last = g.ref(dec, 0)

    up = p.get("upscale") or {}
    do_up = upscale if upscale is not None else up.get("enabled", False)
    if do_up:
        um = g.node("UpscaleModelLoader", model_name=up.get(
            "model", "4x_NMKD-Superscale-SP_178000_G.pth"))
        ui = g.node("ImageUpscaleWithModel", upscale_model=g.ref(um, 0), image=last)
        last = g.ref(ui, 0)

    g.node("SaveImage", images=last,
           filename_prefix=prefix or p.get("output", "styles/out"))
    return g.nodes


def build_hand_fix(p, base_image, mask_image, denoise=None, prompt=None, seed=None):
    """手部 latent 局部重绘：完整编码原图 + 只给 mask 区域加噪，无接缝。

    不用 VAEEncodeForInpaint —— 它会用灰色填充 mask 区域，denoise 不够时残留灰块。
    """
    hf = p.get("hand_fix") or {}
    g = Graph()
    model, clip, vae = _load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip,
                 text=prompt or hf.get("prompt") or p["positive"])
    neg = g.node("CLIPTextEncode", clip=clip, text=p.get("negative", ""))
    im = g.node("LoadImage", image=base_image)
    mk = g.node("LoadImageMask", image=mask_image, channel="red")
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    noise = g.node("SetLatentNoiseMask", samples=g.ref(enc, 0), mask=g.ref(mk, 0))
    ks = g.node(
        "KSampler", model=model,
        seed=int(seed if seed is not None else p.get("seed", 0) + 1),
        steps=int(hf.get("steps", p.get("steps", 28))), cfg=float(p.get("cfg", 5.0)),
        sampler_name=p.get("sampler", "euler"), scheduler=p.get("scheduler", "normal"),
        denoise=float(denoise if denoise is not None else hf.get("denoise", 0.55)),
        positive=g.ref(pos, 0), negative=g.ref(neg, 0), latent_image=g.ref(noise, 0),
    )
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0),
           filename_prefix=(p.get("output", "styles/out") + "_fixed"))
    return g.nodes


# ---------------------------------------------------------------- 提交执行
def submit(wf, tag="", quiet=False):
    data = json.dumps({"prompt": wf, "client_id": "style_lib"}).encode()
    req = urllib.request.Request(SERVER + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        prompt_id = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print("[%s] 提交失败 HTTP %s" % (tag, e.code))
        print(e.read().decode()[:1500])
        return None
    except Exception as e:
        print("[%s] 提交失败：%s" % (tag, e))
        print("  提示：ComfyUI 可能没启动。运行 4_启动_高画质模式.bat")
        return None

    if not quiet:
        print("[%s] 已提交 %s" % (tag, prompt_id))
    t0 = time.time()
    while True:
        try:
            hist = json.load(urllib.request.urlopen(
                SERVER + "/history/" + prompt_id, timeout=30))
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            if status.get("completed"):
                if not quiet:
                    print("[%s] 完成，用时 %.1f 秒" % (tag, time.time() - t0))
                return prompt_id
            for m in status.get("messages", []):
                if m[0] == "execution_error":
                    print("[%s] 执行报错：" % tag)
                    print(json.dumps(m[1], ensure_ascii=False)[:1500])
                    return None
        if time.time() - t0 > 1800:
            print("[%s] 超时" % tag)
            return None
        time.sleep(2)


def snapshot():
    return set(glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True))


def new_files(before):
    now = set(glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True))
    return sorted(now - before, key=os.path.getmtime)


# ---------------------------------------------------------------- 档管理
def profile_path(name):
    return os.path.join(PROFILE_DIR, name + ".json")


def all_names():
    names = []
    for f in sorted(glob.glob(os.path.join(PROFILE_DIR, "*.json"))):
        base = os.path.basename(f)
        if base.startswith("_"):
            continue
        names.append(base[:-5])
    return names


def load(name):
    path = profile_path(name)
    if not os.path.isfile(path):
        raise SystemExit("画风档不存在：%s\n可用档：%s" % (name, ", ".join(all_names())))
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    p.setdefault("name", name)
    return p


def current_name():
    if os.path.isfile(CURRENT_FILE):
        with open(CURRENT_FILE, "r", encoding="utf-8") as f:
            n = f.read().strip()
        if n and os.path.isfile(profile_path(n)):
            return n
    names = all_names()
    return names[0] if names else None


def set_current(name):
    load(name)  # 校验存在
    with open(CURRENT_FILE, "w", encoding="utf-8") as f:
        f.write(name)


def prepare_ref(p, ref):
    """把参考图落到 ComfyUI/input，返回可直接给 LoadImage 的文件名。"""
    ip = p.get("ipadapter") or {}
    ref = ref or ip.get("ref") or None
    if not ref:
        return None
    if os.path.isfile(ref):
        os.makedirs(INPUT_DIR, exist_ok=True)
        target = "ref_%s%s" % (p["name"], os.path.splitext(ref)[1] or ".png")
        shutil.copy2(ref, os.path.join(INPUT_DIR, target))
        return target
    if os.path.isfile(os.path.join(INPUT_DIR, ref)):
        return ref
    raise SystemExit("找不到参考图：%s（既不是本地路径，也不在 ComfyUI/input 下）" % ref)


# ---------------------------------------------------------------- 命令
def _disp_width(s):
    """中文/全角算 2 列，用于表格对齐。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


def _pad(s, width):
    s = str(s)
    return s + " " * max(0, width - _disp_width(s))


def cmd_list(args):
    cur = current_name()
    rows = []
    names = all_names()
    for i, name in enumerate(names, 1):
        p = load(name)
        ip = p.get("ipadapter") or {}
        ad = p.get("animatediff")
        model = p.get("model", "").replace(".safetensors", "").replace(".ckpt", "")
        if p.get("loader") == "diffusers":
            model = "[diffusers] " + model
        rows.append((
            str(i),
            ("*> " if name == cur else "   ") + name,
            p.get("label", ""),
            model,
            "%sx%s" % tuple(p.get("size", ["?", "?"])),
            str(p.get("steps", "?")),
            str(p.get("cfg", "?")),
            p.get("sampler", "?"),
            "%.2f" % ip["weight"] if ip.get("enabled") else "-",
            "%d帧" % ad["frames"] if ad else "-",
        ))
    head = ("#", "档", "画风", "底模", "尺寸", "步", "CFG", "采样器", "IP", "动画")
    widths = [max(_disp_width(r[i]) for r in rows + [head]) for i in range(len(head))]
    def line(vals):
        return "  ".join(_pad(v, widths[i]) for i, v in enumerate(vals)).rstrip()
    print(line(head))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(line(r))
    print("\n当前画风：%s（*> 标记）" % cur)
    print("切换：python style_lib.py use <档名>  或  pick <编号>")
    print("出图：python style_lib.py run [档名] [--ref 图] [--hand] [--upscale]")


def cmd_pick(args):
    names = all_names()
    try:
        idx = int(args.index)
    except ValueError:
        raise SystemExit("编号必须是数字")
    if not 1 <= idx <= len(names):
        raise SystemExit("编号超出范围：1-%d" % len(names))
    set_current(names[idx - 1])
    p = load(names[idx - 1])
    print("已切换到 [%d] %s（%s）" % (idx, p["name"], p.get("label", "")))
    return 0


def cmd_show(args):
    p = load(args.name)
    print(json.dumps(p, ensure_ascii=False, indent=2))


def cmd_use(args):
    set_current(args.name)
    p = load(args.name)
    print("已切换到画风：%s（%s）" % (p["name"], p.get("label", "")))
    print("  底模 %s | %sx%s | %s 步 CFG %s" % (
        p.get("model"), p.get("size", ["?", "?"])[0],
        p.get("size", ["?", "?"])[1], p.get("steps"), p.get("cfg")))
    print("  下次直接 `python style_lib.py run` 就用这个画风")


def cmd_check(args):
    ok = True
    for name in all_names():
        p = load(name)
        missing = []
        def need(kind, fname):
            if not fname:
                return
            d = MODEL_DIRS[kind]
            if not os.path.exists(os.path.join(MODELS, d, fname)):
                missing.append("%s/%s" % (d, fname))
        if p.get("loader") == "diffusers":
            if not os.path.isdir(os.path.join(MODELS, "diffusers", p["model"])):
                missing.append("diffusers/%s" % p["model"])
        else:
            need("checkpoint", p.get("model"))
        need("vae", p.get("vae"))
        for l in p.get("loras") or []:
            need("lora", l["file"])
        ad = p.get("animatediff")
        if ad:
            need("animatediff", ad["model"])
        ip = p.get("ipadapter") or {}
        if ip.get("enabled"):
            need("ipadapter", ip.get("file"))
            need("clip_vision", ip.get("clip_vision"))
        up = p.get("upscale") or {}
        if up.get("model"):
            need("upscale", up["model"])
        if missing:
            ok = False
            print("[缺文件] %s -> %s" % (name, ", ".join(missing)))
        else:
            print("[OK] %s" % name)
    return 0 if ok else 1


def cmd_export(args):
    outdir = os.path.join(PROFILE_DIR, "api")
    os.makedirs(outdir, exist_ok=True)
    names = all_names() if args.all else [args.name or current_name()]
    for name in names:
        p = load(name)
        wf = build(p)
        path = os.path.join(outdir, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=2)
        print("导出 %s -> %s" % (name, path))


def cmd_run(args):
    name = args.name or current_name()
    if not name:
        raise SystemExit("还没有画风档。先在 profiles/ 下建一个 JSON。")
    p = load(name)
    ref = prepare_ref(p, args.ref)

    # 临时覆盖：命令行给的提示词/种子只影响这一次，不写回画风档
    if args.prompt:
        p["positive"] = args.prompt
    if args.neg:
        p["negative"] = args.neg

    size = None
    if args.size:
        w, h = args.size.lower().replace("*", "x").split("x")
        size = (int(w), int(h))

    if args.random:
        seed = random.randint(0, 2 ** 48)
    else:
        seed = args.seed if args.seed is not None else p.get("seed", 0)
    n = max(1, args.n or 1)

    print("画风：%s（%s）" % (name, p.get("label", "")))
    print("底模：%s | %s | 步数 %s | CFG %s | %s" % (
        p.get("model"), "x".join(str(v) for v in (size or p.get("size", []))),
        p.get("steps"), p.get("cfg"), p.get("sampler")))
    print("种子：%s%s" % (seed, "" if args.random else "（固定值，看起来一样就是因为这个）"))
    print("正向：%s" % p["positive"][:120])
    if ref:
        print("喂图：%s（IPAdapter %.2f）" % (ref, p.get("ipadapter", {}).get("weight", 0)))

    made = []
    for i in range(n):
        s = seed + i
        wf = build(p, ref_image=ref, seed=s, size=size,
                   ip_weight=args.ip_weight, upscale=args.upscale)
        before = snapshot()
        if not submit(wf, tag="%s seed=%s" % (name, s)):
            continue
        files = new_files(before)
        made.extend(files)
        for f in files:
            print("  输出：%s" % f)

    if not made:
        print("没有产出图片，请检查上面的报错")
        return 1

    do_hand = args.hand if args.hand is not None else (p.get("hand_fix") or {}).get("enabled", False)
    if do_hand:
        from PIL import Image
        hf = p.get("hand_fix") or {}
        box = hf.get("box", [0.05, 0.44, 0.26, 0.29])
        print("修手：%d 张" % len(made))
        for src in made:
            im = Image.open(src).convert("RGB")
            w, h = im.size
            base_name = "hf_base_%s.png" % name
            mask_name = "hf_mask_%s.png" % name
            im.save(os.path.join(INPUT_DIR, base_name))
            px = (int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3]))
            sys.path.insert(0, BASE)
            import style_workflows_api as W
            W.make_hand_mask(os.path.join(INPUT_DIR, base_name),
                             os.path.join(INPUT_DIR, mask_name), px,
                             feather=hf.get("feather", 14))
            before = snapshot()
            wf = build_hand_fix(p, base_name, mask_name)
            if submit(wf, tag="修手"):
                for f in new_files(before):
                    print("  修手输出：%s" % f)
    return 0


def main():
    ap = argparse.ArgumentParser(description="画风权重库：一个画风一份配置，切换即换档。")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="列出所有画风档")

    s = sub.add_parser("show", help="查看某档完整参数")
    s.add_argument("name")

    s = sub.add_parser("use", help="切换到某画风")
    s.add_argument("name")

    s = sub.add_parser("pick", help="按 list 里的编号切换画风")
    s.add_argument("index")

    sub.add_parser("check", help="检查模型文件是否齐备")

    s = sub.add_parser("export", help="导出 API 格式工作流")
    s.add_argument("name", nargs="?")
    s.add_argument("--all", action="store_true")

    s = sub.add_parser("run", help="用画风档出图")
    s.add_argument("name", nargs="?", help="画风档名，省略则用当前档")
    s.add_argument("--ref", help="参考图路径或 ComfyUI/input 下的文件名")
    s.add_argument("-p", "--prompt", help="本次临时换正向提示词（不写回画风档）")
    s.add_argument("--neg", dest="neg", help="本次临时换负向提示词")
    s.add_argument("--seed", type=int)
    s.add_argument("--random", action="store_true", help="随机种子（每张都不一样）")
    s.add_argument("--size", help="如 832x1216")
    s.add_argument("--n", type=int, default=1, help="连续出几张（seed 递增）")
    s.add_argument("--hand", dest="hand", action="store_true", default=None, help="出图后修手")
    s.add_argument("--no-hand", dest="hand", action="store_false")
    s.add_argument("--upscale", dest="upscale", action="store_true", default=None)
    s.add_argument("--no-upscale", dest="upscale", action="store_false")
    s.add_argument("--ip-weight", type=float)

    args = ap.parse_args()
    if not args.cmd:
        cmd_list(args)
        return 0
    return {"list": cmd_list, "show": cmd_show, "use": cmd_use, "pick": cmd_pick,
            "check": cmd_check, "export": cmd_export, "run": cmd_run}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
