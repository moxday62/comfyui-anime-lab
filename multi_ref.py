# -*- coding: utf-8 -*-
"""多图参考：一张管角色、一张管服装、一张管质感、一张管动作。

四张图分别走两条完全不同的通道，这是关键：

  角色 / 服装 / 质感   -> IP-Adapter 链（视觉特征注入 attention）
  动作                 -> ControlNet openpose（骨架约束空间结构）

  为什么不能全用 IP-Adapter：它对"姿势"几乎无能。
  IP-Adapter 学的是"这张图长什么样"，包括长相、配色、笔触；
  就算 weight 拉到 0.9，它也只能让画面"更接近参考图的构图"，
  不能让你指定"左腿抬起 30 度"。姿势是空间约束，只有 ControlNet 管得了。

  为什么不能全用 ControlNet：它只管形状，管不了"这个角色的脸长什么样"。

  IP-Adapter 链的叠法：三个 IPAdapterAdvanced 串联，
  每个吃上一个输出的 model，相当于在 attention 上叠加三次视觉先验。
  串联会互相稀释，所以权重不能都开 0.6，要往下压：
      角色 0.60  服装 0.45  质感 0.30

  各通道的 weight_type 也不一样：
      角色  linear          保留长相和轮廓
      服装  linear          保留款式和颜色分布
      质感  style transfer  只取笔触/配色，不抄构图（这是它和 linear 的区别）

用法：
  python multi_ref.py --char ref_char.jpg --outfit ref_outfit.jpg
  python multi_ref.py --char A.jpg --outfit B.jpg --style C.jpg --pose pose_jump.png
  python multi_ref.py --char A.jpg --compare     跑 A/B/C/D 四组递进对照
"""
import argparse
import json
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL

OUT_DIR = os.path.join(BASE, "output", "multiref")
IPADAPTER_SDXL = "ip-adapter_sdxl_vit-h.safetensors"
CLIP_VISION_H = "clip_vision_h.safetensors"
CN_OPENPOSE = "controlnet_openpose_sdxl.safetensors"


# ---------------------------------------------------------------- 参考图包
# ref_search.py 扒完图会在目录里留一份 _pack.json。这里负责把它拆回通道：
# 目录名/预设名决定了这张图该进哪一路。

def load_pack(path):
    """读 _pack.json。传目录也行，会自动找里面的 _pack.json。"""
    if os.path.isdir(path):
        path = os.path.join(path, "_pack.json")
    if not os.path.exists(path):
        raise SystemExit("找不到参考图包：%s（先跑一次 ref_search.py）" % path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pack_channel(pack):
    """从包里推断这是哪一路的图。"""
    cat = (pack.get("cat") or "") + " " + (pack.get("preset") or "")
    cat = cat.lower()
    for k in ("pose", "outfit", "style", "scene", "char"):
        if k in cat:
            # char_web 的图是网上扒的陌生人，不该拿来锁你自己的角色。
            # 角色通道要用 refs/char 里那 49 张精选图，这里明确拒绝。
            return k
    return ""


def best_from_pack(pack):
    """取包里分数最高的一条。pose 优先要有骨架的（没骨架进去也是废图）。"""
    items = [x for x in (pack.get("items") or []) if x.get("file")]
    if not items:
        return None
    items.sort(key=lambda x: x.get("rank", 0), reverse=True)
    if pack_channel(pack) == "pose":
        with_skel = [x for x in items if x.get("skeleton")]
        return with_skel[0] if with_skel else None
    return items[0]


def to_input(path, prefix="pk_"):
    """把任意路径的图复制进 ComfyUI/input，返回 LoadImage 用的文件名。"""
    src = path if os.path.isabs(path) else os.path.join(BASE, path)
    if not os.path.exists(src):
        raise SystemExit("图包里的文件不见了：%s" % src)
    dst = os.path.join(SL.INPUT_DIR, prefix + os.path.basename(src))
    shutil.copy2(src, dst)
    return os.path.basename(dst)


def square_crop(src, dst, size=1024, focus=None):
    """把参考图裁成正方形，焦点放在脸上（或指定位置）。

    这一步是 IP-Adapter 能不能生效的关键，日志里写得很清楚：
      "the IPAdapter reference image is not a square,
       CLIPImageProcessor will resize and crop it at the center"
    竖版全身图的中心是躯干 —— 脸和服装细节全被裁掉，
    于是 IP-Adapter 拿到的是"一块躯干"，自然锁不住角色。
    实测就是这个原因导致喂了粉发参考、出来却是深蓝发。
    """
    from PIL import Image
    im = Image.open(src).convert("RGB")
    W, H = im.size

    cx, cy = W / 2.0, H * 0.32
    if focus is None:
        try:
            import staged_prompt as SP
            f = SP.detect_face(src)
            if f:
                cx = (f[0] + f[2]) / 2.0
                cy = (f[1] + f[3]) / 2.0 + (f[3] - f[1]) * 0.9   # 略向下，带进上半身
        except Exception:
            pass
    else:
        cx, cy = focus

    side = min(W, int(H * 0.85))
    side = max(side, int(W * 0.75))
    x0 = max(0, int(cx - side / 2))
    y0 = max(0, int(cy - side / 2))
    x1 = min(W, x0 + side)
    y1 = min(H, y0 + side)
    if x1 - x0 != y1 - y0:                     # 贴边时补齐成正方形
        s2 = min(x1 - x0, y1 - y0)
        x0 = min(max(0, x1 - s2), W - s2)
        y0 = min(max(0, y1 - s2), H - s2)
        x1, y1 = x0 + s2, y0 + s2
    im.crop((x0, y0, x1, y1)).resize((size, size), Image.LANCZOS).save(dst)
    return dst


def build(p, prompt, negative, char=None, outfit=None, style=None, pose=None,
          prefix="multiref/out", w=768, h=1344, seed=None,
          char_w=0.80, outfit_w=0.35, style_w=0.22,
          pose_strength=1.0, pose_end=0.75, ipa_end=0.45):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)

    pos = g.node("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip, text=negative)
    pr, nr = g.ref(pos, 0), g.ref(neg, 0)

    # ---- 动作通道：ControlNet openpose ----
    if pose:
        pimg = g.node("LoadImage", image=pose)
        cn = g.node("ControlNetLoader", control_net_name=CN_OPENPOSE)
        adv = g.node("ControlNetApplyAdvanced",
                     positive=pr, negative=nr, control_net=g.ref(cn, 0),
                     image=g.ref(pimg, 0), strength=pose_strength,
                     start_percent=0.0, end_percent=pose_end)
        pr, nr = g.ref(adv, 0), g.ref(adv, 1)

    # ---- 视觉通道：IP-Adapter 链 ----
    chain = [(char, char_w, "linear", "角色"),
             (outfit, outfit_w, "linear", "服装"),
             (style, style_w, "style transfer", "质感")]
    chain = [c for c in chain if c[0]]
    if chain:
        ipm = g.node("IPAdapterModelLoader", ipadapter_file=IPADAPTER_SDXL)
        cvl = g.node("CLIPVisionLoader", clip_name=CLIP_VISION_H)
        m = g.ref([nid for nid in model] if isinstance(model, list) else model) \
            if False else model
        for img, wt, wtype, _tag in chain:
            li = g.node("LoadImage", image=img)
            m = g.node("IPAdapterAdvanced",
                       model=m, ipadapter=g.ref(ipm, 0), image=g.ref(li, 0),
                       weight=wt, weight_type=wtype, combine_embeds="concat",
                       start_at=0.0, end_at=ipa_end, embeds_scaling="V only",
                       clip_vision=g.ref(cvl, 0))
            m = g.ref(m, 0)
        model = m

    lat = g.node("EmptyLatentImage", width=w, height=h, batch_size=1)
    ks = g.node("KSampler", model=model,
                seed=int(seed if seed is not None else p.get("seed", 0)),
                steps=int(p.get("steps", 28)), cfg=float(p.get("cfg", 5.0)),
                sampler_name="euler_ancestral", scheduler="normal", denoise=1.0,
                positive=pr, negative=nr, latent_image=g.ref(lat, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


# 教训：之前这里只写了皮肤和五官，**没写穿什么**。
# 结果模型直接给了"只穿上衣"，我误判成参考图污染，其实是我自己的提示词空洞。
# 服装必须逐个部位写清楚：上衣、下装、袜子、鞋。
# 光写 "white sailor uniform" 也不够 —— 它只保证上衣，
# 裙子得单独写 "white pleated skirt"，否则模型会自己决定要不要给。
DEFAULT_POS = (
    "1girl, solo, full body, standing, looking at viewer, "
    "pale skin, soft matte skin, delicate features, "
    "pink hair, short hair, blue eyes, "
    "white sailor uniform, short sleeves, navy collar, blue bow, cropped top, "
    "white pleated skirt, white thighhighs, white shoes, "
    "plain wall background, soft studio lighting, flat neutral light, "
    "3d, cgi, octane render, highly detailed, masterpiece, best quality"
)
DEFAULT_NEG = (
    "worst quality, low quality, blurry, bad anatomy, extra limbs, "
    "bad hands, missing fingers, extra fingers, watermark, text, signature, "
    "cropped, out of frame, close-up, cowboy shot, "
    "speech bubble, thought bubble, glossy plastic, "
    # 参考图里有擦边内容时，IP-Adapter 会把那种氛围一起带进来。
    # 负向必须把裸露和低角度构图压死，否则会跟着跑偏。
    "nude, nsfw, naked, topless, bottomless, revealing clothes, "
    "cleavage, underwear only, low angle, from below, kneeling, spread legs"
)


def pose_hint(pose_name):
    """按骨架文件名推断该补哪些姿势词。

    为什么必须补：ControlNet 只约束形状，文字条件还在说"standing、看着镜头"。
    实测（pose_reach + DEFAULT_POS）时，骨架的单臂高举被文字压住，
    出来是双手垂在身前的站姿；一旦提示词补上 "one arm raised up,
    hand on hip"，姿势立刻跟上。所以这两者必须同向描述，
    否则是在让模型同时听两个相反的指令。
    """
    n = (pose_name or "").lower()
    if "jump" in n:
        return ("jumping, mid-air, arms up, raised arms, knee up, "
                "dynamic pose, floating")
    if "reach" in n:
        return ("one arm raised up, arm up, other hand on hip, "
                "hand on hip, body slightly tilted")
    if "sit" in n:
        return "sitting, sitting on chair, knees bent, legs together"
    return ""


def shot(p, tag, stamp, **kw):
    before = SL.snapshot()
    t = time.time()
    wf = build(p, kw.pop("prompt", DEFAULT_POS), kw.pop("neg", None) or DEFAULT_NEG,
               prefix="multiref/%s_%s" % (tag, stamp), **kw)
    if not SL.submit(wf, tag=tag):
        return None
    got = SL.new_files(before)
    if not got:
        return None
    dst = os.path.join(OUT_DIR, "%s_%s.png" % (tag, stamp))
    shutil.copy2(got[-1], dst)
    print("  %-6s %.1fs -> %s" % (tag, time.time() - t, os.path.basename(dst)))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("--char", help="角色参考图（ComfyUI/input 下的文件名）")
    ap.add_argument("--outfit", help="服装参考图")
    ap.add_argument("--style", help="质感/画风参考图")
    ap.add_argument("--pose", help="骨架图（ComfyUI/input 下的文件名）")
    ap.add_argument("--pack", action="append", default=[],
                    help="参考图包：_pack.json 的路径或所在目录，可重复指定。"
                         "按目录名自动分通道（pose_web→pose，outfit_web→outfit …）")
    ap.add_argument("--prompt")
    ap.add_argument("--neg", dest="neg",
                    help="覆盖负向（例如压掉背景场景感、幕布、渐变）")
    ap.add_argument("--size", default="768x1344")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--compare", action="store_true", help="跑 A/B/C/D 递进对照")
    ap.add_argument("--pose-strength", type=float, default=1.0,
                    help="ControlNet 骨架强度，默认 1.0（0.85 会被 IP-Adapter 盖住）")
    ap.add_argument("--pose-end", type=float, default=0.75,
                    help="骨架生效到第几步，默认 0.75（0.55 太早放手，姿势会漂）")
    ap.add_argument("--ipa-end", type=float, default=0.45,
                    help="IP-Adapter 在第几步收手，默认 0.45。0.65 会在人物周围留一圈描边")
    ap.add_argument("--char-w", type=float, default=0.80)
    a = ap.parse_args()

    sk = dict(pose_strength=a.pose_strength, pose_end=a.pose_end,
              ipa_end=a.ipa_end, char_w=a.char_w)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    p = SL.load(a.profile)
    w, h = [int(v) for v in a.size.lower().replace("*", "x").split("x")]
    p["size"] = [w, h]

    # ---- 从参考图包补通道 ----
    # 显式传的参数不覆盖：手动指定永远优先于自动配对。
    pack_hints = []
    for pk_path in a.pack:
        pk = load_pack(pk_path)
        ch = pack_channel(pk)
        it = best_from_pack(pk)
        if not ch:
            print("  [pack] 认不出 %s 属于哪一路，跳过" % pk_path)
            continue
        if it is None:
            print("  [pack] %s 里没有可用条目"
                  "（pose 包可能一张骨架都没提出来）" % pk_path)
            continue
        if ch == "char":
            print("  [pack] 跳过 char：角色通道该用 refs/char 那 49 张精选图，"
                  "网图是陌生人，锁不住你要画的人")
            continue
        if getattr(a, ch):
            print("  [pack] %s 已手动指定，跳过包里的" % ch)
            continue

        if ch == "pose":
            setattr(a, ch, to_input(it["skeleton"]))
            # 骨架文件名是 skel_7325104 这种，pose_hint() 从名字里读不出姿势。
            # 但包里记着当初搜的关键词，那才是这个骨架的意思，得补进提示词，
            # 否则文字说 standing、骨架在跳，模型两头听，姿势就散了。
            q = (pk.get("query") or "").lower()
            hit = ""
            for k, hint in (("jump", "jumping, mid-air, arms up, knee up, dynamic pose"),
                            ("run", "running, arms swinging, one leg forward, one leg back"),
                            ("fly", "flying, floating, arms spread"),
                            ("sit", "sitting, knees bent, legs together"),
                            ("reach", "one arm raised up, arm up"),
                            ("stand", "standing")):
                if k in q:
                    hit = hint
                    break
            if not hit:
                hit = (pk.get("query") or "").split(",")[0].strip()
            if hit:
                pack_hints.append(hit)
            print("  [pack] pose   <- %s（姿势词：%s）"
                  % (os.path.basename(it["skeleton"]), hit or "无"))
        else:
            setattr(a, ch, to_input(it["file"]))
            print("  [pack] %-6s <- %s" % (ch, os.path.basename(it["file"])))

    # 三张参考图统一裁成正方形，否则 IP-Adapter 会中心裁剪掉脸
    def prep(name, tag):
        if not name:
            return None
        src = os.path.join(SL.INPUT_DIR, name)
        if not os.path.exists(src):
            return name
        dst = os.path.join(SL.INPUT_DIR, "sq_%s.png" % tag)
        square_crop(src, dst)
        print("  参考图 %s -> 正方形 %s" % (name, os.path.basename(dst)))
        return os.path.basename(dst)

    a.char = prep(a.char, "char")
    a.outfit = prep(a.outfit, "outfit")
    a.style = prep(a.style, "style")

    # 挂了骨架就必须在提示词里同向描述姿势，否则文字和骨架互相打架
    if a.pose:
        hint = pose_hint(a.pose)
        if hint:
            base = a.prompt or DEFAULT_POS
            if "arm" not in base.lower():
                a.prompt = base + ", " + hint
                print("  姿势词自动补入：%s" % hint)
            else:
                print("  提示词已有手臂描述，跳过自动补入")

    # 图包来的姿势词要单独补：它的骨架叫 skel_7325104，pose_hint 从名字里
    # 读不出任何东西。而且 DEFAULT_POS 里写着 standing，不压掉的话
    # 文字和骨架直接对着干 —— 之前踩过这个坑，出来是"站着但腿是抬的"。
    if pack_hints:
        base = a.prompt or DEFAULT_POS
        add = ", ".join(pack_hints)
        if "standing" in base.lower() and "standing" not in add.lower():
            base = base.replace("standing, ", "").replace(", standing", "")
            base = base.replace("standing", "")
        a.prompt = base.rstrip(", ") + ", " + add
        print("  图包姿势词补入：%s" % add)

    if a.compare:
        if not a.char:
            raise SystemExit("--compare 至少需要 --char")
        print("四组递进对照（同 seed %s）" % a.seed)
        outs = []
        print("\n[A] 只喂角色")
        r = shot(p, "A", stamp, char=a.char, w=w, h=h, seed=a.seed, prompt=a.prompt, neg=a.neg, **sk)
        if r: outs.append(("A 角色", r))

        if a.outfit:
            print("[B] 角色 + 服装")
            r = shot(p, "B", stamp, char=a.char, outfit=a.outfit,
                     w=w, h=h, seed=a.seed, prompt=a.prompt, neg=a.neg, **sk)
            if r: outs.append(("B +服装", r))

        if a.style:
            print("[C] 角色 + 质感")
            r = shot(p, "C", stamp, char=a.char, outfit=a.outfit, style=a.style,
                     w=w, h=h, seed=a.seed, prompt=a.prompt, neg=a.neg, **sk)
            if r: outs.append(("C +质感", r))

        if a.style and a.pose:
            print("[D] 角色 + 质感 + 动作")
            r = shot(p, "D", stamp, char=a.char, outfit=a.outfit, style=a.style,
                     pose=a.pose, w=w, h=h, seed=a.seed, prompt=a.prompt, neg=a.neg, **sk)
            if r: outs.append(("D +动作", r))

        if len(outs) >= 2:
            from PIL import Image, ImageDraw
            tw = 400
            ims = [(t, Image.open(f).convert("RGB")) for t, f in outs]
            ims = [(t, im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS))
                   for t, im in ims]
            hh = min(im.height for _, im in ims)
            W = sum(im.width for _, im in ims) + 10 * (len(ims) - 1)
            cv = Image.new("RGB", (W, hh + 32), (18, 18, 18))
            d = ImageDraw.Draw(cv)
            x = 0
            for t, im in ims:
                cv.paste(im.crop((0, 0, im.width, hh)), (x, 32))
                d.text((x + 6, 10), t, fill=(255, 255, 255))
                x += im.width + 10
            cp = os.path.join(OUT_DIR, "cmp_%s.jpg" % stamp)
            cv.save(cp, "JPEG", quality=93)
            print("\n对比图 %s" % cp)
        return 0

    r = shot(p, "one", stamp, char=a.char, outfit=a.outfit, style=a.style,
             pose=a.pose, w=w, h=h, seed=a.seed, prompt=a.prompt, neg=a.neg, **sk)
    if r:
        print("成品 %s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
