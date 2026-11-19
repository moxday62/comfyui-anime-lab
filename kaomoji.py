# -*- coding: utf-8 -*-
"""颜文字当提示词。四条路都试过了，只有第四条真能用。

  一  直接塞进提示词框   无效。CLIP 词表里没有 ◕ ‿ ◉ 这些字符
  二  img2img 起点       模型把括号画成耳机、弧线画成机械面罩
  三  + ControlNet       更糟，形状更硬但仍然是"物件"不是"表情"
  四  查表映射           ★ 能用。颜文字翻成表情 tag，零训练、瞬出结果

  前三条都想让模型"看着颜文字的形状去画脸"，全卡在同一件事上：
  模型没有"颜文字=表情"这个先验。第四条绕开了形状，直接取语义。

用法：
  python kaomoji.py --tags "(╥﹏╥)"               颜文字转表情 tag（不出图）
  python kaomoji.py --test                       出图对照（跑的是 image 路线）
  python kaomoji.py --render "(◕‿◕)"             只渲染成图看长什么样

  下面先讲前三条为什么不行，再讲第四条怎么用。

  ---------------------------------------------------------------
  一 为什么不能直接塞进提示词框
  ---------------------------------------------------------------
SDXL 的文本编码器是 CLIP，词表只有 49408 个 token，以英文 BPE 为主。
颜文字里的字符，码位是这样的：
    (◕‿◕)   U+28  U+25D5  U+203F  U+25D5  U+29
    >_<     U+3E  U+5F    U+3C
    ◉_◉     U+25C9 U+5F   U+25C9
  "◕"(CIRCLE WITH ALL BUT UPPER LEFT QUADRANT BLACK)、
  "‿"(UNDERTIE)、"◉"(FISHEYE) 这些都不在 CLIP 词表里，
  会被 byte-level fallback 拆成一堆字节 token，语义约等于噪声。
  而 "T_T" 更惨 —— T、_、T 都在词表里，模型读到的是"字母 T 下划线 T"，
  跟"哭"没有半点关系。

  所以"颜文字当文本提示词"不是效果差，是**根本没有信息进入模型**。
  Civitai 上偶尔有人写 (^_^) 好像有用，那是心理作用 —— 它和写 (_) 一样。
  唯一的例外是 "^_^" 这个**具体写法**恰好是 danbooru 的有效 tag，
  但那是因为它被当 tag 收录过，不是因为模型懂颜文字。

二 真正能让模型"看见"的办法
----------------------------------------
  你直觉里那句"应让模型可见而非简单的理解"是对的。
  颜文字本身就是一张极简的脸部线稿：两个点 + 一条弧。
  把它渲染成图片，模型就能当图形读，而不是当文本读。

  三条图像路线，从弱到强：
    A. img2img 起点：把颜文字渲染成图当初始 latent，denoise 0.85-0.92
       模型会把两个黑点"完形"成眼睛、把弧线完形成嘴。
       这是最接近"颜文字当提示词"的字面实现。
    B. ControlNet scribble：用颜文字的笔画当线稿约束
       形状会非常贴颜文字，但脸可能被拉成符号的样子
    C. IP-Adapter：把颜文字图当视觉参考
       字符图和照片差太远，实测意义不大

  这个脚本实现 A（可选叠加 B），并且给出多组颜文字的表情对照。

用法：
  python kaomoji.py --render "(◕‿◕)"            只渲染成图，看长什么样
  python kaomoji.py --test                       跑一组表情对照
  python kaomoji.py -k "(◕‿◕)" --denoise 0.88   单个出图
  python kaomoji.py -k "T_T" --cn                叠加 scribble 线稿约束
"""
import argparse
import os
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL

OUT_DIR = os.path.join(BASE, "output", "kaomoji")
CN_SCRIBBLE = "controlnet_scribble_sdxl.safetensors"

# 优先这几款字体，它们覆盖 U+25xx 几何符号和 U+203F
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\seguisym.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]

BASE_POS = (
    "1girl, solo, close-up face portrait, looking at viewer, "
    "pale skin, soft matte skin, pink hair, short hair, "
    "white sailor uniform, plain background, soft studio lighting, "
    "3d, cgi, octane render, highly detailed, masterpiece, best quality"
)
BASE_NEG = (
    "letter, text, character, symbol, typography, watermark, signature, "
    "multiple faces, deformed, bad anatomy, low quality, blurry"
)


def pick_font(path=None):
    from PIL import ImageFont
    cands = [path] if path else FONT_CANDIDATES
    for f in cands:
        if f and os.path.exists(f):
            try:
                ImageFont.truetype(f, 64)
                return f
            except Exception:
                continue
    raise SystemExit("找不到可用字体")


def render(kaomoji, out_png, size=832, font_path=None, fg=(20, 20, 24), bg=(255, 255, 255)):
    """把颜文字渲染成一张白底黑字的图，字撑满 80% 画面。

    为什么要撑满：img2img 里这图就是"脸的初稿"，
    太小的话模型只看到白底上一个装饰，不会有完形压力。
    """
    from PIL import Image, ImageDraw, ImageFont
    f = pick_font(font_path)
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    # 逐步放大字号直到铺满 80%
    fs = 40
    while fs < size:
        font = ImageFont.truetype(f, fs)
        box = d.textbbox((0, 0), kaomoji, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        if w > size * 0.80 or h > size * 0.55:
            break
        fs += 4
    font = ImageFont.truetype(f, fs)
    box = d.textbbox((0, 0), kaomoji, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((size - w) / 2 - box[0], (size - h) / 2 - box[1]), kaomoji, font=font, fill=fg)
    im.save(out_png)
    return out_png, (w, h)


def build(p, img_name, prompt, denoise, prefix, use_cn=False, cn_strength=0.55, cn_end=0.5):
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip, text=BASE_NEG)
    pr, nr = g.ref(pos, 0), g.ref(neg, 0)

    im = g.node("LoadImage", image=img_name)
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)

    if use_cn:
        cn = g.node("ControlNetLoader", control_net_name=CN_SCRIBBLE)
        adv = g.node("ControlNetApplyAdvanced",
                     positive=pr, negative=nr, control_net=g.ref(cn, 0),
                     image=g.ref(im, 0), strength=cn_strength,
                     start_percent=0.0, end_percent=cn_end)
        pr, nr = g.ref(adv, 0), g.ref(adv, 1)

    ks = g.node("KSampler", model=model, seed=int(p.get("seed", 0)),
                steps=28, cfg=5.5, sampler_name="euler_ancestral",
                scheduler="normal", denoise=denoise,
                positive=pr, negative=nr, latent_image=g.ref(enc, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


# ---------------------------------------------------------------- 映射路线
# 前面试的两条图像路线（img2img / scribble）都想让模型"看着颜文字的形状画脸"，
# 结果它把括号画成耳机、弧线画面罩 —— 因为它没有"颜文字=表情"这个先验。
#
# 但"颜文字当提示词"其实不必走形状这条路。颜文字携带的信息就两样：
#   眼睛长什么样（睁/闭/笑弯/流泪/星星）
#   嘴长什么样（闭/张/撇/大张）
# 这两样用现成的表情 tag 就能精确表达，而且模型全都认识。
# 所以第四条路是：查表。零训练、零显存、瞬间出结果。
# 代价是它只懂表里这四十来个常见款 —— 但人常用的也就这些。
KAOMOJI_MAP = {
    # ---- 开心
    "(◕‿◕)": ["happy", "smiling", "closed mouth"],
    "(＾▽＾)": ["happy", "open mouth", "smile"],
    "(≧▽≦)": ["very happy", "open mouth", "laughing", "^_^"],
    "(*^▽^*)": ["happy", "smile", "open mouth"],
    "(^_^)": ["smile", "happy", "^_^"],
    "(๑˃̵ᴗ˂̵)": ["happy", "smile", "sparkling eyes"],
    "(＾ω＾)": ["closed eyes", "happy", "smile"],
    "(｡◕‿◕｡)": ["happy", "smiling", "innocent"],

    # ---- 闭眼 / 惬意
    "(◡‿◡)": ["closed eyes", "calm", "gentle smile"],
    "(´ω｀)": ["closed eyes", "relaxed", "smile"],
    "(￣▽￣)": ["closed eyes", "smug", "smile"],
    "( ˘ω˘ )": ["closed eyes", "peaceful", "sleeping"],

    # ---- 哭
    "T_T": ["crying", "tears", "closed eyes"],
    "(╥﹏╥)": ["crying", "streaming tears", "sad"],
    "(；ω；)": ["crying", "sad", "teary eyes"],
    "(ಥ﹏ಥ)": ["crying", "streaming tears", "sad"],
    "(´;ω;`)": ["crying", "sad", "tears"],

    # ---- 怒
    "(｀д´)": ["angry", "open mouth", "shouting"],
    "(╬ Ò﹏Ó)": ["angry", "angry eyes", "veins"],
    "(￣ヘ￣)": ["annoyed", "displeased", "closed eyes"],
    "(｀へ´)": ["annoyed", "pouting"],

    # ---- 惊讶
    "(°o°)": ["surprised", "wide eyes", "open mouth"],
    "(⊙_⊙)": ["surprised", "wide eyes"],
    "(O_O)": ["shocked", "wide eyes", "open mouth"],
    "(゜ロ゜)": ["shocked", "open mouth", "wide eyes"],

    # ---- 无语 / 面瘫
    "(・_・)": ["expressionless", "blank stare"],
    "(-_-)": ["expressionless", "annoyed", "closed eyes"],
    "(￣ー￣)": ["expressionless", "smug"],
    "(¬_¬)": ["unamused", "side glance"],

    # ---- 害羞
    "(*/ω＼*)": ["blush", "embarrassed", "covering face"],
    "(⁄ ⁄•⁄ω⁄•⁄ ⁄)": ["blush", "embarrassed", "shy"],
    "(〃▽〃)": ["blush", "embarrassed", "smile"],

    # ---- 星星眼 / 兴奋
    "(✧ω✧)": ["sparkling eyes", "excited", "star-shaped pupils"],
    "(☆▽☆)": ["star-shaped pupils", "excited", "open mouth"],

    # ---- 爱
    "(♥ω♥)": ["heart", "love", "blush"],
    "(´♡‿♡`)": ["love", "blush", "closed eyes"],

    # ---- 汗 / 尴尬
    "(￣▽￣;)": ["sweatdrop", "awkward", "smile"],
    "(；￣Д￣)": ["sweatdrop", "awkward", "annoyed"],

    # ---- 坚定 / 燃
    "(๑•̀ㅂ•́)و✧": ["determined", "confident", "sparkling eyes"],
    "(•̀ω•́)": ["determined", "serious"],

    # ---- 疑惑 / 困
    "(?_?)": ["confused", "question mark"],
    "(－_－)": ["sleeping", "closed eyes", "tired"],
}

_KAO_KEYS = sorted(KAOMOJI_MAP.keys(), key=len, reverse=True)


def parse(text):
    """从一段文本里找出颜文字，翻成表情 tag。

    返回 {"tags": [...], "found": [(颜文字, tags), ...]}。
    按长度降序匹配并占位，避免 "(*/ω＼*)" 里的 "ω" 之类被别的短词抢先命中。
    """
    if not text:
        return {"tags": [], "found": []}
    rest = text
    found = []
    for k in _KAO_KEYS:
        if k in rest:
            found.append((k, KAOMOJI_MAP[k]))
            rest = rest.replace(k, "\x00" * len(k))   # 占位，防止重复匹配
    tags = []
    for _k, ts in found:
        for t in ts:
            if t not in tags:
                tags.append(t)
    return {"tags": tags, "found": found}


def expand(prompt):
    """把提示词里的颜文字就地换成表情 tag，返回 (新提示词, 命中的颜文字)。

    表情 tag 插在最前面 —— SDXL 里靠前的词权重更高，
    表情是这张图的主题，不该被后面的服装/场景压掉。
    """
    r = parse(prompt)
    if not r["tags"]:
        return prompt, []
    rest = prompt
    for k, _ts in r["found"]:
        rest = rest.replace(k, " ")
    rest = " ".join(rest.split())
    rest = rest.replace(" ,", ",").replace(",,", ",").strip(", ")
    new = ", ".join(r["tags"]) + (", " + rest if rest else "")
    return new, [k for k, _ in r["found"]]


def run_one(p, kao, tag, denoise, use_cn, stamp, prompt=None):
    clean = "".join(ch if ch.isalnum() else "_" for ch in kao)[:16]
    png = os.path.join(SL.INPUT_DIR, "kao_%s_%s.png" % (tag, stamp))
    _, wh = render(kao, png)
    print("  渲染 %-8s 字形 %dx%d -> %s" % (kao, wh[0], wh[1], os.path.basename(png)))

    before = SL.snapshot()
    t = time.time()
    wf = build(p, os.path.basename(png), prompt or BASE_POS, denoise,
               "kaomoji/%s_%s" % (tag, stamp), use_cn)
    if not SL.submit(wf, tag=tag):
        return None
    got = SL.new_files(before)
    if not got:
        return None
    dst = os.path.join(OUT_DIR, "%s_%s.png" % (tag, stamp))
    shutil.copy2(got[-1], dst)
    print("  出图 %.1fs -> %s" % (time.time() - t, os.path.basename(dst)))
    return dst


TESTS = [
    ("joy", "(◕‿◕)"),
    ("cry", "(╥﹏╥)"),
    ("angry", "(｀д´)"),
    ("blank", "(・_・)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("-k", "--kaomoji")
    ap.add_argument("--tags", help="只做映射：颜文字 -> 表情 tag，不出图")
    ap.add_argument("--render", help="只渲染不出图")
    ap.add_argument("--test", action="store_true", help="跑一组表情对照")
    ap.add_argument("--denoise", type=float, default=0.88)
    ap.add_argument("--cn", action="store_true", help="叠加 scribble 线稿约束")
    ap.add_argument("--prompt")
    ap.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")

    if a.tags:
        r = parse(a.tags)
        if not r["tags"]:
            print("表里没有这个颜文字。现在收了 %d 个常见款：" % len(KAOMOJI_MAP))
            for i, k in enumerate(sorted(KAOMOJI_MAP), 1):
                print("  %-16s %s" % (k, ", ".join(KAOMOJI_MAP[k][:3])))
                if i % 8 == 0:
                    print()
            return 2
        for k, ts in r["found"]:
            print("%-16s -> %s" % (k, ", ".join(ts)))
        print("\n拼进提示词：%s" % ", ".join(r["tags"]))
        return 0

    if a.render:
        p = os.path.join(OUT_DIR, "render_%s.png" % stamp)
        _, wh = render(a.render, p)
        print("渲染 %s -> %s 字形 %dx%d" % (a.render, p, wh[0], wh[1]))
        return 0

    p = SL.load(a.profile)
    p["seed"] = a.seed

    if a.test:
        print("颜文字表情对照（denoise %.2f，%s）"
              % (a.denoise, "带线稿约束" if a.cn else "纯 img2img"))
        outs = []
        for tag, kao in TESTS:
            r = run_one(p, kao, tag, a.denoise, a.cn, stamp, a.prompt)
            if r:
                outs.append((kao, r))
        if len(outs) >= 2:
            from PIL import Image, ImageDraw
            tiles = [(kao, Image.open(f).convert("RGB")) for kao, f in outs]
            tw = 420
            ims = [(k, im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS))
                   for k, im in tiles]
            h = min(im.height for _, im in ims)
            W = sum(im.width for _, im in ims) + 10 * (len(ims) - 1)
            cv = Image.new("RGB", (W, h + 34), (18, 18, 18))
            d = ImageDraw.Draw(cv)
            x = 0
            for k, im in ims:
                cv.paste(im.crop((0, 0, im.width, h)), (x, 34))
                d.text((x + 6, 11), k, fill=(255, 255, 255))
                x += im.width + 10
            cp = os.path.join(OUT_DIR, "cmp_%s.jpg" % stamp)
            cv.save(cp, "JPEG", quality=93)
            print("\n对比图 %s" % cp)
        return 0

    if not a.kaomoji:
        raise SystemExit("给 -k 一个颜文字，或加 --test / --render")
    r = run_one(p, a.kaomoji, "one", a.denoise, a.cn, stamp, a.prompt)
    if r:
        print("成品 %s" % r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
