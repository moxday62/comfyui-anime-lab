# -*- coding: utf-8 -*-
"""分层作画：把画师的分层流程，翻译成可以逐层验证的机器步骤。

先讲清楚一件事（很重要）：
  扩散模型不是分层渲染器。你不会真的让它"先铺固有色、再二分、再打高光"，
  它一次去噪同时决定结构、明暗、材质。硬要分层，只能把每一层变成
  **一次独立的图像操作**，层与层之间靠遮罩和约束衔接。

  所以这里做的是「空间分层 × 工序分层」：
      空间：脸/眼、手、配饰……各自一块区域
      工序：出图 -> 局部重绘 -> 超分 -> 光效叠加 -> 调色收尾
  这跟画师的"颜色分层"不是一回事，但好处一样：每层都能单独看、单独撤。

五层：
  L1 结构层   骨架/提示词定构图（一次出图，低步数也能定住）
  L2 脸部层   rtmw 自动定位脸 -> 局部重绘眼睛/睫毛/皮肤
  L3 手部层   rtmw 自动找手 -> 局部重绘手指（检测不到就跳过）
  L4 超分层   4x 超分，把已有的高频还原出来（不是无中生有）
  L5 光效层   高光提亮 + 局部对比 + 锐化（纯 PIL）

产物：每一层的 PNG 单独存，另出一张五联对比条，能一眼看出哪层在起作用。

用法：
  python layer_paint.py --profile 3dcg --random
  python layer_paint.py --profile soft --random --no-hand
  python layer_paint.py --profile 3dcg --no-upscale     跳过超分（快一半）
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
import style_workflows_api as SW
from texture_pipeline import post_process

RTMLIB = os.path.join(BASE, "rtmlib_src", "rtmlib-main")
OUT_DIR = os.path.join(BASE, "output", "layers")

FACE_POS = ("detailed eyes, catchlight, clean skin, soft matte skin, "
            "subsurface scattering, keep the same expression, same eye shape, "
            "same gaze direction, masterpiece, best quality")
HAND_POS = ("natural hand, correct anatomy, clean fingers, five fingers, "
            "soft matte skin, subsurface scattering, detailed knuckles, "
            "masterpiece, best quality")


def detect_parts(img_path):
    """rtmw 定位脸和手，返回 {'face': box, 'hand_l': box, 'hand_r': box}（像素坐标）。"""
    sys.path.insert(0, RTMLIB)
    import cv2
    from rtmlib import Wholebody

    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    img = cv2.imread(img_path)
    if img is None:
        return {}
    h, w = img.shape[:2]
    kpts, scores = model(img)
    if kpts is None or len(kpts) == 0:
        return {}
    kp, sc = kpts[0], scores[0]

    def box_from(idx, thr=0.4, pad=0.12):
        pts = [(kp[i], sc[i]) for i in idx if i < len(kp) and sc[i] > thr]
        if len(pts) < max(3, len(idx) // 6):
            return None
        xs = [p[0][0] for p in pts]
        ys = [p[0][1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bw < 8 or bh < 8:
            return None
        x0 = max(0, int(min(xs) - bw * pad))
        y0 = max(0, int(min(ys) - bh * pad))
        x1 = min(w, int(max(xs) + bw * pad))
        y1 = min(h, int(max(ys) + bh * pad))
        return (x0, y0, x1, y1)

    out = {}
    fb = box_from(range(23, 91), thr=0.4, pad=0.10)   # 脸部 68 点
    if fb:
        out["face"] = fb
    for name, idx in (("hand_l", range(91, 112)), ("hand_r", range(112, 133))):
        hb = box_from(idx, thr=0.45, pad=0.18)
        if hb:
            out[name] = hb
    return out


def inpaint(p, base_name, mask_name, prompt, denoise, prefix):
    """局部重绘：完整编码原图 + 只给 mask 加噪（无接缝）。"""
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)
    pos = g.node("CLIPTextEncode", clip=clip, text=prompt)
    neg = g.node("CLIPTextEncode", clip=clip, text=p.get("negative", ""))
    im = g.node("LoadImage", image=base_name)
    mk = g.node("LoadImageMask", image=mask_name, channel="red")
    enc = g.node("VAEEncode", pixels=g.ref(im, 0), vae=vae)
    noise = g.node("SetLatentNoiseMask", samples=g.ref(enc, 0), mask=g.ref(mk, 0))
    ks = g.node("KSampler", model=model, seed=int(p.get("seed", 0) + 7),
                steps=int(p.get("steps", 28)), cfg=float(p.get("cfg", 5.0)),
                sampler_name=p.get("sampler", "euler_ancestral"),
                scheduler=p.get("scheduler", "normal"), denoise=denoise,
                positive=g.ref(pos, 0), negative=g.ref(neg, 0),
                latent_image=g.ref(noise, 0))
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


def upscale_graph(p, base_name, prefix):
    g = SL.Graph()
    im = g.node("LoadImage", image=base_name)
    um = g.node("UpscaleModelLoader",
                model_name=(p.get("upscale") or {}).get(
                    "model", "4x_NMKD-Superscale-SP_178000_G.pth"))
    ui = g.node("ImageUpscaleWithModel", upscale_model=g.ref(um, 0), image=g.ref(im, 0))
    g.node("SaveImage", images=g.ref(ui, 0), filename_prefix=prefix)
    return g.nodes


def contact_sheet(paths, out_path, labels):
    """把各层缩成等高小图横排，一眼看演进。"""
    from PIL import Image, ImageDraw
    th = 640
    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        s = th / float(im.height)
        tiles.append(im.resize((int(im.width * s), th), Image.LANCZOS))
    gap = 8
    W = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (W, th + 26), (22, 22, 22))
    x = 0
    d = ImageDraw.Draw(canvas)
    for t, lab in zip(tiles, labels):
        canvas.paste(t, (x, 26))
        d.text((x + 6, 7), lab, fill=(235, 235, 235))
        x += t.width + gap
    canvas.save(out_path, "JPEG", quality=90)
    return canvas.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("-p", "--prompt")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--face", action="store_true",
                    help="开启脸部层（全脸重绘会把半闭眼改成睁眼，默认关）")
    ap.add_argument("--no-hand", action="store_true")
    ap.add_argument("--hand-auto", action="store_true",
                    help="用 rtmw 自动找手（二次元图经常框到脚上，默认关）")
    ap.add_argument("--hand-box", help="手动指定手部框 x0,y0,x1,y1（像素或 0-1 比例）")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", default=True)
    ap.add_argument("--face-denoise", type=float, default=0.30)
    ap.add_argument("--hand-denoise", type=float, default=0.50)
    a = ap.parse_args()

    import random
    p = SL.load(a.profile)
    if a.prompt:
        p["positive"] = a.prompt
    ref = SL.prepare_ref(p, a.ref)
    seed = a.seed if a.seed is not None else (
        random.randint(0, 2 ** 48) if a.random else p.get("seed", 0))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    stages, labels = [], []

    print("画风 %s | %s | %d 步 CFG %s | 种子 %s"
          % (a.profile, "x".join(map(str, p["size"])), p["steps"], p["cfg"], seed))

    # L1 ------------------------------------------------------------
    print("\n[L1 结构层] 出图 ...")
    before = SL.snapshot()
    t0 = time.time()
    wf = SL.build(p, ref_image=ref, seed=seed, prefix="layers/base_%s" % stamp)
    if not SL.submit(wf, tag="L1"):
        return 1
    files = SL.new_files(before)
    if not files:
        print("出图失败")
        return 1
    cur = files[-1]
    print("  %.1fs -> %s" % (time.time() - t0, os.path.basename(cur)))
    stages.append(cur)
    labels.append("L1 结构")

    # L2/L3 ---------------------------------------------------------
    parts = detect_parts(cur)
    print("  自动定位：%s" % (", ".join(sorted(parts)) or "没检测到部位（会跳过局部层）"))

    def run_part(tag, box, prompt, denoise, base_dn):
        nonlocal cur
        src = os.path.join(SL.INPUT_DIR, "lp_base_%s.png" % tag)
        msk = os.path.join(SL.INPUT_DIR, "lp_mask_%s.png" % tag)
        shutil.copy2(cur, src)
        SW.make_hand_mask(src, msk, box, feather=12)
        print("\n[%s] 重绘 denoise %.2f ..." % (tag, denoise))
        before2 = SL.snapshot()
        t2 = time.time()
        wf2 = inpaint(p, os.path.basename(src), os.path.basename(msk),
                      prompt, denoise, "layers/%s_%s" % (tag[:2], stamp))
        if not SL.submit(wf2, tag=tag):
            print("  失败，跳过这层")
            return
        got = SL.new_files(before2)
        if not got:
            print("  没产出，跳过")
            return
        cur = got[-1]
        print("  %.1fs -> %s" % (time.time() - t2, os.path.basename(cur)))
        stages.append(cur)
        labels.append(tag)

    if parts.get("face") and a.face:
        run_part("L2 脸部", parts["face"], FACE_POS, a.face_denoise, 0)

    if not a.no_hand:
        from PIL import Image as _Im
        W0, H0 = _Im.open(cur).size
        boxes = []
        if a.hand_box:
            vals = [float(v) for v in a.hand_box.split(",")]
            if max(vals) <= 1.0:
                vals = [vals[0] * W0, vals[1] * H0, vals[2] * W0, vals[3] * H0]
            boxes.append(("L3 手部", tuple(int(v) for v in vals)))
        elif a.hand_auto:
            for name in ("hand_l", "hand_r"):
                b = parts.get(name)
                if not b:
                    continue
                # rtmw 在二次元图上常把脚认成手，落在画面下部 70% 以外的一律不信
                if b[1] > 0.72 * H0:
                    print("\n  rtmw 给的 %s 框在 (%.0f,%.0f)，落在身体下部，"
                          "多半是脚，已忽略" % (name, b[0], b[1]))
                    continue
                boxes.append(("L3 手部" if not boxes else "L3 手部b", b))
        for tag, box in boxes:
            run_part(tag, box, HAND_POS, a.hand_denoise, 0)
        if not boxes:
            print("\n[L3 手部] 跳过：自动检测在二次元图上不可靠（这次就把脚认成手），")
            print("  需要修手就手动框：--hand-box x0,y0,x1,y1（像素，或 0-1 比例）")

    # L4 ------------------------------------------------------------
    if a.upscale:
        print("\n[L4 超分层] 4x 超分 ...")
        src = os.path.join(SL.INPUT_DIR, "lp_up_%s.png" % stamp)
        shutil.copy2(cur, src)
        before3 = SL.snapshot()
        t3 = time.time()
        if SL.submit(upscale_graph(p, os.path.basename(src), "layers/up_%s" % stamp),
                     tag="L4"):
            got = SL.new_files(before3)
            if got:
                cur = got[-1]
                print("  %.1fs -> %s" % (time.time() - t3, os.path.basename(cur)))
                stages.append(cur)
                labels.append("L4 超分")

    # L5 ------------------------------------------------------------
    print("\n[L5 光效层] 高光 + 局部对比 + 锐化 ...")
    t4 = time.time()
    final = os.path.join(OUT_DIR, "final_%s.png" % stamp)
    size = post_process(cur, final)
    print("  %.1fs -> %dx%d" % (time.time() - t4, size[0], size[1]))
    stages.append(final)
    labels.append("L5 光效")

    sheet = os.path.join(OUT_DIR, "stages_%s.jpg" % stamp)
    contact_sheet(stages, sheet, labels)
    print("\n五联对比：%s" % sheet)
    print("成品目录：%s" % OUT_DIR)
    try:
        os.startfile(OUT_DIR)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
