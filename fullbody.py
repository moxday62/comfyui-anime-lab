# -*- coding: utf-8 -*-
"""全身立绘：一次出图 -> 4x 超分 -> 光效收尾。

为什么单开一个脚本：
  画风档（profiles/*.json）为了半身构图的观感，尺寸锁在 832x1216（1:1.46），
  提示词里也带了 sitting on bed 这类坐姿。这两项都会让模型自然裁到大腿，
  所以"想要全身"不是改一句提示词能解决的，尺寸和姿态都得一起换。

  全身站立的经验比例是 1:1.7 ~ 1:2，这里默认 768x1344。
  再配 from head to toe / feet visible 这类明确词，负向压掉 cowboy shot。

用法：
  python fullbody.py                          # 3dcg 档，全身站立，随机种子
  python fullbody.py --profile soft           # 换画风
  python fullbody.py -p "你的提示词" --seed 123
  python fullbody.py --no-upscale             # 只要 768x1344，快一半
"""
import argparse
import os
import random
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import style_lib as SL
from texture_pipeline import post_process

OUT_DIR = os.path.join(BASE, "output", "fullbody")

# 全身构图的正向骨架：保留画风档的材质词，只把姿态和取景换掉
FULL_POS = (
    "1girl, solo, petite, slender, full body, standing, from head to toe, "
    "feet visible, white shoes, arms at sides, "
    "pale skin, porcelain skin, soft matte skin, delicate skin, "
    "pink hair, short hair, blue eyes, half-closed eyes, sleepy expression, "
    "white sailor uniform, short sleeves, navy collar, blue bow, cropped top, "
    "white pleated skirt, white thighhighs, "
    "plain wall background, soft studio lighting, indoor, "
    "masterpiece, best quality, amazing quality, very aesthetic, absurdres, "
    "subsurface scattering, 3d, cgi, octane render, "
    "cinematic lighting, soft global illumination, highly detailed"
)

# 负向：把"半身/裁切/特写"这些会把全身挤掉的语义压掉
FULL_NEG_ADD = (
    "cropped, out of frame, close-up, portrait, cowboy shot, upper body, "
    "knees cut off, feet out of frame, sitting, lying, "
    "extra limbs, bad anatomy, bad feet, missing toes, deformed feet"
)


def upscale(p, base_name, prefix):
    """4x 超分。ImageUpscaleWithModel 只能读 input 目录，所以调用前要先拷进去。"""
    g = SL.Graph()
    im = g.node("LoadImage", image=base_name)
    um = g.node("UpscaleModelLoader",
                model_name=(p.get("upscale") or {}).get(
                    "model", "4x_NMKD-Superscale-SP_178000_G.pth"))
    ui = g.node("ImageUpscaleWithModel", upscale_model=g.ref(um, 0), image=g.ref(im, 0))
    g.node("SaveImage", images=g.ref(ui, 0), filename_prefix=prefix)
    return g.nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="3dcg")
    ap.add_argument("-p", "--prompt", help="自定义正向提示词（整句替换，不追加）")
    ap.add_argument("--size", default="768x1344", help="宽x高，默认 768x1344（约 1:1.75）")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--ref", help="参考图（喂图锁角色）")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", default=True)
    a = ap.parse_args()

    p = SL.load(a.profile)
    w, h = [int(v) for v in a.size.lower().replace("*", "x").split("x")]
    p["size"] = [w, h]
    p["positive"] = a.prompt or FULL_POS
    p["negative"] = (p.get("negative", "") + ", " + FULL_NEG_ADD).strip(", ")

    ref = SL.prepare_ref(p, a.ref)
    seed = a.seed if a.seed is not None else (
        random.randint(0, 2 ** 48) if a.random else p.get("seed", 0))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")

    print("画风 %s | %dx%d | %d 步 CFG %s | 种子 %s"
          % (a.profile, w, h, p["steps"], p["cfg"], seed))
    if ref:
        print("喂图 %s（权重 %.2f）" % (ref, (p.get("ipadapter") or {}).get("weight", 0)))

    print("\n[出图] ...")
    before = SL.snapshot()
    t0 = time.time()
    wf = SL.build(p, ref_image=ref, seed=seed, prefix="fullbody/base_%s" % stamp)
    if not SL.submit(wf, tag="出图"):
        return 1
    files = SL.new_files(before)
    if not files:
        print("出图失败")
        return 1
    cur = files[-1]
    print("  %.1fs -> %s" % (time.time() - t0, os.path.basename(cur)))

    if a.upscale:
        print("\n[超分] 4x ...")
        src = os.path.join(SL.INPUT_DIR, "fb_up_%s.png" % stamp)
        shutil.copy2(cur, src)
        before2 = SL.snapshot()
        t1 = time.time()
        if SL.submit(upscale(p, os.path.basename(src), "fullbody/up_%s" % stamp), tag="超分"):
            got = SL.new_files(before2)
            if got:
                cur = got[-1]
                print("  %.1fs -> %s" % (time.time() - t1, os.path.basename(cur)))

        print("\n[光效] 高光 + 局部对比 + 锐化 ...")
        before3 = SL.snapshot()
        t2 = time.time()
        dst_png = os.path.join(OUT_DIR, "final_%s.png" % stamp)
        inp = os.path.join(SL.INPUT_DIR, "fb_ps_%s.png" % stamp)
        shutil.copy2(cur, inp)
        post_process(inp, dst_png)
        print("  %.1fs -> %s" % (time.time() - t2, os.path.basename(dst_png)))
        final = dst_png
    else:
        final = os.path.join(OUT_DIR, "final_%s.png" % stamp)
        shutil.copy2(cur, final)

    from PIL import Image
    im = Image.open(final)
    print("\n成品：%s  (%dx%d)" % (final, im.width, im.height))
    return 0


if __name__ == "__main__":
    sys.exit(main())
