"""对比几种背景/投影提示词，找出"最干净的单色底"。

为什么要单独测这个：
  之前合成的深色轮廓、绿边、悬空感，根子都在源图——
  模型会自己在人物身后画一圈投影（RGB 1,0,1 / alpha 254），
  还会把"plain light gray"读成橄榄绿。
  这些能用提示词解决的话，下游的 despill / 去投影 / 调色全都可以省掉。

指标（客观，不靠肉眼）：
  bg_std    背景区亮度标准差 —— 越小越均匀（理想 < 6）
  bg_hue    背景平均色（BGR）—— 判断有没有被染成绿/蓝
  ring_ratio 人物轮廓外侧 4~14px 一圈的平均亮度 / 背景亮度
             < 0.9 说明有人造投影，越接近 1 越干净
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

import style_lib as SL
import person_mask as PM

OUT = r"D:\AI-Video-Setup\output\bgtest"
os.makedirs(OUT, exist_ok=True)

# 人物主体描述（固定不变，只换背景部分）
BODY = ("1girl, solo, full body, standing, one arm raised up, arm up, "
        "other hand on hip, hand on hip, looking at viewer, "
        "pale skin, soft matte skin, pink hair, short hair, blue eyes, "
        "white sailor uniform, navy collar, blue bow, cropped top, "
        "white pleated skirt, white thighhighs, white shoes, ")

BASE_NEG = ("worst quality, low quality, blurry, bad anatomy, extra limbs, "
            "bad hands, missing fingers, watermark, text, signature, "
            "cropped, out of frame, close-up, cowboy shot, "
            "speech bubble, thought bubble, glossy plastic, "
            "nude, nsfw, naked, revealing clothes, cleavage, low angle, ")

# 三组：A 自然语言描述 / B Danbooru 标签 / C 标签 + 强压投影
VARIANTS = {
    "A_natural": dict(
        pos=BODY + "plain light gray background, solid seamless backdrop, "
                   "flat even lighting, no scenery, studio backdrop",
        neg=BASE_NEG + "dark background, gradient background, spotlight, "
                       "vignette, scenery, detailed background, wall texture, "
                       "cast shadow, drop shadow, rim lighting, dramatic lighting",
    ),
    "B_tags": dict(
        pos=BODY + "grey background, simple background, white background, "
                   "flat color background, no lineart background",
        neg=BASE_NEG + "detailed background, scenery, gradient background, "
                       "spotlight, vignette, shadow, cast shadow, drop shadow",
    ),
    "C_tags_strong": dict(
        pos=BODY + "(grey background:1.3), (simple background:1.3), "
                   "flat color background, plain backdrop",
        neg=BASE_NEG + "detailed background, scenery, gradient background, "
                       "spotlight, vignette, (shadow:1.3), (cast shadow:1.4), "
                       "(drop shadow:1.4), shadow on wall, floor shadow, "
                       "rim lighting, backlight, glow",
    ),
}

PROFILE = "soft"
SIZE = "768x1344"


def measure(path):
    img = cv2.imread(path)
    a = PM.alpha(path)
    bgsel = a < 20
    if bgsel.sum() < 500:
        return None
    bg = img[bgsel].reshape(-1, 3).astype(np.float32)
    lum = bg.mean(axis=1)
    bg_std = float(lum.std())
    bg_mean = bg.mean(axis=0)

    # 轮廓外侧 4~14px 的环
    person = (a > 128).astype(np.uint8)
    d_out = cv2.distanceTransform(1 - person, cv2.DIST_L2, 5)
    ring = (d_out > 4) & (d_out < 14)
    if ring.sum() < 200:
        return None
    ring_lum = float(img[ring].reshape(-1, 3).astype(np.float32).mean())
    return dict(bg_std=bg_std, bg_mean=bg_mean, ring_ratio=ring_lum / max(1.0, lum.mean()),
                ring_px=int(ring.sum()))


def main():
    p = SL.load(PROFILE)
    w, h = [int(v) for v in SIZE.split("x")]
    results = {}
    for name, cfg in VARIANTS.items():
        print("\n=== %s" % name)
        before = SL.snapshot()
        t = time.time()
        # build() 直接读画风档里的 positive/negative，所以改档再跑
        p["positive"] = cfg["pos"]
        p["negative"] = cfg["neg"]
        wf = SL.build(p, seed=20261124, size=(w, h),
                      prefix="bgtest/%s" % name)
        if not SL.submit(wf, tag=name):
            print("  提交失败")
            continue
        got = SL.new_files(before)
        if not got:
            print("  没产出")
            continue
        dst = os.path.join(OUT, "%s.png" % name)
        import shutil
        shutil.copy2(got[-1], dst)
        m = measure(dst)
        print("  %.1fs -> %s" % (time.time() - t, os.path.basename(dst)))
        if m:
            print("  背景均匀度 std=%.1f | 背景色 BGR=(%.0f,%.0f,%.0f) | 轮廓环亮度比 %.3f"
                  % (m["bg_std"], *m["bg_mean"], m["ring_ratio"]))
            results[name] = m

    print("\n==== 汇总（ring 越接近 1.0 越干净，bg_std 越小越均匀）")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["ring_ratio"]):
        print("  %-14s ring=%.3f  std=%.1f  BGR=(%.0f,%.0f,%.0f)"
              % (k, v["ring_ratio"], v["bg_std"], *v["bg_mean"]))


if __name__ == "__main__":
    main()
