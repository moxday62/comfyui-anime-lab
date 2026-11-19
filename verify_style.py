# -*- coding: utf-8 -*-
"""风格模型 A/B 验证 + IPAdapter 锁角色 + 手部修复 端到端脚本。

用法（需先 install_style.py 并重启 ComfyUI）：
  python verify_style.py ab          # A/B 对比 WAI vs Hassaku
  python verify_style.py ip          # IPAdapter 喂图锁角色测试
  python verify_style.py hand        # 手部局部重绘测试
"""
import sys
import os
import json
import time
import uuid
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY
import style_workflows_api as W

SERVER = "http://127.0.0.1:8188"

# 参考图来源。原来是写死的某台机器上的剪贴板临时文件路径，换台机器就跑不了。
# 改成环境变量：
#   Windows:  set REF_SRC=D:\pics\xxx.jpg
#   Linux:    REF_SRC=/home/me/xxx.jpg python verify_style.py ip
# 留空则跳过复制那一步 —— 前提是 ComfyUI/input 里已经放好 ref_original.jpg。
REF_SRC = os.environ.get("REF_SRC", "")


def run(name, workflow):
    data = json.dumps({"prompt": workflow, "client_id": "style_verify"}).encode()
    req = urllib.request.Request(SERVER + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        prompt_id = json.loads(resp.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"[{name}] SUBMIT FAILED: {e.code}")
        print(e.read().decode()[:2000])
        return False

    print(f"[{name}] submitted, prompt_id={prompt_id}")
    t0 = time.time()
    while True:
        try:
            hist = json.load(urllib.request.urlopen(SERVER + "/history/" + prompt_id, timeout=30))
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in hist:
            item = hist[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                elapsed = time.time() - t0
                print(f"[{name}] OK in {elapsed:.1f}s")
                return True
            messages = status.get("messages", [])
            errs = [m for m in messages if m[0] == "execution_error"]
            if errs:
                print(f"[{name}] EXECUTION ERROR:")
                print(json.dumps(errs[0][1], ensure_ascii=False)[:2000])
                return False
        if time.time() - t0 > 1800:
            print(f"[{name}] TIMEOUT")
            return False
        time.sleep(2)


def ab():
    print("== A/B: WAI v17 vs Hassaku v2.2 ==")
    for ckpt, tag in [(W.WAI, "wai"), (W.HASSAKU, "hassaku")]:
        p = W.base_portrait(ckpt=ckpt, prefix=f"style_ab/{tag}")
        run(tag, p)


def ip():
    print("== IPAdapter 喂图锁角色 ==")
    import shutil
    ref = "ref_original.jpg"
    indir = os.path.join(COMFY, "input")
    src = REF_SRC
    if os.path.isfile(src) and not os.path.isfile(os.path.join(indir, ref)):
        shutil.copy2(src, os.path.join(indir, ref))
        print(f"  copied ref -> {ref}")
    for wt, tag in [(0.6, "ip06"), (0.9, "ip09")]:
        p = W.base_portrait(ckpt=W.WAI, ref_image=ref, ip_weight=wt,
                            prefix=f"style_ip/{tag}")
        run(tag, p)


def hand():
    print("== 手部 latent 局部重绘 ==")
    import glob
    indir = os.path.join(COMFY, "input")
    # 优先用 IP 结果当底图，其次任意 output 图
    cands = sorted(glob.glob(r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\style_ip\ip06_*.png"))
    if not cands:
        cands = sorted(glob.glob(r"D:\AI-Video-Setup\output\*final*.jpg"))
    if not cands:
        print("  未找到底图，请先跑 ab 或 ip")
        return
    src = cands[-1]
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.save(os.path.join(indir, "inpaint_base.png"))
    print(f"  base -> {os.path.basename(src)} {im.size}")

    # 手部 box：默认左下 V 字手势区，实际使用请按图调整
    box = (int(w * 0.05), int(h * 0.44), int(w * 0.26), int(h * 0.29))
    W.make_hand_mask(os.path.join(indir, "inpaint_base.png"),
                     os.path.join(indir, "hand_mask.png"), box, feather=14)
    print(f"  mask box={box}")

    ckpt = W.WAI if os.path.isfile(os.path.join(
        r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\models\checkpoints", W.WAI)) else W.ILXL
    hand_prompt = ("1girl, hand doing peace sign, two fingers extended, five fingers, "
                   "natural hand, correct anatomy, clean fingers, cream skin, "
                   "soft diffuse lighting, soft painterly illustration, masterpiece, best quality")
    p = W.hand_inpaint("inpaint_base.png", "hand_mask.png", ckpt=ckpt,
                       steps=28, denoise=0.55, prompt=hand_prompt,
                       prefix="style_hand/fixed")
    run("hand_fix", p)


def final():
    """一键成品：WAI + IPAdapter 锁角色 + 手部局部重绘。"""
    print("== 最终流程：WAI + IPAdapter + 手部修复 ==")
    import glob
    from PIL import Image
    indir = os.path.join(COMFY, "input")
    ref = "ref_original.jpg"
    ref_src = REF_SRC
    if not os.path.isfile(os.path.join(indir, ref)) and os.path.isfile(ref_src):
        import shutil
        shutil.copy2(ref_src, os.path.join(indir, ref))
        print(f"  ref -> {ref}")

    # 1) 出图
    p = W.base_portrait(ckpt=W.WAI, ref_image=ref, ip_weight=0.6, prefix="style_final/raw")
    if not run("final_raw", p):
        print("  出图失败")
        return

    # 2) 找刚出的图 → 修手
    cands = sorted(glob.glob(r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\style_final\raw_*.png"),
                   key=os.path.getmtime)
    if not cands:
        print("  未找到输出")
        return
    im = Image.open(cands[-1]).convert("RGB")
    w, h = im.size
    im.save(os.path.join(indir, "final_base.png"))
    box = (int(w * 0.05), int(h * 0.44), int(w * 0.26), int(h * 0.29))
    W.make_hand_mask(os.path.join(indir, "final_base.png"),
                     os.path.join(indir, "final_mask.png"), box, feather=14)
    hand_prompt = ("1girl, hand doing peace sign, two fingers extended, five fingers, "
                   "natural hand, correct anatomy, clean fingers, cream skin, "
                   "soft diffuse lighting, soft painterly illustration, masterpiece, best quality")
    p2 = W.hand_inpaint("final_base.png", "final_mask.png", ckpt=W.WAI,
                        steps=28, denoise=0.55, prompt=hand_prompt,
                        prefix="style_final/fixed")
    run("final_fixed", p2)


def tgcg():
    """3DCG 手办质感（NOVA 3DCG XL）+ 可选 IPAdapter 锁参考图。"""
    print("== 3DCG 手办质感（系列风格） ==")
    ref = "ref_3d.jpg"
    indir = os.path.join(COMFY, "input")
    ref_src = REF_SRC
    if os.path.isfile(ref_src) and not os.path.isfile(os.path.join(indir, ref)):
        import shutil
        shutil.copy2(ref_src, os.path.join(indir, ref))
        print(f"  ref -> {ref}")

    # 提示词用 W.POS_3DCG 定稿版（soft matte skin + 压气泡）
    run("3dcg_plain", W.nova_3dcg(prefix="style_3dcg/plain"))
    run("3dcg_ip", W.nova_3dcg(ref_image=ref, ip_weight=0.65,
                               prefix="style_3dcg/final"))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    {"ab": ab, "ip": ip, "hand": hand, "final": final, "3dcg": tgcg}.get(which, ab)()
