"""一条命令跑完整条链路：骨架(可选) -> 生成(锁角色) -> 超分 -> 换场景。

设计原则：每一段都能关。
  想精确定制动作 -> 开骨架，指定预设
  想省事随便出   -> 关骨架，提示词里随机塞一句姿势，模型自己发挥
  只想要 768 草图 -> 关超分、关换场景

开关存在 toggles.json 里，命令行参数可以临时覆盖（只在这次生效）。
配 switch.py 可以可视化翻开关，不用记参数。

默认参数是实测出来的，不是猜的：
  profile      3dcg      NOVA 3DCG XL，手办质感
  骨架强度     1.0 / 生效到 0.75 步   —— 低于这个，姿势会被文字压住
  IP-Adapter   权重 0.80 / 0.45 步收手 —— 收到 0.65 会留一圈贴纸描边
  背景         交给提示词，不在正向里描述场景
  换场景       软 alpha + 去投影 + 接触阴影 + 色温匹配

用法：
  python make.py                                  用 toggles.json 直接跑
  python make.py -P reach_up -p "white hair, ..."   临时指定骨架和提示词
  python make.py -P off                           关骨架，随机姿势
  python make.py --scene cherry --upscale         临时开场景和超分
  python make.py --save                           把这次参数写回 toggles.json
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import style_lib as SL
import multi_ref as MR
import matte_composite as MC
import skeleton_gen as SG

BASE = r"D:\AI-Video-Setup"
OUT_ROOT = os.path.join(BASE, "output", "make")
REF_DIR = os.path.join(BASE, "refs", "char")
TOGGLES = os.path.join(BASE, "toggles.json")

PRESETS = ["reach_up", "jump", "stand", "sit"]
POSE_MODES = ["off", "auto"] + PRESETS

# 关骨架时用的姿势池。这些是"模型会画但不会主动想到"的动作，
# 每次随机抽一句塞进提示词，保证不同种子出来的动作有区分度，
# 而不是每次都端端正正站着。
POSE_POOL = [
    ("standing, hands behind back, looking at viewer", ""),
    ("standing, arms crossed, confident expression", ""),
    ("standing, one hand on hip, looking at viewer", ""),
    ("walking, one leg forward, arms swinging", "standing still"),
    ("sitting on a chair, legs crossed, hands on lap", "standing"),
    ("sitting on the floor, hugging knees", "standing"),
    ("leaning against a wall, one leg bent, arms relaxed", ""),
    ("crouching, hands on knees, looking up", "standing"),
    ("stretching, both arms above head, eyes closed", ""),
    ("running, dynamic pose, hair flowing", "standing still"),
    ("side view, standing, looking to the side", "from behind, back view"),
    ("standing on tiptoe, arms slightly out for balance", ""),
]

DEFAULT_TOGGLES = {
    "profile": "3dcg",
    "size": "768x1344",
    "pose": "off",
    "char": True,
    "upscale": True,
    "scene": "",
    "seed": None,
    "prompt_extra": "",
    "ref": "ref_0998187d2dca",
    "pack": [],
    "nl_tags": True,
    "kaomoji": "",
}


def load_toggles():
    if os.path.exists(TOGGLES):
        try:
            with open(TOGGLES, encoding="utf-8") as f:
                d = json.load(f)
            out = dict(DEFAULT_TOGGLES)
            out.update(d)
            return out
        except Exception:
            pass
    return dict(DEFAULT_TOGGLES)


def save_toggles(d):
    with open(TOGGLES, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def resolve_ref(name):
    """把参考图名字解析成 ComfyUI/input 里的文件名。"""
    if not name:
        return None
    direct = os.path.join(SL.INPUT_DIR, name)
    if os.path.exists(direct):
        return name

    cand = name if name.lower().endswith(".png") else name + ".png"
    p = os.path.join(REF_DIR, cand)
    if os.path.exists(p):
        dst = os.path.join(SL.INPUT_DIR, "mk_" + cand)
        shutil.copy2(p, dst)
        return os.path.basename(dst)

    if os.path.exists(name):
        import prep_refs as PR
        im, _ = PR.prepare(name)
        dst = os.path.join(SL.INPUT_DIR, "mk_auto_ref.png")
        im.save(dst)
        print("  现场处理参考图 -> %s" % os.path.basename(dst))
        return os.path.basename(dst)

    raise SystemExit("找不到参考图：%s" % name)


def resolve_pose(mode, size):
    """返回 (骨架文件名 or None, 姿势词, 额外负向, 是否随机)。

    mode:
      off        不用骨架，随机抽一句姿势描述
      auto       随机抽一个骨架预设（动作多样，但有结构约束）
      reach_up / jump / stand / sit   指定骨架
      其它       当作 input 里已有的骨架文件名
    """
    if not mode or mode == "off":
        phrase, bad = random.choice(POSE_POOL)
        return None, phrase, bad, True

    if mode == "auto":
        preset = random.choice(PRESETS)
        print("  自动选骨架：%s" % preset)
    else:
        preset = mode

    w, h = size
    if preset in SG.PRESETS:
        out = os.path.join(SL.INPUT_DIR, "mk_pose_%s.png" % preset)
        SG.draw_pose(SG.PRESETS[preset], w, h, zoom=0.86).save(out)
        print("  生成骨架 %s -> %s" % (preset, os.path.basename(out)))
        return os.path.basename(out), MR.pose_hint(preset), "", False

    if os.path.exists(os.path.join(SL.INPUT_DIR, preset)):
        return preset, MR.pose_hint(preset), "", False

    raise SystemExit("找不到骨架：%s（可选 off / auto / %s）" % (mode, " / ".join(PRESETS)))


def upscale(p, src_path, prefix):
    """4x 超分。UpscaleModelLoader 只能读 input，所以先拷进去。"""
    import fullbody as FB
    base = "mk_up_src.png"
    shutil.copy2(src_path, os.path.join(SL.INPUT_DIR, base))
    before = SL.snapshot()
    t = time.time()
    wf = FB.upscale(p, base, prefix)
    if not SL.submit(wf, tag="超分"):
        return None
    got = SL.new_files(before)
    if not got:
        return None
    print("  超分 %.1fs -> %s" % (time.time() - t, os.path.basename(got[-1])))
    return got[-1]


def ensure_comfy(wait=180):
    """确认 ComfyUI 在跑，没跑就拉起来。

    没有它的话，第一次点"出图"只会看到一串 connection refused，
    新手会以为是自己操作错了。这里顺便把等待时间也打出来。
    """
    import json
    import subprocess
    import urllib.request

    def alive():
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3)
            return True
        except Exception:
            return False

    if alive():
        return True
    print("ComfyUI 没在跑，正在启动（首次约 30-60 秒）…")
    root = r"D:\ComfyUI\ComfyUI_windows_portable"
    py = os.path.join(root, "python_embeded", "python.exe")
    log = os.path.join(BASE, "comfyui_hq.log")
    subprocess.Popen([py, "-s", os.path.join(root, "ComfyUI", "main.py"),
                      "--windows-standalone-build"],
                     cwd=root, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    t = time.time()
    while time.time() - t < wait:
        if alive():
            print("  已就绪（%.0f 秒）" % (time.time() - t))
            return True
        time.sleep(3)
    print("  等太久，放弃。请手动双击 4_启动_高画质模式.bat")
    return False


def fit_pose(src_name, w, h, margin=0.04):
    """把骨架等比缩放、居中放进 w×h 的黑画布，返回新文件名。

    为什么非要这一步 —— 这是排查了一整轮才定位到的：
      系统骨架（skeleton_gen）是按画幅尺寸直接生成的，768x1344 和画幅一模一样，
      接上去立刻生效。
      从网上动作图提取的骨架是 671x1024，形状对但尺寸不匹配，结果 ControlNet
      完全不起作用：喂的是"奔跑"骨架，出来是"靠墙站着扶头"，跟提示词里
      plain wall background + leaning forward 自己长出来的一样。
      关掉角色锁、把 IPA 权重从 0.8 降到 0.6 都没用，因为根子不在那儿。

    为什么用等比 + 黑边而不是直接 resize：
      671x1024 拉到 768x1344 是横向 1.14 倍、纵向 1.31 倍，非等比拉伸会让人
      变瘦变长，骨架比例一歪，ControlNet 就把人画成另一个身材。
    """
    from PIL import Image
    p = os.path.join(SL.INPUT_DIR, src_name)
    if not os.path.exists(p):
        return src_name
    im = Image.open(p).convert("RGB")
    s = min(w * (1 - 2 * margin) / float(im.width),
            h * (1 - 2 * margin) / float(im.height))
    nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im2, ((w - nw) // 2, (h - nh) // 2))
    dst = os.path.join(SL.INPUT_DIR, "fit_" + src_name)
    canvas.save(dst)
    return os.path.basename(dst)


def resolve_pack_pose(pk_path, w=768, h=1344):
    """从参考图包里取骨架和动作词。返回 (骨架文件名, 动作词)。

    动作词必须跟着骨架一起进来。骨架图叫 skel_6823176，pose_hint() 从名字里
    读不出任何东西，而 DEFAULT_POS 写着 standing —— 只换骨架不改词，
    模型就同时收到"站着"和"在跑"两个指令，姿势会散。
    包里记着当初搜的关键词，那才是这个骨架的意思。
    """
    pk = MR.load_pack(pk_path)
    if MR.pack_channel(pk) != "pose":
        print("  [pack] %s 不是动作包，跳过" % pk_path)
        return None, ""
    it = MR.best_from_pack(pk)
    if not it:
        print("  [pack] 包里没有带骨架的图（拔完没提出骨架？）")
        return None, ""

    # 先落进 input，再按画幅等比装进黑画布。顺序不能反：
    # fit_pose 读的是 input 目录里的文件。
    name = fit_pose(MR.to_input(it["skeleton"]), w, h)
    q = (pk.get("query") or "").lower()
    words = ""
    for k, hint in (("jump", "jumping, mid-air, arms up, knee up, dynamic pose"),
                    ("run", "running, arms swinging, one leg forward, one leg back"),
                    ("fly", "flying, floating, arms spread"),
                    ("sit", "sitting, knees bent, legs together"),
                    ("reach", "one arm raised up, arm up")):
        if k in q:
            words = hint
            break
    if not words:
        words = (pk.get("query") or "").split(",")[0].strip()
    print("  [pack] 骨架 %s（动作：%s）"
          % (os.path.basename(it["skeleton"]), words or "无"))
    return name, words


def run(cfg):
    w, h = [int(v) for v in cfg["size"].lower().replace("*", "x").split("x")]
    stamp = time.strftime("%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, stamp)
    os.makedirs(out_dir, exist_ok=True)

    t_all = time.time()
    seed = cfg.get("seed")
    if not seed:
        seed = random.randint(1, 2 ** 31 - 1)
    print("种子 %s" % seed)
    ensure_comfy()

    print("=== 1/4 准备素材")
    ref = resolve_ref(cfg.get("ref")) if cfg.get("char") else None

    # 图包优先于骨架档位：包里自带骨架和当初搜的动作词，
    # 比随机抽一个预设更贴近"我就是要这个动作"的意图。
    # 顺序很重要 —— 先试包、失败才走骨架档位，否则会白抽一次随机姿势，
    # 日志里还会打一句其实没用上的"随机姿势：crouching"。
    from_pack = False
    pose_file, pose_words, pose_neg, randomized = None, "", "", False
    pk_list = cfg.get("pack") or []
    if isinstance(pk_list, str):
        pk_list = [pk_list]
    if pk_list:
        pf, pw = resolve_pack_pose(pk_list[0], w, h)
        if pf:
            pose_file, pose_words = pf, pw
            from_pack = True
    if not from_pack:
        pose_file, pose_words, pose_neg, randomized = resolve_pose(
            cfg.get("pose", "off"), (w, h))
        if randomized and pose_words:
            print("  随机姿势：%s" % pose_words)

    print("=== 2/4 生成（%s / 角色锁 %s / 骨架 %s）"
          % (cfg["profile"], "开" if ref else "关", pose_file or "关"))
    p = SL.load(cfg["profile"])
    p["size"] = [w, h]

    prompt = MR.DEFAULT_POS

    # 颜文字走独立入口，不跟普通提示词混在一起。
    # 展开出来的表情 tag 插在**最前面** —— SDXL 里靠前的词权重更高，
    # 表情是这张图的主题，不该被后面的服装、场景、姿势压掉。
    if cfg.get("kaomoji"):
        import kaomoji as KJ
        r = KJ.parse(cfg["kaomoji"])
        if r["tags"]:
            prompt = ", ".join(r["tags"]) + ", " + prompt
            print("  颜文字 %s -> %s" % (cfg["kaomoji"], ", ".join(r["tags"])))
        else:
            print("  [颜文字] 表里没有 %s，已忽略"
                  "（看支持哪些：python kaomoji.py --tags x）" % cfg["kaomoji"])

    if cfg.get("prompt_extra"):
        prompt += ", " + cfg["prompt_extra"]
    if pose_words:
        if from_pack:
            # 这里有两个必须压掉的东西，都实测踩过：
            #   standing  —— 和包里的动作词直接打架
            #   plain wall background —— 一旦有墙，模型就会把骨架的"举臂抬腿"
            #     理解成"扶着墙摆拍"。同一个奔跑骨架，带这句话出来是扶墙站，
            #     换成 simple background 就老老实实在跑。背景反正后面要换掉，
            #     留着它只会污染姿势。
            prompt = prompt.replace("standing, ", "").replace(", standing", "")
            prompt = prompt.replace("plain wall background, ", "")
            # looking at viewer 是个隐藏的姿势杀手：它要求正面看镜头，
            # 模型为了满足它会把身体拧成"面向镜头的摆拍"，奔跑/跳跃这种
            # 有明确朝向的动作就被掰回去了。极简提示词（无这句）时同一个
            # 骨架完全生效，所以这句必须让路。
            prompt = prompt.replace("looking at viewer, ", "")
        prompt += ", " + pose_words

    # 兜底：用户直接写在 -p 里的颜文字（不是走 --kao）也认
    if any(ord(ch) > 0x2000 for ch in prompt):
        try:
            import kaomoji as KJ
            prompt, hits = KJ.expand(prompt)
            if hits:
                print("  提示词里夹带的颜文字已展开：%s" % ", ".join(hits))
        except Exception:
            pass

    negative = (MR.DEFAULT_NEG + ", detailed background, scenery, "
                "gradient background, spotlight, vignette, "
                "shadow, cast shadow, drop shadow, rim lighting")
    if pose_neg:
        negative += ", " + pose_neg

    before = SL.snapshot()
    t = time.time()
    # 挂了骨架就得让 IP-Adapter 往后退一步。
    # 实测：char_w=0.80 + 参考图是站姿时，骨架被完全压过 —— 明明喂的是
    # "举双手跳起来"的骨架，出来还是靠墙站着（连扶头扶墙的小动作都补上了）。
    # 关掉角色锁用同一个骨架就完全正常，所以问题不在 ControlNet 而在权重。
    # 0.60 能同时保住长相和姿势；ipa_end 也提前到 0.35，让它早点交权。
    extra = {}
    if pose_file:
        extra["char_w"] = 0.60
        extra["ipa_end"] = 0.35
    wf = MR.build(p, prompt, negative, char=ref, pose=pose_file,
                  prefix="make/%s" % stamp, w=w, h=h, seed=seed, **extra)
    if not SL.submit(wf, tag="生成"):
        raise SystemExit("生成失败")
    got = SL.new_files(before)
    if not got:
        raise SystemExit("没有产出")
    shot = os.path.join(out_dir, "01_shot.png")
    shutil.copy2(got[-1], shot)
    print("  出图 %.1fs -> %s" % (time.time() - t, os.path.basename(shot)))
    cur = shot

    print("=== 3/4 超分")
    if cfg.get("upscale"):
        up = upscale(p, cur, "make/%s_up" % stamp)
        if up:
            dst = os.path.join(out_dir, "02_upscaled.png")
            shutil.copy2(up, dst)
            cur = dst
    else:
        print("  关")

    print("=== 4/4 换场景")
    if cfg.get("scene"):
        bg = MC.gen_bg(cfg["scene"], w, h, stamp)
        if bg:
            dst = os.path.join(out_dir, "03_scene.png")
            MC.run(cur, bg, dst, shadow=0.65, env=0.55, despill_s=1.0,
                   debug=False, a_level=0.15, dark_ratio=0.55,
                   luma=0.25, chroma=True)
    else:
        print("  关")

    print("\n=== 完成，用时 %.1f 秒" % (time.time() - t_all))
    from PIL import Image
    for f in sorted(glob.glob(os.path.join(out_dir, "*.png"))):
        im = Image.open(f)
        print("  %-16s %dx%d" % (os.path.basename(f), im.width, im.height))
    print("目录：%s" % out_dir)
    try:
        os.startfile(out_dir)
    except Exception:
        pass
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--ref", help="参考图：refs/char 里的名字 / input 文件名 / 任意图片路径")
    ap.add_argument("-P", "--pose", help="off（关骨架，随机姿势）/ auto（随机骨架）/ %s"
                    % " / ".join(PRESETS))
    ap.add_argument("-p", "--prompt", help="追加到默认主体描述后面的正向提示词")
    ap.add_argument("--profile")
    ap.add_argument("--size")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scene", help="cherry / ruin / night；传 none 关闭")
    ap.add_argument("--upscale", dest="upscale", action="store_true", default=None)
    ap.add_argument("--no-upscale", dest="upscale", action="store_false")
    ap.add_argument("--char", dest="char", action="store_true", default=None)
    ap.add_argument("--no-char", dest="char", action="store_false")
    ap.add_argument("--kao", dest="kaomoji",
                    help="颜文字，例如 \"(╥﹏╥)\"。展开成表情 tag 插在最前面")
    ap.add_argument("--pack", action="append", default=[],
                    help="动作参考包（ref_search 扒完留下的 _pack.json 或目录）。"
                         "给了就用包里的骨架和动作词，覆盖 -P")
    ap.add_argument("--save", action="store_true", help="把参数写回 toggles.json")
    a = ap.parse_args()

    cfg = load_toggles()
    for k in ("profile", "size", "seed"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    if a.ref is not None:
        cfg["ref"] = a.ref
    if a.pose is not None:
        cfg["pose"] = a.pose
    if a.upscale is not None:
        cfg["upscale"] = a.upscale
    if a.char is not None:
        cfg["char"] = a.char
    if a.scene is not None:
        cfg["scene"] = "" if a.scene.lower() in ("none", "off", "") else a.scene
    if a.prompt:
        cfg["prompt_extra"] = a.prompt
    if a.kaomoji is not None:
        cfg["kaomoji"] = a.kaomoji
    if a.pack:
        cfg["pack"] = a.pack        # 命令行这次的，不写盘（除非 --save）

    if a.save:
        save_toggles(cfg)
        print("已写回 toggles.json")

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
