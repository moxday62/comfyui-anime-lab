# -*- coding: utf-8 -*-
"""自进化调参器 —— 通过一张张出图 + 反馈，自动搜索最佳参数。

先说清楚能做和不能做的：

  ✗ 不能做：让 SDXL 边跑边改权重（在线学习）。
    扩散模型没有"单样本在线更新"这回事，一帧一帧喂图学不到东西，
    真要改权重必须离线训练，6GB 显存也扛不住。
  ✓ 能做：**参数搜索**。把 CFG / 步数 / denoise / IPAdapter 权重 / 部位强度
    这些参数当成"可进化的基因"，每出一张图给一个评分，
    用退火爬山逐步逼近你的偏好。这才是"自己调参数"的正确形态。

两种反馈方式：
  auto    自动评分（锐度 + 对比度 + 色彩丰富度 [+ 与参考图相似度]），无人值守
  manual  每轮出 3 张，你选最喜欢的一张，系统据此更新方向（最准，符合"一张张喂图"）

历史会存进 profiles/tuning/<profile>.json，下次接着跑，越用越准。
收敛后可以 --apply 把最优参数写回画风档（自动备份原档）。

用法：
  python auto_tune.py --profile soft --trials 9 --auto
  python auto_tune.py --profile soft --trials 6              # 交互选图
  python auto_tune.py --profile 3dcg --ref 目标图.jpg --auto  # 加相似度约束
  python auto_tune.py --profile soft --apply                 # 把已有最优写回档
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_lib as SL

TUNE_DIR = os.path.join(SL.PROFILE_DIR, "tuning")

# 可进化参数空间：名称 -> (最小值, 最大值, 是否整数)
SPACES = {
    "soft": {"cfg": (3.5, 6.5, False), "steps": (24, 40, True),
             "ip_weight": (0.45, 0.85, False)},
    "3dcg": {"cfg": (4.0, 7.0, False), "steps": (22, 36, True),
             "ip_weight": (0.50, 0.85, False)},
    "vivid": {"cfg": (4.5, 7.5, False), "steps": (28, 44, True),
              "ip_weight": (0.45, 0.80, False)},
    "clear": {"cfg": (3.5, 6.5, False), "steps": (24, 38, True),
              "ip_weight": (0.45, 0.85, False)},
}


# ---------------------------------------------------------------- 评分
def _to_gray_arr(im, size=(256, 256)):
    import numpy as np
    g = im.convert("L").resize(size)
    return np.asarray(g, dtype=np.float32)


def quality_score(path):
    """无参考质量分：锐度 + 对比度 + 色彩丰富度，各自归一化后加权。

    注意：这是代理指标，会被"过锐/过饱和"骗。它适合粗筛，
    真正精细的偏好还是 manual 模式（你自己选图）最准。
    """
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.float32) / 255.0

    # 锐度：拉普拉斯响应方差
    g = _to_gray_arr(im)
    lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    sharp = float(lap.var())
    sharp_n = min(1.0, sharp / 1200.0)

    # 对比度：亮度标准差（理想区间 0.15-0.28，过高过饱和）
    std = float(a.std())
    contrast = 1.0 - abs(std - 0.21) / 0.21
    contrast = max(0.0, min(1.0, contrast))

    # 色彩丰富度：饱和度均值，理想 0.2-0.35
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = float(((mx - mn) / (mx + 1e-6)).mean())
    rich = max(0.0, 1.0 - abs(sat - 0.28) / 0.28)

    # 崩坏惩罚：大面积纯色块（常出现在生成失败时）
    flat = float((np.abs(lap) < 0.5).mean())
    penalty = max(0.0, flat - 0.75) * 2.0

    score = 0.40 * sharp_n + 0.25 * contrast + 0.35 * rich - penalty
    return max(0.0, min(1.0, score))


def ref_similarity(candidate, ref):
    """与参考图的结构 + 色彩相似度（numpy 实现，零依赖）。"""
    import numpy as np
    from PIL import Image
    gc = _to_gray_arr(Image.open(candidate))
    gr = _to_gray_arr(Image.open(ref))
    gc = (gc - gc.mean()) / (gc.std() + 1e-6)
    gr = (gr - gr.mean()) / (gr.std() + 1e-6)
    corr = float((gc * gr).mean())
    corr = (corr + 1) / 2  # -> 0..1

    def hist(p):
        h, _ = np.histogramdd(
            np.asarray(p.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0,
            bins=(8, 8, 8), range=[(0, 1)] * 3)
        h = h.flatten()
        return h / (h.sum() + 1e-6)
    a = hist(Image.open(candidate))
    b = hist(Image.open(ref))
    inter = float(np.minimum(a, b).sum())
    return 0.5 * corr + 0.5 * inter


# ---------------------------------------------------------------- 搜索器
class Tuner(object):
    """退火爬山：前期随机探索，后期在最优附近小步扰动。"""

    def __init__(self, space, history=None):
        self.space = space
        self.history = history or []          # [{params, score, path, ts}]
        self.rng = random.Random(20261118)
        self.explore_ratio = 0.4

    def best(self):
        if not self.history:
            return None
        # 同分时取最新的一条：否则后来的 1.0 永远赢不过最早那条 1.0
        return max(self.history, key=lambda h: (h["score"], h.get("ts", "")))

    def _rand_value(self, name):
        lo, hi, is_int = self.space[name]
        v = self.rng.uniform(lo, hi)
        return int(round(v)) if is_int else round(v, 3)

    def suggest(self):
        n = len(self.history)
        # 前 40% 纯随机探索，保证覆盖
        if n < max(3, len(self.space) * 2) and n < 6:
            return {k: self._rand_value(k) for k in self.space}
        if n / max(1, n + 1) < self.explore_ratio:
            return {k: self._rand_value(k) for k in self.space}

        base = dict(self.best()["params"])
        # 扰动幅度随迭代衰减
        shrink = max(0.15, 1.0 - n / 24.0)
        out = {}
        for k, (lo, hi, is_int) in self.space.items():
            span = (hi - lo) * 0.18 * shrink
            v = base.get(k, (lo + hi) / 2) + self.rng.gauss(0, span)
            v = max(lo, min(hi, v))
            out[k] = int(round(v)) if is_int else round(v, 3)
        return out

    def observe(self, params, score, path):
        self.history.append({"params": params, "score": round(float(score), 4),
                             "path": path, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})


# ---------------------------------------------------------------- 主流程
def load_history(profile):
    os.makedirs(TUNE_DIR, exist_ok=True)
    p = os.path.join(TUNE_DIR, profile + ".json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(profile, history):
    os.makedirs(TUNE_DIR, exist_ok=True)
    p = os.path.join(TUNE_DIR, profile + ".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return p


def build_with_params(p, params, ref, seed, prefix):
    """把搜索到的参数套到画风档上，生成工作流。"""
    q = dict(p)
    q["cfg"] = params.get("cfg", p.get("cfg"))
    q["steps"] = params.get("steps", p.get("steps"))
    wf = SL.build(q, ref_image=ref, seed=seed,
                  ip_weight=params.get("ip_weight"), prefix=prefix)
    return wf


def run_trial(profile, p, params, ref, seed, auto, ref_path):
    prefix = "tune/%s_t%d" % (profile, int(time.time() * 1000) % 100000)
    wf = build_with_params(p, params, ref, seed, prefix)
    before = SL.snapshot()
    if not SL.submit(wf, tag="trial", quiet=True):
        return None, 0.0
    files = SL.new_files(before)
    if not files:
        return None, 0.0
    path = files[-1]

    if auto:
        q = quality_score(path)
        if ref_path:
            q = 0.55 * q + 0.45 * ref_similarity(path, ref_path)
        return path, q
    return path, None  # 交互模式：等用户打分


def apply_best(profile, history):
    best = max(history, key=lambda h: (h["score"], h.get("ts", "")))
    src = SL.profile_path(profile)
    bak = src + ".bak." + time.strftime("%Y%m%d_%H%M%S")
    shutil.copy2(src, bak)
    with open(src, "r", encoding="utf-8") as f:
        p = json.load(f)
    for k, v in best["params"].items():
        if k == "ip_weight":
            p.setdefault("ipadapter", {})["weight"] = v
        else:
            p[k] = v
    with open(src, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)
    print("已写回 %s（备份 %s）" % (os.path.basename(src), os.path.basename(bak)))
    print("最优参数：%s  得分 %.3f" % (json.dumps(best["params"], ensure_ascii=False),
                                     best["score"]))


def main():
    ap = argparse.ArgumentParser(description="自进化调参器")
    ap.add_argument("--profile", default=None, help="画风档名，省略则用当前画风")
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--auto", action="store_true", help="自动评分，无需人工")
    ap.add_argument("--ref", help="参考图路径（加了会算相似度）")
    ap.add_argument("--apply", action="store_true", help="把历史最优写回画风档")
    ap.add_argument("--reset", action="store_true", help="清空历史重新开始")
    a = ap.parse_args()
    a.trials_explicit = any(x == "--trials" or x.startswith("--trials=") for x in sys.argv[1:])

    a.profile = a.profile or SL.current_name()
    if not a.profile:
        print("还没有画风档，先在 profiles/ 下建一个 JSON")
        return 1

    p = SL.load(a.profile)
    space = dict(SPACES.get(a.profile) or SPACES["soft"])

    # 没喂参考图时 ip_weight 不起作用，去掉它，避免浪费试验次数
    ref = SL.prepare_ref(p, a.ref) if (a.ref or (p.get("ipadapter") or {}).get("ref")) else None
    if not ref:
        space.pop("ip_weight", None)

    if a.reset:
        save_history(a.profile, [])
        print("历史已清空（%s）" % a.profile)
        if not a.trials_explicit:
            return 0

    history = load_history(a.profile)
    if a.apply:
        if not history:
            print("还没有历史记录，先跑几轮")
            return 1
        apply_best(a.profile, history)
        return 0

    tuner = Tuner(space, history)
    seed0 = int(p.get("seed", 0))

    print("=== 自进化调参：%s（%s）===" % (a.profile, p.get("label", "")))
    print("参数空间：%s" % json.dumps({k: list(v[:2]) for k, v in space.items()},
                                   ensure_ascii=False))
    print("已有历史 %d 轮 | 本次跑 %d 轮 | 模式 %s\n"
          % (len(history), a.trials, "自动评分" if a.auto else "交互选图"))

    batch = [] if not a.auto else None
    for i in range(a.trials):
        params = tuner.suggest()
        # 交互模式：同一组 3 张共用一颗种子，差异只来自参数，
        # 否则你可能是在挑"构图更好的那张"，而不是"参数更好的那张"。
        seed = seed0 + i if a.auto else seed0 + (i // 3)
        print("[%d/%d] %s" % (i + 1, a.trials, json.dumps(params, ensure_ascii=False)))
        path, score = run_trial(a.profile, p, params, ref, seed, a.auto, a.ref)
        if not path:
            print("      生成失败，跳过")
            continue

        if a.auto:
            tuner.observe(params, score, path)
            print("      得分 %.3f  ->  %s" % (score, os.path.basename(path)))
        else:
            batch.append((params, path))
            print("      出图：%s" % path)
            if len(batch) >= 3 or i == a.trials - 1:
                # 自动弹出目录，省得手动找
                try:
                    os.startfile(os.path.dirname(batch[-1][1]))
                except Exception:
                    pass
                print("\n  本组 %d 张已生成，正在打开目录：" % len(batch))
                for j, (pp, fp) in enumerate(batch, 1):
                    print("   [%d] %s" % (j, os.path.basename(fp)))
                try:
                    pick = int(input("  最喜欢第几张？(1-%d，0=都不行): " % len(batch)).strip() or "0")
                except Exception:
                    pick = 0
                if 1 <= pick <= len(batch):
                    # 赢家 1.0，落选 0.35 —— 落选也要记录，否则只有正反馈分不出高下
                    for idx, (pp, fp) in enumerate(batch, 1):
                        tuner.observe(pp, 1.0 if idx == pick else 0.35, fp)
                    print("  已记录你的偏好（赢家 1.00，落选 0.35）\n")
                else:
                    print("  本组不计分\n")
                batch = []

        save_history(a.profile, tuner.history)

    best = tuner.best()
    if best:
        print("\n=== 当前最优 ===")
        print("参数 %s" % json.dumps(best["params"], ensure_ascii=False))
        print("得分 %.4f   图 %s" % (best["score"], best["path"]))
        print("\n写回画风档：python auto_tune.py --profile %s --apply" % a.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
