# -*- coding: utf-8 -*-
"""用 rtmw/Wholebody（DWPose）从已有图片里提取骨架（133 点，含双手各 21 点）。

这是"模型自动脑补骨架"的第一个环节：图 -> 骨架。
如果现成模型在二次元图上不可靠，就必须自训练 anime pose 模型。

两个容易做错的地方，这里都按 ControlNet 的实际期望来：

  1 画在什么底上
    ControlNet 的 openpose 权重训练时看的是「黑底 + 彩色骨骼」。
    最早这版是 draw_skeleton(img.copy(), ...) —— 直接把骨架画在原图上，
    原图的背景、衣服、头发全留着，等于给 ControlNet 喂了张"带人的照片"，
    它分不清哪些线是骨骼。现在改成先建一张同尺寸纯黑图再往上画。

  2 模型加载
    Wholebody 初始化要几十秒（要读几个 onnx）。原来 run() 每次调用都新建
    一个，批量处理 8 张图就是 8 次初始化，光加载就把时间吃光了。
    现在模块级缓存，run_many() 一次加载跑完一批。

用法：
  python extract_pose.py 图.png 骨架.png
  python extract_pose.py 图.png 骨架.png heavy
"""
import os
import sys

sys.path.insert(0, r"D:/AI-Video-Setup/rtmlib_src/rtmlib-main")

import cv2
import numpy as np
from rtmlib import Wholebody

_MODEL = {}
_MODE = "balanced"


def _get_model(mode=None):
    """模型单例。加载一次几十秒，绝不能每张图重建。"""
    global _MODE
    if mode:
        _MODE = mode
    if _MODE not in _MODEL:
        _MODEL[_MODE] = Wholebody(mode=_MODE, backend="onnxruntime", device="cpu")
    return _MODEL[_MODE]


# COCO 17 点 -> OpenPose 18 点。顺序不对 ControlNet 会画出扭曲的人。
# OpenPose 的 R/L 按"看图的人"的左右算：R 在画面左边，L 在右边。
# COCO 的 left/right 按"被画的人"算，所以正好是反的。
# 这就是下面 Rshoulder 取 coco[6]、Lshoulder 取 coco[5] 的原因 —— 别改。
COCO2OPENPOSE = {
    0: 0,    # nose
    1: 15,   # left_eye  -> Leye
    2: 14,   # right_eye -> Reye
    3: 17,   # left_ear  -> Lear
    4: 16,   # right_ear -> Rear
    5: 5,    # left_shoulder  -> Lshoulder
    6: 2,    # right_shoulder -> Rshoulder
    7: 6,    # left_elbow     -> Lelbow
    8: 3,    # right_elbow    -> Relbow
    9: 7,    # left_wrist     -> Lwrist
    10: 4,   # right_wrist    -> Rwrist
    11: 11,  # left_hip       -> Lhip
    12: 8,   # right_hip      -> Rhip
    13: 12,  # left_knee      -> Lknee
    14: 9,   # right_knee     -> Rknee
    15: 13,  # left_ankle     -> Lankle
    16: 10,  # right_ankle    -> Rankle
}


def _to18(kp, W, H):
    """把 COCO 17 点摊成 OpenPose 18 点（归一化坐标），颈部用双肩中点补。"""
    pts = [(0.5, 0.3)] * 18
    for ci, oi in COCO2OPENPOSE.items():
        x, y = float(kp[ci][0]), float(kp[ci][1])
        pts[oi] = (x / W, y / H)
    # 脖子：双肩中点。检测器不输出颈点，但 18 点骨架必须有它，
    # 少了这根线，头会飘在身体外面。
    sx = (pts[2][0] + pts[5][0]) / 2.0
    sy = (pts[2][1] + pts[5][1]) / 2.0
    pts[1] = (sx, sy)
    return pts


def _refit(pts, margin=0.07):
    """把骨架按自己的外框重新居中、等比撑到画幅内。

    为什么必须做：原图里人物常常只占中间一小块（上下有大量背景留白），
    直接按原图坐标映射，骨架就会缩在画面偏上的位置。ControlNet 照着这个
    构图生成，出来的人物也小一号、位置偏，还得再去裁。
    这里只关心骨架形状，位置和占比由画幅和提示词决定，所以归一化是安全的。
    用 min() 而不是分别缩放，保证等比 —— 否则细长的举手姿势会被拉成矮胖。
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    if bw < 1e-6 or bh < 1e-6:
        return pts
    span = 1.0 - 2.0 * margin
    s = min(span / bw, span / bh)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [(0.5 + (x - cx) * s, 0.5 + (y - cy) * s) for x, y in pts]


def _imread_u(path):
    """cv2.imread 在 Windows 上读不了非 ASCII 路径。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_u(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def extract(src, dst, mode=None, only_first=True, kpt_thr=0.3):
    """抽一张图的骨架，写到 dst。

    返回 dict：
      ok        是否画出了可用骨架
      persons   检测到几个人
      body      body 关键点里置信度 > 阈值的个数（满分 17）
      hand      双手关键点达标数（满分 42）
      mean      平均置信度

    画法用 skeleton_gen.draw_pose，是标准 18 点彩色骨架。
    一开始用的是 rtmlib 自带的 draw_skeleton，它把 wholebody 的 133 点全画出来 ——
    光脸就 68 个点，糊成一团白线。SDXL 的 openpose 权重训练时只见过 18 点，
    多出来那一百多个点它不认识，反而把骨架读坏。
    """
    import skeleton_gen as SG

    model = _get_model(mode)
    img = _imread_u(src)
    if img is None:
        return {"ok": False, "reason": "读不到图", "persons": 0,
                "body": 0, "hand": 0, "mean": 0.0}
    H, W = img.shape[:2]

    kpts, scores = model(img)

    if kpts is None or len(kpts) == 0:
        # 检测不到人也要落一张黑图，调用方好判断"这张废了"
        _imwrite_u(dst, np.zeros((768, 768, 3), np.uint8))
        return {"ok": False, "reason": "没检测到人", "persons": 0,
                "body": 0, "hand": 0, "mean": 0.0}

    # 多人图只留最靠前的那个。筛选时已经要求 solo，但检测器偶尔会在
    # 背景里多认一个人，两张骨架叠一起 ControlNet 会画出一团怪物。
    if only_first and len(kpts) > 1:
        best = int(np.argmax([s[:17].mean() for s in scores]))
        kpts, scores = kpts[best:best + 1], scores[best:best + 1]

    kp, sc = kpts[0], scores[0]
    n_body = int((sc[:17] > kpt_thr).sum())
    n_hand = int((sc[91:133] > kpt_thr).sum()) if len(sc) >= 133 else -1

    pts = _refit(_to18(kp, W, H))
    # 输出尺寸跟骨架自身比例走，不跟原图走 —— 原图的留白比例不该带进来。
    # 一张 3000x1500 的宽幅动作图，骨架其实是个瘦高的人，输出也该是竖的。
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    aspect = (max(xs) - min(xs)) / max(1e-6, (max(ys) - min(ys)))
    if aspect <= 1.0:
        oh, ow = 1024, max(64, int(1024 * aspect))
    else:
        ow, oh = 1024, max(64, int(1024 / aspect))
    SG.draw_pose(pts, w=ow, h=oh).save(dst)    # PIL 的 save 支持中文路径

    # 身体点少于一半就说明这骨架没法用（侧脸、背影、被遮挡、Q 版头身比）
    ok = n_body >= 9
    return {"ok": ok, "reason": "" if ok else "身体点太少",
            "persons": len(kpts), "body": n_body, "hand": n_hand,
            "mean": float(sc.mean())}


def run_many(pairs, mode=None, only_first=True, quiet=False):
    """批量抽骨架，模型只加载一次。pairs 是 [(src, dst), ...]。"""
    out = []
    _get_model(mode)
    for src, dst in pairs:
        r = extract(src, dst, only_first=only_first)
        if not quiet:
            print("  %-28s body %2d/17  hand %2d/42  %s"
                  % (os.path.basename(src), r["body"], max(r["hand"], 0),
                     "OK" if r["ok"] else "废(" + r["reason"] + ")"))
        out.append(r)
    return out


def run(src, dst, mode="balanced"):
    """老接口，保持兼容。"""
    r = extract(src, dst, mode)
    if not r["ok"] and r["persons"] == 0:
        print("NO_PERSON_DETECTED")
        return 0
    print("persons", r["persons"], "| body pts>0.3:", r["body"], "/17 |"
          " hand pts>0.3:", r["hand"], "/42")
    print("mean score %.3f" % r["mean"])
    return r["persons"]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("用法：python extract_pose.py 图.png 骨架.png [balanced|heavy|light]")
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "balanced")
