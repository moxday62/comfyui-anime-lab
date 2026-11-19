# -*- coding: utf-8 -*-
"""骨架 / 线稿生成器 —— 把"姿势"和"手指数量"变成显式结构图。

为什么要这个：
  扩散模型不知道人的骨骼在哪，也不知道一只手该有几根手指。
  它只能靠统计先验去猜，猜错就出现六指、关节反折。
  所以要把结构信息"画出来"喂给它，而不是写在提示词里祈祷。

两种结构图，分工完全不同（这点最容易搞错）：
  OpenPose 骨架  -> 只管身体姿势、比例、朝向。**它不包含手指信息**，
                    所以单靠骨架永远解决不了六指。
  线稿 sketch    -> 管轮廓、形状、**手指数量**、衣褶走向。
                    想画 5 根就画 5 根，模型照着描。

用法：
  python skeleton_gen.py pose --preset stand --out pose.png
  python skeleton_gen.py pose --preset v_sign --out pose.png
  python skeleton_gen.py hand --fingers 5 --style open --out hand.png
  python skeleton_gen.py hand --fingers 5 --style v --out hand_v.png
"""
import argparse
import math
import os

from PIL import Image, ImageDraw

# OpenPose 18 点标准配色（BGR 转 RGB），ControlNet 训练时就是这套
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# 肢体连接（0-based，顺序即配色顺序）
LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# 正面站立：nose, neck, Rsh, Relb, Rwr, Lsh, Lelb, Lwr,
#            Rhip, Rknee, Rankle, Lhip, Lknee, Lankle,
#            Reye, Leye, Rear, Lear
PRESETS = {
    "stand": [
        (0.500, 0.100), (0.500, 0.180),
        (0.420, 0.200), (0.385, 0.320), (0.360, 0.440),
        (0.580, 0.200), (0.615, 0.320), (0.640, 0.440),
        (0.455, 0.480), (0.448, 0.680), (0.440, 0.900),
        (0.545, 0.480), (0.552, 0.680), (0.560, 0.900),
        (0.480, 0.088), (0.520, 0.088), (0.462, 0.092), (0.538, 0.092),
    ],
    "v_sign": [
        (0.500, 0.100), (0.500, 0.180),
        (0.420, 0.200), (0.360, 0.300), (0.330, 0.220),
        (0.580, 0.200), (0.640, 0.300), (0.670, 0.220),
        (0.455, 0.480), (0.448, 0.680), (0.440, 0.900),
        (0.545, 0.480), (0.552, 0.680), (0.560, 0.900),
        (0.480, 0.088), (0.520, 0.088), (0.462, 0.092), (0.538, 0.092),
    ],
    "sit": [
        (0.500, 0.120), (0.500, 0.200),
        (0.420, 0.220), (0.390, 0.340), (0.370, 0.460),
        (0.580, 0.220), (0.610, 0.340), (0.630, 0.460),
        (0.455, 0.500), (0.330, 0.560), (0.320, 0.760),
        (0.545, 0.500), (0.670, 0.560), (0.680, 0.760),
        (0.480, 0.108), (0.520, 0.108), (0.462, 0.112), (0.538, 0.112),
    ],
    # 腾空跳跃：双臂上举张开、右腿屈膝抬起、左腿向下伸直。
    # 这个姿态对模型是最难的 —— 四肢都不在常规位置，
    # 没有骨架约束时模型会本能地把它画成站立。
    "jump": [
        (0.500, 0.150), (0.500, 0.230),
        (0.428, 0.250), (0.352, 0.170), (0.310, 0.075),
        (0.572, 0.250), (0.648, 0.170), (0.690, 0.075),
        (0.458, 0.510), (0.398, 0.640), (0.352, 0.735),
        (0.542, 0.510), (0.588, 0.700), (0.606, 0.890),
        (0.480, 0.138), (0.520, 0.138), (0.462, 0.142), (0.538, 0.142),
    ],
    # 单臂高举 + 另一手叉腰 + 身体侧倾
    # 注意手腕 y 别低于 0.10：0.03 时手掌会被画面顶边裁掉，
    # 全身构图要留出头顶余量
    "reach_up": [
        (0.520, 0.130), (0.520, 0.205),
        (0.448, 0.228), (0.404, 0.135), (0.372, 0.055),
        (0.592, 0.228), (0.664, 0.320), (0.576, 0.420),
        (0.480, 0.490), (0.462, 0.690), (0.452, 0.900),
        (0.560, 0.490), (0.580, 0.690), (0.592, 0.900),
        (0.500, 0.118), (0.540, 0.118), (0.482, 0.122), (0.558, 0.122),
    ],
}


def draw_pose(points, w=832, h=1216, stick=6, joint=8, zoom=1.0):
    """画 OpenPose 风格骨架：彩色肢体线 + 圆形关节点。

    zoom < 1 会把整个人缩小并居中，给画幅留出头顶/脚底余量。
    为什么需要：骨架贴着画幅边缘时，模型画出来的手臂往往比骨架更长，
    于是举手的位置就顶出画面被裁掉（实测 768x1344 时手腕顶边）。
    缩小到 0.86 后头顶和脚底都有余量，成品不再裁手裁脚。
    """
    im = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(im)
    if zoom != 1.0:
        points = [(0.5 + (x - 0.5) * zoom, 0.5 + (y - 0.5) * zoom)
                  for x, y in points]
    px = [(int(x * w), int(y * h)) for x, y in points]
    for i, (a, b) in enumerate(LIMBS):
        if a < len(px) and b < len(px):
            d.line([px[a], px[b]], fill=COLORS[i % len(COLORS)], width=stick)
    for i, p in enumerate(px):
        c = COLORS[i % len(COLORS)]
        d.ellipse([p[0] - joint, p[1] - joint, p[0] + joint, p[1] + joint], fill=c)
    return im


# ---------------------------------------------------------------- 手部线稿
# 每根手指：根部相对手掌中心的位置 + 三段长度（按比例）+ 朝向角度（-90 为正上方）
FINGER_LAYOUT = [
    # (name, root_dx, root_dy, seg_len, base_angle_deg)
    ("thumb", -0.30, 0.04, [0.20, 0.17, 0.14], -128),
    ("index", -0.20, -0.24, [0.22, 0.18, 0.14], -102),
    ("middle", -0.06, -0.26, [0.24, 0.19, 0.15], -91),
    ("ring", 0.08, -0.25, [0.22, 0.18, 0.14], -80),
    ("pinky", 0.21, -0.21, [0.18, 0.15, 0.12], -68),
]


def draw_hand_sketch(w=768, h=768, fingers=5, style="open", spread=1.0,
                     line=9, joint=7):
    """手部线稿：白底黑线，手掌轮廓 + 每根手指的指节折线。

    fingers: 画几根就是几根 —— 这是控制六指的唯一可靠手段。
    style: open(张开) / v(剪刀手) / fist(握拳)
    """
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    cx, cy = w * 0.5, h * 0.60
    s = min(w, h) * 0.46  # 手掌尺度

    # 手掌轮廓（圆角矩形），给出"这是一只手"的先验
    d.rounded_rectangle(
        [cx - 0.36 * s, cy - 0.28 * s, cx + 0.36 * s, cy + 0.20 * s],
        radius=int(0.12 * s), outline=(0, 0, 0), width=line)
    # 手腕
    d.line([(cx - 0.18 * s, cy + 0.20 * s), (cx - 0.18 * s, cy + 0.44 * s)],
           fill=(0, 0, 0), width=line)
    d.line([(cx + 0.18 * s, cy + 0.20 * s), (cx + 0.18 * s, cy + 0.44 * s)],
           fill=(0, 0, 0), width=line)
    d.line([(cx - 0.18 * s, cy + 0.44 * s), (cx + 0.18 * s, cy + 0.44 * s)],
           fill=(0, 0, 0), width=line)

    use = FINGER_LAYOUT[:max(1, min(5, fingers))]
    for idx, (name, rdx, rdy, segs, ang) in enumerate(use):
        folded = (style == "fist") or (style == "v" and name in ("ring", "pinky"))
        if folded:
            bend = [150.0, 150.0, 0.0]
        elif style == "v" and name == "thumb":
            bend = [40.0, 30.0, 0.0]
        else:
            bend = [0.0, abs(math.sin(idx * 1.3)) * 8 * spread,
                    abs(math.cos(idx * 0.9)) * 7 * spread]

        x, y = cx + rdx * s, cy + rdy * s
        a = math.radians(ang * (0.7 + 0.3 * spread))
        pts = [(x, y)]
        for k, seg in enumerate(segs):
            a += math.radians(bend[k])
            x += math.cos(a) * seg * s
            y += math.sin(a) * seg * s
            pts.append((x, y))
        d.line(pts, fill=(0, 0, 0), width=line, joint="curve")
        for p in pts:
            d.ellipse([p[0] - joint, p[1] - joint, p[0] + joint, p[1] + joint],
                      fill=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser(description="骨架 / 手部线稿生成器")
    ap.add_argument("kind", choices=["pose", "hand"])
    ap.add_argument("--preset", default="stand", choices=list(PRESETS))
    ap.add_argument("--fingers", type=int, default=5)
    ap.add_argument("--style", default="open", choices=["open", "v", "fist"])
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--size", default="832x1216")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="骨架整体缩放，<1 留出画幅余量（0.86 可避免裁手裁脚）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w, h = [int(v) for v in a.size.lower().split("x")]
    if a.kind == "pose":
        im = draw_pose(PRESETS[a.preset], w, h, zoom=a.zoom)
    else:
        im = draw_hand_sketch(w, h, fingers=a.fingers, style=a.style, spread=a.spread)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    im.save(a.out)
    print("saved", a.out, im.size)


if __name__ == "__main__":
    main()
