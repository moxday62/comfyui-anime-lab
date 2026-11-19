# -*- coding: utf-8 -*-
"""把中文描述粗翻成图站 tag。纯离线词典映射，不联网、不调模型。

定位先说清楚：这是"聊胜于无"，不是真的语言理解。
  能做的：常见名词、颜色、动作、场景 → 对应 tag（覆盖两百来个词）
  做不到：句法、修饰关系、生僻词、抽象概念
  例：
    "樱花树下奔跑的白发少女"  →  running + white_hair + 1girl（场景词进加分项）
    "穿着黑色大衣站在雨夜街头" →  standing + black_coat + 1boy?…（"街头"能中，"雨夜"能中，
                                但"站在"和"街头"的关系它并不懂，只是各取所需）

为什么不上大模型：
  接 LLM 要 API key + 网络，而网络这块实测很不稳（见 ref_search.py 头部）。
  为了一句翻译去赌网络，不如退一步用词典 —— 失败了也是零成本。

为什么输出分 core / extra 两组（这点最重要）：
  图站的 tag 之间是 AND。塞五个 tag 进去大概率归零（实测 jumping+dynamic 就是 0 条）。
  所以只拿 2-3 个"最能定义这张图"的词做真查询，其余的当加分项，
  在结果排序里用 —— 有它更好，没它也不至于搜不到。

用法：
  python nl2tags.py "樱花树下奔跑的白发少女"
  python nl2tags.py "雨夜霓虹街道" --preset scene
"""
import sys

# (tag, 类别)。类别决定它能当 core 还是只能当加分项。
# 类别含义：
#   who    人数/性别 —— 几乎总能当 core
#   hair / eyes / cloth / effect   外观特征 —— 能当 core，但不宜太多
#   action 动作 —— 优先级最高的 core
#   scene  场景 —— 一般只当加分，除非 preset=scene
#   shot   构图/视角 —— 一般只当加分，容易和别的冲突
DICT = {
    # ---------------- 人物
    "少女": ("1girl", "who"), "女孩": ("1girl", "who"),
    "女生": ("1girl", "who"), "女性": ("1girl", "who"),
    "男孩": ("1boy", "who"), "少年": ("1boy", "who"),
    "男生": ("1boy", "who"), "男性": ("1boy", "who"),
    "两人": ("2girls", "who"), "双人": ("2girls", "who"),
    "单人": ("solo", "who"), "一个人": ("solo", "who"),

    # ---------------- 头发
    "白发": ("white_hair", "hair"), "银发": ("silver_hair", "hair"),
    "黑发": ("black_hair", "hair"), "金发": ("blonde_hair", "hair"),
    "棕发": ("brown_hair", "hair"), "褐发": ("brown_hair", "hair"),
    "红发": ("red_hair", "hair"), "蓝发": ("blue_hair", "hair"),
    "绿发": ("green_hair", "hair"), "紫发": ("purple_hair", "hair"),
    "粉发": ("pink_hair", "hair"), "粉红发": ("pink_hair", "hair"),
    "灰发": ("grey_hair", "hair"), "橙发": ("orange_hair", "hair"),
    "长发": ("long_hair", "hair"), "短发": ("short_hair", "hair"),
    "中长发": ("medium_hair", "hair"),
    "双马尾": ("twintails", "hair"), "马尾": ("ponytail", "hair"),
    "单马尾": ("ponytail", "hair"), "丸子头": ("hair_bun", "hair"),
    "卷发": ("curly_hair", "hair"), "直发": ("straight_hair", "hair"),
    "呆毛": ("ahoge", "hair"), "刘海": ("bangs", "hair"),
    "编发": ("braid", "hair"), "麻花辫": ("braid", "hair"),

    # ---------------- 眼睛
    "蓝眼": ("blue_eyes", "eyes"), "蓝瞳": ("blue_eyes", "eyes"),
    "红眼": ("red_eyes", "eyes"), "红瞳": ("red_eyes", "eyes"),
    "金眼": ("yellow_eyes", "eyes"), "金瞳": ("yellow_eyes", "eyes"),
    "绿眼": ("green_eyes", "eyes"), "绿瞳": ("green_eyes", "eyes"),
    "紫眼": ("purple_eyes", "eyes"), "紫瞳": ("purple_eyes", "eyes"),
    "异色瞳": ("heterochromia", "eyes"),

    # ---------------- 服装
    "水手服": ("sailor_uniform", "cloth"),
    "校服": ("school_uniform", "cloth"),
    "制服": ("uniform", "cloth"),
    "连衣裙": ("dress", "cloth"), "裙子": ("skirt", "cloth"),
    "百褶裙": ("pleated_skirt", "cloth"),
    "和服": ("kimono", "cloth"), "浴衣": ("yukata", "cloth"),
    "旗袍": ("china_dress", "cloth"),
    "大衣": ("coat", "cloth"), "外套": ("jacket", "cloth"),
    "西装": ("suit", "cloth"), "衬衫": ("shirt", "cloth"),
    "毛衣": ("sweater", "cloth"), "卫衣": ("hoodie", "cloth"),
    "斗篷": ("cape", "cloth"), "披风": ("cape", "cloth"),
    "短裤": ("shorts", "cloth"), "长裤": ("pants", "cloth"),
    "丝袜": ("pantyhose", "cloth"), "过膝袜": ("thighhighs", "cloth"),
    "长筒袜": ("thighhighs", "cloth"), "短袜": ("socks", "cloth"),
    "靴子": ("boots", "cloth"), "高跟鞋": ("high_heels", "cloth"),
    "帽子": ("hat", "cloth"), "贝雷帽": ("beret", "cloth"),
    "围巾": ("scarf", "cloth"), "手套": ("gloves", "cloth"),
    "领带": ("necktie", "cloth"), "蝴蝶结": ("bow", "cloth"),
    "盔甲": ("armor", "cloth"), "机甲": ("mecha", "cloth"),
    "黑色": ("black", "color"), "白色": ("white", "color"),
    "红色": ("red", "color"), "蓝色": ("blue", "color"),
    "金色": ("gold", "color"), "银色": ("silver", "color"),

    # ---------------- 动作
    "站": ("standing", "action"), "站立": ("standing", "action"),
    "坐着": ("sitting", "action"), "坐": ("sitting", "action"),
    "蹲下": ("crouching", "action"), "蹲着": ("crouching", "action"),
    "奔跑": ("running", "action"), "跑": ("running", "action"),
    "跳": ("jumping", "action"), "跳跃": ("jumping", "action"),
    "走路": ("walking", "action"), "行走": ("walking", "action"),
    "躺着": ("lying", "action"), "趴着": ("on_stomach", "action"),
    "举手": ("arm_up", "action"), "抬手": ("arm_up", "action"),
    "挥手": ("waving", "action"), "回头": ("looking_back", "action"),
    "伸手": ("outstretched_arm", "action"),
    "抱膝": ("hugging_own_legs", "action"),
    "双手抱胸": ("crossed_arms", "action"),
    "叉腰": ("hand_on_hip", "action"),
    "看书": ("reading", "action"), "弹琴": ("playing_instrument", "action"),
    "持剑": ("holding_sword", "action"), "拿剑": ("holding_sword", "action"),

    # ---------------- 场景
    "樱花": ("cherry_blossoms", "scene"),
    "樱花树": ("cherry_blossoms", "scene"),
    "花瓣": ("falling_petals", "scene"),
    "教室": ("classroom", "scene"),
    "学校": ("school", "scene"),
    "街道": ("street", "scene"),
    "城市": ("city", "scene"),
    "夜晚": ("night", "scene"), "夜": ("night", "scene"),
    "黄昏": ("sunset", "scene"), "日落": ("sunset", "scene"),
    "雨天": ("rain", "scene"), "下雨": ("rain", "scene"),
    "雪": ("snow", "scene"), "下雪": ("snow", "scene"),
    "海边": ("beach", "scene"), "沙滩": ("beach", "scene"),
    "森林": ("forest", "scene"), "树林": ("forest", "scene"),
    "花园": ("garden", "scene"),
    "神社": ("shrine", "scene"), "寺庙": ("temple", "scene"),
    "废墟": ("ruins", "scene"), "遗迹": ("ruins", "scene"),
    "城堡": ("castle", "scene"),
    "霓虹": ("neon_lights", "scene"),
    "星空": ("starry_sky", "scene"),
    "天空": ("sky", "scene"),
    "房间": ("indoors", "scene"), "室内": ("indoors", "scene"),
    "咖啡厅": ("cafe", "scene"), "图书馆": ("library", "scene"),
    "车站": ("train_station", "scene"),
    "教堂": ("church", "scene"),
    "水面": ("water", "scene"), "湖": ("lake", "scene"),

    # ---------------- 构图 / 视角
    "全身": ("full_body", "shot"), "半身": ("upper_body", "shot"),
    "特写": ("close-up", "shot"), "头像": ("portrait", "shot"),
    "正面": ("front_view", "shot"), "侧面": ("side_view", "shot"),
    "背影": ("from_behind", "shot"),
    "仰视": ("from_below", "shot"), "俯视": ("from_above", "shot"),
    "看镜头": ("looking_at_viewer", "shot"),

    # ---------------- 种族 / 特效
    "猫耳": ("cat_ears", "effect"), "猫娘": ("cat_ears", "effect"),
    "兽耳": ("animal_ears", "effect"),
    "狐狸耳": ("fox_ears", "effect"),
    "精灵": ("pointy_ears", "effect"),
    "天使": ("angel", "effect"), "恶魔": ("demon", "effect"),
    "翅膀": ("wings", "effect"),
    "光环": ("halo", "effect"),
    "魔法阵": ("magic_circle", "effect"),
    "发光": ("glowing", "effect"),
    "纹身": ("tattoo", "effect"),
    "眼罩": ("eyepatch", "effect"),
}

# 长词优先，避免"过膝袜"被拆成"袜"
_KEYS = sorted(DICT.keys(), key=len, reverse=True)

# 每种用途下，哪些类别有资格进 AND 查询（其余只能当加分项）。
# 判断标准是"这个词卡错了会导致搜不到吗"：
#   动作错了 → 出来的图完全不是要的东西，必须卡
#   发色错了 → IP-Adapter 会把颜色从参考图带出来，不必卡
CORE_CATS = {
    "pose": ("action", "who"),
    "outfit": ("cloth", "who"),
    "char": ("who", "hair", "eyes"),
    "scene": ("scene", "shot"),
    "style": ("who",),
}


def parse(text, preset=""):
    """返回 {"core": [...], "extra": [...], "all": [...], "miss": [...]}。

    core  拿去真查（AND），只取 2-3 个
    extra 当加分项，在结果排序里用
    """
    used = [False] * len(text)
    hits = []
    for k in _KEYS:
        start = 0
        while True:
            i = text.find(k, start)
            if i < 0:
                break
            if not any(used[i:i + len(k)]):
                for j in range(i, i + len(k)):
                    used[j] = True
                hits.append((i, i + len(k), DICT[k][0], DICT[k][1]))
            start = i + len(k)

    hits.sort()

    # 颜色 + 衣物/头发 要合成一个 tag。
    # "黑色大衣" 如果拆成 black + coat，black 会单独进核心查询，
    # 而它在图站上是个极泛的词（黑背景也算），等于把查询稀释掉。
    # 合成 black_coat / black_hair 之后才是准确的那个词。
    MERGE_INTO = {"hair", "cloth"}
    merged = []
    i = 0
    while i < len(hits):
        s_, e_, tag, cat = hits[i]
        if cat == "color" and i + 1 < len(hits):
            s2, _e2, tag2, cat2 = hits[i + 1]
            if s2 == e_ and cat2 in MERGE_INTO:
                merged.append((s_, tag + "_" + tag2, cat2))
                i += 2
                continue
        merged.append((s_, tag, cat))
        i += 1

    seen, tags = set(), []
    for _i, tag, cat in merged:
        if tag in seen:
            continue
        seen.add(tag)
        tags.append((tag, cat))

    if not tags:
        return {"core": [], "extra": [], "all": [],
                "miss": [c for c in text if not c.isspace()]}

    # core 的挑选顺序跟 preset 有关：
    #   动作类：动作 > 人物 > 外观 > 场景
    #   场景类：场景 > 构图 > 人物 > 动作
    # 因为 AND 查询的"主词"决定结果池子长什么样，主词选错就全跑偏。
    if preset == "scene":
        order = {"scene": 0, "shot": 1, "who": 2, "action": 3,
                 "hair": 4, "eyes": 5, "cloth": 4, "effect": 6, "color": 7}
    elif preset == "char":
        order = {"who": 0, "hair": 1, "eyes": 2, "shot": 3,
                 "cloth": 4, "action": 5, "scene": 6, "effect": 6, "color": 7}
    else:
        order = {"action": 0, "who": 1, "hair": 2, "cloth": 3,
                 "effect": 4, "eyes": 5, "shot": 6, "scene": 7, "color": 8}

    ranked = sorted(tags, key=lambda t: order.get(t[1], 9))

    # core 只从"结构性"的类别里挑。颜色、发色、服装这些**不进查询**，
    # 只当加分项。原因是 AND 逻辑：白发 + 跑步 + 1girl 三个词一起查，
    # 很可能一条都不剩（跑步的人不一定会被打 white_hair 标）。
    # 而发色和服装本来就会由 IP-Adapter 从参考图带出来，不必靠查询去卡。
    # 动作和人数才是"这张图是不是我要的那类",必须卡住。
    allow = CORE_CATS.get(preset, ("action", "who"))
    core = [t for t, c in ranked if c in allow][:3]
    if not core:
        # 一个结构性词都没有（比如只说了"白发的她"），退而求其次
        core = [t for t, _c in ranked[:2]]
    extra = [t for t, _c in ranked if t not in core]
    return {"core": core, "extra": extra,
            "all": [t for t, _c in ranked], "miss": []}


def main():
    if len(sys.argv) < 2:
        print('用法：python nl2tags.py "樱花树下奔跑的白发少女"')
        return 1
    text = sys.argv[1]
    preset = ""
    if "--preset" in sys.argv:
        i = sys.argv.index("--preset")
        if i + 1 < len(sys.argv):
            preset = sys.argv[i + 1]
    r = parse(text, preset)
    if not r["all"]:
        print("一个词都没认出来。词典是按常见词收的，换个说法试试。")
        return 2
    print("原文：%s" % text)
    print("查询：%s" % " ".join(r["core"]))
    if r["extra"]:
        print("加分：%s" % " ".join(r["extra"]))
    print("全部：%s" % " ".join(r["all"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
