# -*- coding: utf-8 -*-
"""参考图检索：给一句 tag 描述，自动扒图、筛图、存进 refs/<类别>/。

为什么要有这个东西：
  之前多图参考（multi_ref.py）全靠手动丢图。可每换一个动作、换一套衣服，
  就得重新找图、重新上传，一次只能丢五张。这条脚本把「找图」也自动化 ——
  你说"要一张落樱庭院的背景"，它去图站按 tag 检索、筛掉不合格的、存好，
  顺手拼一张缩略图墙给你挑。

图源的现实情况（2026-11 全部实测，别照抄网上的老教程）：
  danbooru.donmai.us   超时，国内直连无望。挂了梯子才可用。
  safebooru.donmai.us  403 + Cloudflare 挑战页（Cf-Mitigated: challenge），
                       纯 HTTP 客户端过不去，除非上浏览器内核。
  konachan / yande.re  超时。
  safebooru.org        能连，但这是 gelbooru 的镜像，分级数据是 gelbooru 的：
                       抽 300 条 1girl，299 条是 q 级（擦边）。而且 dapi 不支持
                       服务端 rating 过滤（rating=safe 参数被忽略）。不能用。
  ★ www.zerochan.net  通，2.8 秒响应。审核制、严格 SFW、库大、按 tag 检索、
                       还能服务端按方向筛（d=portrait / landscape）。← 默认用这个

  Zerochan 的 URL 加 ?json 会直接吐 JSON（前面有个多余的 "{"，得跳过），
  每条带 id / width / height / md5 / 完整 tags（50 个左右）。
  这比解析 HTML 稳得多，所以在看到 JSON 之后就没再写 HTML 解析。

  图片直链不入 JSON，但能从 thumbnail 反推：
      thumbnail  https://s3.zerochan.net/240/00/41/4034550.avif
                        ↓ 把尺寸段换掉
      sample     https://s3.zerochan.net/600/00/41/4034550.jpg     ≈700KB
      full       https://s3.zerochan.net/full/00/41/4034550.png    通常 4MB+
  只有 240 / 600 / full 三档（800、1000、1200 都是 404，试过了）。

  为什么默认下 600 而不是 full：
      IP-Adapter 用 CLIP ViT-H，输入 224×224；ControlNet 的骨架检测也在千级。
      600 档够用，而 full 档单张 4MB、实测 107KB/s，下 10 张要 6 分钟。
      要拿去训 LoRA 才需要 full，加 --size full。

筛图的四个硬条件（少一个都会污染参考库）：
  1 分级     只留 SFW。Zerochan 全站审核过，天然安全；
             走 safebooru 分支时仍然会滤掉 q/e（那个站不滤不行）。
  2 分辨率   按【原图】短边筛，不是按下载下来的 600 档。
             原图都糊的图，缩到 600 只会更糊。
  3 宽高比   比例 1:6 的长条是条漫，裁方形必然切掉关键部分。
  4 tag 数   tag 太少的，多半是随手涂的草稿、截图、表情包。
             实测 Zerochan 每条都有 40+ 个 tag，这个门槛几乎不误杀。

tag 写法的坑（这条最重要）：
  图站是 AND 逻辑，只要有一个 tag 不存在，整条查询就归零，界面上看起来
  就是"没图"，你根本不知道是自己写错了一个词。实测：
      jumping         20805 条
      jumping + solo   9787 条
      jumping + dynamic     0 条   ← dynamic 在 safebooru 不是有效 tag
  所以脚本做两级容错：
      一级  逐个单查，剔掉"站上根本没有的这个词"
      二级  贪心累加，从主词开始一个个加，加进去归零就跳过这个
  贪心而不是"从后往前砍"，是因为后者会误伤 solo 这种有价值的限定词 ——
  它会把 "jumping solo dynamic" 砍成只剩 "jumping"，把单人约束一起扔掉。

用法：
  python ref_search.py -q "white hair, nekomimi" --preset char -n 10
  python ref_search.py --preset pose -q "jumping" -n 8
  python ref_search.py --preset scene -q "japanese garden, cherry blossoms" -n 6
  python ref_search.py --preset outfit -q "gothic lolita" -n 10
  python ref_search.py -q "..." -c tmp --dry        只看筛出几张，不下载
  python ref_search.py --list-presets               看预设规则
  python ref_search.py -q "..." --site safebooru    换备用站（会滤分级，命中率低）
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = r"D:\AI-Video-Setup"
# 提骨架要 import extract_pose，它自己再 import rtmlib。
# 不把项目目录塞进 sys.path 的话，在别的目录跑这个脚本就 ModuleNotFoundError。
if BASE not in sys.path:
    sys.path.insert(0, BASE)
REF_ROOT = os.path.join(BASE, "refs")
INDEX_PATH = os.path.join(REF_ROOT, "_index.json")
TOGGLES = os.path.join(BASE, "toggles.json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

ZEROCHAN = "https://www.zerochan.net"
SAFEBOORU = "https://safebooru.org/index.php"
DANBOORU = "https://danbooru.donmai.us/posts.json"

# 每个类别一套筛图规则。
# require  硬条件，搜不到就说明这个类别不适合这个站
# prefer   加分项，同批里优先挑这些
# ratio    宽高比区间（w/h），跟用途强相关：
#          半身可以接近方形，服装和动作必须竖幅全身，场景必须横幅
# dir      发给 Zerochan 的服务端方向筛选，能先把一半废图挡在门外
PRESETS = {
    "char": {
        "require": [],
        "prefer": ["upper_body", "portrait", "looking_at_viewer", "detailed",
                   "highres", "front_view", "close-up"],
        "ratio": (0.55, 1.70),
        "dir": "portrait",
    },
    "outfit": {
        "require": ["solo"],
        "prefer": ["full_body", "detailed_clothes", "standing", "front_view",
                   "dress", "uniform", "highres"],
        "ratio": (0.42, 0.80),
        "dir": "portrait",
    },
    "pose": {
        "require": ["solo"],
        "prefer": ["full_body", "dynamic_pose", "jumping", "running",
                   "flying", "outstretched_arm", "standing"],
        "ratio": (0.38, 0.80),
        "dir": "portrait",
    },
    "style": {
        "require": [],
        "prefer": ["highres", "detailed", "illustration"],
        "ratio": (0.45, 2.00),
        "dir": "",
    },
    "scene": {
        "require": ["no_humans"],
        "prefer": ["scenery", "detailed_background", "depth_of_field", "highres"],
        "ratio": (1.10, 2.40),
        "dir": "landscape",
    },
}

# Zerochan 用日式/英文 Title Case 词表，跟 danbooru 的下划线词表不一样。
# 这几个是最容易踩空的，值给空串表示"这个词在目标站没有对应，直接删掉"。
ZERO_TAG_MAP = {
    "1girl": "Female",
    "1boy": "Male",
    "solo": "",
    "cat_ears": "Nekomimi",
    "dog_ears": "Inumimi",
    "fox_ears": "Kitsunemimi",
    "wolf_ears": "Okamimimi",
    "animal_ears": "Kemonomimi",
    "no_humans": "No Humans",
    "looking_at_viewer": "Looking At Viewer",
    "long_hair": "Long Hair",
    "short_hair": "Short Hair",
    "white_hair": "White Hair",
    "black_hair": "Black Hair",
    "blonde_hair": "Blonde Hair",
    "blue_eyes": "Blue Eyes",
    "red_eyes": "Red Eyes",
    "school_uniform": "School Uniform",
}


# ---------------------------------------------------------------- 网络

def http_get(url, timeout=40, retries=3, proxy="", referer="", slow=True):
    """带重试和退避的 GET，返回 bytes。

    slow=True 时会尊重站点 1 req/s 的礼貌间隔（Zerochan 查得太快会被掐）。
    """
    headers = {
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer:
        headers["Referer"] = referer

    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = "HTTP %s" % e.code
            if e.code == 404:
                break
            if e.code in (403, 429):
                time.sleep(3.0 * (attempt + 1))     # 被限速了，老实退避
                continue
            if e.code == 503:
                # Zerochan 被打急的时候回 503（不是 429）。这个退避必须够长，
                # 否则越试越死：实测连续几次后要等半分钟以上才恢复。
                time.sleep(12.0 * (attempt + 1))
                continue
            time.sleep(1.0)
        except Exception as e:
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("请求失败：%s（%s）" % (last, url[:90]))


def nl_enabled():
    """中文 → tag 的开关，存在 toggles.json 的 nl_tags，默认开。

    为什么不默认关：这个功能完全离线、零成本、最多几百毫秒，
    开着不会拖慢什么；真想手写英文加 --no-nl 就行。
    """
    try:
        with open(TOGGLES, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("nl_tags", True))
    except Exception:
        return True


def norm_tags(text):
    """把 "cat ears, white hair" 这类人话切成 tag 列表。

    也兼容已经写好的 "cat_ears white_hair"。逗号和空格都当分隔。
    """
    parts = re.split(r"[,\s]+", (text or "").strip())
    out = []
    for p in parts:
        p = p.strip().strip("_")
        if not p:
            continue
        p = p.replace("(", "\x00").replace(")", "\x01")
        p = p.replace(" ", "_").lower()
        p = p.replace("\x00", "(").replace("\x01", ")")
        out.append(p)
    return out


def has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def to_zero_tag(t):
    """danbooru 风格 tag → Zerochan 风格。

    white_hair → White Hair（首字母大写，下划线换空格）
    已经是大写开头的（用户直接抄的站内 tag）保持原样。
    """
    if t in ZERO_TAG_MAP:
        return ZERO_TAG_MAP[t]
    if any(c.isupper() for c in t):
        return t.replace("_", " ")
    return " ".join(w[:1].upper() + w[1:] for w in t.replace("_", " ").split())


# ---------------------------------------------------------------- provider

def _zero_urls(thumb):
    """从 thumbnail 反推 600 档和原图地址。

    thumbnail 形如 https://s3.zerochan.net/240/00/41/4034550.avif
    中间那两段（00/41）是分片目录，换尺寸只改第一段。
    """
    m = re.match(r"https://s3\.zerochan\.net/\d+/([^/]+/[^/]+)/(\d+)\.(\w+)", thumb)
    if not m:
        return [], []
    d, pid = m.group(1), m.group(2)
    # 600 档实测只有 jpg；原图 png 和 jpg 都出现过，所以都留一个候选
    sample = ["https://s3.zerochan.net/600/%s/%s.jpg" % (d, pid)]
    full = ["https://s3.zerochan.net/full/%s/%s.png" % (d, pid),
            "https://s3.zerochan.net/full/%s/%s.jpg" % (d, pid)]
    return sample, full


def search_zerochan(tags, page=1, direction="", proxy="", limit=48):
    """Zerochan 检索。返回 (posts, total)。

    total 是站点估的命中数；拿不到就拿本页条数顶一下。
    """
    ztags = []
    for t in tags:
        z = to_zero_tag(t)
        if z:
            ztags.append(z)
    if not ztags:
        return [], 0

    q = {"p": page, "json": ""}
    if direction in ("portrait", "landscape", "square"):
        q["d"] = direction
    url = "%s/%s?%s" % (ZEROCHAN,
                        urllib.parse.quote(",".join(ztags), safe=","),
                        urllib.parse.urlencode(q))
    raw = http_get(url, timeout=40, proxy=proxy, referer=ZEROCHAN + "/")
    text = raw.decode("utf-8", "replace")

    # 响应前面会多一个 "{"，后面还跟着页脚，所以 raw_decode 只吃第一个对象
    m = re.search(r'\{\s*"items"', text)
    if not m:
        return [], 0
    try:
        data, _ = json.JSONDecoder().raw_decode(text[m.start():])
    except Exception:
        return [], 0

    posts = []
    for it in data.get("items") or []:
        try:
            w, h = int(it.get("width") or 0), int(it.get("height") or 0)
        except ValueError:
            continue
        if w < 8 or h < 8:
            continue
        samp, full = _zero_urls(it.get("thumbnail") or "")
        # Zerochan 的 tags 是 Title Case，统一压成 danbooru 风格方便打分和去重
        tags_l = [t.lower().replace(" ", "_") for t in (it.get("tags") or [])]
        posts.append({
            "site": "zerochan",
            "post_id": str(it.get("id") or ""),
            "md5": it.get("md5") or "",
            "w": w, "h": h,
            "rating": "s",                      # 全站审核过，没有 q/e
            "tags": tags_l,
            "full": full,
            "sample": samp,
            "page": "%s/%s" % (ZEROCHAN, it.get("id") or ""),
            "source": it.get("source") or "",
        })
    return posts, len(posts)


def search_safebooru(tags, limit, pid=0, proxy=""):
    """safebooru.org 的 dapi。能通，但分级里 q 级占 98%，靠客户端滤。"""
    q = urllib.parse.urlencode({
        "page": "dapi", "s": "post", "q": "index",
        "tags": " ".join(tags), "limit": limit, "pid": pid,
    })
    raw = http_get("%s?%s" % (SAFEBOORU, q), timeout=35, proxy=proxy,
                   referer="https://safebooru.org/")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return [], 0

    total = int(root.get("count") or 0)
    posts = []
    for p in root.findall("post"):
        try:
            w, h = int(p.get("width") or 0), int(p.get("height") or 0)
        except ValueError:
            continue
        posts.append({
            "site": "safebooru",
            "post_id": str(p.get("id") or ""),
            "md5": p.get("md5") or "",
            "w": w, "h": h,
            "rating": (p.get("rating") or "").lower(),
            "tags": (p.get("tags") or "").strip().split(),
            "full": [p.get("file_url")] if p.get("file_url") else [],
            "sample": [p.get("sample_url")] if p.get("sample_url") else [],
            "page": "https://safebooru.org/index.php?page=post&s=view&id=%s" % p.get("id"),
        })
    return posts, total


def search_danbooru(tags, limit, page=1, proxy=""):
    """danbooru JSON API。国内要梯子，但带 score 排序，筛图最准。"""
    q = urllib.parse.urlencode({"tags": " ".join(tags), "limit": limit, "page": page})
    raw = http_get("%s?%s" % (DANBOORU, q), timeout=30, proxy=proxy,
                   referer="https://danbooru.donmai.us/")
    try:
        arr = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return [], 0
    if not isinstance(arr, list):
        return [], 0

    posts = []
    for p in arr:
        if not p.get("file_url"):
            continue
        posts.append({
            "site": "danbooru",
            "post_id": str(p.get("id") or ""),
            "md5": p.get("md5") or "",
            "w": p.get("image_width") or 0,
            "h": p.get("image_height") or 0,
            "rating": (p.get("rating") or "").lower(),
            "tags": (p.get("tag_string") or "").split(),
            "score": p.get("score") or 0,
            "full": [p.get("file_url")],
            "sample": [p.get("large_file_url") or p.get("file_url")],
            "page": "https://danbooru.donmai.us/posts/%s" % p.get("id"),
        })
    return posts, len(posts)


def fetch_page(tags, site, page, proxy="", limit=100, direction=""):
    """统一入口。两个站的翻页语义不同，在这里抹平：
    zerochan / danbooru 的 page 从 1 开始，safebooru 的 pid 从 0 开始。
    """
    if site == "zerochan":
        return search_zerochan(tags, page + 1, direction, proxy)
    if site == "danbooru":
        return search_danbooru(tags, limit, page + 1, proxy)
    return search_safebooru(tags, limit, page, proxy)


def probe_tag(tag, site="zerochan", proxy=""):
    """单独查一个 tag 有多少命中。返回条数，失败返回 -1。

    为什么要这一步：图站的 tag 之间是 AND。只要有一个 tag 不存在，
    整条查询直接归零，返回空列表 —— 界面上看起来就是"没图"，
    用户根本不知道是自己写错了一个词。
    """
    try:
        posts, tot = fetch_page([tag], site, 0, proxy, limit=1)
        return len(posts) if tot <= 0 else tot
    except Exception:
        return -1


# ---------------------------------------------------------------- 筛选

# 这些标记一出现，图基本就没法当参考了。实测 safebooru.org 放宽到 q 级以后，
# 前几张全是同人志封面、卡片、Q 版 logo 图 —— 这个站没有评分体系，
# 好画师的作品大多在 danbooru 上（那边有 score 排序），所以垃圾比例很高。
JUNK_TAGS = {
    # 带文字的：同人志封面、漫画截图、素材图
    "text", "english_text", "japanese_text", "chinese_text", "korean_text",
    "bilingual_text", "translation_request", "commentary", "commentary_request",
    # 分镜/拼贴：一格里塞好几个画面，裁方形必然切坏
    "comic", "4koma", "multiple_views", "split_screen", "collage",
    # 比例不对或不是画师成稿
    "chibi", "artist_request", "sketch", "monochrome", "greyscale",
    # 卡片/杂志版式
    "card", "trading_card", "magazine_scan", "scan", "border",
    # 明显的水印/logo
    "watermark", "artist_logo", "english_commentary",
}


def judge(it, min_px, ratio_min, ratio_max, min_tags, allow_q=False):
    """返回 (是否合格, 原因)。原因留着，被滤掉多少张得能说清楚。

    分级这里必须卡死。safebooru.org 的 rating 字段是 gelbooru 的原始标记：
    抽 300 条 1girl，299 条是 q。q 在 gelbooru 的定义是"擦边"（泳装、内衣、
    轻度暴露），不是色情，但塞进创作参考库不合适 —— 用户拿它画正经角色，
    参考图却是泳装，构图和服装全带偏。
    所以默认只留 s；确实想要更多候选时用 --allow-q 自己开。
    """
    w, h = it["w"], it["h"]
    if w < 8 or h < 8:
        return False, "尺寸无效"
    if not allow_q and it["rating"] != "s":
        return False, "分级"
    if allow_q and it["rating"] not in ("s", "q"):
        return False, "分级"
    if JUNK_TAGS.intersection(it["tags"]):
        return False, "垃圾图"
    if min(w, h) < min_px:
        return False, "短边"
    r = w / float(h)
    if r < ratio_min or r > ratio_max:
        return False, "比例"
    if len(it["tags"]) < min_tags:
        return False, "tag少"
    if not (it["full"] or it["sample"]):
        return False, "无地址"
    return True, ""


def rank(it, prefer):
    """排序分：tag 越多越可能是精修图；命中 prefer 每个加分；danbooru 有 score 就用。"""
    s = len(it["tags"]) * 0.05
    tset = set(it["tags"])
    for k, t in enumerate(prefer):
        if t in tset:
            s += 3.0 - k * 0.1
    if it.get("score"):
        s += min(it["score"], 300) / 30.0
    for flag in ("absurdres", "highres", "detailed"):
        if flag in tset:
            s += 1.5
    return s


# ---------------------------------------------------------------- 下载

def _ext_from(url, blob):
    low = url.lower().split("?")[0]
    for e in (".png", ".jpg", ".jpeg", ".webp"):
        if low.endswith(e):
            return ".jpg" if e == ".jpeg" else e
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:4] == b"RIFF":
        return ".webp"
    return ".jpg"


def download_one(it, dst_dir, size="sample", proxy="", timeout=90):
    """下载一张。返回 (ok, 路径或错误, 字节数)。

    候选地址逐个试：Zerochan 的 600 档基本都是 jpg，但原图 png/jpg 都有，
    猜错就是 404，所以留两手。404 要立刻换下一个，不能傻等重试。
    """
    urls = it["sample"] if size == "sample" else it["full"]
    if not urls:
        urls = it["full"] or it["sample"]
    if not urls:
        return False, "无地址", 0

    blob, used = None, ""
    last = ""
    for u in urls:
        if not u:
            continue
        try:
            blob = http_get(u, timeout=timeout, retries=2, proxy=proxy,
                            referer="https://www.zerochan.net/")
            used = u
            break
        except Exception as e:
            last = str(e)[:50]
            continue
    if blob is None:
        return False, last or "全失败", 0

    if len(blob) < 20000:
        return False, "只有 %d 字节" % len(blob), 0
    head = blob[:16].lstrip()
    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html"):
        return False, "返回的是网页", 0

    ext = _ext_from(used, blob)
    name = "%s_%s%s" % (it["site"][:3], it["post_id"], ext)
    dst = os.path.join(dst_dir, name)
    tmp = dst + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, dst)
    return True, dst, len(blob)


# ---------------------------------------------------------------- 图墙

def imread_u(path):
    """cv2 在 Windows 上读不了非 ASCII 路径，绕过去。"""
    import numpy as np
    import cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(path, img):
    import cv2
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def contact_sheet(paths, out_path, cols=5, cell=250):
    """拼缩略图墙。懒得一张张点开，看这一张就够。

    编号必须和命令行输出顺序一致，所以调用方要保证 paths 已排好序。
    """
    import numpy as np
    import cv2

    imgs = [p for p in (imread_u(x) for x in paths) if p is not None]
    if not imgs:
        return None
    cols = min(cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 22, np.uint8)

    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        s = min(cell / float(w), cell / float(h)) * 0.94
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y = (i // cols) * cell + (cell - nh) // 2
        x = (i % cols) * cell + (cell - nw) // 2
        sheet[y:y + nh, x:x + nw] = small
        cv2.rectangle(sheet, (x + 3, y + 3), (x + 28, y + 24), (0, 0, 0), -1)
        cv2.putText(sheet, str(i + 1), (x + 7, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    imwrite_u(out_path, sheet)
    return out_path


# ---------------------------------------------------------------- 索引

def load_index():
    if os.path.isfile(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_index(idx):
    os.makedirs(REF_ROOT, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 主流程

def run(args):
    preset = PRESETS.get(args.preset) or {"require": [], "prefer": [],
                                          "ratio": (0.4, 2.4), "dir": ""}
    ratio_min = args.ratio_min if args.ratio_min is not None else preset["ratio"][0]
    ratio_max = args.ratio_max if args.ratio_max is not None else preset["ratio"][1]
    direction = args.dir or preset["dir"]

    # pose 通道是 ControlNet openpose 的输入，它只吃黑底骨架。
    # 动作类扒下来的照片不过一道骨架提取，对 pose 通道就是废图，所以默认开。
    if args.skeleton is None:
        args.skeleton = (args.preset == "pose")

    # --- 中文 → tag ---
    # 纯词典映射，不联网不调模型。开关在 toggles.json 的 nl_tags，
    # 关掉就退回原来的行为（提示你手写英文）。
    nl_extra = []
    if has_cjk(args.query):
        if args.no_nl or not nl_enabled():
            print("[提示] 中文翻译已关。图站只认英文 tag（danbooru / Zerochan 词表）。")
            print("       例子：白发 white hair / 樱花 cherry blossoms / 奔跑 running")
            if has_cjk(args.query):
                # 中文直接拿去查必定一条都没有，白跑一轮请求，直接拦下
                print("（查询里还有中文。换成英文，或者把开关打开：")
                print("  出图开关面板按 9，或改 toggles.json 的 nl_tags）")
                return 2
        else:
            import nl2tags as NL
            r = NL.parse(args.query, args.preset)
            if r["core"]:
                print("中文解析：%s" % " ".join(r["core"]))
                if r["extra"]:
                    print("  加分项：%s（只参与排序，不进查询）" % " ".join(r["extra"]))
                args.query = ", ".join(r["core"])
                nl_extra = r["extra"]
            else:
                print("[提示] 这句没认出任何词。词典只收常见词（两百来个），")
                print("       换成具体的东西试试：白发少女 / 黑色大衣 / 雨天街道 / 蹲着")

    tags = norm_tags(args.query)
    for t in preset["require"]:
        if t not in tags:
            tags.append(t)
    if not tags:
        print('没给 tag。至少写一个，例如 -q "white hair, nekomimi"')
        return 2

    # 中文解析出来的"加分项"接到排序偏好后面。
    # 这些词不参与 AND 查询（多一个词就多一次归零），只在挑图时加权。
    prefer = list(preset["prefer"]) + nl_extra

    for t in norm_tags(args.neg):
        tags.append("-" + t)
    pick = [t for t in tags if not t.startswith("-")]
    negs = [t for t in tags if t.startswith("-")]

    print("检索：%s" % " ".join(tags))
    print("规则：短边>=%d  比例 %.2f~%.2f  tag>=%d  分类=%s  方向=%s"
          % (args.min_px, ratio_min, ratio_max, args.min_tags,
             args.cat, direction or "不限"))

    # --- 找图 ---
    # 请求预算很紧（Zerochan 大约 1 req/s，连发十几次就 503），所以顺序很讲究：
    #   1 先按原样搜一次 —— 绝大多数情况用户写的词是对的，一次就中
    #   2 只有归零了，才值得花请求去查是哪个词的问题
    # 之前的写法是每次都先跑 5 次 tag 校验再搜，等于把预算花在大概率用不上的地方。
    pool = []
    sites = [args.site] if args.site != "auto" else ["safebooru", "zerochan", "danbooru"]
    for site in sites:
        use = list(pick)
        skipped = []
        try:
            batch, _ = fetch_page(use + negs, site, 0, args.proxy, direction=direction)
        except Exception as e:
            print("  [%s] 直搜失败：%s" % (site, str(e)[:70]))
            batch = []

        if not batch and len(pick) > 1:
            # --- 一级：剔掉站上根本不存在的词（这个可以一次问清，省得反复试）---
            if not args.no_check:
                bad = []
                for t in pick:
                    n = probe_tag(t, site, args.proxy)
                    if n == 0:
                        bad.append(t)
                    time.sleep(1.5)
                if bad:
                    print("  [%s] 站上没有这些 tag，剔除：%s" % (site, ", ".join(bad)))
                    use = [t for t in use if t not in bad]

            # --- 二级：贪心累加，加进去归零就跳过 ---
            # 不用"从后往前砍"，那会把 solo 这类有价值的限定词一起扔掉：
            # "jumping solo dynamic" 会被砍成只剩 jumping，单人约束没了。
            if use:
                cur = []
                for t in use:
                    if not cur:
                        cur.append(t)
                        continue
                    try:
                        posts_try, _ = fetch_page(cur + [t] + negs, site, 0,
                                                  args.proxy, direction=direction)
                    except Exception:
                        cur.append(t)      # 网络抽风，别乱丢用户的词
                        continue
                    if posts_try:
                        cur.append(t)
                    else:
                        skipped.append(t)
                    time.sleep(1.5)
                use = cur
                if skipped:
                    print("  [%s] 和主词没交集，跳过：%s" % (site, ", ".join(skipped)))
                if use:
                    try:
                        batch, _ = fetch_page(use + negs, site, 0, args.proxy,
                                              direction=direction)
                    except Exception as e:
                        print("  [%s] 重搜失败：%s" % (site, str(e)[:70]))
                        batch = []

        if not batch:
            continue
        if skipped or (use != pick):
            print("  [%s] 实际检索：%s" % (site, " ".join(use)))

        scanned, page = 0, 0
        if args.random_page:
            page = random.randint(1, 12)     # 随机翻页，换一批，别老是第一页
        while True:
            if batch:
                pool.extend(batch)
                scanned += len(batch)
                print("  [%s] 第 %d 页 %d 条（累计 %d）"
                      % (site, page + 1, len(batch), scanned))
            if scanned >= args.max_scan or len(batch) < 40:
                break
            page += 1
            time.sleep(2.5)                  # 站点礼貌间隔，打太急会被掐
            try:
                batch, _ = fetch_page(use + negs, site, page, args.proxy,
                                      direction=direction)
            except Exception as e:
                print("  [%s] 第 %d 页失败：%s" % (site, page + 1, str(e)[:70]))
                break

        if pool:
            break                  # 第一个能出图的站就够了，不折腾后面的

    if not pool:
        print("一张都没拿到。可能是网络问题，或者 tag 太刁钻。")
        print("试试：--preset style -q \"anime girl\"   先确认链路通不通")
        return 3

    print("拿到 %d 条，开始筛..." % len(pool))

    # --- 筛 ---
    keep, drop = [], {}
    seen_md5 = set()
    idx = load_index()
    for it in pool:
        ok, why = judge(it, args.min_px, ratio_min, ratio_max, args.min_tags, args.allow_q)
        key = "%s:%s" % (it["site"], it["post_id"])
        if ok and it["md5"] and it["md5"] in seen_md5:
            ok, why = False, "重复"
        if ok and key in idx:
            ok, why = False, "本地已有"
        if not ok:
            drop[why] = drop.get(why, 0) + 1
            continue
        if it["md5"]:
            seen_md5.add(it["md5"])
        keep.append(it)

    if drop:
        print("筛掉 %d 条：%s" % (sum(drop.values()),
                                "  ".join("%s %d" % kv for kv in
                                          sorted(drop.items(), key=lambda kv: -kv[1]))))
    print("合格 %d 条" % len(keep))

    if not keep:
        print("筛完一张不剩。三个旋钮，从宽到窄依次调：")
        print("  --min-px 600      放宽分辨率（Zerochan 的原图短边常常只有 600）")
        print("  --min-tags 20     放宽 tag 数")
        print("  --preset style    取消 no_humans / 方向限制")
        return 4

    keep.sort(key=lambda x: rank(x, prefer), reverse=True)
    picks = keep[:args.num]

    if args.dry:
        print("\n[dry] 会下这几张（只列表，不下载）：")
        for it in picks:
            print("  %-9s %5dx%-5d 比例%.2f  %s"
                  % (it["site"], it["w"], it["h"], it["w"] / float(it["h"]),
                     " ".join(it["tags"][:8])))
        return 0

    # --- 下载 ---
    out_dir = os.path.join(REF_ROOT, args.cat)
    os.makedirs(out_dir, exist_ok=True)
    print("\n下载 %d 张到 refs/%s/ ..." % (len(picks), args.cat))

    order = {id(it): n for n, it in enumerate(picks)}
    got = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, it, out_dir, args.size, args.proxy): it
                for it in picks}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            it = futs[fut]
            try:
                ok, path, nbytes = fut.result()
            except Exception as e:
                ok, path, nbytes = False, str(e)[:50], 0
            if ok:
                got.append((path, it))
                print("  [%d/%d] OK   %s  %dKB" % (i, len(picks),
                                                  os.path.basename(path), nbytes // 1024))
            else:
                print("  [%d/%d] 跳过 %s（%s）" % (i, len(picks), it["post_id"], path))

    if not got:
        print("全部失败。多半是被限速了，隔一会儿再试，或者 --workers 2")
        return 5

    got.sort(key=lambda r: order[id(r[1])])
    elapsed = time.time() - t0

    size_sum = sum(os.path.getsize(p) for p, _ in got)
    print("\n完成 %d 张，%.1f 秒，平均 %.0f KB/s"
          % (len(got), elapsed, size_sum / 1024.0 / max(elapsed, 0.1)))

    # --- 提骨架 ---
    # pose 通道走的是 ControlNet openpose，它只认黑底彩色骨骼，不认照片。
    # 所以动作类扒下来必须再过一道骨架提取，否则这张图对 pose 通道毫无用处。
    # 这也是为什么 pose preset 默认开骨架：不开的话这批图等于白下。
    if args.skeleton:
        import extract_pose as EP
        pairs = [(path, os.path.join(out_dir, "skel_%s.png" % it["post_id"]))
                 for path, it in got]
        print("\n提骨架（%d 张，模型要加载一次，约 20-40 秒）..." % len(pairs))
        t1 = time.time()
        res = EP.run_many(pairs, quiet=False)
        print("  骨架用时 %.1fs" % (time.time() - t1))
        for (path, it), r, (_src, skel) in zip(got, res, pairs):
            it["skeleton"] = os.path.relpath(skel, BASE) if r["ok"] else ""
            it["body_pts"] = r["body"]
        n_ok = sum(1 for it in (x[1] for x in got) if it.get("skeleton"))
        print("  可用骨架 %d/%d 张" % (n_ok, len(got)))
        if n_ok == 0:
            print("  一张都没提出骨架。二次元图的肢体常常和真人比例差太多，")
            print("  检测器认不出来。这种情况直接 skeleton_gen.py 捏骨架更靠谱。")

    # --- 记索引（下次自动跳过下过的图）---
    idx = load_index()
    for path, it in got:
        idx["%s:%s" % (it["site"], it["post_id"])] = {
            "file": os.path.relpath(path, BASE).replace("\\", "/"),
            "cat": args.cat,
            "w": it["w"], "h": it["h"],
            "tags": it["tags"],
            "query": args.query,
            "page": it["page"],
            "t": time.strftime("%Y-%m-%d %H:%M"),
        }
    save_index(idx)

    # --- 参考图包 ---
    # 给 multi_ref.py / make.py 直接吃用。它们不需要知道图是从哪个站扒的、
    # 怎么筛的，只关心"这批图里哪张排第一、骨架在哪"。
    pack = {
        "cat": args.cat,
        "query": args.query,
        "preset": args.preset,
        "site": pool[0]["site"] if pool else "",
        "t": time.strftime("%Y-%m-%d %H:%M"),
        "items": [
            {
                "file": os.path.relpath(path, BASE).replace("\\", "/"),
                "skeleton": it.get("skeleton", ""),
                "body_pts": it.get("body_pts", 0),
                "w": it["w"], "h": it["h"],
                "rank": round(rank(it, prefer), 2),
                "page": it["page"],
                "tags": it["tags"][:20],
            }
            for path, it in got
        ],
    }
    pack_path = os.path.join(out_dir, "_pack.json")
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=1)

    # --- 图墙 ---
    sheet = None
    if not args.no_sheet:
        sheet = contact_sheet([p for p, _ in got],
                              os.path.join(out_dir, "_sheet_%s.jpg"
                                           % time.strftime("%m%d_%H%M%S")))

    print("\n存到：refs/%s/" % args.cat)
    for n, (path, it) in enumerate(got, 1):
        print("  %2d  %-24s %5dx%-5d  %s"
              % (n, os.path.basename(path), it["w"], it["h"],
                 it["page"]))
    if sheet:
        print("\n图墙：%s" % sheet)

    print("\n参考图包：%s" % pack_path)
    print("下一步，让它自己去包里配对：")
    print('  python multi_ref.py --pack "%s"' % pack_path)
    if any(it.get("skeleton") for _, it in got):
        print('  或手动指定某一对：')
        print('  python multi_ref.py --pose "%s" --char ref_0998187d2dca.png'
              % [it["skeleton"] for _, it in got if it.get("skeleton")][0])
    return 0


def interactive():
    """交互式向导。命令行参数记不住的时候用这个，双击 bat 就进这里。"""
    cats = [("char", "角色长相 / 半身"), ("outfit", "服装 / 全身"),
            ("pose", "动作 / 全身"), ("style", "画风"), ("scene", "场景背景")]

    print("=" * 58)
    print(" 找参考图 —— 按 tag 检索，筛完存进 refs/<类别>/")
    print("=" * 58)
    print(" 图源说明：国内直连能用的只有 safebooru.org（老库，质量参差）。")
    print(" 挂着梯子的话，命令行加 --site danbooru 效果最好。")
    print("")

    while True:
        print("要哪一类？")
        for i, (k, label) in enumerate(cats, 1):
            print("  %d  %-6s %s" % (i, k, label))
        print("  q  退出")
        c = input("选：").strip().lower()
        if c in ("q", "", "quit", "exit"):
            return 0
        if not c.isdigit() or not (1 <= int(c) <= len(cats)):
            print("输个 1-%d 的数字。\n" % len(cats))
            continue
        preset = cats[int(c) - 1][0]

        q = input("关键词（英文，逗号分隔。例：white hair, nekomimi）：").strip()
        if not q:
            print("没写关键词，回上一步。\n")
            continue

        n = input("要几张？（回车=8）：").strip()
        n = int(n) if n.isdigit() and int(n) > 0 else 8

        print("")
        # 注意存到 <preset>_web 而不是 <preset>：
        # refs/char/ 里是 prep_refs.py 产出的 49 张训练集，混进网图会把训练数据搅脏。
        argv = ["--preset", preset, "-q", q, "-n", str(n), "-c", preset + "_web"]
        if preset in ("pose", "outfit", "style"):
            # 这两类放宽到 q 是划算的：动作和画风跟穿多少无关，
            # 而 s 级在这个站只占 1.5%，卡死就出不图。
            # 服装类不放宽 —— q 级服装图多半是泳装，参考价值是负的。
            if preset != "outfit":
                a = input("允许 q 级（泳装/内衣这类擦边图）？动作和画风建议允许 [y/N]：").strip().lower()
                if a == "y":
                    argv.append("--allow-q")
                    print("")

        rc = run(_parse(argv))
        print("\n" + "-" * 58 + "\n")
        return rc


def _parse(argv):
    """给 interactive 用：把参数列表解析成 args。"""
    return main_parser().parse_args(argv)


def main(argv=None):
    ap = main_parser()
    args = ap.parse_args(argv)

    if args.ui:
        return interactive()

    if args.list_presets:
        for k, v in PRESETS.items():
            print("[%s]  方向=%s  比例 %.2f~%.2f"
                  % (k, v["dir"] or "不限", v["ratio"][0], v["ratio"][1]))
            print("   必须  %s" % (" ".join(v["require"]) or "（无）"))
            print("   加分  %s" % " ".join(v["prefer"]))
        return 0

    return run(args)


def main_parser():
    ap = argparse.ArgumentParser(
        description="按 tag 检索参考图，筛完存进 refs/<类别>/",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--query", default="", help="tag 描述，逗号或空格分隔，英文")
    ap.add_argument("-c", "--cat", default="misc", help="存到 refs/<cat>/，默认 misc")
    ap.add_argument("-n", "--num", type=int, default=8, help="要几张，默认 8")
    ap.add_argument("--preset", default="",
                    choices=["", "char", "outfit", "pose", "style", "scene"],
                    help="筛图预设，决定方向、宽高比、加分词")
    ap.add_argument("--neg", default="", help="排除词，逗号分隔，例如 comic,text")
    ap.add_argument("--site", default="auto",
                    choices=["auto", "zerochan", "safebooru", "danbooru"])
    ap.add_argument("--dir", default="", choices=["", "portrait", "landscape", "square"],
                    help="方向筛选，默认跟着 preset")
    ap.add_argument("--min-px", type=int, default=700, help="原图短边下限，默认 700")
    ap.add_argument("--ratio-min", type=float, default=None)
    ap.add_argument("--ratio-max", type=float, default=None)
    ap.add_argument("--min-tags", type=int, default=15, help="tag 数下限，默认 15")
    ap.add_argument("--size", default="sample", choices=["sample", "full"],
                    help="sample=600档（快），full=原图（慢，训 LoRA 才需要）")
    ap.add_argument("--workers", type=int, default=4, help="并发下载数，默认 4")
    ap.add_argument("--max-scan", type=int, default=900,
                    help="最多翻多少条，默认 900（s 级只占约 1.5%%，扫得少就出不了图）")
    ap.add_argument("--random-page", action="store_true", help="随机翻页，换一批")
    ap.add_argument("--proxy", default="", help="代理，例如 http://127.0.0.1:7890")
    ap.add_argument("--allow-q", action="store_true",
                    help="放行 q 级（擦边：泳装/内衣），默认只留 s 级")
    ap.add_argument("--dry", action="store_true", help="只筛选不下载")
    ap.add_argument("--no-sheet", action="store_true", help="不拼图墙")
    ap.add_argument("--skeleton", dest="skeleton", action="store_true", default=None,
                    help="下载后提骨架（pose 类默认开，因为 pose 通道只认骨架不认照片）")
    ap.add_argument("--no-skeleton", dest="skeleton", action="store_false",
                    help="不提骨架")
    ap.add_argument("--no-check", action="store_true", help="跳过 tag 校验（快，但拼错一个词就搜不到）")
    ap.add_argument("--no-nl", action="store_true",
                    help="关掉中文自动翻译（默认开，词典映射不联网）")
    ap.add_argument("--ui", action="store_true", help="交互式向导")
    ap.add_argument("--list-presets", action="store_true")
    return ap


if __name__ == "__main__":
    sys.exit(main())
