"""交互式开关面板：翻开关 -> 直接出图，不用记命令行参数。

为什么单独做一个开关面板：
  make.py 的参数已经够少了，但"今天想换个动作""这张不想超分"这种事
  每次都要敲一遍命令行，实际用起来还是会嫌烦。
  这里把每个能力做成一行开关，输编号就翻，翻完按 r 直接跑。

开着 / 关了分别是什么：
  骨架 off   不用 ControlNet，提示词里随机塞一句姿势，模型自由发挥
  骨架 auto  随机抽一个预设骨架（动作多样，但有结构约束）
  骨架 指定  固定那个动作，每次都一样
  角色锁     开 = IP-Adapter 拿 refs/char 里的图锁长相
  超分       开 = 出图后再 4x（约 +20 秒，3072x5376）
  换场景     关 / cherry（樱花庭院）/ ruin（废教堂）/ night（霓虹夜街）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = r"D:\AI-Video-Setup"
TOGGLES = os.path.join(BASE, "toggles.json")
REF_DIR = os.path.join(BASE, "refs", "char")
REF_ROOT = os.path.join(BASE, "refs")

POSE_CYCLE = ["off", "auto", "reach_up", "jump", "stand", "sit"]
SCENE_CYCLE = ["", "cherry", "ruin", "night"]
PROFILE_CYCLE = ["3dcg", "soft", "vivid", "clear"]
SIZE_CYCLE = ["768x1344", "832x1216", "512x896", "768x1216"]

SCENE_LABEL = {"": "关", "cherry": "樱花庭院", "ruin": "废教堂", "night": "霓虹夜街"}
POSE_LABEL = {"off": "关（随机姿势）", "auto": "自动（随机骨架）",
              "reach_up": "单臂高举", "jump": "跳跃腾空",
              "stand": "站立", "sit": "坐姿"}


def load():
    import make as MK
    return MK.load_toggles()


def save(cfg):
    import make as MK
    MK.save_toggles(cfg)


def refs():
    if not os.path.isdir(REF_DIR):
        return []
    fs = [f for f in os.listdir(REF_DIR) if f.lower().endswith(".png")]
    return sorted(fs)


def packs():
    """列出扒过图、留下 _pack.json 的目录。"""
    if not os.path.isdir(REF_ROOT):
        return []
    out = []
    for d in sorted(os.listdir(REF_ROOT)):
        if os.path.exists(os.path.join(REF_ROOT, d, "_pack.json")):
            out.append("refs/" + d)
    return out


def cycle(lst, cur):
    if cur not in lst:
        return lst[0]
    return lst[(lst.index(cur) + 1) % len(lst)]


def show(cfg):
    ref = cfg.get("ref") or "-"
    print()
    print("=" * 62)
    print("  出图开关")
    print("=" * 62)
    pk_list = cfg.get("pack") or []
    pk = pk_list[0] if pk_list else ""
    print("  1  骨架        %s%s"
          % (POSE_LABEL.get(cfg.get("pose"), cfg.get("pose")),
             "（被动作包覆盖）" if pk else ""))
    print("  2  角色锁      %s   [%s]" % ("开" if cfg.get("char") else "关", ref))
    print("  3  4x 超分     %s" % ("开（+20 秒）" if cfg.get("upscale") else "关"))
    print("  4  换场景      %s" % SCENE_LABEL.get(cfg.get("scene"), cfg.get("scene")))
    print("  5  底模        %s" % cfg.get("profile"))
    print("  6  尺寸        %s" % cfg.get("size"))
    ex = cfg.get("prompt_extra") or "-"
    if len(ex) > 46:
        ex = ex[:46] + "..."
    print("  7  追加提示词  %s" % ex)
    print("  8  动作包      %s" % (pk or "关"))
    print("  9  中文解析    %s" % ("开（中文自动转 tag）"
                                  if cfg.get("nl_tags", True) else "关"))
    kao = cfg.get("kaomoji") or ""
    print("  k  颜文字      %s" % (kao or "无"))
    print("  c  换参考图（共 %d 张可用）" % len(refs()))
    print("-" * 62)
    print("  输入编号翻开关    k = 写颜文字    r = 立刻出图    q = 退出")
    print()


def main():
    cfg = load()
    while True:
        show(cfg)
        try:
            c = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return 0

        if c == "q":
            save(cfg)
            print("已保存，退出。")
            return 0

        if c == "1":
            cfg["pose"] = cycle(POSE_CYCLE, cfg.get("pose"))
        elif c == "2":
            cfg["char"] = not cfg.get("char")
        elif c == "3":
            cfg["upscale"] = not cfg.get("upscale")
        elif c == "4":
            cfg["scene"] = cycle(SCENE_CYCLE, cfg.get("scene"))
        elif c == "5":
            cfg["profile"] = cycle(PROFILE_CYCLE, cfg.get("profile"))
        elif c == "6":
            cfg["size"] = cycle(SIZE_CYCLE, cfg.get("size"))
        elif c == "7":
            print("当前：%s" % (cfg.get("prompt_extra") or "（空）"))
            v = input("新的追加提示词（留空清掉）: ").strip()
            cfg["prompt_extra"] = v
        elif c == "8":
            ps = [""] + packs()
            cur = (cfg.get("pack") or [""])[0]
            nxt = cycle(ps, cur)
            cfg["pack"] = [nxt] if nxt else []
            if nxt:
                import multi_ref as MR
                try:
                    info = MR.load_pack(nxt)
                    print("  已选 %s（%d 张，动作：%s）"
                          % (nxt, len(info.get("items") or []), info.get("query")))
                except Exception as e:
                    print("  读不了这个包：%s" % e)
        elif c == "9":
            cfg["nl_tags"] = not cfg.get("nl_tags", True)
            print("  中文解析%s。关掉之后 ref_search 只认英文 tag。"
                  % ("已开" if cfg["nl_tags"] else "已关"))
        elif c == "k":
            print("当前：%s" % (cfg.get("kaomoji") or "（空）"))
            v = input("颜文字（留空清掉，? 看支持哪些）: ").strip()
            if v == "?":
                import kaomoji as KJ
                for i, kk in enumerate(sorted(KJ.KAOMOJI_MAP), 1):
                    print("  %-16s %s" % (kk, ", ".join(KJ.KAOMOJI_MAP[kk][:3])))
                    if i % 6 == 0:
                        print()
                continue
            cfg["kaomoji"] = v
            if v:
                import kaomoji as KJ
                r = KJ.parse(v)
                print("  已设 %s -> %s" % (v, ", ".join(r["tags"]) or "（表里没有，出图时会忽略）"))
            else:
                print("  已清掉")
        elif c == "c":
            rs = refs()
            if not rs:
                print("refs/char 里没有图，先跑 prep_refs.py")
                continue
            for i, f in enumerate(rs[:20], 1):
                print("  %2d  %s" % (i, f))
            v = input("选编号（回车取消，也可直接粘贴文件名/路径）: ").strip()
            if not v:
                continue
            if v.isdigit():
                i = int(v) - 1
                if 0 <= i < min(20, len(rs)):
                    cfg["ref"] = rs[i].replace(".png", "")
            else:
                cfg["ref"] = v
        elif c == "r":
            save(cfg)
            import make as MK
            print("\n开始出图…（不想等就按 Ctrl+C，已提交的任务会继续跑完）")
            try:
                MK.run(cfg)
            except KeyboardInterrupt:
                print("\n已中断")
            except SystemExit as e:
                print("失败：%s" % e)
            except Exception as e:
                print("出错：%s" % e)
            input("\n回车回到开关面板")
        else:
            print("看不懂：%s" % c)


if __name__ == "__main__":
    sys.exit(main())
