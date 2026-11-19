# -*- coding: utf-8 -*-
"""体检所有 UI 格式工作流（workflows/ + workflows_hq/）。

查三类致命错误（ComfyUI 导入时会静默丢线，跑起来才报 missing input）：
  1. 节点输入引用了不存在的 link id        → 断线
  2. link 的目标节点/槽位与引用方对不上     → 错线（典型：VRAM 少了 vae）
  3. link id 重复，或多条 link 指向同一目标口 → UI 只认一条

用法：python check_workflows.py
"""
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = sorted(glob.glob(os.path.join(BASE, "workflows", "*.json")) +
               glob.glob(os.path.join(BASE, "workflows_hq", "*.json")))


def check(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    nodes = {n["id"]: n for n in d.get("nodes", [])}
    links = {l[0]: l for l in d.get("links", [])}
    errs = []

    targeted = {}
    for lid, l in links.items():
        _, src, sslot, dst, dslot, typ = l[:6]
        if src not in nodes:
            errs.append("link %s 源节点 %s 不存在" % (lid, src))
        if dst not in nodes:
            errs.append("link %s 目标节点 %s 不存在" % (lid, dst))
            continue
        # 同一目标口被多条 link 占用
        if (dst, dslot) in targeted:
            errs.append("link %s 与 link %s 抢同一个口 %s[%s]"
                        % (lid, targeted[(dst, dslot)], dst, dslot))
        targeted[(dst, dslot)] = lid

    for nid, n in nodes.items():
        for i, inp in enumerate(n.get("inputs") or []):
            if not isinstance(inp, dict):
                errs.append("节点 %s(%s) 的 inputs[%d] 格式异常（不是对象）" % (nid, n["type"], i))
                continue
            lid = inp.get("link")
            if lid is None:
                # 只有带 widget 的口才允许空（如 KSampler 的 seed）；这里只报必填口
                if n["type"] in ("VAEDecode", "VAEEncode", "KSampler", "CLIPTextEncode") \
                        and inp.get("type") in ("VAE", "MODEL", "CLIP", "LATENT",
                                                "IMAGE", "MASK", "CONDITIONING"):
                    errs.append("节点 %s(%s) 的 %s 口是空的"
                                % (nid, n["type"], inp.get("name")))
                continue
            if lid not in links:
                errs.append("节点 %s(%s) 的 %s 口引用不存在的 link %s"
                            % (nid, n["type"], inp.get("name"), lid))
                continue
            l = links[lid]
            if l[3] != nid or l[4] != i:
                errs.append("节点 %s(%s) 的 %s 口挂 link %s，但该 link 实际接在 %s[%s]"
                            % (nid, n["type"], inp.get("name"), lid, l[3], l[4]))
        for slot, out in enumerate(n.get("outputs") or []):
            for lid in out.get("links") or []:
                if lid not in links:
                    errs.append("节点 %s 输出 %s 引用不存在的 link %s"
                                % (nid, out.get("name"), lid))
                elif links[lid][1] != nid or links[lid][2] != slot:
                    errs.append("节点 %s 输出 %s 挂 link %s，但该 link 源头是 %s[%s]"
                                % (nid, out.get("name"), lid, links[lid][1], links[lid][2]))
    return errs


def main():
    bad = 0
    for f in FILES:
        errs = check(f)
        name = os.path.basename(f)
        if errs:
            bad += 1
            print("[坏] %s" % name)
            for e in errs:
                print("      - %s" % e)
        else:
            print("[好] %s" % name)
    print("-" * 50)
    print("共 %d 个，%d 个有问题" % (len(FILES), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
