# -*- coding: utf-8 -*-
"""端到端验证高质量工作流：默认跑 SDXL 立绘，可用参数切换。"""
import json
import sys
import time
import urllib.request

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hq_workflows_api import sdxl_portrait, sdxl_anim, flux_portrait, soft_painterly

BASE = "http://127.0.0.1:8188"


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=60))


def run(name, prompt, timeout=3600):
    print(f"\n=== 提交 {name} ===")
    res = post("/prompt", {"prompt": prompt, "client_id": "hq_verify"})
    if res.get("errors"):
        print("提交失败:", json.dumps(res["errors"], ensure_ascii=False)[:3000])
        return False
    pid = res["prompt_id"]
    print("任务ID:", pid, "（质量优先模式，可能要几分钟到十几分钟）")
    start = time.time()
    last_report = 0
    while True:
        try:
            hist = get(f"/history/{pid}")
        except Exception:
            hist = {}
        if hist.get(pid):
            item = hist[pid]
            status = item.get("status", {})
            if status.get("completed"):
                print(f"完成! 耗时 {time.time()-start:.1f}s")
                for nid, out in item.get("outputs", {}).items():
                    for img in out.get("images", []):
                        print(f"  输出: {img.get('filename')}")
                return True
            if status.get("status_str") == "error":
                print("失败:", json.dumps(status, ensure_ascii=False)[:3000])
                return False
        elapsed = time.time() - start
        if elapsed - last_report >= 60:
            print(f"  运行中... {elapsed/60:.1f} 分钟")
            last_report = elapsed
        if elapsed > timeout:
            print("超时")
            return False
        time.sleep(5)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sdxl_portrait"
    mapping = {
        "sdxl_portrait": lambda: sdxl_portrait(),
        "soft_painterly": lambda: soft_painterly(),
        "sdxl_anim": lambda: sdxl_anim(frames=8),
        "flux": lambda: flux_portrait(),
    }
    if which not in mapping:
        print("用法: python verify_hq.py [sdxl_portrait|soft_painterly|sdxl_anim|flux]")
        sys.exit(1)
    ok = run(which, mapping[which]())
    sys.exit(0 if ok else 1)
