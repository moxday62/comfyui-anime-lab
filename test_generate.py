"""端到端验证：提交一个最小动画生成任务并等待结果。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8188"


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


prompt = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "MeinaV7-pruned.safetensors"}},
    "2": {"class_type": "LoraLoader",
          "inputs": {"model": ["1", 0], "clip": ["1", 1],
                     "lora_name": "lcm-lora-sdv1-5.safetensors",
                     "strength_model": 1.0, "strength_clip": 1.0}},
    "3": {"class_type": "ADE_AnimateDiffLoaderGen1",
          "inputs": {"model": ["2", 0], "model_name": "v3_sd15_mm.ckpt",
                     "beta_schedule": "autoselect"}},
    "4": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 1],
                     "text": "masterpiece, best quality, anime style, 1girl, cute, smiling, waving hand, simple background"}},
    "5": {"class_type": "CLIPTextEncode",
          "inputs": {"clip": ["2", 1],
                     "text": "worst quality, low quality, blurry, bad anatomy, watermark, text"}},
    "6": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 256, "height": 256, "batch_size": 8}},
    "7": {"class_type": "KSampler",
          "inputs": {"model": ["3", 0], "seed": 12345, "steps": 8, "cfg": 1.0,
                     "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                     "positive": ["4", 0], "negative": ["5", 0],
                     "latent_image": ["6", 0]}},
    "8": {"class_type": "VAEDecode",
          "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
    "9": {"class_type": "SaveImage",
          "inputs": {"images": ["8", 0], "filename_prefix": "verify_test/frame"}},
}

print("提交验证任务...")
res = post("/prompt", {"prompt": prompt, "client_id": "verify_client"})
if "errors" in res and res["errors"]:
    print("提交失败:", json.dumps(res, ensure_ascii=False))
    raise SystemExit(1)
pid = res["prompt_id"]
print("任务ID:", pid)

start = time.time()
while True:
    hist = get(f"/history/{pid}")
    if hist.get(pid):
        item = hist[pid]
        status = item.get("status", {})
        if status.get("completed"):
            print(f"生成成功! 耗时 {time.time()-start:.1f}s")
            print("输出:", [o.get("filename") for o in item.get("outputs", {}).get("9", {}).get("images", [])][:3], "...")
            break
        if status.get("status_str") == "error":
            print("生成失败:", json.dumps(status, ensure_ascii=False)[:2000])
            break
    if time.time() - start > 300:
        print("超时")
        break
    time.sleep(3)
