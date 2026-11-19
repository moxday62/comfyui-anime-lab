# -*- coding: utf-8 -*-
"""分层结构注入 —— 把画师的作画顺序映射到去噪的步数轴上。

关键认识：
  扩散模型没有"思考"，只有去噪。但它有一条隐式规律：
  早期步定大局（低频、构图、姿势），后期步补细节（高频、质感）。
  所以"先骨架、再草线、后质感"在工程上就等于：
  **给不同结构条件分配不同的生效步数区间**。

  这正是 ControlNet 的 start_percent / end_percent 干的事：
      骨架 OpenPose   0% ~ 45%   strength 1.0   先把姿势和比例钉死
      线稿 Scribble  15% ~ 70%   strength 0.80  再叠轮廓、衣褶、手指数量
      剩余 30% 步数   不加约束                    放开让模型补光影质感

  两类结构图分工不同，最容易搞错：
      骨架只管身体姿势和比例，**不含手指信息** —— 单靠骨架永远治不了六指
      线稿能精确表达手指数量 —— 想五根就画五根，模型照着描

用法：
  python struct_paint.py pose --pose output/gen_pose_v.png --size 832x1216
  python struct_paint.py hand --sketch output/gen_hand_v.png --size 768x768
  python struct_paint.py pose --pose p.png --sketch s.png     # 两层叠加
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style_lib as SL
import skeleton_gen as SG

CN_POSE = "controlnet_openpose_sdxl.safetensors"
CN_SKETCH = "controlnet_scribble_sdxl.safetensors"


def wf_struct(profile, pose_img=None, sketch_img=None, w=832, h=1216,
              steps=None, cfg=None, seed=None, prompt=None, prefix="struct/out",
              pose_strength=1.0, pose_start=0.0, pose_end=0.45,
              sketch_strength=0.8, sketch_start=0.15, sketch_end=0.70,
              init_img=None, denoise=1.0):
    """构建分层结构注入工作流。

    init_img 给了就走 img2img：保留原图的画风、肤色、光照、上下文，
    只让线稿去修正形状和手指数量。这是"既控结构又不丢画风"的关键。
    从白底凭空生成线稿图会崩成 3D 塑料手（实测）。
    """
    p = SL.load(profile)
    g = SL.Graph()
    model, clip, vae = SL._load_model(g, p)

    pos = g.node("CLIPTextEncode", clip=clip, text=prompt or p["positive"])
    neg = g.node("CLIPTextEncode", clip=clip, text=p.get("negative", ""))
    pos_ref, neg_ref = g.ref(pos, 0), g.ref(neg, 0)

    # 骨层：早期步数强约束姿势
    if pose_img:
        cn = g.node("ControlNetLoader", control_net_name=CN_POSE)
        im = g.node("LoadImage", image=pose_img)
        a = g.node("ControlNetApplyAdvanced", positive=pos_ref, negative=neg_ref,
                   control_net=g.ref(cn, 0), image=g.ref(im, 0),
                   strength=float(pose_strength),
                   start_percent=float(pose_start), end_percent=float(pose_end),
                   vae=vae)
        pos_ref, neg_ref = g.ref(a, 0), g.ref(a, 1)

    # 线稿层：中段约束轮廓与手指数量
    if sketch_img:
        cn = g.node("ControlNetLoader", control_net_name=CN_SKETCH)
        im = g.node("LoadImage", image=sketch_img)
        b = g.node("ControlNetApplyAdvanced", positive=pos_ref, negative=neg_ref,
                   control_net=g.ref(cn, 0), image=g.ref(im, 0),
                   strength=float(sketch_strength),
                   start_percent=float(sketch_start), end_percent=float(sketch_end),
                   vae=vae)
        pos_ref, neg_ref = g.ref(b, 0), g.ref(b, 1)

    if init_img:
        src = g.node("LoadImage", image=init_img)
        enc = g.node("VAEEncode", pixels=g.ref(src, 0), vae=vae)
        lat_ref = g.ref(enc, 0)
    else:
        lat = g.node("EmptyLatentImage", width=int(w), height=int(h), batch_size=1)
        lat_ref = g.ref(lat, 0)
    ks = g.node("KSampler", model=model,
                seed=int(seed if seed is not None else p.get("seed", 0)),
                steps=int(steps or p.get("steps", 28)),
                cfg=float(cfg or p.get("cfg", 5.0)),
                sampler_name=p.get("sampler", "euler"),
                scheduler=p.get("scheduler", "normal"),
                denoise=float(denoise),
                positive=pos_ref, negative=neg_ref, latent_image=lat_ref)
    dec = g.node("VAEDecode", samples=g.ref(ks, 0), vae=vae)
    g.node("SaveImage", images=g.ref(dec, 0), filename_prefix=prefix)
    return g.nodes


def main():
    ap = argparse.ArgumentParser(description="分层结构注入（骨架 + 线稿）")
    ap.add_argument("kind", choices=["pose", "hand"])
    ap.add_argument("--profile", default="soft")
    ap.add_argument("--pose", help="骨架图文件名（ComfyUI/input 下）")
    ap.add_argument("--sketch", help="线稿图文件名（ComfyUI/input 下）")
    ap.add_argument("--size", default="832x1216")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--prompt")
    ap.add_argument("--prefix", default="struct/out")
    ap.add_argument("--pose-end", type=float, default=0.45)
    ap.add_argument("--sketch-end", type=float, default=0.70)
    ap.add_argument("--sketch-strength", type=float, default=0.8)
    ap.add_argument("--init", help="底图文件名：给了就做 img2img，保留画风与上下文")
    ap.add_argument("--denoise", type=float, default=1.0)
    a = ap.parse_args()

    # 本地生成的图要先搬进 ComfyUI/input
    def prep(path):
        if not path:
            return None
        if os.path.isfile(path):
            name = os.path.basename(path)
            from PIL import Image
            Image.open(path).convert("RGB").save(os.path.join(SL.INPUT_DIR, name))
            return name
        return path

    w, h = [int(v) for v in a.size.lower().split("x")]
    wf = wf_struct(a.profile, pose_img=prep(a.pose), sketch_img=prep(a.sketch),
                   w=w, h=h, steps=a.steps, cfg=a.cfg, seed=a.seed,
                   prompt=a.prompt, prefix=a.prefix,
                   pose_end=a.pose_end, sketch_end=a.sketch_end,
                   sketch_strength=a.sketch_strength,
                   init_img=prep(a.init), denoise=a.denoise)
    before = SL.snapshot()
    if not SL.submit(wf, tag=a.kind):
        return 1
    for f in SL.new_files(before):
        print("输出：", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
