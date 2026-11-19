# -*- coding: utf-8 -*-
"""临时诊断：检查 FLUX 文本编码器是否真的把不同提示词编码成不同向量。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMFY

# ComfyUI 自己的包（folder_paths / comfy）要单独加进 sys.path
sys.path.insert(0, COMFY)
import torch
import folder_paths
import comfy.sd

p_clip = folder_paths.get_full_path_or_raise("text_encoders", "clip_l.safetensors")
p_t5 = folder_paths.get_full_path_or_raise("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
print("CLIP", p_clip)
print("T5", p_t5)

clip = comfy.sd.load_clip(
    ckpt_paths=[p_clip, p_t5],
    embedding_directory=folder_paths.get_folder_paths("embeddings")[0],
    clip_type=comfy.sd.CLIPType.FLUX,
)


def encode(text, verbose=False):
    tokens = clip.tokenize(text)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if verbose:
        print("COND_KEYS", sorted(cond[0][1].keys()))
        for k, v in cond[0][1].items():
            try:
                print(" ", k, "shape", tuple(v.shape), "norm", round(v.float().norm().item(), 4))
            except AttributeError:
                print(" ", k, repr(v)[:200])
    return cond[0][0].detach().float().cpu(), {k: len(v) for k, v in tokens.items()}


a, ta = encode("a single bright red cube on a blue background", verbose=True)
b, tb = encode("a green sphere on a yellow background")
c, tc = encode("")

print("TOKEN_LENS", ta, tb, tc)
for name, x in (("red_cube", a), ("green_sphere", b), ("empty", c)):
    print(name, "shape", tuple(x.shape), "norm", round(x.norm().item(), 4),
          "mean", round(x.mean().item(), 6), "std", round(x.std().item(), 6))

d1 = (a - b).abs()
d2 = (a - c).abs()
print("DIFF red_cube vs green_sphere: mean", round(d1.mean().item(), 6), "max", round(d1.max().item(), 6))
print("DIFF red_cube vs empty:        mean", round(d2.mean().item(), 6), "max", round(d2.max().item(), 6))
