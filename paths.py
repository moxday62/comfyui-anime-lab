# -*- coding: utf-8 -*-
"""路径解析：把写死的盘符路径收在一处。

原来十几个脚本各自写着 `r"D:\\AI-Video-Setup"` 和
`r"D:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI"`，仓库挪个位置就全废。
这里统一成：

  BASE    脚本所在目录（自动，不需要配）
  COMFY   ComfyUI 根目录，按优先级找：
            1  环境变量  COMFY_PATH
            2  和本仓库同级的 ComfyUI 便携版
            3  常见的便携版落点
            4  D:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI（作者环境，兜底）

用法：
    from paths import BASE, COMFY
    indir = os.path.join(COMFY, "input")

跨目录调用时这样设：
    Windows:  set COMFY_PATH=E:\\ComfyUI\\ComfyUI_windows_portable\\ComfyUI
    Linux:    COMFY_PATH=/opt/ComfyUI python make.py
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

_CANDIDATES = [
    os.environ.get("COMFY_PATH", ""),
    os.path.join(BASE, "ComfyUI_windows_portable", "ComfyUI"),
    os.path.join(os.path.dirname(BASE), "ComfyUI_windows_portable", "ComfyUI"),
    os.path.join(os.path.dirname(BASE), "ComfyUI"),
    r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI",
    os.path.expanduser("~/ComfyUI"),
    "/opt/ComfyUI",
]


def find_comfy():
    """返回第一个真实存在的 ComfyUI 目录；都不存在就返回兜底那个，
    让调用方在打开文件时自己报错，而不是在这里抛异常。"""
    for c in _CANDIDATES:
        if c and os.path.isdir(os.path.join(c, "models")):
            return c
    return _CANDIDATES[4]


COMFY = find_comfy()
