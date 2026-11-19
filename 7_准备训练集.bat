@echo off
title 准备 LoRA 训练集
cd /d "D:\AI-Video-Setup"
set "PY=D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"

echo ================================================================
echo    准备 LoRA 训练集
echo    去重 / 筛尺寸 / 统一 PNG / 整理成 kohya 目录结构
echo ================================================================
echo.
echo 原图请先全部复制进：D:\AI-Video-Setup\lora\inbox
echo.
"%PY%" lora\prepare_dataset.py %*

echo.
echo ----------------------------------------------------------------
echo 清洗后的图在 D:\AI-Video-Setup\lora\dataset\4_akistyle
echo 训练步骤看 D:\AI-Video-Setup\lora\README.txt
echo.
pause
