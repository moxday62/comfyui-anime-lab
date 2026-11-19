@echo off
title Frames to Sprite Sheet
cd /d "D:\AI-Video-Setup"
set PY="D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"
%PY% make_sprite.py --input "D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\anime_sprite" --name hero_run --cols 4 --fps 8 --pingpong
echo.
echo Done. Output folder: D:\AI-Video-Setup\output
pause
