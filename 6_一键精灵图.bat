@echo off
title 一键精灵图
cd /d "D:\AI-Video-Setup"
set "PY=D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"

echo ================================================================
echo    一键精灵图
echo    16 帧  +  sprite sheet  +  GIF  +  MP4  +  TripoSG 单帧
echo ================================================================
echo.
"%PY%" oneshot_sprite.py %*

echo.
echo ----------------------------------------------------------------
echo 成品目录：D:\AI-Video-Setup\output\sprite_run
echo.
pause
