@echo off
title Anime Video - ComfyUI
cd /d "D:\ComfyUI\ComfyUI_windows_portable"
echo Starting ComfyUI...
echo Open http://127.0.0.1:8188 when ready
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 12; Start-Process 'http://127.0.0.1:8188'"
".\python_embeded\python.exe" -s ComfyUI\main.py --windows-standalone-build
pause
