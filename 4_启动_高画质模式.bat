@echo off
title Anime Video - ComfyUI HIGH QUALITY
cd /d "D:\ComfyUI\ComfyUI_windows_portable"
echo ============================================
echo  HIGH QUALITY MODE - VRAM hungry, quality first
echo  Default normal VRAM: keeps as much resident as possible, offloads the rest
echo  Open http://127.0.0.1:8188 when ready
echo ============================================
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 15; Start-Process 'http://127.0.0.1:8188'"
".\python_embeded\python.exe" -s ComfyUI\main.py --windows-standalone-build
pause
