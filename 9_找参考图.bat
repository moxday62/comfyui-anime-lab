@echo off
title ÕÒ²Î¿¼Í¼
cd /d "D://AI-Video-Setup"
"D://ComfyUI//ComfyUI_windows_portable//python_embeded//python.exe" ref_search.py --ui
if errorlevel 1 pause
