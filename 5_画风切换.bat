@echo off
setlocal enabledelayedexpansion
cd /d "D:\AI-Video-Setup"
set "PY=D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe"
title 画风权重库 - 切换

:menu
cls
echo ================================================================
echo    画风权重库   一份配置一个画风，切换即换档
echo ================================================================
"%PY%" style_lib.py list
echo.
echo ----------------------------------------------------------------
echo    输入编号切换画风      t = 自进化调参      c = 查模型    w = 查工作流    q = 退出
echo ----------------------------------------------------------------
set "SEL="
set /p "SEL=请输入: "
if "%SEL%"=="" goto menu
if /i "%SEL%"=="q" exit /b
if /i "%SEL%"=="t" goto tune
if /i "%SEL%"=="c" (
    "%PY%" style_lib.py check
    echo.
    pause
    goto menu
)
if /i "%SEL%"=="w" (
    "%PY%" check_workflows.py
    echo.
    pause
    goto menu
)

"%PY%" style_lib.py pick %SEL%
if errorlevel 1 (
    echo.
    pause
    goto menu
)

echo.
set "GO="
set /p "GO=现在就用这个画风出图？(y/N): "
if /i not "%GO%"=="y" goto menu

echo.
echo 提示词：直接输英文描述；留空 = 用画风档自带的提示词。
echo 　　　　想跟前一张不一样，就换提示词，或让下面那项随机种子挑 Y。
set "PR="
set /p "PR=正向提示词: "

set "SEEDARG=--random"
set "RS="
set /p "RS=随机种子（换个画面）？(Y/n): "
if /i "%RS%"=="n" set "SEEDARG="

echo.
echo 参考图：可以把图片直接拖进这个窗口，或粘贴完整路径；留空则不喂图。
set "REF="
set /p "REF=参考图: "
set "REF=%REF:"=%"
set "HANDARG="
set "HAND="
set /p "HAND=出图后自动修手？(y/N): "
if /i "%HAND%"=="y" set "HANDARG=--hand"

set "PARGS=%SEEDARG% %HANDARG%"
set "PROMPTARG="
if not "%PR%"=="" set PROMPTARG=--prompt "%PR%"

echo.
echo ================================================================
if "%REF%"=="" (
    "%PY%" style_lib.py run %PARGS% %PROMPTARG%
) else (
    "%PY%" style_lib.py run --ref "%REF%" %PARGS% %PROMPTARG%
)
echo ================================================================
echo 图片在 D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\styles\
echo.
pause
goto menu

:tune
cls
echo ================================================================
echo    自进化调参 —— 一张张出图 + 你的反馈，自动搜索最佳参数
echo ================================================================
"%PY%" style_lib.py list
echo.
set "TP="
set /p "TP=要调哪个画风的编号（回车=保持当前画风）: "
if not "%TP%"=="" "%PY%" style_lib.py pick %TP%
echo.
set "TN="
set /p "TN=跑几轮（回车=6，每轮约 25-50 秒）: "
if "%TN%"=="" set "TN=6"
echo.
echo   1 = 交互选图：每轮出 3 张，你选最好的（推荐，最准）
echo   2 = 自动评分：无人值守，但只是代理指标
set "TM="
set /p "TM=模式（回车=1）: "
if "%TM%"=="" set "TM=1"
echo.
echo ----------------------------------------------------------------
if "%TM%"=="2" (
    "%PY%" auto_tune.py --trials %TN% --auto
) else (
    "%PY%" auto_tune.py --trials %TN%
)
echo ----------------------------------------------------------------
echo.
set "TA="
set /p "TA=把最优参数写回画风档？(y/N): "
if /i "%TA%"=="y" "%PY%" auto_tune.py --apply
echo.
pause
goto menu
