@echo off
chcp 65001 >nul
setlocal

REM 切换到脚本所在目录
cd /d "%~dp0"

title 打包 gui_classifier.exe

echo ========================================
echo 开始打包 gui_classifier.exe
echo 当前目录：%cd%
echo ========================================
echo.

REM 检查 python 是否可用
python --version >nul 2>nul
if errorlevel 1 (
    echo 未检测到 python 命令。
    echo 请先安装 Python，或在 Anaconda Prompt / 已激活的虚拟环境中运行本脚本。
    pause
    exit /b 1
)

REM 检查 PyInstaller 是否已安装
python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo 当前 Python 环境未安装 PyInstaller，正在尝试自动安装...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo.
        echo PyInstaller 安装失败。
        echo 请先手动执行：python -m pip install pyinstaller
        pause
        exit /b 1
    )
)

REM 可选：清理旧构建
if exist build (
    echo 正在删除旧的 build 目录...
    rmdir /s /q build
)
if exist dist (
    echo 正在删除旧的 dist 目录...
    rmdir /s /q dist
)
if exist gui_classifier.spec (
    echo 正在删除旧的 gui_classifier.spec ...
    del /f /q gui_classifier.spec
)

echo.
echo 正在执行 PyInstaller 打包...
python -m PyInstaller -F -w gui_classifier.py ^
  -n "御3T分图工具" ^
  --icon favicon2.ico ^
  --hidden-import extension ^
  --hidden-import button1 ^
  --hidden-import button2 ^
  --hidden-import button3 ^
  --hidden-import openpyxl ^
  --hidden-import exifread

if errorlevel 1 (
    echo.
    echo ========================================
    echo 打包失败，请检查上方报错信息
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo 打包完成！
echo 输出文件：
echo %cd%\dist\gui_classifier.exe
echo ========================================

if exist "%cd%\dist" (
    start "" "%cd%\dist"
)

pause