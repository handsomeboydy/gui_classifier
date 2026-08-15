# updater_github.py
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
import subprocess
from typing import Optional, Tuple

# ====== 发布配置（与构建产物保持一致）======
REPO = "handsomeboydy/gui_classifier"
ASSET_NAME_PATTERN = r"^御3T分图工具\.exe$"  # 与构建产物名保持一致（D-08）
SHA256_SUFFIX = ".sha256"                    # 校验文件资产后缀（E2）
CHECK_INTERVAL_SEC = 6 * 60 * 60               # 自动检查间隔 6 小时（E4）
# ==========================================

def _api_latest_release_url() -> str:
    return f"https://api.github.com/repos/{REPO}/releases/latest"

def _user_agent_headers() -> dict:
    return {"User-Agent": "gui-classifier-updater"}

def _strip_v(tag: str) -> str:
    return tag[1:] if tag.startswith(("v", "V")) else tag

def _parse_ver(v: str):
    """
    简易语义版本比较：1.2.10 > 1.2.3
    非数字段会被忽略。
    """
    v = _strip_v(v).strip()
    parts = re.split(r"[^\d]+", v)
    nums = [int(p) for p in parts if p.isdigit()]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])

def get_current_exe_path() -> str:
    return os.path.abspath(sys.executable if getattr(sys, "frozen", False) else sys.argv[0])

def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_remove(path: str):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass

def fetch_latest_release(timeout=8) -> dict:
    req = urllib.request.Request(_api_latest_release_url(), headers=_user_agent_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def select_asset(release_json: dict) -> dict:
    assets = release_json.get("assets", [])
    pat = re.compile(ASSET_NAME_PATTERN, re.IGNORECASE)
    for a in assets:
        name = a.get("name", "")
        if pat.search(name):
            return a
    raise RuntimeError("未在 Release assets 中找到匹配的 exe（请检查 ASSET_NAME_PATTERN 或 Release 资产名）。")

def find_digest_asset(release_json: dict, asset_name: str) -> Optional[dict]:
    expected = asset_name + SHA256_SUFFIX
    for a in release_json.get("assets", []):
        if (a.get("name") or "").lower() == expected.lower():
            return a
    return None

def _extract_sha256(text: str) -> Optional[str]:
    m = re.search(r"[0-9a-fA-F]{64}", text)
    return m.group(0).lower() if m else None

def _fetch_expected_sha256(release_json: dict, asset: dict) -> str:
    digest_asset = find_digest_asset(release_json, asset.get("name", ""))
    if not digest_asset:
        raise RuntimeError("Release 缺少 SHA-256 校验文件（" + (asset.get("name", "") + SHA256_SUFFIX) + "），已取消更新。")
    url = digest_asset.get("browser_download_url")
    if not url:
        raise RuntimeError("SHA-256 校验文件缺少下载地址。")
    tmp = os.path.join(tempfile.gettempdir(), "gui_classifier_digest.tmp")
    download_file(url, tmp)
    try:
        with open(tmp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    finally:
        _safe_remove(tmp)
    expected = _extract_sha256(text)
    if not expected:
        raise RuntimeError("SHA-256 校验文件内容无效。")
    return expected

def has_update(current_version: str, latest_tag: str) -> bool:
    return _parse_ver(latest_tag) > _parse_ver(current_version)

def _cache_path() -> str:
    # 启动自动检查用：避免频繁打 GitHub API（防止触发 rate limit）
    return os.path.join(tempfile.gettempdir(), "gui_classifier_update_cache.json")

def should_check_now(min_interval_sec: Optional[int] = None) -> bool:
    """默认 6 小时内只查一次（E4）。"""
    interval = CHECK_INTERVAL_SEC if min_interval_sec is None else min_interval_sec
    p = _cache_path()
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
        last = float(data.get("last_check", 0))
        return (time.time() - last) > interval
    except Exception:
        return True

def mark_checked():
    p = _cache_path()
    try:
        json.dump({"last_check": time.time()}, open(p, "w", encoding="utf-8"))
    except Exception:
        pass

def download_file(url: str, dest: str):
    # GitHub docs 推荐用 browser_download_url 下载 release asset
    req = urllib.request.Request(url, headers=_user_agent_headers())
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            f.write(b)

def create_apply_update_bat(app_path: str, new_path: str) -> str:
    bat_path = os.path.join(tempfile.gettempdir(), "apply_gui_classifier_update.bat")
    # 注意用双引号包裹，兼容中文/空格/括号路径
    bat = fr"""@echo off
setlocal
set "APP={app_path}"
set "NEW={new_path}"
set "BAK={app_path}.bak"

REM 等待主程序退出释放文件锁
ping 127.0.0.1 -n 3 >nul

if not exist "%NEW%" (
  echo [update] new file missing
  exit /b 2
)

REM 备份旧版本
if exist "%APP%" (
  copy /Y "%APP%" "%BAK%" >nul
  if errorlevel 1 (
    echo [update] backup failed
    exit /b 3
  )
)

REM 覆盖新版本；失败时恢复备份
copy /Y "%NEW%" "%APP%" >nul
if errorlevel 1 (
  if exist "%BAK%" copy /Y "%BAK%" "%APP%" >nul
  echo [update] replace failed, backup restored
  exit /b 4
)

if not exist "%APP%" (
  if exist "%BAK%" copy /Y "%BAK%" "%APP%" >nul
  echo [update] app missing after replace, backup restored
  exit /b 5
)

REM 启动新版本并清理
start "" "%APP%"
del "%NEW%" >nul
del "%~f0" >nul
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat)
    return bat_path

def prepare_update(current_version: str, verify_sha256: Optional[str] = None, require_verify: bool = True) -> Tuple[str, str, str]:
    """
    返回：(latest_tag, notes, tmp_exe_path)
    校验失败时删除下载文件并抛错（E2/E3）。
    """
    release = fetch_latest_release()
    latest_tag = release.get("tag_name", "").strip()
    notes = release.get("body", "") or ""
    if not latest_tag:
        raise RuntimeError("Release 缺少 tag_name。")

    if not has_update(current_version, latest_tag):
        return latest_tag, notes, ""

    asset = select_asset(release)
    url = asset.get("browser_download_url")
    if not url:
        raise RuntimeError("Release asset 缺少 browser_download_url。")

    tmp_exe = os.path.join(tempfile.gettempdir(), f"gui_classifier_{_strip_v(latest_tag)}.exe")
    try:
        download_file(url, tmp_exe)
        expected = verify_sha256
        if not expected:
            if not require_verify:
                return latest_tag, notes, tmp_exe
            expected = _fetch_expected_sha256(release, asset)
        actual = sha256sum(tmp_exe).lower()
        if actual != expected.lower().strip():
            raise RuntimeError("更新包 SHA-256 校验失败，已取消更新并删除下载文件。")
    except Exception:
        _safe_remove(tmp_exe)
        raise
    return latest_tag, notes, tmp_exe

def apply_update_and_restart(tmp_exe: str):
    if not os.path.isfile(tmp_exe) or os.path.getsize(tmp_exe) <= 0:
        raise RuntimeError("更新文件无效，已停止更新。")
    app_path = get_current_exe_path()
    bat = create_apply_update_bat(app_path, tmp_exe)
    subprocess.Popen(["cmd", "/c", bat], creationflags=subprocess.CREATE_NEW_CONSOLE)

