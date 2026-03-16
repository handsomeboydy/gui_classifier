# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui_classifier.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['extension', 'button1', 'button2', 'button3', 'openpyxl', 'exifread'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='御3T分图工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['favicon2.ico'],
)
