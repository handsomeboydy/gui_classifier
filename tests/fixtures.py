"""脱敏测试夹具：虚构线路、杆塔坐标、台账与带 EXIF 的合成照片。

所有坐标均为虚构值，位于 23°30'N / 113°15'E 附近；不包含任何真实线路、
真实坐标或人员信息。测试输出目录全部使用临时目录。
"""

import os
from fractions import Fraction as F

import pandas as pd
from PIL import Image


LINE_NAME = "测试线路A"

# 虚构杆塔坐标：塔间距约 150 米（0.0015° 经度）
TOWERS = {
    1: (23.5, 113.2500),
    2: (23.5, 113.2515),
    3: (23.5, 113.2530),
    4: (23.5, 113.2545),
    5: (23.5, 113.2560),
}


def dms(value):
    """十进制坐标转 DMS 有理数，供 Pillow EXIF 写入。"""
    v = abs(float(value))
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60.0) * 3600.0
    den = 1000
    s_num = int(round(s * den))
    return (F(d, 1), F(m, 1), F(s_num, den))


def make_photo(path, lat, lon, dt=None, color=(100, 100, 100), size=(64, 64)):
    """生成带 GPS（可选拍摄时间）的合成 JPEG。"""
    exif = Image.Exif()
    if dt:
        exif.get_ifd(0x8769)[0x9003] = dt
        exif[0x0132] = dt
    gps = exif.get_ifd(0x8825)
    gps[0] = b"\x02\x03\x00\x00"
    gps[1] = "N"
    gps[2] = dms(lat)
    gps[3] = "E"
    gps[4] = dms(lon)
    Image.new("RGB", size, color).save(path, format="JPEG", exif=exif)


def make_plain_photo(path, color=(200, 200, 200)):
    """生成无 EXIF 的 JPEG（模拟无 GPS 照片）。"""
    Image.new("RGB", (64, 64), color).save(path, format="JPEG")


def build_ledger(path, towers=None, include_line=True, missing=()):
    """生成虚构线路经纬度台账 xlsx。"""
    if towers is None:
        towers = list(range(1, 6))
    rows = []
    for i in towers:
        lat, lon = TOWERS[int(i)]
        row = {"杆塔编号": "%03d" % int(i), "经度": lon, "纬度": lat}
        if include_line:
            row["线路名称"] = LINE_NAME
        for col in missing:
            row.pop(col, None)
        rows.append(row)
    pd.DataFrame(rows).to_excel(path, index=False)


def build_double_tower(path, side_str="1-5:左"):
    """生成虚构双回塔台账 xlsx（与主台账同目录）。"""
    pd.DataFrame([{
        "杆塔1名称": LINE_NAME,
        "杆塔2名称": LINE_NAME,
        "杆塔1方位": side_str,
        "杆塔2方位": side_str,
    }]).to_excel(path, index=False)


def near(tower, lat_off=0.0, lon_off=0.0):
    """返回指定杆塔附近的虚构坐标（北为正、东为正）。"""
    lat, lon = TOWERS[tower]
    return lat + lat_off, lon + lon_off


def collect_files(root):
    """收集目录下所有文件的相对路径（统一使用 / 分隔）。"""
    result = set()
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = os.path.join(dirpath, name)
            result.add(os.path.relpath(full, root).replace(os.sep, "/"))
    return result
