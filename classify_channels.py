import os
import glob
import shutil
import pandas as pd
import exifread
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import re

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点（十进制度）间的大圆距离（米）
    """
    R = 6371000
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ, Δλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1) * cos(φ2) * sin(Δλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_exif(img_path):
    """
    从照片 EXIF 中提取 GPS 和拍摄时间；找不到经纬度则返回 (None, None, None)
    """
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        def _deg(v):
            d, m, s = [float(x.num) / float(x.den) for x in v.values]
            return d + m/60 + s/3600

        lat = _deg(tags['GPS GPSLatitude'])
        if tags['GPS GPSLatitudeRef'].printable != 'N':
            lat = -lat
        lon = _deg(tags['GPS GPSLongitude'])
        if tags['GPS GPSLongitudeRef'].printable != 'E':
            lon = -lon
    except KeyError:
        return None, None, None

    # 提取拍摄时间
    tstr = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    ts = None
    if tstr:
        try:
            ts = datetime.strptime(tstr.printable, '%Y:%m:%d %H:%M:%S')
        except Exception:
            pass
    return lat, lon, ts

def load_ledger(path, line_name):
    """
    读取台账Excel，并根据线路名称过滤，返回包含杆塔编号与坐标的DataFrame
    """
    df = pd.read_excel(path, dtype={"杆塔编号": str})
    df = df[df['线路名称'] == line_name].dropna(subset=['杆塔编号'])
    df['杆塔编号'] = df['杆塔编号'].astype(str)
    return df[["杆塔编号", "经度", "纬度"]].rename(columns={"杆塔编号": "tower"})

def classify_channels(ledger_file, src_folder, output_root, line_name, threshold):
    """
    仅提取“通道照片”，并在遇到台账错误时跳过相关塔号的精细化和通道照片
    """
    # —— 第一步：删除文件名含 "_T" 但不含 "_T_" 的红外照片违例文件 —— #
    for p in glob.glob(os.path.join(src_folder, '*.*')):
        fn = os.path.basename(p)
        if '_T' in fn and '_T_' not in fn:
            os.remove(p)

    # 读取台账，并获取所有合法塔号
    ledger = load_ledger(ledger_file, line_name)
    towers = ledger['tower'].tolist()

    # 收集所有照片的 metadata（路径、GPS 和拍摄时间）
    imgs = []
    for p in glob.glob(os.path.join(src_folder, '*.*')):
        lat, lon, ts = get_exif(p)
        imgs.append({
            'path': p,
            'lat': lat,
            'lon': lon,
            'time': ts,
            'tower': None,
            'cat': None
        })

    # 用于记录每个塔的“可见光精细化”照片的时间窗
    tower_times = {t: {'min': None, 'max': None} for t in towers}

    # —— 第一轮：识别“精细化”和“红外照片”，记录时间窗，并跳过台账错误塔号 —— #
    candidates = []  # 距离阈值之外或未判定的图片，供第二轮使用
    skip_mode = False
    skipped_towers = set()

    for img in imgs:
        name = os.path.basename(img['path'])
        if img['lat'] is None:
            continue

        # 计算当前照片到所有台账点的距离
        ledger['dist'] = ledger.apply(
            lambda r: haversine(img['lat'], img['lon'], r['纬度'], r['经度']), axis=1
        )
        min_dist = ledger['dist'].min()

        # 如果处于跳过模式，持续跳过，直到遇到可匹配的精细化照片
        if skip_mode:
            if min_dist <= threshold:
                skip_mode = False
            else:
                continue

        # 如果最小距离超过阈值，判断是否进入跳过模式
        if min_dist > threshold:
            # 如果是自主飞行的精细化或红外照片，进入跳过模式
            if '_V_' in name or '_T_' in name:
                skip_mode = True
                # 尝试从文件名中提取塔号
                m = re.match(r'(\d+)', name)
                if m:
                    skipped_towers.add(m.group(1))
                continue
            # 否则，延后第二轮处理
            candidates.append(img)
            continue

        # 找到最近的塔
        row = ledger.loc[ledger['dist'].idxmin()]
        img['tower'] = row['tower']

        # 根据文件名简单判别类别：“_V_”代表可见光精细化，“_T_”代表红外照片
        if '_V_' in name:
            img['cat'] = '精细化'
            if img['time']:
                tt = tower_times[img['tower']]
                if tt['min'] is None or img['time'] < tt['min']:
                    tt['min'] = img['time']
                if tt['max'] is None or img['time'] > tt['max']:
                    tt['max'] = img['time']
        elif '_T_' in name:
            img['cat'] = '红外'
        else:
            # 不符合上述两类，作为候选在第二轮决定是否为“通道照片”
            candidates.append(img)

    # 输出被跳过的杆塔号日志
    if skipped_towers:
        try:
            sorted_ids = sorted(map(int, skipped_towers))
            sorted_ids = list(map(str, sorted_ids))
        except ValueError:
            sorted_ids = list(skipped_towers)
        print(f"[跳过杆塔] {'，'.join(sorted_ids)}，可能存在坐标错误")

    # —— 构建“通道”时间窗 —— #
    seq = sorted(towers, key=lambda x: int(x))
    windows = []

    # 1) 正序（升号）航线：塔 i 到 i+1 之间的时间窗
    for i in range(len(seq) - 1):
        t_low, t_high = seq[i], seq[i+1]
        end_low = tower_times[t_low]['max']
        start_high = tower_times[t_high]['min']
        if end_low and start_high and end_low < start_high:
            windows.append((t_low, end_low, start_high))

    # 2) 逆序（降号）航线：塔 i+1 到 i 之间的时间窗
    for i in range(len(seq) - 1):
        t_low, t_high = seq[i], seq[i+1]
        end_high = tower_times[t_high]['max']
        start_low = tower_times[t_low]['min']
        if end_high and start_low and end_high < start_low:
            windows.append((t_high, end_high, start_low))

    # 3) 考虑最后一个采集精细化后，没有后续精细化的通道照片
    towers_with_windows = {w[0] for w in windows}
    for t in seq:
        t_end = tower_times[t]['max']
        if t_end and t not in towers_with_windows:
            windows.append((t, t_end, None))

    # —— 第二轮：从候选图片中提取“通道照片” —— #
    for img in candidates:
        if img['time'] is None:
            continue
        for t, t_start, t_end in windows:
            # 如果窗口有结束边界
            if t_end is not None:
                if t_start < img['time'] < t_end:
                    if '_T' in os.path.basename(img['path']):
                        os.remove(img['path'])
                        break
                    img['tower'] = t
                    img['cat'] = '通道'
                    break
            else:
                if img['time'] > t_start:
                    if '_T' in os.path.basename(img['path']):
                        os.remove(img['path'])
                        break
                    img['tower'] = t
                    img['cat'] = '通道'
                    break

    # —— 最终只输出“通道照片”到磁盘 —— #
    base = os.path.join(output_root, line_name, '通道')
    os.makedirs(base, exist_ok=True)

    for img in imgs:
        if img.get('cat') == '通道' and img.get('tower') and os.path.isfile(img['path']):
            dst = os.path.join(base, img['tower'])
            os.makedirs(dst, exist_ok=True)
            shutil.copy(img['path'], dst)
            print(f"[通道] {img['path']} → {dst}")

    # 统计输出结果
    records = [img for img in imgs + candidates if img.get('cat') == '通道']
    if not records:
        print("⚠️ 未找到任何“通道照片”，请检查 src_folder 路径或拍摄时间是否满足时间窗条件。")
    else:
        df = pd.DataFrame(records)
        print("“通道照片”统计：")
        print(df['tower'].value_counts())
    print('“通道照片”提取完成。')
