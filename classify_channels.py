import os
import glob
import shutil
import pandas as pd
import exifread
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两点（十进制度）间的大圆距离（米）
    """
    R = 6371000
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ, Δλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_exif(img_path):
    """
    从照片 EXIF 中提取 GPS（十进制度）和拍摄时间；
    如果缺 GPS，则返回 (None, None, None)
    """
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        def _deg(v):
            d, m, s = [float(x.num)/float(x.den) for x in v.values]
            return d + m/60 + s/3600

        lat = _deg(tags['GPS GPSLatitude'])
        if tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
        lon = _deg(tags['GPS GPSLongitude'])
        if tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
    except KeyError:
        return None, None, None

    tstr = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    ts = None
    if tstr:
        try:
            ts = datetime.strptime(tstr.printable, '%Y:%m:%d %H:%M:%S')
        except:
            pass
    return lat, lon, ts

def load_ledger(path, line_name):
    """
    读取台账 Excel，根据线路名称过滤，返回包含 ['tower','经度','纬度'] 的 DataFrame
    """
    df = pd.read_excel(path, dtype={"杆塔编号": str})
    df = df[df['线路名称']==line_name].dropna(subset=['杆塔编号'])
    df = df.rename(columns={'杆塔编号':'tower','经度':'经度','纬度':'纬度'})
    df['tower'] = df['tower'].astype(str)
    return df[['tower','经度','纬度']]

def classify_channels(ledger_file, src_folder, output_root, line_name, threshold):
    """
    只输出“通道照片”，并生成 skip_ir 列表供其他脚本使用。
    """
    # 1. 加载台账与照片 metadata
    ledger = load_ledger(ledger_file, line_name)
    towers = ledger['tower'].tolist()

    imgs = []
    for p in glob.glob(os.path.join(src_folder, '*.*')):
        lat, lon, ts = get_exif(p)
        imgs.append({'path': p, 'lat': lat, 'lon': lon, 'time': ts, 'tower': None, 'cat': None})

    # 2. 第一轮：标记精细化(V) 与 初步红外(T)
    candidates = []
    tower_times = {t:{'min':None,'max':None} for t in towers}
    for img in imgs:
        name = os.path.basename(img['path'])
        if img['lat'] is None: continue
        ledger['dist'] = ledger.apply(lambda r: haversine(img['lat'], img['lon'], r['纬度'], r['经度']), axis=1)
        min_dist = ledger['dist'].min()
        if min_dist > threshold:
            candidates.append(img)
            continue
        row = ledger.loc[ledger['dist'].idxmin()]
        img['tower'] = row['tower']
        if '_V_' in name:
            img['cat'] = '精细化'
            if img['time']:
                t0 = tower_times[img['tower']]
                t0['min'] = img['time'] if t0['min'] is None or img['time'] < t0['min'] else t0['min']
                t0['max'] = img['time'] if t0['max'] is None or img['time'] > t0['max'] else t0['max']
        elif '_T_' in name:
            img['cat'] = '红外'
        else:
            candidates.append(img)

    # 3. 构建通道时间窗
    seq = sorted(towers, key=lambda x:int(x))
    windows = []
    for i in range(len(seq)-1):
        t1,t2 = seq[i],seq[i+1]
        end1 = tower_times[t1]['max']; start2 = tower_times[t2]['min']
        if end1 and start2 and end1 < start2: windows.append((t1,end1,start2))
    used = {w[0] for w in windows}
    for t in seq:
        if tower_times[t]['max'] and t not in used:
            windows.append((t, tower_times[t]['max'], None))

    # 4. 第二轮：提取通道照片
    for img in candidates:
        if img['time'] is None: continue
        for t,t_start,t_end in windows:
            in_win = (t_start < img['time'] < t_end) if t_end else (img['time'] > t_start)
            if in_win and '_T' not in os.path.basename(img['path']):
                img['tower'], img['cat'] = t, '通道'
                break

    # 5. 生成 skip_ir 列表并写入文件
    skip_ir = set()
    for img in imgs:
        if img.get('cat') == '通道':
            bn = os.path.basename(img['path'])
            if '_V_' in bn: skip_ir.add(bn.replace('_V_','_T_'))

    os.makedirs(os.path.join(output_root, line_name), exist_ok=True)
    skip_file = os.path.join(output_root, line_name, 'skip_ir.txt')
    with open(skip_file,'w') as f:
        for name in skip_ir: f.write(name + '\n')

    # 6. 仅输出通道照片
    base = os.path.join(output_root, line_name, '通道')
    os.makedirs(base, exist_ok=True)
    for img in candidates:
        if img.get('cat') == '通道' and img.get('tower'):
            dst = os.path.join(base, img['tower'])
            os.makedirs(dst, exist_ok=True)
            shutil.copy(img['path'], dst)
            print(f"[通道] {img['path']} → {dst}")

    print('classify_channels: 通道照片提取完成，skip_ir 写入', skip_file)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='classify_channels: 只输出通道并生成 skip_ir')
    parser.add_argument('--ledger-file', required=True)
    parser.add_argument('--src-folder',   required=True)
    parser.add_argument('--output-root',  required=True)
    parser.add_argument('--line-name',    required=True)
    parser.add_argument('--threshold',    type=float, default=50.0)
    args=parser.parse_args()
    classify_channels(args.ledger_file, args.src_folder, args.output_root, args.line_name, args.threshold)