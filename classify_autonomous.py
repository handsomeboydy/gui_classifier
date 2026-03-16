import os
import glob
import shutil
import pandas as pd
import exifread
from math import radians, sin, cos, sqrt, atan2
from side_parser import get_expected_side_for_tower

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

def get_image_gps(img_path):
    """从照片 EXIF 中提取 GPS，经纬度十进制度；找不到则返回 (None,None)"""
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        d, m, s = [float(x.num)/float(x.den) for x in tags['GPS GPSLatitude'].values]
        lat = d + m/60 + s/3600
        if tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
        d, m, s = [float(x.num)/float(x.den) for x in tags['GPS GPSLongitude'].values]
        lon = d + m/60 + s/3600
        if tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
        return lat, lon
    except Exception:
        return None, None

def load_ledger(path):
    df = pd.read_excel(path, dtype={'杆塔编号': str})
    df = df.rename(columns={'杆塔编号':'tower','经度':'经度','纬度':'纬度'})
    return df[['tower','经度','纬度']]

def load_double_tower_side(ledger_file, line_name):
    import glob
    dirpath = os.path.dirname(ledger_file)
    files = glob.glob(os.path.join(dirpath,'*双回塔台账文件.xlsx'))
    if not files: return None
    df = pd.read_excel(files[0], dtype=str)
    mask1 = df['杆塔1名称']==line_name
    mask2 = df['杆塔2名称']==line_name
    if mask1.any(): return df.loc[mask1,'杆塔1方位'].iat[0]
    if mask2.any(): return df.loc[mask2,'杆塔2方位'].iat[0]
    return None

def determine_side(curr, adj, img):
    vx, vy = adj[1]-curr[1], adj[0]-curr[0]
    wx, wy = img[1]-curr[1], img[0]-curr[0]
    cross = vx*wy - vy*wx
    if cross>0: return '左'
    if cross<0: return '右'
    return None

def classify_autonomous(ledger_file, src_folder, output_root, line_name, threshold):
    # 加载 skip_ir 列表
    skip_path = os.path.join(output_root, line_name, 'skip_ir.txt')
    if os.path.isfile(skip_path):
        with open(skip_path) as f:
            skip_ir = {l.strip() for l in f if l.strip()}
    else:
        skip_ir = set()
    # 加载台账与侧别配置
    ledger = load_ledger(ledger_file)
    side_str = load_double_tower_side(ledger_file, line_name)
    # 准备输出目录
    fine_base = os.path.join(output_root, line_name, '精细化')
    ir_base   = os.path.join(output_root, line_name, '红外照片')
    os.makedirs(fine_base, exist_ok=True)
    os.makedirs(ir_base, exist_ok=True)
    # 排序塔序
    df_sorted = ledger.copy()
    df_sorted['idx']=df_sorted['tower'].astype(int)
    df_sorted.sort_values('idx', inplace=True)
    seq = df_sorted['tower'].tolist()
    # 遍历分类
    for img_path in glob.glob(os.path.join(src_folder,'*.*')):
        fn = os.path.basename(img_path)
        # 跳过标记 IR
        if fn.endswith('.jpg') and '_T_' in fn and fn in skip_ir:
            continue
        lat, lon = get_image_gps(img_path)
        if lat is None:
            print(f"[跳过无GPS] {fn}")
            continue
        ledger['dist'] = ledger.apply(lambda r: haversine(lat,lon,r['纬度'],r['经度']),axis=1)
        row = ledger.loc[ledger['dist'].idxmin()]
        dist = ledger['dist'].min()
        tower = row['tower']
        # 距离内才分类
        if dist>threshold:
            continue
        # 侧别过滤
        expected = get_expected_side_for_tower(side_str,int(tower))
        if expected in ('左','右'):
            idx = seq.index(tower)
            adj = seq[idx+1] if idx<len(seq)-1 else seq[idx-1]
            curr = df_sorted[df_sorted['tower']==tower][['纬度','经度']].iloc[0]
            other = df_sorted[df_sorted['tower']==adj][['纬度','经度']].iloc[0]
            actual = determine_side((curr['纬度'],curr['经度']),(other['纬度'],other['经度']), (lat,lon))
            if actual!=expected:
                continue
        # 分类并输出
        if '_T_' in fn:
            dst = os.path.join(ir_base, tower)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(img_path,dst)
            print(f"[红外] {fn} → {dst}")
        elif '_V_' in fn:
            dst = os.path.join(fine_base, tower)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(img_path,dst)
            print(f"[精细化] {fn} → {dst}")
    print('自主飞行 IR & 精细化 分类完成。')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger-file',required=True)
    parser.add_argument('--src-folder',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--line-name',required=True)
    parser.add_argument('--threshold',type=float,default=50.0)
    args = parser.parse_args()
    classify_autonomous(args.ledger_file, args.src_folder, args.output_root, args.line_name, args.threshold)
