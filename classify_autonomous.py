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
    R = 6371000  # 地球平均半径，米
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ, Δλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1) * cos(φ2) * sin(Δλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def get_image_gps(img_path):
    """从照片 EXIF 中提取 GPS，经纬度十进制度；找不到时返回 (None, None)"""
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        # 纬度
        d, m, s = [float(x.num) / float(x.den) for x in tags['GPS GPSLatitude'].values]
        lat = d + m/60 + s/3600
        if tags['GPS GPSLatitudeRef'].printable != 'N':
            lat = -lat
        # 经度
        d, m, s = [float(x.num) / float(x.den) for x in tags['GPS GPSLongitude'].values]
        lon = d + m/60 + s/3600
        if tags['GPS GPSLongitudeRef'].printable != 'E':
            lon = -lon
        return lat, lon
    except Exception:
        return None, None


def load_ledger(path):
    """读取台账Excel，返回包含杆塔编号与坐标的DataFrame"""
    df = pd.read_excel(path, dtype={'杆塔编号': str})
    return df[['杆塔编号', '经度', '纬度']]


def load_double_tower_side(ledger_file, line_name):
    """
    在与当前线路台账同一目录下，查找 '*双回塔台账文件.xlsx'（支持如 '1双回塔台账文件.xlsx'），
    如果存在且包含 line_name，则返回该行的原始侧别字符串；否则返回 None
    """
    import glob
    dirpath = os.path.dirname(ledger_file)
    candidates = glob.glob(os.path.join(dirpath, '*双回塔台账文件.xlsx'))
    if not candidates:
        return None
    double_path = candidates[0]
    df = pd.read_excel(double_path, dtype=str)
    mask1 = df['杆塔1名称'] == line_name
    mask2 = df['杆塔2名称'] == line_name
    if mask1.any():
        return df.loc[mask1, '杆塔1方位'].iloc[0]
    if mask2.any():
        return df.loc[mask2, '杆塔2方位'].iloc[0]
    return None


def determine_side(curr_coord, adj_coord, img_coord):
    """
    用二维叉乘判断 img_coord 相对于从 curr_coord 指向 adj_coord 的有向线段位置：
    返回 '左' / '右' / None
    """
    lon_curr, lat_curr = curr_coord[1], curr_coord[0]
    lon_adj, lat_adj = adj_coord[1], adj_coord[0]
    lon_img, lat_img = img_coord[1], img_coord[0]
    vx, vy = lon_adj - lon_curr, lat_adj - lat_curr
    wx, wy = lon_img - lon_curr, lat_img - lat_curr
    cross = vx * wy - vy * wx
    if cross > 0:
        return '左'
    if cross < 0:
        return '右'
    return None


def classify_autonomous(ledger_file, src_folder, output_root, line_name, threshold):
    """
    对自主飞行的照片进行精细化和红外分类，支持全线或部分同塔分段侧别配置
    且跳过已属于“通道”目录的照片，避免混入
    """
    # 1. 加载台账 & 侧别配置
    ledger = load_ledger(ledger_file)
    raw_side_str = load_double_tower_side(ledger_file, line_name)

    # 2. 准备输出目录（精细化、红外照片）
    fine_base = os.path.join(output_root, line_name, '精细化')
    ir_base   = os.path.join(output_root, line_name, '红外照片')
    os.makedirs(fine_base, exist_ok=True)
    os.makedirs(ir_base,   exist_ok=True)

    # 3. 收集“通道”目录中已有文件，跳过它们
    chan_base = os.path.join(output_root, line_name, '通道')
    chan_files = set()
    if os.path.isdir(chan_base):
        for root, _, files in os.walk(chan_base):
            for fname in files:
                chan_files.add(fname)

    # 4. 台账按杆塔编号排序
    ledger_sorted = ledger.copy()
    ledger_sorted['__num'] = ledger_sorted['杆塔编号'].astype(int)
    ledger_sorted = ledger_sorted.sort_values('__num').reset_index(drop=True)
    towers_seq = ledger_sorted['杆塔编号'].tolist()

    # 5. 遍历并分类照片，跳过通道照片
    for img_path in glob.glob(os.path.join(src_folder, '*.*')):
        fname = os.path.basename(img_path)
        if fname in chan_files:
            continue

        lat, lon = get_image_gps(img_path)
        if lat is None:
            print(f"[跳过，无 GPS] {img_path}")
            continue

        # 5.1 找到最近杆塔
        ledger['dist'] = ledger.apply(
            lambda r: haversine(lat, lon, r['纬度'], r['经度']), axis=1
        )
        closest = ledger.loc[ledger['dist'].idxmin()]
        if closest['dist'] > threshold:
            print(f"[未匹配] {img_path}，最小距离 {closest['dist']:.1f}m")
            continue

        tower = closest['杆塔编号']
        tower_int = int(tower)
        expected_side = get_expected_side_for_tower(raw_side_str, tower_int)

        # 5.2 侧别过滤（仅对“左”/“右”生效）
        if expected_side in ('左', '右'):
            try:
                idx = towers_seq.index(tower)
            except ValueError:
                print(f"[跳过，台账中找不到塔号] {img_path} → 塔 {tower}")
                continue
            adj = towers_seq[idx+1] if idx < len(towers_seq)-1 else towers_seq[idx-1]
            curr_row = ledger_sorted[ledger_sorted['杆塔编号'] == tower].iloc[0]
            adj_row  = ledger_sorted[ledger_sorted['杆塔编号'] == adj].iloc[0]
            actual_side = determine_side(
                (curr_row['纬度'], curr_row['经度']),
                (adj_row['纬度'], adj_row['经度']),
                (lat, lon)
            )
            if actual_side != expected_side:
                print(f"[跳过，侧别不符] {img_path} → 塔 {tower} (期望 {expected_side}, 实际 {actual_side})")
                continue

        # 5.3 分类并拷贝到精细化或红外目录
        category = '红外照片' if '_T' in fname else '精细化'
        dest = os.path.join(output_root, line_name, category, tower)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img_path, dest)
        print(f"[已分类] {img_path} → {dest}")

    print("自主飞行精细化 & 红外分类完成。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='自主飞行照片分类工具')
    parser.add_argument('--ledger-file', required=True)
    parser.add_argument('--src-folder',   required=True)
    parser.add_argument('--output-root',  required=True)
    parser.add_argument('--line-name',    required=True)
    parser.add_argument('--threshold',    type=float, default=50.0, help='距离阈值（米）')
    args = parser.parse_args()

    classify_autonomous(
        ledger_file=args.ledger_file,
        src_folder=args.src_folder,
        output_root=args.output_root,
        line_name=args.line_name,
        threshold=args.threshold
    )

if __name__ == '__main__':
    main()
