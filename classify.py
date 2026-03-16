# classify.py
import os
import glob
import shutil
import pandas as pd
import exifread
from math import radians, sin, cos, sqrt, atan2

from side_parser import get_expected_side_for_tower


def haversine(lat1, lon1, lat2, lon2):
    """计算两点（十进制度）间的大圆距离（米）"""
    R = 6371000  # 地球平均半径，米
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def get_image_gps(img_path):
    """从照片 EXIF 中提取 GPS，经纬度十进制度；找不到时返回 (None, None)"""
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        def _to_deg(value):
            d, m, s = [float(x.num) / float(x.den) for x in value.values]
            return d + m/60 + s/3600

        lat_ref = tags["GPS GPSLatitudeRef"].printable
        lon_ref = tags["GPS GPSLongitudeRef"].printable
        lat = _to_deg(tags["GPS GPSLatitude"])
        lon = _to_deg(tags["GPS GPSLongitude"])
        if lat_ref != "N": lat = -lat
        if lon_ref != "E": lon = -lon
        return lat, lon
    except KeyError:
        return None, None


def load_ledger(path):
    """读取台账Excel，返回包含杆塔编号与坐标的DataFrame"""
    df = pd.read_excel(path, dtype={"杆塔编号": str})
    return df[["杆塔编号", "经度", "纬度"]]


def load_double_tower_side(ledger_file, line_name):
    """
    在与当前线路台账同一目录下，查找 '1双回塔台账文件.xlsx'，
    如果存在且包含当前 line_name，则返回该行的原始侧别字符串；
    否则返回 None
    """
    dirpath = os.path.dirname(ledger_file)
    double_path = os.path.join(dirpath, "1双回塔台账文件.xlsx")
    if not os.path.isfile(double_path):
        return None

    df = pd.read_excel(double_path, dtype=str)
    mask1 = df["杆塔1名称"] == line_name
    mask2 = df["杆塔2名称"] == line_name
    if mask1.any():
        return df.loc[mask1, "杆塔1方位"].iloc[0]
    if mask2.any():
        return df.loc[mask2, "杆塔2方位"].iloc[0]
    return None


def determine_side(prev_coord, curr_coord, next_coord, img_coord):
    """
    根据前后两塔线段折线判断 img_coord 相对于当前塔位置的侧别：
      - 若同时存在 prev_coord 和 next_coord：用入向量(prev→curr)与出向量(curr→next)的角平分线做分侧线；
      - 若只有一侧邻塔：退回到原来单线段判断逻辑。
    返回："左" / "右" / None
    """
    lon_curr, lat_curr = curr_coord[1], curr_coord[0]

    # 缺少一侧时，沿原有逻辑
    if prev_coord is None or next_coord is None:
        adj = next_coord if prev_coord is None else prev_coord
        lon_adj, lat_adj = adj[1], adj[0]
        vx, vy = lon_adj - lon_curr, lat_adj - lat_curr
        wx, wy = img_coord[1] - lon_curr, img_coord[0] - lat_curr
        cross = vx * wy - vy * wx
        if cross > 0: return "左"
        if cross < 0: return "右"
        return None

    # 同时有前后邻塔：计算角平分线
    lon_prev, lat_prev = prev_coord[1], prev_coord[0]
    lon_next, lat_next = next_coord[1], next_coord[0]
    v_in  = (lon_curr - lon_prev, lat_curr - lat_prev)
    v_out = (lon_next - lon_curr, lat_next - lat_curr)
    len_in  = sqrt(v_in[0]**2 + v_in[1]**2)
    len_out = sqrt(v_out[0]**2 + v_out[1]**2)
    u_in   = (v_in[0]/len_in,  v_in[1]/len_in)   if len_in > 0 else (0.0, 0.0)
    u_out  = (v_out[0]/len_out, v_out[1]/len_out) if len_out > 0 else (0.0, 0.0)
    bis_x, bis_y = u_in[0] + u_out[0], u_in[1] + u_out[1]
    norm_b = sqrt(bis_x**2 + bis_y**2)
    if norm_b < 1e-6:
        vx, vy = v_out
    else:
        vx, vy = bis_x, bis_y

    wx, wy = img_coord[1] - lon_curr, img_coord[0] - lat_curr
    cross = vx * wy - vy * wx
    if cross > 0: return "左"
    if cross < 0: return "右"
    return None


def classify(ledger_file, src_folder, output_root, line_name, threshold):
    """
    对手动飞行的照片进行分类（支持部分同塔分段侧别配置）。
    """
    # 1. 加载台账与侧别配置
    ledger = load_ledger(ledger_file)
    raw_side_str = load_double_tower_side(ledger_file, line_name)

    # 2. 准备输出目录
    for sub in ("精细化", "红外照片", "通道"):
        os.makedirs(os.path.join(output_root, line_name, sub), exist_ok=True)

    # 3. 对台账按杆塔编号排序，生成序列
    ledger_sorted = ledger.copy()
    ledger_sorted['__num'] = ledger_sorted['杆塔编号'].astype(int)
    ledger_sorted = ledger_sorted.sort_values(by='__num').reset_index(drop=True)
    towers_seq = ledger_sorted['杆塔编号'].tolist()

    # 4. 遍历照片
    for img_path in glob.glob(os.path.join(src_folder, '*.*')):
        lat, lon = get_image_gps(img_path)
        if lat is None:
            print(f"[跳过，无 GPS] {img_path}")
            continue

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

        # 5. 如果 expected_side 指定了 "左" 或 "右"，则进行侧别过滤
        if expected_side in ('左', '右'):
            try:
                idx = towers_seq.index(tower)
            except ValueError:
                print(f"[跳过，台账中找不到塔号] {img_path} → 塔 {tower}")
                continue

            curr_row = ledger_sorted[ledger_sorted['杆塔编号'] == tower].iloc[0]
            # 中间基杆，两侧都存在：用折线角平分判断
            if idx > 0 and idx < len(towers_seq) - 1:
                prev_id, next_id = towers_seq[idx - 1], towers_seq[idx + 1]
                prev_row = ledger_sorted[ledger_sorted['杆塔编号'] == prev_id].iloc[0]
                next_row = ledger_sorted[ledger_sorted['杆塔编号'] == next_id].iloc[0]
                actual_side = determine_side(
                    (prev_row['纬度'], prev_row['经度']),
                    (curr_row['纬度'], curr_row['经度']),
                    (next_row['纬度'], next_row['经度']),
                    (lat, lon)
                )
            else:
                # 端点基杆，只能用单边
                adj_id = towers_seq[idx + 1] if idx < len(towers_seq) - 1 else towers_seq[idx - 1]
                adj_row = ledger_sorted[ledger_sorted['杆塔编号'] == adj_id].iloc[0]
                actual_side = determine_side(
                    None,
                    (curr_row['纬度'], curr_row['经度']),
                    (adj_row['纬度'], adj_row['经度']),
                    (lat, lon)
                )

            if actual_side != expected_side:
                print(f"[跳过，侧别不符] {img_path} → 塔 {tower} (期望 {expected_side}，实际 {actual_side})")
                continue

        # 6. 分类并拷贝
        name = os.path.basename(img_path)
        category = "红外照片" if '_T' in name else "精细化"
        dest = os.path.join(output_root, line_name, category, tower)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(img_path, dest)
        print(f"[已分类] {img_path} → {dest}")

        # 7. 生成空通道文件夹
        if category == "精细化":
            chan_dest = os.path.join(output_root, line_name, "通道", tower)
            os.makedirs(chan_dest, exist_ok=True)

    print("手动飞行分类完成。")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="手动飞行照片分类工具")
    parser.add_argument('--ledger-file', required=True)
    parser.add_argument('--src-folder', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--line-name', required=True)
    parser.add_argument('--threshold', type=float, default=50.0)
    args = parser.parse_args()

    classify(
        ledger_file=args.ledger_file,
        src_folder=args.src_folder,
        output_root=args.output_root,
        line_name=args.line_name,
        threshold=args.threshold
    )
    print("分类完成。")
