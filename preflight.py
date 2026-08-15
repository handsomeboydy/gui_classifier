# preflight.py
"""分类前预检：错误/警告/信息分级，供 GUI 与命令行共用。

预检只读取文件与目录状态，不修改任何文件；支持格式按 D-07 约定。
"""

import glob
import os

import exifread
import pandas as pd


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}  # D-07

LEVEL_ERROR = "错误"
LEVEL_WARNING = "警告"
LEVEL_INFO = "信息"

EXIF_SAMPLE_LIMIT = 20


def is_supported_file(path):
    """是否为支持的照片扩展名（D-07）。"""
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def iter_src_files(src_folder):
    """扫描源目录顶层文件（与现有 glob('*.*') 行为一致，仅文件）。"""
    for p in glob.glob(os.path.join(src_folder, "*.*")):
        if os.path.isfile(p):
            yield p


class PreflightResult:
    def __init__(self):
        self.issues = []

    def add(self, level, category, message):
        self.issues.append({"级别": level, "类别": category, "说明": message})

    def has_errors(self):
        return any(i["级别"] == LEVEL_ERROR for i in self.issues)

    def count(self, level):
        return sum(1 for i in self.issues if i["级别"] == level)

    def rows(self):
        return list(self.issues)


def preflight(ledger_file, src_dirs, output_root, line_name, threshold, mode="manual"):
    """执行分类前预检，返回 PreflightResult；不修改任何文件。"""
    result = PreflightResult()

    # 0. 阈值参数
    try:
        thr = float(threshold)
        if thr <= 0:
            result.add(LEVEL_ERROR, "参数", "距离阈值必须为正数")
    except (TypeError, ValueError):
        result.add(LEVEL_ERROR, "参数", "距离阈值必须是数字")

    # 1. 台账存在
    if not os.path.isfile(ledger_file):
        result.add(LEVEL_ERROR, "台账", f"未找到台账文件: {ledger_file}")
        return result

    # 2. 台账读取与必需列
    try:
        df = pd.read_excel(ledger_file, dtype={"杆塔编号": str})
    except Exception as e:
        result.add(LEVEL_ERROR, "台账", f"台账读取失败: {e}")
        return result

    required = ["杆塔编号", "经度", "纬度"]
    if mode == "auto":
        required.append("线路名称")
    missing = [c for c in required if c not in df.columns]
    if missing:
        for col in missing:
            result.add(LEVEL_ERROR, "台账", f"缺少必需列: {col}")
        return result

    # 3. 坐标可转换
    try:
        lat_num = pd.to_numeric(df["纬度"], errors="coerce")
        lon_num = pd.to_numeric(df["经度"], errors="coerce")
        bad_coord = int((lat_num.isna() | lon_num.isna()).sum())
        if bad_coord:
            result.add(LEVEL_WARNING, "台账", f"{bad_coord} 行经纬度无法转换，将不参与匹配")
    except Exception:
        result.add(LEVEL_WARNING, "台账", "经纬度列无法解析为数字")

    # 4. 塔号可排序（整数）
    tower_vals = df["杆塔编号"].dropna().astype(str)
    bad_tower = 0
    for v in tower_vals:
        try:
            int(v)
        except (TypeError, ValueError):
            bad_tower += 1
    if bad_tower:
        result.add(LEVEL_WARNING, "台账", f"{bad_tower} 个杆塔编号不是整数，当前版本无法排序或侧别判断")
    dup_tower = int(tower_vals.duplicated().sum())
    if dup_tower:
        result.add(LEVEL_WARNING, "台账", f"存在 {dup_tower} 个重复杆塔编号")

    # 5. 线路过滤（自主模式）
    if mode == "auto":
        if "线路名称" not in df.columns:
            result.add(LEVEL_ERROR, "台账", "自主模式需要“线路名称”列")
            return result
        filtered = df[df["线路名称"] == line_name]
        if filtered.empty:
            result.add(LEVEL_ERROR, "台账", f"线路名称“{line_name}”在台账中没有匹配行")
        else:
            result.add(LEVEL_INFO, "台账", f"线路“{line_name}”匹配杆塔 {len(filtered)} 基")
    else:
        result.add(LEVEL_INFO, "台账", f"台账共 {len(df)} 行")

    # 6. 源目录
    for src in src_dirs:
        if not os.path.isdir(src):
            result.add(LEVEL_ERROR, "源目录", f"源目录不存在: {src}")
            continue
        files = list(iter_src_files(src))
        if not files:
            result.add(LEVEL_WARNING, "源目录", f"源目录为空: {src}")
            continue
        supported = [p for p in files if is_supported_file(p)]
        unsupported = len(files) - len(supported)
        result.add(LEVEL_INFO, "源目录", f"{src}: 共 {len(files)} 个文件，支持格式 {len(supported)} 个")
        if unsupported:
            result.add(LEVEL_WARNING, "源目录", f"{src}: {unsupported} 个文件扩展名不受支持，将记录为“格式不支持”")
        # EXIF 抽样
        sample = supported[:EXIF_SAMPLE_LIMIT]
        gps_ok = 0
        time_ok = 0
        parse_fail = 0
        for p in sample:
            try:
                with open(p, "rb") as f:
                    tags = exifread.process_file(f, details=False)
                if "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
                    gps_ok += 1
                if "EXIF DateTimeOriginal" in tags or "Image DateTime" in tags:
                    time_ok += 1
            except Exception:
                parse_fail += 1
        if sample:
            result.add(LEVEL_INFO, "EXIF", f"{src}: 抽样 {len(sample)} 张，含 GPS {gps_ok} 张，含拍摄时间 {time_ok} 张，解析失败 {parse_fail} 张")
            if gps_ok == 0:
                result.add(LEVEL_WARNING, "EXIF", f"{src}: 抽样照片均无 GPS，分类结果可能大量跳过")
            elif gps_ok < len(sample):
                result.add(LEVEL_INFO, "EXIF", f"{src}: {len(sample) - gps_ok} 张抽样照片无 GPS")
        # 命名统计
        names = [os.path.basename(p) for p in files]
        if mode == "auto":
            v_count = sum(1 for n in names if "_V_" in n)
            t_count = sum(1 for n in names if "_T_" in n)
            other = len(names) - v_count - t_count
            result.add(LEVEL_INFO, "命名", f"{src}: _V_ {v_count} 张，_T_ {t_count} 张，其他 {other} 张")
            if v_count == 0 and t_count == 0:
                result.add(LEVEL_WARNING, "命名", f"{src}: 未发现 _V_/_T_ 命名照片，自主分类可能无输出")
        else:
            t_loose = sum(1 for n in names if "_T" in n)
            result.add(LEVEL_INFO, "命名", f"{src}: 含 _T 的红外命名 {t_loose} 张，其余视为精细化")

    # 7. 双回塔侧别台账
    ledger_dir = os.path.dirname(ledger_file)
    if mode == "manual":
        double = os.path.join(ledger_dir, "1双回塔台账文件.xlsx")
        if os.path.isfile(double):
            result.add(LEVEL_INFO, "侧别", "已找到 1双回塔台账文件.xlsx，将按侧别过滤")
        else:
            result.add(LEVEL_INFO, "侧别", "未找到 1双回塔台账文件.xlsx，将不进行侧别过滤")
    else:
        doubles = glob.glob(os.path.join(ledger_dir, "*双回塔台账文件.xlsx"))
        if doubles:
            result.add(LEVEL_INFO, "侧别", f"已找到双回塔台账 {os.path.basename(doubles[0])}，将按侧别过滤")
        else:
            result.add(LEVEL_INFO, "侧别", "未找到 *双回塔台账文件.xlsx，将不进行侧别过滤")

    # 8. 输出目录与冲突
    if os.path.isdir(output_root):
        line_dir = os.path.join(output_root, line_name)
        if os.path.isdir(line_dir):
            existing = sum(len(files2) for _r, _d, files2 in os.walk(line_dir))
            if existing:
                result.add(LEVEL_WARNING, "输出目录", f"输出目录 {line_dir} 已存在 {existing} 个文件，正式分类可能覆盖同名文件")
            else:
                result.add(LEVEL_INFO, "输出目录", "输出目录已存在但为空")
        else:
            result.add(LEVEL_INFO, "输出目录", "输出目录将自动创建")
    else:
        result.add(LEVEL_INFO, "输出目录", f"输出根目录不存在，将自动创建: {output_root}")

    return result
