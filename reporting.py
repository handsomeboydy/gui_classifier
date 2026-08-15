# reporting.py
"""结构化分类结果记录：一文件一结果、UTF-8 BOM CSV、脱敏输出与运行摘要。

本模块只负责记录与导出，不参与分类决策，不改变现有分类结果。
"""

import csv
import os
import time
from collections import Counter


CSV_FILENAME_PREFIX = "分图结果清单"

COLUMNS = [
    "运行编号",
    "源目录",
    "文件名",
    "源文件完整路径",
    "EXIF状态",
    "纬度",
    "经度",
    "最近杆塔",
    "最近距离",
    "期望侧别",
    "实际侧别",
    "分类结果",
    "结果原因",
    "目标路径",
    "冲突处理",
    "处理时间",
]


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ResultRecorder:
    """按（源目录, 文件名）维护一文件一结果的记录集合。"""

    def __init__(self, run_id=None):
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self._started = time.time()
        self._records = {}
        self._last_csv_path = None

    def _key(self, src_dir, filename):
        return (src_dir, filename)

    def has_result(self, src_dir, filename):
        return self._key(src_dir, filename) in self._records

    def record(self, src_dir, filename, **fields):
        """新增或更新某文件的结果记录（同一文件只保留一条最终结果）。"""
        key = self._key(src_dir, filename)
        rec = self._records.setdefault(key, {})
        rec.setdefault("运行编号", self.run_id)
        rec.setdefault("源目录", src_dir)
        rec.setdefault("文件名", filename)
        rec.setdefault("源文件完整路径", os.path.join(src_dir, filename))
        rec.setdefault("EXIF状态", "")
        rec.setdefault("分类结果", "")
        rec.setdefault("结果原因", "")
        rec.setdefault("冲突处理", "")
        rec.setdefault("处理时间", _now())
        for field, value in fields.items():
            rec[field] = value
        return rec

    def records(self):
        return list(self._records.values())

    def write_csv(self, output_root, line_name, sanitize=False):
        """写出 UTF-8 BOM CSV：<输出根目录>/<线路名称>/分图结果清单_<运行编号>.csv"""
        line_dir = os.path.join(output_root, line_name)
        os.makedirs(line_dir, exist_ok=True)
        path = os.path.join(line_dir, "%s_%s.csv" % (CSV_FILENAME_PREFIX, self.run_id))
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for rec in self._records.values():
                writer.writerow(self._sanitize(rec) if sanitize else rec)
        self._last_csv_path = path
        return path

    @staticmethod
    def _sanitize(rec):
        """脱敏：坐标保留两位小数，路径只保留文件名（D-04）。"""
        out = dict(rec)
        for coord in ("纬度", "经度"):
            value = out.get(coord)
            if value not in (None, ""):
                out[coord] = round(float(value), 2)
        for path_field in ("源文件完整路径", "目标路径"):
            value = out.get(path_field)
            if value:
                out[path_field] = os.path.basename(value)
        src = out.get("源目录")
        if src:
            out["源目录"] = os.path.basename(src)
        return out

    def summary(self):
        result_counter = Counter()
        reason_counter = Counter()
        conflict_counter = Counter()
        for rec in self._records.values():
            result_counter[rec.get("分类结果") or "未记录"] += 1
            if rec.get("结果原因"):
                reason_counter[rec.get("结果原因")] += 1
            if rec.get("冲突处理"):
                conflict_counter[rec.get("冲突处理")] += 1
        return {
            "总数": len(self._records),
            "分类结果": dict(result_counter),
            "结果原因": dict(reason_counter),
            "冲突处理": dict(conflict_counter),
            "耗时秒": round(time.time() - self._started, 1),
            "清单位置": self._last_csv_path or "",
        }
