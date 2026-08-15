import os
import shutil
import tempfile
import unittest

import pandas as pd

from preflight import LEVEL_ERROR, LEVEL_WARNING, preflight

from fixtures import LINE_NAME, build_ledger, build_double_tower, make_photo, make_plain_photo, near


class PreflightBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_pre_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_preflight(self, mode="manual", src_dirs=None, threshold=200.0):
        return preflight(
            ledger_file=self.ledger_file,
            src_dirs=src_dirs if src_dirs is not None else [self.src],
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=threshold,
            mode=mode,
        )


class PreflightErrorTest(PreflightBase):
    def test_missing_ledger_error(self):
        result = preflight(
            ledger_file=os.path.join(self.ledger_dir, "不存在.xlsx"),
            src_dirs=[self.src],
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=200.0,
            mode="manual",
        )
        self.assertTrue(result.has_errors())

    def test_missing_required_column_error(self):
        build_ledger(self.ledger_file, missing=("经度",))
        result = self.run_preflight()
        self.assertTrue(result.has_errors())
        self.assertTrue(any("经度" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_ERROR))

    def test_auto_empty_line_filter_error(self):
        pd.DataFrame([
            {"杆塔编号": "001", "经度": 113.25, "纬度": 23.5, "线路名称": "其他线路"},
            {"杆塔编号": "002", "经度": 113.2515, "纬度": 23.5, "线路名称": "其他线路"},
        ]).to_excel(self.ledger_file, index=False)
        result = self.run_preflight(mode="auto")
        self.assertTrue(result.has_errors())
        self.assertTrue(any("没有匹配行" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_ERROR))

    def test_invalid_threshold_error(self):
        result = self.run_preflight(threshold="abc")
        self.assertTrue(result.has_errors())

    def test_missing_src_dir_error(self):
        result = self.run_preflight(src_dirs=[os.path.join(self.tmp, "missing")])
        self.assertTrue(result.has_errors())


class PreflightWarningTest(PreflightBase):
    def test_bad_coordinates_warning(self):
        pd.DataFrame([
            {"杆塔编号": "001", "经度": 113.25, "纬度": "未知", "线路名称": LINE_NAME},
            {"杆塔编号": "002", "经度": 113.2515, "纬度": 23.5, "线路名称": LINE_NAME},
        ]).to_excel(self.ledger_file, index=False)
        result = self.run_preflight()
        self.assertFalse(result.has_errors())
        self.assertTrue(any("经纬度无法转换" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))

    def test_non_numeric_tower_warning(self):
        pd.DataFrame([
            {"杆塔编号": "001", "经度": 113.25, "纬度": 23.5, "线路名称": LINE_NAME},
            {"杆塔编号": "A2", "经度": 113.2515, "纬度": 23.5, "线路名称": LINE_NAME},
        ]).to_excel(self.ledger_file, index=False)
        result = self.run_preflight()
        self.assertTrue(any("不是整数" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))

    def test_empty_src_warning(self):
        result = self.run_preflight()
        self.assertTrue(any("源目录为空" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))

    def test_unsupported_extension_warning(self):
        with open(os.path.join(self.src, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("hello")
        make_photo(os.path.join(self.src, "ok.jpg"), *near(1, lat_off=0.00005))
        result = self.run_preflight()
        self.assertTrue(any("格式不支持" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))

    def test_existing_output_warning(self):
        make_photo(os.path.join(self.src, "ok.jpg"), *near(1, lat_off=0.00005))
        line_dir = os.path.join(self.out, LINE_NAME)
        os.makedirs(os.path.join(line_dir, "精细化", "001"), exist_ok=True)
        with open(os.path.join(line_dir, "精细化", "001", "old.jpg"), "wb") as f:
            f.write(b"x")
        result = self.run_preflight()
        self.assertTrue(any("已存在" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))


class PreflightInfoTest(PreflightBase):
    def test_clean_input_no_errors(self):
        make_photo(os.path.join(self.src, "001_V_0001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:00")
        build_double_tower(os.path.join(self.ledger_dir, "1双回塔台账文件.xlsx"), "1-5:左")
        result = self.run_preflight()
        self.assertFalse(result.has_errors())
        self.assertTrue(any(i["级别"] == "信息" for i in result.rows()))

    def test_no_gps_sample_warning(self):
        make_plain_photo(os.path.join(self.src, "n1.jpg"))
        make_plain_photo(os.path.join(self.src, "n2.jpg"))
        result = self.run_preflight()
        self.assertTrue(any("均无 GPS" in i["说明"] for i in result.rows() if i["级别"] == LEVEL_WARNING))


if __name__ == "__main__":
    unittest.main()
