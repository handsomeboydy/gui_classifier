import csv
import os
import shutil
import tempfile
import unittest

import classify
import classify_autonomous
import classify_channels
from gui_classifier import ClassifierGUI
from reporting import COLUMNS, ResultRecorder

from fixtures import (
    LINE_NAME,
    build_ledger,
    make_photo,
    make_plain_photo,
    near,
)


class RecorderBasicsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_rep_")
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_upsert_and_summary(self):
        rec = ResultRecorder(run_id="20260815_120000")
        rec.record("srcA", "a.jpg", 分类结果="精细化", 目标路径="x/a.jpg")
        rec.record("srcA", "a.jpg", 分类结果="精细化", 目标路径="x/a.jpg", 冲突处理="覆盖")
        rec.record("srcB", "b.jpg", 分类结果="跳过", 结果原因="无GPS")
        self.assertEqual(len(rec.records()), 2)
        self.assertTrue(rec.has_result("srcA", "a.jpg"))
        summary = rec.summary()
        self.assertEqual(summary["总数"], 2)
        self.assertEqual(summary["分类结果"]["精细化"], 1)
        self.assertEqual(summary["结果原因"]["无GPS"], 1)
        self.assertEqual(summary["冲突处理"]["覆盖"], 1)

    def test_csv_with_bom_and_columns(self):
        rec = ResultRecorder(run_id="20260815_120000")
        rec.record("srcA", "a.jpg", 分类结果="精细化", 目标路径="x/a.jpg")
        path = rec.write_csv(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(path), "分图结果清单_20260815_120000.csv")
        with open(path, "rb") as f:
            self.assertEqual(f.read(3), bytes([0xEF, 0xBB, 0xBF]))
        with open(path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["文件名"], "a.jpg")
        self.assertEqual(rows[0]["分类结果"], "精细化")
        self.assertEqual(list(rows[0].keys()), COLUMNS)

    def test_sanitize(self):
        rec = ResultRecorder(run_id="20260815_120000")
        rec.record(
            "srcA",
            "b.jpg",
            纬度=23.5678,
            经度=113.12345,
            源文件完整路径="C:/ledger/srcA/b.jpg",
            目标路径="C:/out/线路/精细化/001/b.jpg",
            分类结果="精细化",
        )
        path = rec.write_csv(self.out, LINE_NAME, sanitize=True)
        with open(path, "r", encoding="utf-8-sig") as f:
            row = list(csv.DictReader(f))[0]
        self.assertEqual(row["纬度"], "23.57")
        self.assertEqual(row["经度"], "113.12")
        self.assertEqual(row["源文件完整路径"], "b.jpg")
        self.assertEqual(row["目标路径"], "b.jpg")
        self.assertEqual(row["源目录"], "srcA")


class ManualRecordingTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_rep_manual_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_record_per_file_with_statuses(self):
        make_photo(os.path.join(self.src, "A_001.jpg"), *near(1, lat_off=0.00005))
        make_photo(os.path.join(self.src, "A_002_T.jpg"), *near(2, lat_off=0.00005))
        make_plain_photo(os.path.join(self.src, "NOGPS.jpg"))
        make_photo(os.path.join(self.src, "FAR.jpg"), *near(3, lat_off=0.006))
        rec = ResultRecorder()
        classify.classify(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
            recorder=rec,
        )
        records = rec.records()
        self.assertEqual(len(records), 4)
        by_name = {r["文件名"]: r for r in records}
        self.assertEqual(by_name["A_001.jpg"]["分类结果"], "精细化")
        self.assertEqual(by_name["A_002_T.jpg"]["分类结果"], "红外照片")
        self.assertEqual(by_name["NOGPS.jpg"]["结果原因"], "无GPS")
        self.assertEqual(by_name["FAR.jpg"]["结果原因"], "超阈值")
        path = rec.write_csv(self.out, LINE_NAME)
        with open(path, "r", encoding="utf-8-sig") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 4)


class AutonomousPipelineRecordingTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_rep_auto_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def seed(self):
        make_photo(os.path.join(self.src, "001_V_0001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:00")
        make_photo(os.path.join(self.src, "002_V_0001.jpg"), *near(2, lat_off=0.00005), dt="2026:08:15 10:01:00")
        make_photo(os.path.join(self.src, "003_V_0001.jpg"), *near(3, lat_off=0.00005), dt="2026:08:15 10:02:00")
        make_photo(os.path.join(self.src, "CH_001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:30")
        make_photo(os.path.join(self.src, "001_V_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "001_T_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "003_T_0001.jpg"), *near(3, lat_off=0.00005), dt="2026:08:15 10:02:30")

    def test_pipeline_records_one_final_result_per_file(self):
        self.seed()
        rec = ResultRecorder()
        classify_channels.classify_channels(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD, recorder=rec
        )
        classify_autonomous.classify_autonomous(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD, recorder=rec
        )
        ClassifierGUI._cleanup_skip_ir(None, self.out, LINE_NAME)
        records = rec.records()
        src_files = [f for f in os.listdir(self.src) if os.path.isfile(os.path.join(self.src, f))]
        self.assertEqual(len(records), len(src_files))
        keys = {(r["源目录"], r["文件名"]) for r in records}
        self.assertEqual(len(keys), len(records))
        by_name = {r["文件名"]: r for r in records}
        self.assertEqual(by_name["CH_001.jpg"]["分类结果"], "通道")
        self.assertEqual(by_name["001_V_0900.jpg"]["分类结果"], "通道")
        self.assertEqual(by_name["001_T_0900.jpg"]["结果原因"], "skip_ir 对应红外")
        self.assertEqual(by_name["003_T_0001.jpg"]["分类结果"], "红外照片")
        self.assertEqual(by_name["002_V_0001.jpg"]["分类结果"], "精细化")
        path = rec.write_csv(self.out, LINE_NAME)
        with open(path, "r", encoding="utf-8-sig") as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(records))
        self.assertEqual(rec.summary()["总数"], len(records))


if __name__ == "__main__":
    unittest.main()
