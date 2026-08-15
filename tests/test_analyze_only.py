import os
import shutil
import tempfile
import unittest

import classify
import classify_autonomous
import classify_channels
from gui_classifier import ClassifierGUI
from reporting import ResultRecorder

from fixtures import LINE_NAME, build_ledger, make_photo, near


class ManualDryRunTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_dry_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_creates_nothing_but_records(self):
        make_photo(os.path.join(self.src, "A_001.jpg"), *near(1, lat_off=0.00005))
        rec = ResultRecorder()
        classify.classify(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD,
            recorder=rec, dry_run=True,
        )
        self.assertEqual(len(rec.records()), 1)
        self.assertEqual(rec.records()[0]["分类结果"], "精细化")
        self.assertTrue(rec.records()[0]["目标路径"])
        line = os.path.join(self.out, LINE_NAME)
        self.assertFalse(os.path.isdir(os.path.join(line, "精细化")))
        self.assertFalse(os.path.isdir(os.path.join(line, "红外照片")))
        self.assertFalse(os.path.isdir(os.path.join(line, "通道")))
        self.assertFalse(os.path.isdir(line))
        csv_path = rec.write_csv(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(csv_path))


class AutonomousDryRunTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_dry_auto_")
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
        make_photo(os.path.join(self.src, "CH_001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:30")
        make_photo(os.path.join(self.src, "001_V_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "001_T_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "003_T_0001.jpg"), *near(3, lat_off=0.00005), dt="2026:08:15 10:02:30")

    def test_auto_dry_run_no_output_no_skipir_full_records(self):
        self.seed()
        rec = ResultRecorder()
        classify_channels.classify_channels(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD,
            recorder=rec, dry_run=True,
        )
        classify_autonomous.classify_autonomous(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD,
            recorder=rec, dry_run=True,
        )
        ClassifierGUI._cleanup_skip_ir(None, self.out, LINE_NAME)
        src_count = len([f for f in os.listdir(self.src) if os.path.isfile(os.path.join(self.src, f))])
        self.assertEqual(len(rec.records()), src_count)
        by_name = {r["文件名"]: r for r in rec.records()}
        self.assertEqual(by_name["CH_001.jpg"]["分类结果"], "通道")
        self.assertEqual(by_name["001_T_0900.jpg"]["结果原因"], "skip_ir 对应红外")
        line = os.path.join(self.out, LINE_NAME)
        self.assertFalse(os.path.isdir(os.path.join(line, "通道")))
        self.assertFalse(os.path.isdir(os.path.join(line, "精细化")))
        self.assertFalse(os.path.isdir(os.path.join(line, "红外照片")))
        self.assertFalse(os.path.isfile(os.path.join(line, "skip_ir.txt")))
        csv_path = rec.write_csv(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(csv_path))


class ConflictPolicyTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_conf_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skip_policy_keeps_existing_file(self):
        make_photo(os.path.join(self.src, "DUP_001.jpg"), *near(1, lat_off=0.00005), color=(10, 10, 10))
        classify.classify(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD,
            conflict_policy="覆盖",
        )
        target = os.path.join(self.out, LINE_NAME, "精细化", "001", "DUP_001.jpg")
        with open(target, "rb") as f:
            first = f.read()
        rec = ResultRecorder()
        classify.classify(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD,
            recorder=rec, conflict_policy="跳过",
        )
        with open(target, "rb") as f:
            self.assertEqual(f.read(), first)
        self.assertEqual(rec.records()[0]["分类结果"], "跳过")
        self.assertEqual(rec.records()[0]["结果原因"], "输出冲突")


if __name__ == "__main__":
    unittest.main()
