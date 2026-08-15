import os
import shutil
import tempfile
import unittest

import classify_autonomous
import classify_channels

from fixtures import (
    LINE_NAME,
    build_double_tower,
    build_ledger,
    collect_files,
    make_photo,
    near,
)


class AutonomousBase(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_auto_")
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

    def run_pipeline(self, with_side=None):
        if with_side:
            build_double_tower(os.path.join(self.ledger_dir, "1双回塔台账文件.xlsx"), with_side)
        classify_channels.classify_channels(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
        )
        classify_autonomous.classify_autonomous(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
        )


class AutonomousClassificationTest(AutonomousBase):
    def test_phase2_fine_ir_and_skip_ir(self):
        self.seed()
        self.run_pipeline()
        line = os.path.join(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(os.path.join(line, "红外照片", "003", "003_T_0001.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(line, "精细化", "001", "001_V_0001.jpg")))
        copied = collect_files(self.out)
        self.assertNotIn("001_T_0900.jpg", "|".join(copied))
        self.assertTrue(os.path.isfile(os.path.join(line, "skip_ir.txt")))

    def test_no_time_v_photo_still_classified(self):
        self.seed()
        make_photo(os.path.join(self.src, "002_V_NOTIME.jpg"), *near(2, lat_off=0.00005), dt=None)
        self.run_pipeline()
        line = os.path.join(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(os.path.join(line, "精细化", "002", "002_V_NOTIME.jpg")))

    def test_autonomous_side_filter(self):
        self.seed()
        self.run_pipeline(with_side="1-5:左")
        make_photo(os.path.join(self.src, "004_V_LEFT.jpg"), *near(4, lat_off=0.00015), dt="2026:08:15 10:03:00")
        make_photo(os.path.join(self.src, "004_V_RIGHT.jpg"), *near(4, lat_off=-0.00015), dt="2026:08:15 10:03:10")
        classify_autonomous.classify_autonomous(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
        )
        line = os.path.join(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(os.path.join(line, "精细化", "004", "004_V_LEFT.jpg")))
        self.assertFalse(os.path.isfile(os.path.join(line, "精细化", "004", "004_V_RIGHT.jpg")))


if __name__ == "__main__":
    unittest.main()
