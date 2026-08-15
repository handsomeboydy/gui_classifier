import os
import shutil
import tempfile
import unittest

import classify_autonomous
import classify_channels
from gui_classifier import ClassifierGUI

from fixtures import LINE_NAME, build_ledger, collect_files, make_photo, near


class SkipIrLifecycleTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_skip_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)
        make_photo(os.path.join(self.src, "001_V_0001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:00")
        make_photo(os.path.join(self.src, "CH_001.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:30")
        make_photo(os.path.join(self.src, "001_V_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "001_T_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def skip_path(self):
        return os.path.join(self.out, LINE_NAME, "skip_ir.txt")

    def run_pipeline(self):
        classify_channels.classify_channels(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD
        )
        classify_autonomous.classify_autonomous(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD
        )

    def test_skip_ir_exists_between_phases_and_removed_after_cleanup(self):
        classify_channels.classify_channels(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD
        )
        self.assertTrue(os.path.isfile(self.skip_path()))
        classify_autonomous.classify_autonomous(
            self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD
        )
        self.assertTrue(os.path.isfile(self.skip_path()))
        ClassifierGUI._cleanup_skip_ir(None, self.out, LINE_NAME)
        self.assertFalse(os.path.exists(self.skip_path()))
        ClassifierGUI._cleanup_skip_ir(None, self.out, LINE_NAME)

    def test_full_pipeline_repeat_identical(self):
        out1 = os.path.join(self.tmp, "out1")
        out2 = os.path.join(self.tmp, "out2")
        for out in (out1, out2):
            classify_channels.classify_channels(
                self.ledger_file, self.src, out, LINE_NAME, self.THRESHOLD
            )
            classify_autonomous.classify_autonomous(
                self.ledger_file, self.src, out, LINE_NAME, self.THRESHOLD
            )
            ClassifierGUI._cleanup_skip_ir(None, out, LINE_NAME)
        self.assertEqual(collect_files(out1), collect_files(out2))


if __name__ == "__main__":
    unittest.main()
