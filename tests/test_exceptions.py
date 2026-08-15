import os
import shutil
import tempfile
import unittest

import classify
import pandas as pd

from fixtures import LINE_NAME, build_ledger, make_photo, near


class LedgerExceptionTest(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_exc_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def classify(self):
        classify.classify(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
        )

    def test_empty_ledger_raises(self):
        pd.DataFrame(columns=["杆塔编号", "经度", "纬度", "线路名称"]).to_excel(self.ledger_file, index=False)
        make_photo(os.path.join(self.src, "A_001.jpg"), *near(1, lat_off=0.00005))
        with self.assertRaises(ValueError):
            self.classify()

    def test_missing_required_column_raises(self):
        build_ledger(self.ledger_file, missing=("经度",))
        with self.assertRaises(KeyError):
            self.classify()

    def test_non_numeric_tower_raises(self):
        pd.DataFrame([
            {"杆塔编号": "001", "经度": 113.25, "纬度": 23.5, "线路名称": LINE_NAME},
            {"杆塔编号": "A2", "经度": 113.2515, "纬度": 23.5, "线路名称": LINE_NAME},
            {"杆塔编号": "003", "经度": 113.2530, "纬度": 23.5, "线路名称": LINE_NAME},
        ]).to_excel(self.ledger_file, index=False)
        with self.assertRaises(ValueError):
            self.classify()


if __name__ == "__main__":
    unittest.main()
