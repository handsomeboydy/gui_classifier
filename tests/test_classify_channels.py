import os
import shutil
import tempfile
import unittest

import classify_channels

from fixtures import LINE_NAME, build_ledger, collect_files, make_photo, near


class ChannelBase(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_chan_")
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
        make_photo(os.path.join(self.src, "CH_002.jpg"), *near(1, lat_off=0.00005), dt="2026:08:15 10:00:40")
        make_photo(os.path.join(self.src, "001_V_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "001_T_0900.jpg"), *near(1, lat_off=0.006), dt="2026:08:15 10:00:50")
        make_photo(os.path.join(self.src, "CH_003.jpg"), *near(1, lat_off=0.00005), dt=None)
        make_photo(os.path.join(self.src, "003_T_0001.jpg"), *near(3, lat_off=0.00005), dt="2026:08:15 10:02:30")

    def run_channels(self):
        classify_channels.classify_channels(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD,
        )

    def skip_path(self):
        return os.path.join(self.out, LINE_NAME, "skip_ir.txt")


class ChannelExtractionTest(ChannelBase):
    def test_channel_windows_and_skip_ir(self):
        self.seed()
        self.run_channels()
        line = os.path.join(self.out, LINE_NAME)
        chan1 = os.path.join(line, "通道", "001")
        self.assertTrue(os.path.isfile(os.path.join(chan1, "CH_001.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(chan1, "CH_002.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(chan1, "001_V_0900.jpg")))
        with open(self.skip_path(), "r", encoding="utf-8") as f:
            skip = {line_txt.strip() for line_txt in f if line_txt.strip()}
        self.assertEqual(skip, {"001_T_0900.jpg"})
        copied = collect_files(self.out)
        self.assertNotIn("测试线路A/通道/001/CH_003.jpg", copied)


if __name__ == "__main__":
    unittest.main()
