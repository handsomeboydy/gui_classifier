import os
import shutil
import tempfile
import unittest

import classify

from fixtures import (
    LINE_NAME,
    build_double_tower,
    build_ledger,
    collect_files,
    make_photo,
    make_plain_photo,
    near,
)


class ManualBase(unittest.TestCase):
    THRESHOLD = 200.0

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg_manual_")
        self.ledger_dir = os.path.join(self.tmp, "ledger")
        self.src = os.path.join(self.tmp, "src")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.ledger_dir)
        os.makedirs(self.src)
        self.ledger_file = os.path.join(self.ledger_dir, LINE_NAME + "经纬度台账.xlsx")
        build_ledger(self.ledger_file)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def with_side(self, side_str="1-5:左"):
        build_double_tower(os.path.join(self.ledger_dir, "1双回塔台账文件.xlsx"), side_str)

    def classify(self, threshold=None):
        classify.classify(
            ledger_file=self.ledger_file,
            src_folder=self.src,
            output_root=self.out,
            line_name=LINE_NAME,
            threshold=self.THRESHOLD if threshold is None else threshold,
        )


class ManualClassificationTest(ManualBase):
    def test_nearest_tower_category_and_channel_dir(self):
        make_photo(os.path.join(self.src, "IMG_001.jpg"), *near(1, lat_off=0.00005))
        make_photo(os.path.join(self.src, "IMG_002_T.jpg"), *near(2, lat_off=0.00005))
        self.classify()
        line = os.path.join(self.out, LINE_NAME)
        self.assertTrue(os.path.isfile(os.path.join(line, "精细化", "001", "IMG_001.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(line, "红外照片", "002", "IMG_002_T.jpg")))
        self.assertTrue(os.path.isdir(os.path.join(line, "通道", "001")))

    def test_out_of_threshold_skipped(self):
        make_photo(os.path.join(self.src, "FAR_001.jpg"), *near(1, lat_off=0.006))
        self.classify()
        copied = collect_files(self.out)
        self.assertNotIn("测试线路A/精细化/001/FAR_001.jpg", copied)

    def test_no_gps_skipped(self):
        make_plain_photo(os.path.join(self.src, "NOGPS_001.jpg"))
        self.classify()
        self.assertEqual(collect_files(self.out), set())

    def test_corrupt_image_skipped_without_crash(self):
        with open(os.path.join(self.src, "BAD_001.jpg"), "wb") as f:
            f.write(b"this is not a jpeg at all" * 100)
        self.classify()
        self.assertNotIn("BAD_001.jpg", "|".join(collect_files(self.out)))

    def test_side_left_filters_right_side(self):
        self.with_side("1-5:左")
        make_photo(os.path.join(self.src, "L_001.jpg"), *near(1, lat_off=0.00015))
        make_photo(os.path.join(self.src, "R_002.jpg"), *near(2, lat_off=-0.00015))
        self.classify()
        copied = collect_files(self.out)
        self.assertIn("测试线路A/精细化/001/L_001.jpg", copied)
        self.assertNotIn("R_002.jpg", "|".join(copied))

    def test_side_single_no_filter(self):
        self.with_side("1-5:单")
        make_photo(os.path.join(self.src, "N_001.jpg"), *near(1, lat_off=0.00015))
        make_photo(os.path.join(self.src, "S_002.jpg"), *near(2, lat_off=-0.00015))
        self.classify()
        copied = collect_files(self.out)
        self.assertIn("测试线路A/精细化/001/N_001.jpg", copied)
        self.assertIn("测试线路A/精细化/002/S_002.jpg", copied)

    def test_endpoint_tower_side(self):
        self.with_side("1-5:左")
        make_photo(os.path.join(self.src, "E1L_001.jpg"), *near(1, lat_off=0.00015))
        make_photo(os.path.join(self.src, "E1R_002.jpg"), *near(1, lat_off=-0.00015))
        self.classify()
        copied = collect_files(self.out)
        self.assertIn("测试线路A/精细化/001/E1L_001.jpg", copied)
        self.assertNotIn("E1R_002.jpg", "|".join(copied))

    def test_without_double_tower_no_side_filter(self):
        make_photo(os.path.join(self.src, "N_001.jpg"), *near(1, lat_off=0.00015))
        make_photo(os.path.join(self.src, "S_002.jpg"), *near(2, lat_off=-0.00015))
        self.classify()
        copied = collect_files(self.out)
        self.assertIn("测试线路A/精细化/001/N_001.jpg", copied)
        self.assertIn("测试线路A/精细化/002/S_002.jpg", copied)

    def test_near_boundary_inside_and_outside(self):
        make_photo(os.path.join(self.src, "IN_001.jpg"), *near(1, lat_off=0.0009))
        make_photo(os.path.join(self.src, "OUT_002.jpg"), *near(2, lat_off=0.002))
        self.classify()
        copied = collect_files(self.out)
        self.assertIn("测试线路A/精细化/001/IN_001.jpg", copied)
        self.assertNotIn("OUT_002.jpg", "|".join(copied))

    def test_duplicate_basename_overwrite_same_tower(self):
        src2 = os.path.join(self.tmp, "src2")
        os.makedirs(src2)
        make_photo(os.path.join(self.src, "DUP_001.jpg"), *near(1, lat_off=0.00005), color=(10, 10, 10))
        make_photo(os.path.join(src2, "DUP_001.jpg"), *near(1, lat_off=0.00005), color=(250, 250, 250))
        classify.classify(self.ledger_file, self.src, self.out, LINE_NAME, self.THRESHOLD)
        classify.classify(self.ledger_file, src2, self.out, LINE_NAME, self.THRESHOLD)
        target = os.path.join(self.out, LINE_NAME, "精细化", "001", "DUP_001.jpg")
        self.assertTrue(os.path.isfile(target))
        with open(target, "rb") as f:
            with open(os.path.join(src2, "DUP_001.jpg"), "rb") as g:
                self.assertEqual(f.read(), g.read())

    def test_source_photos_not_modified(self):
        p = os.path.join(self.src, "KEEP_001.jpg")
        make_photo(p, *near(1, lat_off=0.00005))
        with open(p, "rb") as f:
            before = f.read()
        self.classify()
        with open(p, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_repeat_run_identical_outputs(self):
        make_photo(os.path.join(self.src, "A_001.jpg"), *near(1, lat_off=0.00005))
        make_photo(os.path.join(self.src, "B_002_T.jpg"), *near(2, lat_off=-0.00005))
        out1 = os.path.join(self.tmp, "out1")
        out2 = os.path.join(self.tmp, "out2")
        classify.classify(self.ledger_file, self.src, out1, LINE_NAME, self.THRESHOLD)
        classify.classify(self.ledger_file, self.src, out2, LINE_NAME, self.THRESHOLD)
        self.assertEqual(collect_files(out1), collect_files(out2))


if __name__ == "__main__":
    unittest.main()
