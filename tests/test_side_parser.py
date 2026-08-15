import unittest

from side_parser import get_expected_side_for_tower, parse_side_mapping


class SideParserTest(unittest.TestCase):
    def test_whole_line_left(self):
        self.assertEqual(get_expected_side_for_tower("左", 8), "左")

    def test_whole_line_right(self):
        self.assertEqual(get_expected_side_for_tower("右", 8), "右")

    def test_segmented_ranges(self):
        raw = "1-15:左,16-20:单,21-25:右"
        self.assertEqual(get_expected_side_for_tower(raw, 3), "左")
        self.assertIsNone(get_expected_side_for_tower(raw, 18))
        self.assertEqual(get_expected_side_for_tower(raw, 23), "右")

    def test_single_point(self):
        raw = "1:左, 2:单, 3-5:右"
        self.assertEqual(get_expected_side_for_tower(raw, 1), "左")
        self.assertIsNone(get_expected_side_for_tower(raw, 2))
        self.assertEqual(get_expected_side_for_tower(raw, 5), "右")

    def test_fullwidth_punctuation(self):
        raw = "1-15：左，16-20：单，21-25：右"
        self.assertEqual(get_expected_side_for_tower(raw, 15), "左")
        self.assertIsNone(get_expected_side_for_tower(raw, 20))
        self.assertEqual(get_expected_side_for_tower(raw, 25), "右")

    def test_invalid_input(self):
        self.assertIsNone(get_expected_side_for_tower("", 1))
        self.assertIsNone(get_expected_side_for_tower(None, 1))
        self.assertIsNone(get_expected_side_for_tower("不知道", 1))
        self.assertIsNone(get_expected_side_for_tower("10-5:左", 7))

    def test_parse_mapping_shape(self):
        mapping = parse_side_mapping("1-15:左,16-20:单,21-25:右")
        self.assertEqual(mapping, [(1, 15, "左"), (16, 20, "单"), (21, 25, "右")])


if __name__ == "__main__":
    unittest.main()
