# -*- coding: utf-8 -*-
"""危害等级边界单元测试（不加载模型、不读取真实图片）。

覆盖 src/hazard.py 害虫规则的类别数边界：
- 主要类别数 <= 2 -> 轻；3-5 -> 中；> 5（即 >= 6）-> 重
- 密度阈值仍为 >= 5 中、>= 15 重
并覆盖未检出时新增的 ``none`` 标记。

运行：python -B -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hazard import compute_hazard_level  # noqa: E402


# 与 config.yaml hazard 段一致的阈值；1000x1000 = 1 MP
HAZARD_CFG = {
    "weed": {
        "light_max_per_mp": 10.0,
        "medium_max_per_mp": 40.0,
    },
    "pest": {
        "light_max_per_mp": 5.0,
        "medium_max_per_mp": 15.0,
        "major_classes_light_max": 2,
        "major_classes_medium_max": 5,
    },
}


def level(num, classes=0, task="pest", width=1000, height=1000):
    return compute_hazard_level(
        num, width, height, HAZARD_CFG, task=task, major_class_count=classes
    )


class NoneFlagTests(unittest.TestCase):
    def test_pest_no_detection_keeps_light_and_marks_none(self):
        r = level(0, 0)
        self.assertTrue(r["none"])
        self.assertEqual(r["level"], "轻")
        self.assertEqual(r["label"], "轻（未检出）")
        self.assertEqual(r["severity"], 0)

    def test_weed_no_detection_keeps_light_and_marks_none(self):
        r = level(0, task="weed")
        self.assertTrue(r["none"])
        self.assertEqual(r["level"], "轻")
        self.assertEqual(r["label"], "轻（未检出）")
        self.assertEqual(r["severity"], 0)

    def test_detection_clears_none(self):
        self.assertFalse(level(1, 1)["none"])
        self.assertFalse(level(1, task="weed")["none"])

    def test_existing_keys_preserved(self):
        r = level(0, 0)
        for key in (
            "level",
            "label",
            "description",
            "severity",
            "density_per_mp",
            "major_class_count",
            "num_detections",
            "image_area_mp",
            "thresholds",
            "note",
        ):
            self.assertIn(key, r)


class PestDensityBoundaryTests(unittest.TestCase):
    """类别数 0 时只由密度决定；1 MP 下 num_detections 数值即密度。

    这里用浮点检测数只为精确命中 4.99 / 14.99 边界，函数本身只做除法与比较。
    """

    def test_density_below_light(self):
        r = level(4.99, 0)
        self.assertEqual(r["density_per_mp"], 4.99)
        self.assertEqual(r["level"], "轻")

    def test_density_at_light_boundary_is_medium(self):
        r = level(5.0, 0)
        self.assertEqual(r["density_per_mp"], 5.0)
        self.assertEqual(r["level"], "中")

    def test_density_below_heavy(self):
        r = level(14.99, 0)
        self.assertEqual(r["density_per_mp"], 14.99)
        self.assertEqual(r["level"], "中")

    def test_density_at_heavy_boundary_is_heavy(self):
        r = level(15.0, 0)
        self.assertEqual(r["density_per_mp"], 15.0)
        self.assertEqual(r["level"], "重")

    def test_integer_counts_round_trip(self):
        self.assertEqual(level(4, 0)["level"], "轻")
        self.assertEqual(level(5, 0)["level"], "中")
        self.assertEqual(level(14, 0)["level"], "中")
        self.assertEqual(level(15, 0)["level"], "重")


class PestClassCountBoundaryTests(unittest.TestCase):
    """低密度（1 个/MP）下等级只由类别数决定。"""

    def test_zero_to_two_classes_light(self):
        for n in (0, 1, 2):
            with self.subTest(classes=n):
                self.assertEqual(level(1, n)["level"], "轻")

    def test_three_to_five_classes_medium(self):
        for n in (3, 5):
            with self.subTest(classes=n):
                self.assertEqual(level(1, n)["level"], "中")

    def test_six_classes_heavy(self):
        self.assertEqual(level(1, 6)["level"], "重")


class PestCombinationTests(unittest.TestCase):
    def test_two_classes_density_4_17_is_light(self):
        # 实测场景：800x600 = 0.48 MP，2 框 -> 密度 4.17，2 类 -> 轻
        r = compute_hazard_level(
            2, 800, 600, HAZARD_CFG, task="pest", major_class_count=2
        )
        self.assertEqual(r["density_per_mp"], 4.17)
        self.assertEqual(r["level"], "轻")
        self.assertEqual(r["label"], "轻")

    def test_five_classes_low_density_is_medium(self):
        self.assertEqual(level(1, 5)["level"], "中")

    def test_six_classes_low_density_is_heavy(self):
        self.assertEqual(level(1, 6)["level"], "重")

    def test_high_density_with_one_class_is_heavy(self):
        self.assertEqual(level(15.0, 1)["level"], "重")

    def test_three_classes_with_low_density_is_medium(self):
        self.assertEqual(level(4.17, 3)["level"], "中")


if __name__ == "__main__":
    unittest.main()
