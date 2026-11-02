# -*- coding: utf-8 -*-
"""防治建议等级映射测试（不加载模型）。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from advice import get_advice  # noqa: E402


ADVICE_CFG = {
    "weed": {
        "light": "LIGHT",
        "medium": "MEDIUM",
        "heavy": "HEAVY",
        "none": "NONE",
        "disclaimer": "D",
    },
    "pest": {
        "light": "P_LIGHT",
        "medium": "P_MEDIUM",
        "heavy": "P_HEAVY",
        "none": "P_NONE",
        "disclaimer": "D",
    },
}


class AdviceMappingTests(unittest.TestCase):
    def test_no_detection_uses_none_text(self):
        self.assertEqual(get_advice("轻", ADVICE_CFG, task="weed", level_str="轻（未检出）")["advice"], "NONE")
        self.assertEqual(get_advice("轻", ADVICE_CFG, task="pest", level_str="轻（未检出）")["advice"], "P_NONE")

    def test_normal_levels(self):
        self.assertEqual(get_advice("轻", ADVICE_CFG, task="weed")["advice"], "LIGHT")
        self.assertEqual(get_advice("中", ADVICE_CFG, task="weed")["advice"], "MEDIUM")
        self.assertEqual(get_advice("重", ADVICE_CFG, task="weed")["advice"], "HEAVY")

    def test_missing_none_falls_back_to_light(self):
        cfg = {"weed": {"light": "LIGHT", "disclaimer": "D"}}
        self.assertEqual(get_advice("轻", cfg, task="weed", level_str="轻（未检出）")["advice"], "LIGHT")

    def test_level_and_disclaimer_preserved(self):
        out = get_advice("轻", ADVICE_CFG, task="weed", level_str="轻（未检出）")
        self.assertEqual(out["level"], "轻（未检出）")
        self.assertEqual(out["disclaimer"], "D")


if __name__ == "__main__":
    unittest.main()
