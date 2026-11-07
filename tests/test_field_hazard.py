# -*- coding: utf-8 -*-
"""按「实际拍摄面积」口径的危害等级追加/覆盖逻辑单元测试。

覆盖 src/hazard.py 新增的三个函数：
- to_square_meters：㎡ 原值、亩换算、各类非法输入返回 None
- compute_field_hazard：密度分级、边界、未检出、类别数升级、无效面积
- apply_field_area：无面积保持旧结构、有面积保留全部旧键、幂等性、
  不修改传入 result、缺 field_area 段优雅降级

测试使用自造 cfg dict 与自造最小 result dict，不加载模型、不读取真实图片、
不依赖 config.yaml 文件加载。

运行：python -B -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from advice import get_advice  # noqa: E402
from hazard import (  # noqa: E402
    apply_field_area,
    compute_field_hazard,
    compute_hazard_level,
    to_square_meters,
)


# 自造 cfg：含 hazard.weed/pest 旧阈值 + field_area 新阈值 + advice 文本
CFG = {
    "hazard": {
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
        "field_area": {
            "default_unit": "m2",
            "mu_in_m2": 666.6667,
            "units": ["m2", "mu"],
            "weed": {
                "light_max_per_m2": 0.5,
                "medium_max_per_m2": 2.0,
            },
            "pest": {
                "light_max_per_m2": 0.5,
                "medium_max_per_m2": 2.0,
                "major_classes_light_max": 2,
                "major_classes_medium_max": 5,
            },
            "prototype_note": (
                "按实际拍摄面积估算的检出密度分级；阈值为占位演示值，"
                "未经农学专家校准，不构成防治处方或正式农业决策依据。"
            ),
        },
    },
    "advice": {
        "weed": {
            "light": "杂草轻度建议",
            "medium": "杂草中度建议",
            "heavy": "杂草重度建议",
            "none": "杂草未检出建议",
            "disclaimer": "杂草免责声明",
        },
        "pest": {
            "light": "害虫轻度建议",
            "medium": "害虫中度建议",
            "heavy": "害虫重度建议",
            "none": "害虫未检出建议",
            "disclaimer": "害虫免责声明",
        },
    },
}

HAZARD_CFG = CFG["hazard"]


def make_result(n=7, width=1000, height=1000, mcc=0, task="weed"):
    """自造最小 result dict，结构与 weed_inference / pest_inference 返回一致。"""
    hazard = compute_hazard_level(
        n, width, height, HAZARD_CFG, task=task, major_class_count=mcc
    )
    hazard["decision_conf"] = 0.44
    advice = get_advice(
        hazard["level"], CFG["advice"], task=task, level_str=hazard["label"]
    )
    return {
        "num_detections": n,
        "width": width,
        "height": height,
        "decision_conf": 0.44,
        "major_class_count": mcc,
        "hazard": hazard,
        "advice": advice,
    }


class ToSquareMetersTests(unittest.TestCase):
    def test_m2_unchanged(self):
        self.assertEqual(to_square_meters(10, "m2", HAZARD_CFG), 10.0)
        self.assertEqual(to_square_meters(10.5, "m2", HAZARD_CFG), 10.5)

    def test_mu_conversion(self):
        self.assertAlmostEqual(
            to_square_meters(1, "mu", HAZARD_CFG), 666.6667, places=4
        )
        self.assertAlmostEqual(
            to_square_meters(2, "mu", HAZARD_CFG), 1333.3334, places=4
        )

    def test_invalid_values_return_none(self):
        for v in (None, 0, -1, -0.5, math.nan, math.inf, -math.inf,
                  "10", "abc", True, False, [], {}):
            with self.subTest(value=v):
                self.assertIsNone(to_square_meters(v, "m2", HAZARD_CFG))

    def test_invalid_units_return_none(self):
        for unit in (None, "", "亩", "acre", "MU", "M2", "m", 1):
            with self.subTest(unit=unit):
                self.assertIsNone(to_square_meters(10, unit, HAZARD_CFG))

    def test_missing_mu_in_m2_uses_default(self):
        cfg = {"field_area": {}}
        self.assertAlmostEqual(
            to_square_meters(1, "mu", cfg), 666.6667, places=4
        )


class FieldScenarioTests(unittest.TestCase):
    """用户截图场景：正式检测框数 / 实际拍摄面积。"""

    def test_7_in_10_m2_is_medium(self):
        r = make_result(n=7)
        hz = apply_field_area(r, CFG, field_area_value=10, field_area_unit="m2")["hazard"]
        self.assertEqual(hz["density_per_m2"], 0.70)
        self.assertEqual(hz["level"], "中")
        self.assertIn("10.00", hz["description"])
        self.assertIn("0.70", hz["description"])

    def test_7_in_100_m2_is_light(self):
        r = make_result(n=7)
        hz = apply_field_area(r, CFG, field_area_value=100, field_area_unit="m2")["hazard"]
        self.assertEqual(hz["density_per_m2"], 0.07)
        self.assertEqual(hz["level"], "轻")

    def test_7_in_2_m2_is_heavy(self):
        r = make_result(n=7)
        hz = apply_field_area(r, CFG, field_area_value=2, field_area_unit="m2")["hazard"]
        self.assertEqual(hz["density_per_m2"], 3.50)
        self.assertEqual(hz["level"], "重")

    def test_mu_input_goes_through_conversion(self):
        # 0.015 亩 = 10.0000005 ㎡，7 框 -> 0.70 -> 中
        r = make_result(n=7)
        out = apply_field_area(r, CFG, field_area_value=0.015, field_area_unit="mu")
        hz = out["hazard"]
        self.assertAlmostEqual(hz["field_area_m2"], 10.0, places=3)
        self.assertEqual(hz["density_per_m2"], 0.70)
        self.assertEqual(hz["level"], "中")
        self.assertEqual(hz["field_area_value"], 0.015)
        self.assertEqual(hz["field_area_unit"], "mu")


class BoundaryTests(unittest.TestCase):
    """轻 <0.5，中 0.5–2，重 >=2 个/㎡（`<` 轻 / `>=` 中 / `>=` 重）。"""

    def test_below_light_is_light(self):
        hz = compute_field_hazard(4, 10.0, HAZARD_CFG)  # 0.4
        self.assertEqual(hz["density_per_m2"], 0.4)
        self.assertEqual(hz["level"], "轻")

    def test_density_at_0_5_is_medium(self):
        hz = compute_field_hazard(5, 10.0, HAZARD_CFG)  # 0.5
        self.assertEqual(hz["density_per_m2"], 0.5)
        self.assertEqual(hz["level"], "中")

    def test_density_just_below_2_is_medium(self):
        hz = compute_field_hazard(19, 10.0, HAZARD_CFG)  # 1.9
        self.assertEqual(hz["density_per_m2"], 1.9)
        self.assertEqual(hz["level"], "中")

    def test_density_at_2_is_heavy(self):
        hz = compute_field_hazard(20, 10.0, HAZARD_CFG)  # 2.0
        self.assertEqual(hz["density_per_m2"], 2.0)
        self.assertEqual(hz["level"], "重")


class NoAreaTests(unittest.TestCase):
    def _out(self, **kw):
        r = make_result()
        return r, apply_field_area(r, CFG, field_area_value=None, **kw)

    def test_displayable_false_and_reason(self):
        _, out = self._out()
        hz = out["hazard"]
        self.assertIs(hz["displayable"], False)
        self.assertEqual(hz["reason"], "field_area_not_provided")
        self.assertEqual(hz["assessment_basis"], "not_available")
        self.assertIsNone(hz["field_area_m2"])
        self.assertIsNone(hz["density_per_m2"])

    def test_all_legacy_keys_preserved_with_same_values_and_types(self):
        r, out = self._out()
        old, hz = r["hazard"], out["hazard"]
        for key in ("level", "label", "severity", "density_per_mp", "none",
                    "thresholds"):
            with self.subTest(key=key):
                self.assertIn(key, hz)
                self.assertEqual(type(hz[key]), type(old[key]))
                self.assertEqual(hz[key], old[key])
        # note 仍是字符串键，仅为「未填写面积」说明文本
        self.assertIn("note", hz)
        self.assertIsInstance(hz["note"], str)
        self.assertTrue(hz["note"])

    def test_advice_keeps_original_text_and_computed_false(self):
        r, out = self._out()
        adv = out["advice"]
        self.assertIs(adv["computed"], False)
        self.assertEqual(adv["reason"], "field_area_not_provided")
        self.assertEqual(adv["advice"], r["advice"]["advice"])
        self.assertIsInstance(adv["advice"], str)
        self.assertEqual(adv["disclaimer"], r["advice"]["disclaimer"])
        self.assertIsInstance(adv["disclaimer"], str)
        self.assertEqual(adv["level"], r["advice"]["level"])

    def test_invalid_area_values_behave_like_no_area(self):
        for v in (0, -1, math.nan, math.inf, "10", None):
            with self.subTest(value=v):
                hz = apply_field_area(make_result(), CFG, field_area_value=v)["hazard"]
                self.assertIs(hz["displayable"], False)
                self.assertEqual(hz["reason"], "field_area_not_provided")

    def test_invalid_unit_behaves_like_no_area(self):
        hz = apply_field_area(
            make_result(), CFG, field_area_value=10, field_area_unit="亩"
        )["hazard"]
        self.assertIs(hz["displayable"], False)


class WithAreaTests(unittest.TestCase):
    def test_displayable_true_and_computed_advice(self):
        out = apply_field_area(make_result(), CFG, field_area_value=10)
        hz = out["hazard"]
        self.assertIs(hz["displayable"], True)
        self.assertEqual(hz["assessment_basis"], "field_area")
        self.assertEqual(hz["reason"], "ok")
        self.assertIs(out["advice"]["computed"], True)
        self.assertEqual(out["advice"]["reason"], "ok")

    def test_density_per_mp_still_present(self):
        hz = apply_field_area(make_result(), CFG, field_area_value=10)["hazard"]
        self.assertIn("density_per_mp", hz)
        self.assertIsInstance(hz["density_per_mp"], float)
        self.assertIn("image_area_mp", hz)

    def test_thresholds_have_both_per_mp_and_per_m2(self):
        hz = apply_field_area(make_result(), CFG, field_area_value=10)["hazard"]
        th = hz["thresholds"]
        self.assertIn("light_max_per_mp", th)
        self.assertIn("medium_max_per_mp", th)
        self.assertIn("light_max_per_m2", th)
        self.assertIn("medium_max_per_m2", th)
        self.assertEqual(th["basis"], "field_area")

    def test_new_level_overrides_legacy(self):
        # 7 框在 1000x1000 旧口径：7 / 1MP = 7 <10 -> 轻；
        # 新口径 7 / 10㎡ = 0.70 -> 中
        r = make_result(n=7)
        self.assertEqual(r["hazard"]["level"], "轻")
        hz = apply_field_area(r, CFG, field_area_value=10)["hazard"]
        self.assertEqual(hz["level"], "中")
        self.assertEqual(hz["label"], "中")


class IdempotencyTests(unittest.TestCase):
    def test_no_area_then_area_equals_direct_area(self):
        r = make_result()
        direct = apply_field_area(r, CFG, field_area_value=10, field_area_unit="m2")
        one = apply_field_area(r, CFG)  # 无面积
        two = apply_field_area(one, CFG, field_area_value=10, field_area_unit="m2")
        self.assertEqual(two["hazard"], direct["hazard"])
        self.assertEqual(two["advice"], direct["advice"])

    def test_area_then_other_area_is_recomputed(self):
        r = make_result()
        a = apply_field_area(r, CFG, field_area_value=10, field_area_unit="m2")
        b = apply_field_area(a, CFG, field_area_value=100, field_area_unit="m2")
        direct = apply_field_area(r, CFG, field_area_value=100, field_area_unit="m2")
        self.assertEqual(b["hazard"], direct["hazard"])
        self.assertEqual(b["advice"], direct["advice"])
        self.assertEqual(b["hazard"]["level"], "轻")


class NoMutationTests(unittest.TestCase):
    def test_input_result_is_not_modified(self):
        r = make_result()
        hazard_obj, advice_obj = r["hazard"], r["advice"]
        hazard_snapshot = dict(r["hazard"])
        advice_snapshot = dict(r["advice"])
        apply_field_area(r, CFG, field_area_value=10, field_area_unit="m2")
        self.assertIs(r["hazard"], hazard_obj)
        self.assertIs(r["advice"], advice_obj)
        self.assertEqual(r["hazard"], hazard_snapshot)
        self.assertEqual(r["advice"], advice_snapshot)


class PestClassCountTests(unittest.TestCase):
    def test_low_density_but_many_classes_is_heavy(self):
        # 1 框 / 1000㎡ = 0.001（极低密度），但 6 类 > medium_max(5) -> 重
        r = make_result(n=1, mcc=6, task="pest")
        hz = apply_field_area(
            r, CFG, task="pest", field_area_value=1000, field_area_unit="m2"
        )["hazard"]
        self.assertLess(hz["density_per_m2"], 0.5)
        self.assertEqual(hz["level"], "重")

    def test_three_classes_low_density_is_medium(self):
        r = make_result(n=1, mcc=3, task="pest")
        hz = apply_field_area(
            r, CFG, task="pest", field_area_value=1000, field_area_unit="m2"
        )["hazard"]
        self.assertEqual(hz["level"], "中")


class ZeroDetectionTests(unittest.TestCase):
    def test_no_detection_marks_none_and_label(self):
        out = apply_field_area(
            make_result(n=0), CFG, field_area_value=10, field_area_unit="m2"
        )
        hz = out["hazard"]
        self.assertIs(hz["none"], True)
        self.assertIn("未检出", hz["label"])
        self.assertEqual(hz["density_per_m2"], 0.0)
        self.assertEqual(hz["severity"], 0)


class ComputeFieldHazardDirectTests(unittest.TestCase):
    def test_invalid_area_returns_undisplayable_fragment(self):
        hz = compute_field_hazard(7, None, HAZARD_CFG)
        self.assertIs(hz["displayable"], False)
        self.assertIsNone(hz["level"])
        self.assertEqual(hz["label"], "未评估（未填写拍摄面积）")
        self.assertEqual(hz["reason"], "field_area_not_provided")

    def test_valid_area_returns_displayable_fragment(self):
        hz = compute_field_hazard(7, 10.0, HAZARD_CFG)
        self.assertIs(hz["displayable"], True)
        self.assertEqual(hz["density_per_m2"], 0.70)
        self.assertEqual(hz["level"], "中")
        self.assertEqual(hz["thresholds"]["basis"], "field_area")
        self.assertIn(hz["note"], HAZARD_CFG["field_area"]["prototype_note"])

    def test_all_required_keys_present(self):
        hz = compute_field_hazard(7, 10.0, HAZARD_CFG)
        for key in ("displayable", "field_area_m2", "density_per_m2", "level",
                    "label", "severity", "description", "none", "thresholds",
                    "note", "reason", "num_detections", "assessment_basis"):
            with self.subTest(key=key):
                self.assertIn(key, hz)


class DegradeTests(unittest.TestCase):
    def test_missing_field_area_section_degrades_to_no_area(self):
        cfg = {
            "hazard": {
                "weed": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0},
            },
            "advice": CFG["advice"],
        }
        out = apply_field_area(make_result(), cfg, field_area_value=10)
        self.assertIs(out["hazard"]["displayable"], False)
        self.assertEqual(out["hazard"]["reason"], "field_area_not_provided")
        self.assertIs(out["advice"]["computed"], False)


if __name__ == "__main__":
    unittest.main()
