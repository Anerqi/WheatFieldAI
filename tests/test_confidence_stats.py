# -*- coding: utf-8 -*-
"""置信度统计与业务阈值拆分的单元测试（不加载任何模型权重）。

覆盖：
- decision.confidence_stats：空列表、单条、4 位小数舍入、全低置信
- decision.split_by_conf：空列表、全部低于阈值、混合、阈值边界（== decision_conf 算正式检测）
- 推理返回 dict 的字段契约：max_decided_confidence 只统计正式检测，无正式检测时为 None；
  旧字段 max_confidence 仍反映原始最高置信度（向后兼容）

运行：python -B -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from decision import confidence_stats, get_decision_conf, split_by_conf  # noqa: E402


def _dets(*confs):
    """构造最小检测 dict 列表（decision 模块只读 confidence 键）。"""
    return [{"confidence": c, "category_name": "a"} for c in confs]


def _cfg():
    os.environ.setdefault("WHEATWEED_DEVICE", "cpu")
    from config import load_config
    return load_config()


class ConfidenceStatsTests(unittest.TestCase):
    def test_empty_returns_none_pair(self):
        self.assertEqual(confidence_stats([]), (None, None))

    def test_single_detection(self):
        self.assertEqual(confidence_stats(_dets(0.987654)), (0.9877, 0.9877))

    def test_mean_and_max_rounded_to_4_decimals(self):
        mean, mx = confidence_stats(_dets(0.123456, 0.987654))
        self.assertEqual(mean, 0.5556)  # (0.123456 + 0.987654) / 2 = 0.555555
        self.assertEqual(mx, 0.9877)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(mx, float)

    def test_all_low_confidence_still_returns_values(self):
        # 统计函数本身不做阈值过滤：过滤由 split_by_conf 负责
        mean, mx = confidence_stats(_dets(0.05, 0.001))
        self.assertAlmostEqual(mean, 0.0255, places=4)
        self.assertEqual(mx, 0.05)


class SplitByConfTests(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(split_by_conf([], 0.5), ([], []))

    def test_all_below_threshold(self):
        dets = _dets(0.4999, 0.2, 0.001)
        decided, candidates = split_by_conf(dets, 0.5)
        self.assertEqual(decided, [])
        self.assertEqual([d["confidence"] for d in candidates], [0.4999, 0.2, 0.001])
        self.assertEqual(confidence_stats(decided), (None, None))  # 无正式检测 -> max 为 None

    def test_all_at_or_above_threshold(self):
        dets = _dets(0.9, 0.5)
        decided, candidates = split_by_conf(dets, 0.5)
        self.assertEqual([d["confidence"] for d in decided], [0.9, 0.5])
        self.assertEqual(candidates, [])

    def test_mixed_keeps_order(self):
        dets = _dets(0.9, 0.49, 0.6, 0.1)
        decided, candidates = split_by_conf(dets, 0.5)
        self.assertEqual([d["confidence"] for d in decided], [0.9, 0.6])
        self.assertEqual([d["confidence"] for d in candidates], [0.49, 0.1])
        self.assertEqual(confidence_stats(decided)[1], 0.9)

    def test_boundary_equal_counts_as_decided(self):
        dets = _dets(0.5, 0.499999)
        decided, candidates = split_by_conf(dets, 0.5)
        self.assertEqual([d["confidence"] for d in decided], [0.5])
        self.assertEqual([d["confidence"] for d in candidates], [0.499999])

    def test_partition_is_disjoint_and_complete(self):
        dets = _dets(0.9, 0.5, 0.3, 0.1)
        decided, candidates = split_by_conf(dets, 0.5)
        self.assertEqual(len(decided) + len(candidates), len(dets))
        self.assertEqual(len(dets), 4)  # 输入列表未被修改


class _Tensor:
    """张量替身：只实现推理代码用到的 .cpu().numpy() 与 len()。"""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values

    def __len__(self):
        return len(self._values)


class _Boxes:
    def __init__(self, confs):
        n = len(confs)
        self.xyxy = _Tensor([[0.0, 0.0, 10.0, 10.0]] * n)
        self.conf = _Tensor(confs)
        self.cls = _Tensor([0] * n)

    def __len__(self):
        return len(self.conf)


class _StubModel:
    """模拟 Ultralytics 单图输出：置信度可控，不加载真实权重。"""

    def __init__(self, confs):
        self._boxes = _Boxes(confs)

    def predict(self, img, **kwargs):
        return [type("R", (), {"boxes": self._boxes})()]


class _StubWeedManager:
    def __init__(self, model):
        self._model = model

    def device_str(self):
        return "cpu"

    def get_yolo11(self):
        return self._model


class _StubPestManager:
    device = "cpu"

    def __init__(self, model):
        self._model = model

    def get(self, key):
        return self._model

    def class_names(self):
        return ["class_0"]


class InferenceConfidenceFieldTests(unittest.TestCase):
    """推理返回 dict 的置信度字段契约（单模型路径，不加载权重）。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg()
        cls.img = np.zeros((1080, 810, 3), dtype=np.uint8)

    def _weed(self, confs):
        from weed_inference import infer_weed_single_image
        return infer_weed_single_image(self.img, self.cfg,
                                       _StubWeedManager(_StubModel(confs)), mode="yolo11")

    def _pest(self, confs):
        from pest_inference import infer_pest_single_image
        return infer_pest_single_image(self.img, self.cfg,
                                       _StubPestManager(_StubModel(confs)), mode="fast")

    def test_weed_no_decided_gives_none(self):
        r = self._weed([0.20, 0.05])
        self.assertEqual(r["num_detections"], 0)
        self.assertIsNone(r["max_decided_confidence"])
        self.assertEqual(r["max_confidence"], 0.2)  # 旧字段仍反映原始最高

    def test_weed_mixed(self):
        r = self._weed([0.91, 0.62, 0.20])
        self.assertEqual(r["num_detections"], 2)
        self.assertEqual(r["max_decided_confidence"], 0.91)
        self.assertAlmostEqual(r["mean_confidence"], 0.765, places=4)
        self.assertEqual(r["max_confidence"], 0.91)

    def test_weed_boundary_equals_decision_conf(self):
        # 任务 26：阈值从配置读取，避免调阈值时测试硬编码失效；
        # 断言的是「== decision_conf 计入正式检测」这一契约本身。
        conf = get_decision_conf(self.cfg["weed"])
        r = self._weed([conf, conf - 0.01])
        self.assertEqual(r["num_detections"], 1)
        self.assertAlmostEqual(r["max_decided_confidence"], conf, places=6)
        self.assertEqual(len(r["low_conf_candidates"]), 1)

    def test_pest_no_decided_gives_none(self):
        r = self._pest([0.29, 0.05])
        self.assertEqual(r["num_detections"], 0)
        self.assertIsNone(r["max_decided_confidence"])
        self.assertEqual(r["max_confidence"], 0.29)

    def test_pest_mixed(self):
        r = self._pest([0.88, 0.31, 0.10])
        self.assertEqual(r["num_detections"], 2)
        self.assertEqual(r["max_decided_confidence"], 0.88)

    def test_pest_boundary_equals_decision_conf(self):
        r = self._pest([0.30, 0.299])
        self.assertEqual(r["num_detections"], 1)
        self.assertEqual(r["max_decided_confidence"], 0.3)


if __name__ == "__main__":
    unittest.main()
