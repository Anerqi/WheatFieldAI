# -*- coding: utf-8 -*-
"""不依赖模型权重的单元/回归测试：业务决策阈值、危害等级、WBF 基本行为。

运行：python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from decision import split_by_conf, get_decision_conf, class_counts  # noqa: E402
from hazard import compute_hazard_level  # noqa: E402
from wbf import wbf_fuse_boxes  # noqa: E402
from pest_wbf import classwise_wbf, load_class_configs  # noqa: E402


def _cfg():
    import os
    os.environ.setdefault("WHEATWEED_DEVICE", "cpu")
    return load_config()


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        import torch
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return len(self.conf)


class _FakeYOLO:
    """模拟 Ultralytics 在 conf=0.001 下的输出：大量低置信噪声 + 少量真实目标。"""

    def __init__(self, n_noise=150, n_real=0, n_classes=1, seed=0):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(0, 900, size=(n_noise + n_real, 2))
        self.xyxy = np.hstack([xy, xy + 40]).tolist()
        self.conf = list(rng.uniform(0.001, 0.2, n_noise)) + [0.8] * n_real
        self.cls = list(rng.integers(0, n_classes, n_noise + n_real))

    def predict(self, img, **kw):
        assert kw["conf"] <= 0.01, "推理阈值应保持评测口径"
        return [type("R", (), {"boxes": _FakeBoxes(self.xyxy, self.conf, self.cls)})()]


class _FakeWeedManager:
    def __init__(self, model):
        self._m = model

    def device_str(self):
        return "cpu"

    def get_yolo11(self):
        return self._m


class _FakePestManager:
    device = "cpu"

    def __init__(self, model, names):
        self._m, self._names = model, names

    def get(self, key):
        return self._m

    def class_names(self):
        return self._names


class DecisionConfTests(unittest.TestCase):
    def test_config_has_decision_conf(self):
        cfg = _cfg()
        self.assertEqual(get_decision_conf(cfg["weed"]), 0.25)
        self.assertEqual(get_decision_conf(cfg["pest"]), 0.25)

    def test_invalid_decision_conf_rejected(self):
        with self.assertRaises(ValueError):
            get_decision_conf({"decision_conf": 1.5})

    def test_split_keeps_order_and_boundary(self):
        dets = [{"confidence": c, "category_name": "a"} for c in (0.9, 0.25, 0.2499, 0.01)]
        decided, cand = split_by_conf(dets, 0.25)
        self.assertEqual([d["confidence"] for d in decided], [0.9, 0.25])
        self.assertEqual([d["confidence"] for d in cand], [0.2499, 0.01])

    def test_class_counts_sorted(self):
        dets = [{"category_name": n} for n in ("b", "a", "b")]
        self.assertEqual(list(class_counts(dets).items()), [("b", 2), ("a", 1)])


class NoiseRegressionTests(unittest.TestCase):
    """回归：无目标图像的大量低置信候选框不得把危害等级推到"重"。"""

    img = np.zeros((1080, 810, 3), dtype=np.uint8)

    def test_weed_noise_only_is_light(self):
        from weed_inference import infer_weed_single_image
        r = infer_weed_single_image(self.img, _cfg(), _FakeWeedManager(_FakeYOLO(150, 0)), mode="yolo11")
        self.assertEqual(r["num_detections"], 0)
        self.assertEqual(r["num_candidates_total"], 150)
        self.assertEqual(r["hazard"]["severity"], 0)
        self.assertEqual(len(r["low_conf_candidates"]), 150)

    def test_weed_real_targets_counted(self):
        from weed_inference import infer_weed_single_image
        r = infer_weed_single_image(self.img, _cfg(), _FakeWeedManager(_FakeYOLO(150, 30)), mode="yolo11")
        self.assertEqual(r["num_detections"], 30)
        self.assertEqual(r["hazard"]["level"], "中")  # 30 / 0.8748MP ≈ 34.3

    def test_pest_noise_only_is_light(self):
        from pest_inference import infer_pest_single_image
        names = [str(i) for i in range(32)]
        r = infer_pest_single_image(self.img, _cfg(), _FakePestManager(_FakeYOLO(80, 0, 32), names), mode="fast")
        self.assertEqual(r["num_detections"], 0)
        self.assertEqual(r["major_class_count"], 0)
        self.assertEqual(r["class_counts"], {})
        self.assertEqual(r["hazard"]["severity"], 0)


class HazardTests(unittest.TestCase):
    def setUp(self):
        self.h = _cfg()["hazard"]

    def test_weed_thresholds(self):
        # 1000x1000 = 1 MP
        self.assertEqual(compute_hazard_level(9, 1000, 1000, self.h, "weed")["level"], "轻")
        self.assertEqual(compute_hazard_level(10, 1000, 1000, self.h, "weed")["level"], "中")
        self.assertEqual(compute_hazard_level(40, 1000, 1000, self.h, "weed")["level"], "重")

    def test_pest_class_count_rule(self):
        self.assertEqual(compute_hazard_level(3, 1000, 1000, self.h, "pest", major_class_count=1)["level"], "轻")
        self.assertEqual(compute_hazard_level(3, 1000, 1000, self.h, "pest", major_class_count=2)["level"], "中")
        self.assertEqual(compute_hazard_level(3, 1000, 1000, self.h, "pest", major_class_count=5)["level"], "重")


class WbfTests(unittest.TestCase):
    def test_weed_wbf_merges_overlapping(self):
        b = np.array([[0, 0, 10, 10]], float)
        fb, fs = wbf_fuse_boxes([b, b + 0.5, b + 50], [np.array([0.9]), np.array([0.8]), np.array([0.7])],
                                [1, 1, 1], iou_thr=0.65, skip_box_thr=0.01, conf_type="max")
        self.assertEqual(len(fb), 2)
        self.assertAlmostEqual(float(fs.max()), 0.9)

    def test_classwise_config_covers_all_classes(self):
        cfg = _cfg()
        cc = load_class_configs(cfg["pest"]["classwise_config"])
        self.assertEqual(sorted(cc), list(range(32)))
        pred = (np.array([[0, 0, 10, 10]], float), np.array([3]), np.array([0.9]))
        fb, fc, fs = classwise_wbf([pred, pred, pred], cc)
        self.assertEqual(fc.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
