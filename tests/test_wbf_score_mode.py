# -*- coding: utf-8 -*-
"""任务 26 · WBF 分数调制（score_mode）回归与正确性测试。

关键约束：
1. score_mode="raw"（默认）时，输出必须与任务 05 原实现逐值一致 —— 保证
   任何旧调用方（评测脚本、历史复算）行为不变。
2. geo_agree 的数值必须落在 [0,1]，不破坏置信度的百分比语义。
3. geo_agree 与线性 n^beta 在 n<=cap 时排序等价 —— 这是采用几何形式的理由。

运行：python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wbf import wbf_fuse_boxes  # noqa: E402


def _legacy_wbf(boxes_list, scores_list, model_weights, iou_thr, skip_box_thr, conf_type):
    """任务 05 原实现的独立复刻，仅用于等价性对照（不含任何调制）。"""
    boxes = np.concatenate(boxes_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    mw = np.concatenate([np.full(len(s), w) for s, w in zip(scores_list, model_weights)])
    keep = scores > skip_box_thr
    boxes, scores, mw = boxes[keep], scores[keep], mw[keep]
    order = scores.argsort()[::-1]
    boxes, scores, mw = boxes[order], scores[order], mw[order]

    coord_sum, coord_w, max_score, reps = [], [], [], []
    for i in range(len(boxes)):
        bx, sc, w = boxes[i], scores[i], mw[i]
        best_iou, best_c = 0.0, -1
        for c in range(len(reps)):
            a, b = bx, reps[c]
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_c = iou, c
        if best_iou >= iou_thr:
            coord_sum[best_c] = coord_sum[best_c] + bx * sc * w
            coord_w[best_c] += sc * w
            max_score[best_c] = max(max_score[best_c], sc)
            if coord_w[best_c] > 0:
                reps[best_c] = coord_sum[best_c] / coord_w[best_c]
        else:
            coord_sum.append(bx * sc * w)
            coord_w.append(sc * w)
            max_score.append(sc)
            reps.append(bx.copy())

    out_b, out_s = [], []
    for c in range(len(reps)):
        if coord_w[c] <= 0:
            continue
        score = max_score[c] if conf_type == "max" else max_score[c]
        if score >= skip_box_thr:
            out_b.append(coord_sum[c] / coord_w[c])
            out_s.append(score)
    return np.array(out_b), np.array(out_s)


def _synthetic_case():
    """构造三模型、含重叠与孤立框的确定性输入。"""
    boxes = [
        np.array([[10.0, 10.0, 50.0, 50.0],
                  [100.0, 100.0, 140.0, 140.0]]),
        np.array([[11.0, 9.0, 51.0, 49.0],
                  [200.0, 200.0, 260.0, 260.0]]),
        np.array([[9.0, 11.0, 49.0, 51.0]]),
    ]
    scores = [
        np.array([0.80, 0.35]),
        np.array([0.70, 0.55]),
        np.array([0.60]),
    ]
    weights = [1.0, 1.0, 1.0]
    return boxes, scores, weights


class TestScoreModeRawIsBackwardCompatible(unittest.TestCase):
    """默认参数必须逐值复现任务 05 原实现。"""

    def test_raw_matches_legacy_conf_max(self):
        boxes, scores, weights = _synthetic_case()
        b_new, s_new = wbf_fuse_boxes(boxes, scores, weights,
                                      iou_thr=0.65, skip_box_thr=0.01, conf_type="max")
        b_old, s_old = _legacy_wbf(boxes, scores, weights, 0.65, 0.01, "max")
        self.assertEqual(len(b_new), len(b_old))
        np.testing.assert_allclose(b_new, b_old, rtol=0, atol=1e-12)
        np.testing.assert_allclose(s_new, s_old, rtol=0, atol=1e-12)

    def test_explicit_raw_equals_default(self):
        boxes, scores, weights = _synthetic_case()
        b1, s1 = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                skip_box_thr=0.01, conf_type="avg")
        b2, s2 = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                skip_box_thr=0.01, conf_type="avg", score_mode="raw")
        np.testing.assert_allclose(b1, b2, rtol=0, atol=0)
        np.testing.assert_allclose(s1, s2, rtol=0, atol=0)

    def test_zero_beta_geo_equals_raw(self):
        """geo_beta=0 时几何调制退化为原分数。"""
        boxes, scores, weights = _synthetic_case()
        b1, s1 = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                skip_box_thr=0.01, conf_type="avg")
        b2, s2 = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                skip_box_thr=0.01, conf_type="avg",
                                score_mode="geo_agree", geo_beta=0.0, geo_cap=8.0)
        np.testing.assert_allclose(b1, b2, rtol=0, atol=0)
        np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-12)


class TestGeoAgree(unittest.TestCase):
    def test_scores_stay_in_unit_interval(self):
        boxes, scores, weights = _synthetic_case()
        _, s = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                              skip_box_thr=0.0, conf_type="avg",
                              score_mode="geo_agree", geo_beta=0.412, geo_cap=8.0)
        self.assertTrue(np.all(s >= 0.0))
        self.assertTrue(np.all(s <= 1.0), f"score exceeded 1: {s.max()}")

    def test_boxes_unchanged(self):
        """调制只改分数，不得改坐标。"""
        boxes, scores, weights = _synthetic_case()
        b_raw, _ = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                  skip_box_thr=0.01, conf_type="avg")
        b_geo, _ = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                                  skip_box_thr=0.01, conf_type="avg",
                                  score_mode="geo_agree", geo_beta=0.412, geo_cap=8.0)
        self.assertEqual(len(b_raw), len(b_geo))
        np.testing.assert_allclose(b_raw, b_geo, rtol=0, atol=1e-12)

    def test_agreement_is_rewarded(self):
        """同分数下，簇内框数更多的检测应得到更高分数。"""
        # 两个位置：A 被 3 个模型覆盖，B 只被 1 个模型覆盖，两者 avg 分数接近
        boxes = [
            np.array([[10.0, 10.0, 50.0, 50.0]]),
            np.array([[10.5, 10.5, 50.5, 50.5]]),
            np.array([[9.5, 9.5, 49.5, 49.5]]),
        ]
        scores = [np.array([0.50]), np.array([0.50]), np.array([0.50])]
        weights = [1.0, 1.0, 1.0]
        b, s = wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                              skip_box_thr=0.01, conf_type="avg",
                              score_mode="geo_agree", geo_beta=0.412, geo_cap=8.0)
        self.assertEqual(len(b), 1)
        # n=3, cap=8, g=0.412: 0.5^0.588 * (3/8)^0.412
        expected = (0.5 ** (1 - 0.412)) * ((3 / 8) ** 0.412)
        self.assertAlmostEqual(float(s[0]), expected, places=10)

    def test_ordering_equivalent_to_linear_power(self):
        """n<=cap 时，geo 与线性 n^beta 的排序一致（beta = g/(1-g)）。"""
        g = 0.412
        beta = g / (1 - g)
        cap = 8.0
        ns = [1, 2, 3, 5, 8]
        avgs = [0.20, 0.35, 0.50, 0.65, 0.80]
        geo, lin = [], []
        for n in ns:
            for a in avgs:
                geo.append((a ** (1 - g)) * ((min(n, cap) / cap) ** g))
                lin.append(a * (n ** beta))
        order_geo = sorted(range(len(geo)), key=lambda i: geo[i])
        order_lin = sorted(range(len(lin)), key=lambda i: lin[i])
        self.assertEqual(order_geo, order_lin)

    def test_unknown_score_mode_raises(self):
        boxes, scores, weights = _synthetic_case()
        with self.assertRaises(ValueError):
            wbf_fuse_boxes(boxes, scores, weights, iou_thr=0.65,
                           skip_box_thr=0.01, conf_type="avg", score_mode="nope")


class TestConfigWiring(unittest.TestCase):
    """config.yaml 必须与 wbf.py 支持的取值一致。"""

    def test_config_values_are_valid(self):
        from config import load_config
        import os
        os.environ.setdefault("WHEATWEED_DEVICE", "cpu")
        cfg = load_config(str(ROOT / "config.yaml"))
        wbf = cfg["weed"]["wbf"]
        self.assertIn(wbf["conf_type"], ("avg", "max"))
        self.assertIn(wbf.get("score_mode", "raw"), ("raw", "geo_agree"))
        if wbf.get("score_mode") == "geo_agree":
            self.assertGreater(wbf["geo_beta"], 0.0)
            self.assertLess(wbf["geo_beta"], 1.0)
            self.assertGreater(wbf["geo_cap"], 0.0)

    def test_decision_conf_in_range(self):
        from config import load_config
        import os
        os.environ.setdefault("WHEATWEED_DEVICE", "cpu")
        cfg = load_config(str(ROOT / "config.yaml"))
        conf = cfg["weed"]["decision_conf"]
        self.assertGreater(conf, 0.0)
        self.assertLess(conf, 1.0)


if __name__ == "__main__":
    unittest.main()
