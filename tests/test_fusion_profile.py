# -*- coding: utf-8 -*-
"""杂草融合参数档位（六模型 / 五模型退回）的单元测试（不加载任何模型权重）。

回归目标：D-FINE-S 权重或 onnxruntime 缺失时，融合退回五模型，且 iou_thr 与
decision_conf 必须成套切换到任务 26 五模型尺度（0.65 / 0.40），而不是沿用为
六模型校准的 0.68 / 0.44；权重在但会话创建失败时，空结果不得作为第 6 路参与融合。

运行：python -B -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_profile import DEFAULT_FIVE_MODEL_FALLBACK, resolve_fusion_profile  # noqa: E402


def _cfg():
    os.environ.setdefault("WHEATWEED_DEVICE", "cpu")
    from config import load_config
    return load_config()


def _weed_cfg(dfine_enabled=True, fallback=None):
    wbf = {"iou_thr": 0.68, "decision_conf_unused": None}
    if fallback is not None:
        wbf["five_model_fallback"] = fallback
    return {"wbf": wbf, "decision_conf": 0.44, "dfine": {"enabled": dfine_enabled}}


class ResolveProfileTests(unittest.TestCase):
    def test_six_model_uses_main_params(self):
        p = resolve_fusion_profile(_weed_cfg(), dfine_used=True)
        self.assertEqual(p["profile"], "six_model")
        self.assertEqual((p["iou_thr"], p["decision_conf"]), (0.68, 0.44))
        self.assertFalse(p["degraded"])

    def test_fallback_uses_configured_five_model_params(self):
        p = resolve_fusion_profile(
            _weed_cfg(fallback={"iou_thr": 0.6, "decision_conf": 0.35}), dfine_used=False
        )
        self.assertEqual(p["profile"], "five_model_fallback")
        self.assertEqual((p["iou_thr"], p["decision_conf"]), (0.6, 0.35))
        self.assertTrue(p["degraded"])

    def test_fallback_defaults_when_section_missing(self):
        p = resolve_fusion_profile(_weed_cfg(), dfine_used=False)
        self.assertEqual(p["iou_thr"], DEFAULT_FIVE_MODEL_FALLBACK["iou_thr"])
        self.assertEqual(p["decision_conf"], DEFAULT_FIVE_MODEL_FALLBACK["decision_conf"])

    def test_dfine_disabled_is_not_degraded(self):
        p = resolve_fusion_profile(_weed_cfg(dfine_enabled=False), dfine_used=False)
        self.assertEqual(p["profile"], "five_model_fallback")
        self.assertFalse(p["degraded"])

    def test_invalid_fallback_decision_conf_rejected(self):
        with self.assertRaises(ValueError):
            resolve_fusion_profile(_weed_cfg(fallback={"decision_conf": 1.2}), dfine_used=False)

    def test_shipped_config_profiles(self):
        weed = _cfg()["weed"]
        six = resolve_fusion_profile(weed, dfine_used=True)
        five = resolve_fusion_profile(weed, dfine_used=False)
        self.assertEqual((six["iou_thr"], six["decision_conf"]), (0.68, 0.44))
        self.assertEqual((five["iou_thr"], five["decision_conf"]), (0.65, 0.40))


# ---------------------------------------------------------------------------
# 端到端（假模型）：infer_weed_single_image 在各种 D-FINE 状态下选对档位
# ---------------------------------------------------------------------------

def _fake_boxes(n=6, conf=0.9, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 800, size=(n, 2))
    return np.hstack([xy, xy + 40]).astype(float), np.full(n, conf, dtype=float)


class _FakeDfine:
    def __init__(self, ensure_ok=True, infer_ok=True):
        self.ensure_ok, self.infer_ok = ensure_ok, infer_ok
        self.last_ok = False
        self.infer_calls = 0

    def available(self):
        return True

    def _ensure(self):
        return self.ensure_ok

    def infer(self, img):
        self.infer_calls += 1
        self.last_ok = self.ensure_ok and self.infer_ok
        if not self.last_ok:
            return np.zeros((0, 4)), np.zeros((0,))
        return _fake_boxes()


class _FakeManager:
    def __init__(self, dfine):
        self._dfine = dfine

    def device_str(self):
        return "cpu"

    def get_yolo11(self):
        return object()

    def get_yolo11_variant(self, name):
        return object()

    def get_yolox(self, size):
        return object()

    def get_dfine_onnx(self):
        return self._dfine


class FusionFallbackTests(unittest.TestCase):
    img = np.zeros((960, 960, 3), dtype=np.uint8)

    def _run(self, dfine):
        import weed_inference as wi
        fake = lambda *a, **k: _fake_boxes()  # noqa: E731
        with mock.patch.object(wi, "run_yolo11_single", fake), \
                mock.patch.object(wi, "run_yolox_model", fake):
            return wi.infer_weed_single_image(self.img, _cfg(), _FakeManager(dfine), mode="fusion")

    def _assert_five(self, r):
        self.assertNotIn("D-FINE-S", r["models_used"])
        self.assertEqual(r["wbf_params"]["profile"], "five_model_fallback")
        self.assertEqual(r["wbf_params"]["iou_thr"], 0.65)
        self.assertEqual(r["decision_conf"], 0.40)
        self.assertEqual(len(r["wbf_params"]["weights"]), 5)
        self.assertTrue(r["fusion_degraded"])

    def test_six_model_when_dfine_ok(self):
        r = self._run(_FakeDfine())
        self.assertIn("D-FINE-S", r["models_used"])
        self.assertEqual(r["wbf_params"]["profile"], "six_model")
        self.assertEqual(r["wbf_params"]["iou_thr"], 0.68)
        self.assertEqual(r["decision_conf"], 0.44)
        self.assertFalse(r["fusion_degraded"])

    def test_fallback_when_runner_missing(self):
        self._assert_five(self._run(None))

    def test_fallback_when_session_cannot_start(self):
        # 权重在但 onnxruntime 缺失：不得调用 infer，更不得把空结果当第 6 路
        d = _FakeDfine(ensure_ok=False)
        self._assert_five(self._run(d))
        self.assertEqual(d.infer_calls, 0)

    def test_fallback_when_inference_fails(self):
        self._assert_five(self._run(_FakeDfine(infer_ok=False)))


if __name__ == "__main__":
    unittest.main()
