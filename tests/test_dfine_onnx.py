# -*- coding: utf-8 -*-
"""任务 30 · D-FINE-S ONNX 第 6 支成员的单元测试。

覆盖四条契约：
1. 未启用 / 权重缺失时 build_from_config 返回 None 或 available()=False，
   调用方（weed_inference）必须能安全退回五模型；
2. infer() 在不可用时返回空数组而不是抛异常（界面不能因缺权重而崩）；
3. 预处理契约：输入张量为 (1,3,imgsz,imgsz)、float32、值域 [0,1]、尺寸 int64 且为 (h,w)；
4. score_thr 过滤生效。

运行：python -B -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dfine_onnx import DFineOnnxWeed, build_from_config  # noqa: E402


class _StubSession:
    """替身：记录入参并返回预制的 (labels, boxes, scores)。"""

    def __init__(self, boxes, scores):
        self.boxes = boxes
        self.scores = scores
        self.seen = None

    def run(self, _outputs, feeds):
        self.seen = feeds
        return (np.zeros((1, len(self.scores)), dtype=np.int64),
                self.boxes[None], self.scores[None])


class TestBuildFromConfig(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(build_from_config({"dfine": {"enabled": False, "onnx_path": "x.onnx"}}))

    def test_missing_node_returns_none(self):
        self.assertIsNone(build_from_config({}))

    def test_enabled_returns_runner(self):
        runner = build_from_config({"dfine": {"enabled": True, "onnx_path": "x.onnx",
                                              "score_thr": 0.05, "imgsz": 960}})
        self.assertIsInstance(runner, DFineOnnxWeed)
        self.assertEqual(runner.score_thr, 0.05)
        self.assertEqual(runner.imgsz, 960)


class TestAvailability(unittest.TestCase):
    def test_missing_file_unavailable(self):
        runner = DFineOnnxWeed(str(ROOT / "models" / "weed" / "dfine_s" / "_nope.onnx"))
        self.assertFalse(runner.available())

    def test_infer_returns_empty_when_unavailable(self):
        runner = DFineOnnxWeed(str(ROOT / "models" / "weed" / "dfine_s" / "_nope.onnx"))
        boxes, scores = runner.infer(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertIsNotNone(runner.error)

    def test_shipped_weights_exist(self):
        onnx_path = ROOT / "models" / "weed" / "dfine_s" / "best.onnx"
        if not onnx_path.exists():
            self.skipTest("ONNX 权重未随工作区提供")
        self.assertGreater(onnx_path.stat().st_size, 10 * 1024 * 1024)


class TestPreprocessContract(unittest.TestCase):
    def test_input_tensor_and_sizes(self):
        runner = DFineOnnxWeed("dummy.onnx", score_thr=0.0, imgsz=960)
        stub = _StubSession(boxes=np.array([[1.0, 2.0, 30.0, 40.0]]), scores=np.array([0.9]))
        runner._session = stub          # 绕开真实 ONNX 会话，只验我们的输入构造
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes, scores = runner.infer(img)
        feeds = stub.seen
        self.assertEqual(feeds["images"].shape, (1, 3, 960, 960))
        self.assertEqual(feeds["images"].dtype, np.float32)
        self.assertGreaterEqual(float(feeds["images"].min()), 0.0)
        self.assertLessEqual(float(feeds["images"].max()), 1.0)
        self.assertEqual(feeds["orig_target_sizes"].dtype, np.int64)
        np.testing.assert_array_equal(feeds["orig_target_sizes"], np.array([[480, 640]]))
        np.testing.assert_allclose(boxes, np.array([[1.0, 2.0, 30.0, 40.0]], dtype=float))
        np.testing.assert_allclose(scores, np.array([0.9], dtype=float))

    def test_score_threshold_filters(self):
        runner = DFineOnnxWeed("dummy.onnx", score_thr=0.5, imgsz=960)
        stub = _StubSession(boxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float),
                            scores=np.array([0.9, 0.1]))
        runner._session = stub
        boxes, scores = runner.infer(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(float(scores[0]), 0.9)
        self.assertEqual(boxes.shape, (1, 4))

    def test_bgr_to_rgb(self):
        runner = DFineOnnxWeed("dummy.onnx", score_thr=0.0, imgsz=8)
        stub = _StubSession(boxes=np.zeros((0, 4)), scores=np.zeros((0,)))
        runner._session = stub
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[:, :, 0] = 255      # BGR 的 B 通道拉满
        runner.infer(img)
        rgb = stub.seen["images"][0]          # CHW
        self.assertAlmostEqual(float(rgb[0].max()), 0.0, places=5)   # R 应为 0
        self.assertAlmostEqual(float(rgb[2].max()), 1.0, places=5)   # B 变成第 3 通道


class TestConfigWiring(unittest.TestCase):
    def test_weights_match_enabled_branches(self):
        from config import load_config
        cfg = load_config(str(ROOT / "config.yaml"))
        weed = cfg["weed"]
        n_weights = len(weed["wbf"]["weights"])
        n_branches = 5 + (1 if (weed.get("dfine") or {}).get("enabled") else 0)
        self.assertEqual(n_weights, n_branches,
                         "weed.wbf.weights 数量必须与启用分支数一致")

    def test_iou_thr_and_dfine_path(self):
        from config import load_config
        cfg = load_config(str(ROOT / "config.yaml"))
        self.assertAlmostEqual(float(cfg["weed"]["wbf"]["iou_thr"]), 0.68)
        self.assertTrue(cfg["weed"]["paths"]["dfine_onnx"].endswith("best.onnx"))


if __name__ == "__main__":
    unittest.main()