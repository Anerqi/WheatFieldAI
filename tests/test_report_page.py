# -*- coding: utf-8 -*-
"""汇报页（pages/2_汇报.py）的 AppTest 集成测试。

覆盖只读审查提出的页面级问题：
- 空状态三个 tab、0 exception（页面可渲染）；
- R3：单图 / 批量 tab 任务不同时，多次 rerun 不再反复 switch_task / free_gpu
  （用假 manager 计数，不需要 GPU）；
- R10：0 检出时页面显示“未检出”，不显示“轻（未检出）（严重度 0/3）”
  （打桩推理函数，不依赖真实权重）；
- R2：批量页 dup.jpg / dup.png 走完整导出链路不报错，表格列名为“正式最高置信度”
  （ZIP 内条目唯一性由 test_report_data.TestBatchZip 直接断言）；
- 单图 tab 真实图片 + 快速单模型：0 exception（缺权重时自动跳过）。

运行：python -B -m unittest discover -s tests -v
"""

import sys
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import models  # noqa: E402
import report_data as rd  # noqa: E402
import report_html as rh  # noqa: E402
import weed_inference  # noqa: E402

PAGE = ROOT / "pages" / "2_汇报.py"
WEED_IMAGE = ROOT / "test_images" / "weed" / "DJI_weed_0001.jpg"


def _zero_result():
    """0 检出的推理结果（字段与 weed_inference 返回结构一致）。"""
    return {
        "status": "ok",
        "mode": "fast",
        "mode_display": "快速单模型（YOLO11s）",
        "models_used": ["YOLO11s"],
        "inference_time_ms": 12.0,
        "net_inference_ms": 10.0,
        "model_load_ms": 2.0,
        "width": 640,
        "height": 480,
        "num_detections": 0,
        "num_candidates_total": 0,
        "decision_conf": 0.5,
        "mean_confidence": None,
        "max_confidence": None,
        "max_decided_confidence": None,
        "detections": [],
        "low_conf_candidates": [],
        "hazard": {
            "level": "轻",
            "label": "轻（未检出）",
            "severity": 0,
            "none": True,
            "density_per_mp": 0.0,
            "num_detections": 0,
            "thresholds": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0},
        },
        "advice": {"advice": "本次未检出目标。", "disclaimer": "原型规则提示。"},
        "wbf_params": {"iou_thr": 0.65},
        "device": "cpu",
        "class_mapping": {"category_name": "Obonianghao"},
    }


def _one_detection_result(width=1280, height=1280):
    """1 个正式检测 + 1 个低置信候选，用于验证导出图只含正式检测。"""
    result = _zero_result()
    result.update({
        "width": width,
        "height": height,
        "num_detections": 1,
        "num_candidates_total": 2,
        "mean_confidence": 0.81,
        "max_confidence": 0.88,
        "max_decided_confidence": 0.81,
        "detections": [{
            "category_name": "Obonianghao",
            "confidence": 0.81,
            "bbox_xyxy": [120.0, 120.0, 420.0, 420.0],
        }],
        "low_conf_candidates": [{
            "category_name": "Obonianghao",
            "confidence": 0.22,
            "bbox_xyxy": [700.0, 700.0, 980.0, 980.0],
        }],
        "hazard": {
            "level": "轻",
            "label": "轻",
            "severity": 1,
            "none": False,
            "density_per_mp": 0.61,
            "num_detections": 1,
            "thresholds": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0},
        },
    })
    return result


def _jpeg(img_bgr, quality=88):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


class _FakeManager:
    """假模型管理器：只统计任务切换/释放次数，不加载任何模型。

    真实 ``switch_task`` 在任务变化时调用 ``free_gpu``，这里用 ``free_calls`` 复刻该行为，
    从而在 AppTest 里直接观察“rerun 是否反复卸载模型”。
    """

    instances = []

    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = threading.RLock()
        self.active_task = None
        self.switch_calls = []
        self.free_calls = 0
        self._pest = object()
        _FakeManager.instances.append(self)

    @property
    def lock(self):
        return self._lock

    def switch_task(self, task):
        with self._lock:
            if task == self.active_task:
                return
            self.free_calls += 1  # 真实实现：切换即 free_gpu + empty_cache
            self.active_task = task
            self.switch_calls.append(task)

    def check_weights(self, task, mode="fusion"):
        return [], True

    def free_gpu(self):
        self.free_calls += 1

    def device_str(self):
        return "cpu"


def _weed_weights_available():
    try:
        from config import load_config

        return Path(load_config()["weed"]["paths"]["yolo11_weights"]).is_file()
    except Exception:  # noqa: BLE001 - 缺依赖/缺配置时跳过真实推理测试
        return False


def _uploader(at, key):
    return next(item for item in at.file_uploader if item.key == key)


class ReportPageTestBase(unittest.TestCase):
    def setUp(self):
        self._orig_manager = models.DualModelManager
        self._orig_infer = weed_inference.infer_weed_single_image
        _FakeManager.instances.clear()
        st.cache_resource.clear()

    def tearDown(self):
        models.DualModelManager = self._orig_manager
        weed_inference.infer_weed_single_image = self._orig_infer
        st.cache_resource.clear()


class TestReportPageEmpty(ReportPageTestBase):
    def test_empty_state_has_three_tabs_and_no_exception(self):
        at = AppTest.from_file(str(PAGE), default_timeout=90)
        at.run()

        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual(len(at.tabs), 3)
        self.assertEqual(len(at.file_uploader), 2)
        text = " ".join(m.value for m in at.markdown)
        self.assertTrue("上传一张图片开始生成报告" in text or "模型未配置" in text)


class TestReportPageTaskSwitch(ReportPageTestBase):
    def test_rerun_does_not_switch_task_or_free_gpu(self):
        """R3：单图 tab 选害虫、批量 tab 保持杂草，连续 rerun 不应触发任何切换。"""
        models.DualModelManager = _FakeManager
        at = AppTest.from_file(str(PAGE), default_timeout=90)
        at.run()
        at.session_state["rp_task"] = "害虫检测"
        at.run()
        at.run()
        at.run()

        self.assertEqual([e.value for e in at.exception], [])
        manager = _FakeManager.instances[-1]
        self.assertEqual(manager.switch_calls, [])
        self.assertEqual(manager.free_calls, 0)


class TestReportPageZeroDetection(ReportPageTestBase):
    def test_zero_detection_shows_none(self):
        """R10：0 检出时显示“未检出 / 严重度 —”。"""
        if not WEED_IMAGE.is_file():
            self.skipTest("缺少测试图片")

        models.DualModelManager = _FakeManager
        weed_inference.infer_weed_single_image = (
            lambda img, cfg, manager, mode="fusion": _zero_result()
        )

        at = AppTest.from_file(str(PAGE), default_timeout=90)
        at.run()
        _uploader(at, "rp_single_upload").set_value(
            (WEED_IMAGE.name, WEED_IMAGE.read_bytes(), "image/jpeg")
        )
        at.run()

        self.assertEqual([e.value for e in at.exception], [])
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("危害等级", text)
        self.assertIn("未检出", text)
        self.assertNotIn("轻（未检出）", text)
        self.assertNotIn("严重度 0/3", text)


class TestReportPageBatchExport(ReportPageTestBase):
    def test_duplicate_names_batch_export_runs(self):
        """R2：dup.jpg / dup.png 走完整批量导出链路不报错，列名为“正式最高置信度”。"""
        if not WEED_IMAGE.is_file():
            self.skipTest("缺少测试图片")

        models.DualModelManager = _FakeManager
        weed_inference.infer_weed_single_image = (
            lambda img, cfg, manager, mode="fusion": _zero_result()
        )

        at = AppTest.from_file(str(PAGE), default_timeout=90)
        at.run()
        data = WEED_IMAGE.read_bytes()
        _uploader(at, "rp_batch_upload").set_value([
            ("dup.jpg", data, "image/jpeg"),
            ("dup.png", data, "image/png"),
        ])
        at.run()
        next(b for b in at.button if b.key == "rp_batch_run").click()
        at.run()

        self.assertEqual([e.value for e in at.exception], [])
        frames = [d.value for d in at.dataframe]
        batch_df = next(df for df in frames if "正式最高置信度" in df.columns)
        self.assertEqual(len(batch_df), 2)
        # 0 检出批量行显示“未检出”，不是“轻（未检出）”
        self.assertEqual(list(batch_df["危害"]), ["未检出", "未检出"])
        labels = [d.label for d in at.download_button]
        for expected in ("下载 CSV", "下载 HTML", "下载 ZIP"):
            self.assertIn(expected, labels)


class TestReportPageSingleInference(ReportPageTestBase):
    @unittest.skipUnless(WEED_IMAGE.is_file() and _weed_weights_available(),
                         "缺少杂草权重或测试图片")
    def test_single_tab_real_quick_inference(self):
        """真实图片 + 快速单模型：页面 0 exception，叠加候选后导出图注仍为“仅正式检测”。"""
        at = AppTest.from_file(str(PAGE), default_timeout=180)
        at.run()
        at.session_state["rp_mode"] = "快速单模型"
        _uploader(at, "rp_single_upload").set_value(
            (WEED_IMAGE.name, WEED_IMAGE.read_bytes(), "image/jpeg")
        )
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("标注图" in m.value for m in at.markdown))
        self.assertEqual(len(at.image), 1)

        next(c for c in at.checkbox if c.key == "rp_show_low").set_value(True)
        at.run()
        self.assertEqual([e.value for e in at.exception], [])
        captions = " ".join(c.value for c in at.caption)
        # 页面显示说明叠加了候选；导出说明仍然只含正式检测
        self.assertIn("灰色细虚线", captions)
        self.assertIn("导出报告只含正式检测", captions)
        self.assertIn("仅正式检测", captions)


class TestReportPageExportImage(ReportPageTestBase):
    def test_export_html_image_equals_formal_only(self):
        """R1：勾选“叠加低置信候选”后，导出 HTML 的标注图必须仍等于“仅正式检测”图。

        通过 spy ``report_html.render_single_report_html`` 捕获页面真正传给导出的 JPEG，
        再与 compose_annotation_image 产出的 export / display 两张图逐字节比对。
        """
        if not WEED_IMAGE.is_file():
            self.skipTest("缺少测试图片")

        models.DualModelManager = _FakeManager
        result = _one_detection_result()
        weed_inference.infer_weed_single_image = lambda *a, **k: result

        captured = {}
        original_render = rh.render_single_report_html

        def spy(payload, annotated_jpeg=None, annotated_alt=None):
            captured["jpeg"] = annotated_jpeg
            return original_render(payload, annotated_jpeg, annotated_alt=annotated_alt)

        rh.render_single_report_html = spy
        try:
            at = AppTest.from_file(str(PAGE), default_timeout=90)
            at.run()
            _uploader(at, "rp_single_upload").set_value(
                (WEED_IMAGE.name, WEED_IMAGE.read_bytes(), "image/jpeg")
            )
            at.run()
            next(c for c in at.checkbox if c.key == "rp_show_low").set_value(True)
            at.run()
        finally:
            rh.render_single_report_html = original_render

        self.assertEqual([e.value for e in at.exception], [])
        img = cv2.imdecode(np.frombuffer(WEED_IMAGE.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        display, export = rd.compose_annotation_image(
            img, result["detections"], result["low_conf_candidates"], show_low=True
        )
        self.assertEqual(captured["jpeg"], _jpeg(export))
        self.assertNotEqual(captured["jpeg"], _jpeg(display))


class TestReportPageExportInvariant(ReportPageTestBase):
    def test_page_exports_formal_only_image(self):
        """结构守卫：页面必须用 compose_annotation_image 的 export 图导出。

        行为断言（导出图像素不含候选）见 test_report_data.TestAnnotationImage。
        """
        source = PAGE.read_text(encoding="utf-8")
        self.assertIn("rd.compose_annotation_image(", source)
        self.assertIn("display_bgr, export_bgr = rd.compose_annotation_image(", source)
        # 旧实现：把 detections + candidates 合成一个列表直接画
        self.assertNotIn("draw_list", source)
        self.assertNotIn("export_bgr = annotated_bgr", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
