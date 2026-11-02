# -*- coding: utf-8 -*-
"""report_data / report_html 的单元测试（不加载模型、不依赖 任务输出/ 真实文件）。

覆盖：
- 置信度直方图分箱与边界（0.1 落第二箱、1.0 落最后一箱、越界丢弃）；
- 框尺寸分桶（COCO 32/96 阈值）、空输入、统计量；
- 单图 payload 结构、max_confidence=None 显示口径、JSON 可序列化；
- 批量汇总的计数、危害分桶、失败行保留、类别合计；
- 评测 JSON 解析（用临时目录的合成 fixture，不代表任何真实指标）；
- 缺文件时 available=False（页面显示“数据未找到”的数据侧保证）；
- HTML 报告内嵌 base64、无外部 CDN 链接、文件名转义。

运行：python -B -m unittest discover -s tests -v
"""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import report_data as rd  # noqa: E402
import report_html as rh  # noqa: E402


def _detection(conf, box=(10, 10, 50, 50), name="Obonianghao"):
    return {"category_name": name, "confidence": conf, "bbox_xyxy": list(box)}


def _weed_result(num_decided=2, num_candidates=3, max_conf=0.83):
    detections = [_detection(max_conf), _detection(0.61)]
    candidates = [_detection(0.12), _detection(0.05), _detection(0.30)]
    return {
        "status": "ok",
        "mode": "fusion",
        "mode_display": "高精度融合（五模型 WBF）",
        "models_used": ["YOLO11s", "YOLO11s-box8"],
        "inference_time_ms": 812.4,
        "net_inference_ms": 120.3,
        "model_load_ms": 692.1,
        "width": 2048,
        "height": 1024,
        "num_detections": num_decided,
        "num_candidates_total": num_decided + num_candidates,
        "decision_conf": 0.5,
        "mean_confidence": 0.72,
        "max_confidence": max_conf,
        "detections": detections[:num_decided],
        "low_conf_candidates": candidates[:num_candidates],
        "hazard": {
            "level": "轻",
            "label": "轻",
            "severity": 1,
            "density_per_mp": 0.95,
            "thresholds": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0},
            "note": "原型规则",
        },
        "advice": {"level": "轻", "advice": "保持巡查。", "disclaimer": "原型规则提示。"},
        "wbf_params": {"iou_thr": 0.65, "weights": [1, 1, 1, 1, 1]},
        "device": "cuda",
        "class_mapping": {"category_name": "Obonianghao"},
    }


def _pest_result():
    result = _weed_result()
    result["class_counts"] = {"Ostrinia furnacalis": 2, "Armyworm": 1}
    result["major_class_count"] = 2
    result["detections"] = [
        _detection(0.71, name="Ostrinia furnacalis"),
        _detection(0.66, name="Ostrinia furnacalis"),
        _detection(0.55, name="Armyworm"),
    ]
    result["num_detections"] = 3
    result["mode_display"] = "高精度融合（三模型 classwise WBF）"
    result["decision_conf"] = 0.3
    return result


class TestConfidenceHistogram(unittest.TestCase):
    def test_bins_and_boundaries(self):
        decided = [_detection(1.0), _detection(0.1), _detection(0.5)]
        candidates = [_detection(0.05), _detection(0.0)]
        rows = rd.confidence_histogram(decided, candidates)

        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["decided"], 0)
        self.assertEqual(rows[1]["decided"], 1)  # 0.1 属于 0.1–0.2
        self.assertEqual(rows[5]["decided"], 1)  # 0.5 属于 0.5–0.6
        self.assertEqual(rows[9]["decided"], 1)  # 1.0 属于最后一箱（闭区间）
        self.assertEqual(rows[0]["candidates"], 2)
        self.assertEqual(sum(r["total"] for r in rows), 5)

    def test_out_of_range_and_invalid_ignored(self):
        rows = rd.confidence_histogram([_detection(1.5), _detection(-0.2), {"confidence": None}],
                                       ["bad", None])
        self.assertEqual(sum(r["total"] for r in rows), 0)

    def test_empty(self):
        rows = rd.confidence_histogram([], [])
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["total"] == 0 for r in rows))
        self.assertEqual(rows[0]["label"], "0.0–0.1")


class TestBoxSizeDistribution(unittest.TestCase):
    def test_buckets_by_side(self):
        dets = [
            _detection(0.9, box=(0, 0, 20, 20)),      # side 20 -> 小
            _detection(0.9, box=(0, 0, 50, 50)),      # side 50 -> 中
            _detection(0.9, box=(0, 0, 200, 200)),    # side 200 -> 大
        ]
        dist = rd.box_size_distribution(dets)
        counts = {b["key"]: b["count"] for b in dist["buckets"]}
        self.assertEqual(counts, {"small": 1, "medium": 1, "large": 1})
        self.assertEqual(dist["total"], 3)
        self.assertAlmostEqual(sum(b["share"] for b in dist["buckets"]), 1.0, places=3)
        self.assertAlmostEqual(dist["stats"]["median_side_px"], 50.0, places=1)
        self.assertAlmostEqual(dist["stats"]["mean_aspect"], 1.0, places=2)

    def test_empty(self):
        dist = rd.box_size_distribution([])
        self.assertEqual(dist["total"], 0)
        self.assertTrue(all(b["count"] == 0 for b in dist["buckets"]))
        self.assertIsNone(dist["stats"]["median_side_px"])
        self.assertIsNone(dist["stats"]["mean_aspect"])

    def test_invalid_box_ignored(self):
        dist = rd.box_size_distribution([{"bbox_xyxy": [1, 2]}, {"bbox_xyxy": None}])
        self.assertEqual(dist["total"], 0)


class TestSingleReport(unittest.TestCase):
    def test_payload_structure_and_json(self):
        payload = rd.build_single_report(
            _weed_result(), "weed", "DJI_weed_0001.jpg",
            generated_at="2026-10-30T19:40:00+08:00",
        )
        self.assertEqual(payload["schema"], rd.REPORT_SCHEMA)
        self.assertEqual(payload["task_label"], "杂草检测")
        self.assertEqual(payload["image"]["width"], 2048)
        self.assertEqual(payload["image"]["megapixels"], 2.097)
        self.assertEqual(payload["metrics"]["num_detections"], 2)
        self.assertEqual(payload["metrics"]["max_confidence"], 0.83)
        self.assertEqual(payload["metrics"]["max_decided_confidence"], 0.83)
        self.assertEqual(payload["metrics"]["density_per_mp"], 0.95)
        self.assertEqual(len(payload["detections"]), 2)
        self.assertEqual(len(payload["low_conf_candidates"]), 3)
        self.assertEqual(payload["provenance"]["generated_at"], "2026-10-30T19:40:00+08:00")
        self.assertEqual(payload["provenance"]["decision_conf"], 0.5)
        # JSON 导出必须可序列化
        json.dumps(payload, ensure_ascii=False)

    def test_missing_max_confidence(self):
        result = _weed_result()
        result["max_confidence"] = None
        result["detections"] = []
        result["num_detections"] = 0
        payload = rd.build_single_report(result, "weed", "empty.jpg")
        self.assertIsNone(payload["metrics"]["max_confidence"])
        self.assertIsNone(payload["metrics"]["max_decided_confidence"])
        self.assertEqual(payload["class_distribution"], [])
        self.assertEqual(payload["box_sizes"]["total"], 0)

    def test_max_decided_confidence_excludes_candidates(self):
        """正式最高置信度只能来自 detections；max_confidence 含候选，两者不能混用。"""
        result = _weed_result()
        result["max_confidence"] = 0.97          # 模型原始最高分（可能来自低置信候选区）
        result["detections"] = [_detection(0.66), _detection(0.42)]
        result["low_conf_candidates"] = [_detection(0.97)]
        result["max_decided_confidence"] = None  # 缺省 -> 回退 detections 的 max
        payload = rd.build_single_report(result, "weed", "a.jpg")
        self.assertEqual(payload["metrics"]["max_confidence"], 0.97)
        self.assertEqual(payload["metrics"]["max_decided_confidence"], 0.66)
        self.assertEqual(payload["provenance"]["max_decided_confidence"], 0.66)
        self.assertEqual(payload["provenance"]["max_confidence"], 0.97)

        result["max_decided_confidence"] = 0.66
        payload = rd.build_single_report(result, "weed", "a.jpg")
        self.assertEqual(payload["metrics"]["max_decided_confidence"], 0.66)

    def test_pest_class_distribution_from_counts(self):
        payload = rd.build_single_report(_pest_result(), "pest", "pest24_0000002.jpg")
        names = [row["name"] for row in payload["class_distribution"]]
        self.assertEqual(names, ["Ostrinia furnacalis", "Armyworm"])
        self.assertEqual(payload["task_label"], "害虫检测")

    def test_inference_thresholds_from_cfg(self):
        cfg = {"weed": {"inference": {"yolo11_conf": 0.001, "yolo11_imgsz": 960}}, "pest": {}}
        payload = rd.build_single_report(_weed_result(), "weed", "a.jpg", cfg=cfg)
        self.assertEqual(payload["provenance"]["inference_thresholds"],
                         {"yolo11_conf": 0.001, "yolo11_imgsz": 960})

    def test_rejects_non_dict(self):
        with self.assertRaises(TypeError):
            rd.build_single_report([], "weed", "a.jpg")

    def test_private_paths_sanitized(self):
        result = _pest_result()
        result["wbf_params"] = {
            "type": "classwise_WBF",
            "config_source": r"C:\Users\someone\Desktop\work\configs\classwise.json",
        }
        result["class_mapping"] = {
            "num_classes": 32,
            "source": r"C:\Users\someone\Desktop\work\configs\dataset.yaml",
            "note": "32 类英文名称严格来自提交包 configs/dataset.yaml（唯一数据源）",
        }
        payload = rd.build_single_report(result, "pest", "a.jpg")
        dumped = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("Users", dumped)
        self.assertNotIn("Desktop", dumped)
        self.assertEqual(payload["provenance"]["wbf_params"]["config_source"], "classwise.json")
        self.assertEqual(payload["provenance"]["class_mapping"]["source"], "dataset.yaml")
        # 含斜杠的描述性文本不能被误判为路径
        self.assertIn("configs/dataset.yaml", payload["provenance"]["class_mapping"]["note"])
        # 只清洗展示副本，不改动原始推理结果
        self.assertIn("Desktop", result["wbf_params"]["config_source"])


class TestPathDisplay(unittest.TestCase):
    """R9：界面与导出不显示本机绝对路径。"""

    def test_relative_path_text_hides_absolute_prefix(self):
        root = Path(r"C:\work")
        self.assertEqual(rd.relative_path_text(r"C:\work\a\b.json", root), "a/b.json")
        # 不在 root 下时只保留文件名
        self.assertEqual(rd.relative_path_text(r"D:\other\c.json", root), "c.json")


class TestAnnotationImage(unittest.TestCase):
    """R1：页面显示图可含低置信候选，导出图必须只含正式检测。"""

    def setUp(self):
        # 白底图：任何绘制都会改变像素，便于断言“哪里被画了”
        self.img = np.full((120, 180, 3), 255, dtype=np.uint8)
        self.formal = [_detection(0.90, box=(10, 10, 60, 60))]
        self.candidates = [_detection(0.20, box=(110, 60, 170, 110))]

    def test_export_image_has_no_candidate_marks(self):
        display, export = rd.compose_annotation_image(
            self.img, self.formal, self.candidates, show_low=True
        )
        # 候选框区域：导出图保持原样，显示图被画上灰色虚线
        self.assertTrue(np.array_equal(export[60:111, 110:171], self.img[60:111, 110:171]))
        self.assertFalse(np.array_equal(display[60:111, 110:171], self.img[60:111, 110:171]))
        # 正式框两张图都有
        self.assertFalse(np.array_equal(export[10, 10:61], self.img[10, 10:61]))
        self.assertFalse(np.array_equal(display[10, 10:61], self.img[10, 10:61]))
        # 显示图 = 导出图 + 候选；导出图不含灰色候选色
        def _has_gray(image):
            diff = np.abs(image.reshape(-1, 3).astype(int) - 150)
            return bool(np.all(diff <= 8, axis=1).any())

        self.assertFalse(_has_gray(export))
        self.assertTrue(_has_gray(display))

    def test_without_overlay_display_equals_export(self):
        display, export = rd.compose_annotation_image(
            self.img, self.formal, self.candidates, show_low=False
        )
        self.assertTrue(np.array_equal(display, export))

    def test_candidate_count_capped(self):
        # 8 个互不重叠的候选，只画前 3 个
        many = [_detection(0.1, box=(i * 15, 80, i * 15 + 10, 100)) for i in range(8)]
        display, export = rd.compose_annotation_image(
            self.img, self.formal, many, show_low=True, max_candidates=3
        )
        # 第 4 个候选（x 45–55）未被绘制：显示图在该区域与导出图一致
        # （从 y=81 起算，避开第 3 个候选标签背景覆盖的第 80 行）
        self.assertTrue(np.array_equal(display[81:100, 45:56], export[81:100, 45:56]))
        # 第 3 个候选（x 30–40）被绘制：显示图在该区域与导出图不同
        self.assertFalse(np.array_equal(display[80:101, 30:41], export[80:101, 30:41]))


class TestBatchSummary(unittest.TestCase):
    def test_totals_hazard_and_classes(self):
        items = [
            {"file_name": "a.jpg", "status": "ok", "result": _pest_result()},
            {"file_name": "b.jpg", "status": "ok", "result": _pest_result()},
            {"file_name": "c.jpg", "status": "error", "error": "文件为空（0 字节）。"},
        ]
        summary = rd.build_batch_summary(items, "pest", "高精度融合", decision_conf=0.3)
        self.assertEqual(summary["totals"]["images"], 3)
        self.assertEqual(summary["totals"]["ok"], 2)
        self.assertEqual(summary["totals"]["failed"], 1)
        self.assertEqual(summary["totals"]["detections"], 6)
        self.assertEqual(summary["totals"]["mean_density_per_mp"], 0.95)

        hazard = {row["level"]: row["count"] for row in summary["hazard_distribution"]}
        self.assertEqual(hazard, {"未检出": 0, "轻": 2, "中": 0, "重": 0})

        classes = {row["name"]: row["count"] for row in summary["class_distribution"]}
        self.assertEqual(classes["Ostrinia furnacalis"], 4)
        self.assertEqual(classes["Armyworm"], 2)

        failed_row = [row for row in summary["rows"] if row["status"] == "失败"][0]
        self.assertEqual(failed_row["error"], "文件为空（0 字节）。")
        self.assertIsNone(failed_row["num_detections"])
        self.assertEqual(failed_row["hazard_label"], "—")
        json.dumps(summary, ensure_ascii=False)

    def test_zero_detection_counts_as_none(self):
        result = _weed_result()
        result["detections"] = []
        result["num_detections"] = 0
        result["hazard"] = {"level": "轻", "label": "轻（未检出）", "severity": 0,
                            "none": True, "density_per_mp": 0.0, "thresholds": {}}
        summary = rd.build_batch_summary(
            [{"file_name": "z.jpg", "status": "ok", "result": result}], "weed", "快速单模型"
        )
        hazard = {row["level"]: row["count"] for row in summary["hazard_distribution"]}
        self.assertEqual(hazard["未检出"], 1)
        # 0 检出统一显示“未检出”，不再把“轻（未检出）”写进批量表
        self.assertEqual(summary["rows"][0]["hazard_label"], "未检出")
        self.assertEqual(summary["rows"][0]["hazard_level"], "轻")
        self.assertIsNone(summary["rows"][0]["max_decided_confidence"])
        self.assertEqual(rd.hazard_label(result), "未检出")
        self.assertEqual(rd.hazard_badge(result), {"badge": "none", "text": "未检出"})

    def test_empty_items(self):
        summary = rd.build_batch_summary([], "weed", "快速单模型")
        self.assertEqual(summary["totals"]["images"], 0)
        self.assertIsNone(summary["totals"]["mean_density_per_mp"])
        self.assertEqual(len(summary["hazard_distribution"]), 4)


class TestBatchZip(unittest.TestCase):
    """R2：重名文件（a.jpg / a.png）不得在 ZIP 内互相覆盖。"""

    def _items(self):
        return [
            {"file_name": "dup.jpg", "status": "ok", "result": _weed_result(),
             "annotated_jpeg": b"\xff\xd8jpg"},
            {"file_name": "dup.png", "status": "ok", "result": _weed_result(),
             "annotated_jpeg": b"\xff\xd8png"},
            {"file_name": "bad.jpg", "status": "error", "error": "文件为空（0 字节）。"},
        ]

    def test_duplicate_stems_do_not_overwrite(self):
        items = self._items()
        payload = rd.build_batch_summary(
            items, "weed", "快速单模型",
            generated_at="2026-10-30T20:00:00+08:00", decision_conf=0.5,
        )
        data = rd.build_batch_zip(
            payload, items, summary_csv=b"a,b\n1,2\n", summary_html="<html>x</html>",
            generated_at="2026-10-30T20:00:00+08:00",
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            self.assertEqual(len(names), len(set(names)), f"ZIP 条目重名：{names}")
            for expected in ("summary.csv", "summary.html", "batch_summary.json",
                             "json/dup.json", "json/dup_2.json",
                             "annotated/dup.jpg", "annotated/dup_2.jpg"):
                self.assertIn(expected, names)
            # 失败图不写文件
            self.assertNotIn("json/bad.json", names)
            # 两个重名条目内容各自独立，没有被覆盖
            self.assertEqual(zf.read("annotated/dup.jpg"), b"\xff\xd8jpg")
            self.assertEqual(zf.read("annotated/dup_2.jpg"), b"\xff\xd8png")
            for name in names:
                if name.startswith("json/"):
                    json.loads(zf.read(name).decode("utf-8"))
            manifest = json.loads(zf.read("batch_summary.json").decode("utf-8"))
            self.assertEqual(len(manifest["files"]), 3)
            self.assertEqual(manifest["files"][2]["zip_files"], [])
            self.assertEqual(manifest["totals"]["ok"], 2)

    def test_zip_rows_match_summary(self):
        """ZIP 内成功图文件数 == payload rows 中的成功行数（CSV/HTML/ZIP 行数一致）。"""
        items = self._items()
        payload = rd.build_batch_summary(items, "weed", "快速单模型", decision_conf=0.5)
        data = rd.build_batch_zip(payload, items)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
        ok_rows = [row for row in payload["rows"] if row["status"] == "成功"]
        self.assertEqual(sum(1 for n in names if n.startswith("json/")), len(ok_rows))
        self.assertEqual(sum(1 for n in names if n.startswith("annotated/")), len(ok_rows))


class TestModelPerformance(unittest.TestCase):
    """用临时目录的合成 fixture 验证解析逻辑（数值仅用于测试，不代表任何真实指标）。"""

    def _write(self, base, rel, payload):
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_parsing_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task24 = base / "任务输出" / "24_杂草精度误差分析与优化" / "outputs"
            task22 = base / "任务输出" / "22_decision_conf阈值校准" / "results"
            configs = base / "WheatFieldAI" / "configs"

            self._write(task24, "app_five_model/summary.json", {
                "images": 10, "failures": [], "mAP50-95": 0.5, "mAP50": 0.8,
                "AP75": 0.45, "seconds_per_image": 0.1, "peak_reserved_gb": 1.0,
                "models_used": ["A", "B"],
            })
            self._write(task24, "fusion_candidates_eval.json", {
                "base3+1280+box8": {
                    "members": ["A", "B"], "n_predictions": 12,
                    "original": {"mAP50-95": 0.49, "mAP50": 0.79, "AP75": 0.44},
                    "v2": {"mAP50-95": 0.51, "mAP50": 0.81, "AP75": 0.47},
                }
            })
            self._write(task24, "val_v2_eval.json", {
                "fusion": {"original": {"mAP50-95": 0.4}, "v2": {"mAP50-95": 0.42}},
                "yolo11_baseline": {"original": {"mAP50-95": 0.39}, "v2": {"mAP50-95": 0.41}},
            })
            self._write(task22, "summary.json", {
                "generated_at": "2026-10-28T21:43:58+08:00",
                "datasets": {"weed_images": 10},
                "picks": {"weed_fusion": {"best_f1": {"threshold": 0.5, "f1": 0.77}}},
            })
            self._write(configs, "classwise_ensemble_11m11l11s_960_refined_current.json", {
                "n_images": 10, "map50": 0.8, "map50_95": 0.5,
                "weights": ["a.pt", "b.pt", "c.pt"],
                "per_class": {"0": 0.3, "1": 0.6},
                "class_configs": {"0": {"source": "C:\\private\\path.json"}},
            })
            configs.mkdir(parents=True, exist_ok=True)
            (configs / "dataset.yaml").write_text(
                "names:\n  0: Alpha\n  1: Beta\n", encoding="utf-8"
            )

            perf = rd.load_model_performance(root=base)

            self.assertTrue(perf["sources"]["five_model_app"]["exists"])
            self.assertEqual(perf["weed"]["five_model_run"]["images"], 10)
            labels = [row["label"] for row in perf["weed"]["models"]]
            self.assertTrue(any("五模型" in label for label in labels))
            self.assertTrue(any("YOLO11s 单模型" in label for label in labels))
            self.assertTrue(perf["weed"]["available"])

            pest = perf["pest"]
            self.assertTrue(pest["available"])
            self.assertEqual(pest["n_classes"], 2)
            self.assertEqual(pest["n_models"], 3)
            self.assertEqual([row["name"] for row in pest["per_class"]], ["Beta", "Alpha"])
            # 私有路径字段不得出现在返回结构中
            self.assertNotIn("C:\\private\\path.json", json.dumps(perf, ensure_ascii=False))

            self.assertTrue(perf["decision_conf"]["available"])
            self.assertEqual(perf["decision_conf"]["picks"]["weed_fusion"]["best_f1"]["f1"], 0.77)

            # R8：说明文字由 JSON 生成，不写死 0.8232 / 787 等常量
            notes = " ".join(perf["notes"])
            self.assertNotIn("0.8232", notes)
            self.assertNotIn("28,133", notes)
            self.assertIn("10 图复算", notes)
            self.assertIn("0.8000", notes)

    def test_all_missing_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            perf = rd.load_model_performance(root=Path(tmp))
            self.assertFalse(perf["weed"]["available"])
            self.assertFalse(perf["pest"]["available"])
            self.assertFalse(perf["decision_conf"]["available"])
            for src in perf["sources"].values():
                self.assertFalse(src["exists"])
            json.dumps(perf, ensure_ascii=False)

    def test_corrupt_json_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task24 = base / "任务输出" / "24_杂草精度误差分析与优化" / "outputs"
            task24.mkdir(parents=True)
            (task24 / "app_five_model").mkdir(parents=True)
            (task24 / "app_five_model" / "summary.json").write_text("{not json", encoding="utf-8")
            perf = rd.load_model_performance(root=base)
            src = perf["sources"]["five_model_app"]
            self.assertTrue(src["exists"])
            self.assertIsNotNone(src["error"])
            self.assertFalse(perf["weed"]["available"])


class TestHtmlReport(unittest.TestCase):
    def test_single_report_embeds_image_and_escapes(self):
        payload = rd.build_single_report(
            _weed_result(), "weed", "<img src=x onerror=alert(1)>.jpg",
            generated_at="2026-10-30T19:40:00+08:00",
        )
        html = rh.render_single_report_html(payload, b"\xff\xd8fake-jpeg")
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("危害刻度", html)
        self.assertIn("置信度分布", html)

    def test_single_report_uses_formal_max_confidence(self):
        """R5：单图报告展示“正式最高置信度”，原始最高分只作溯源。"""
        result = _weed_result()
        result["max_confidence"] = 0.97
        result["detections"] = [_detection(0.66)]
        result["max_decided_confidence"] = 0.66
        payload = rd.build_single_report(result, "weed", "a.jpg")
        html = rh.render_single_report_html(payload, b"\xff\xd8x")
        self.assertIn("正式最高置信度", html)
        self.assertIn("0.660", html)
        self.assertIn("模型原始最高置信度（含低置信候选）", html)

    def test_batch_report_and_scale_states(self):
        summary = rd.build_batch_summary(
            [{"file_name": "a.jpg", "status": "ok", "result": _pest_result()}],
            "pest", "高精度融合", decision_conf=0.3,
        )
        html = rh.render_batch_report_html(
            summary,
            [{"file_name": "a.jpg", "jpeg": b"\xff\xd8x", "badge": "light",
              "badge_text": "轻", "caption": "3 框"}],
        )
        self.assertIn("逐图汇总", html)
        self.assertIn("data:image/jpeg;base64,", html)

        # 未检出 -> 中性刻度状态
        neutral = rh.hazard_scale_html(
            {"level": "轻", "label": "轻（未检出）", "severity": 0, "density_per_mp": 0.0,
             "thresholds": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0}},
            task="weed", decision_conf=0.5,
        )
        self.assertIn("中性状态", neutral)
        self.assertIn('class="scale neutral"', neutral)

    def test_pest_major_class_scale_text_matches_hazard_rule(self):
        """R4：hazard.py 用 > 5 判“重”，文案必须写“重 ≥ 6 类”。"""
        scale = rh.hazard_scale_html(
            {"level": "中", "label": "中", "severity": 2, "density_per_mp": 8.0,
             "major_class_count": 4,
             "thresholds": {"light_max_per_mp": 5.0, "medium_max_per_mp": 15.0,
                            "major_classes_light_max": 2, "major_classes_medium_max": 5}},
            task="pest", decision_conf=0.3,
        )
        self.assertIn("轻 ≤ 2 类", scale)
        self.assertIn("中 3–5 类", scale)
        self.assertIn("重 ≥ 6 类", scale)
        self.assertNotIn("重 ≥ 5 类", scale)

    def test_zero_detection_advice_card(self):
        """R10：0 检出时危害等级显示“未检出”，严重度显示“—”。"""
        result = _weed_result()
        result["detections"] = []
        result["low_conf_candidates"] = []
        result["num_detections"] = 0
        result["max_decided_confidence"] = None
        result["hazard"] = {"level": "轻", "label": "轻（未检出）", "severity": 0,
                            "none": True, "density_per_mp": 0.0,
                            "thresholds": {"light_max_per_mp": 10.0, "medium_max_per_mp": 40.0}}
        result["advice"] = {"advice": "本次未检出目标。", "disclaimer": "原型规则提示。"}
        payload = rd.build_single_report(result, "weed", "none.jpg")
        html = rh.render_single_report_html(payload, b"\xff\xd8x")
        self.assertIn("危害等级：未检出", html)
        self.assertIn("严重度 —", html)
        self.assertNotIn("严重度 0/3", html)
        self.assertNotIn("轻（未检出）", html)

    def test_batch_failed_thumbnail_has_no_empty_img(self):
        """R11：失败图不输出 <img src="">，尺寸缺失显示“—”。"""
        summary = rd.build_batch_summary(
            [{"file_name": "bad.jpg", "status": "error", "error": "文件为空（0 字节）。"}],
            "weed", "快速单模型", decision_conf=0.5,
        )
        html = rh.render_batch_report_html(
            summary,
            [{"file_name": "bad.jpg", "jpeg": None, "badge": "none",
              "badge_text": "失败", "caption": "文件为空（0 字节）。"}],
        )
        self.assertNotIn('src=""', html)
        self.assertIn("无缩略图", html)
        self.assertNotIn("—×—", html)
        self.assertIn(">—<", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
