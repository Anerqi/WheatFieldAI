# -*- coding: utf-8 -*-
"""批量总览数据层单元测试（不加载模型、不读取真实图片、不依赖 Streamlit）。

覆盖 src/batch_summary.py：
- 文件名安全化与重名去重（ZIP 内不互相覆盖）
- 单图摘要（正式框数 / 正式最高置信度 / 危害等级 / 未检出中性状态）
- 单图 JSON payload 字段集合（与单图导出一致，批量 ZIP 不丢字段）
- 批量 ZIP 结构：成功图 2 个文件、失败图跳过并写入 batch_summary.json
- 缩略图只缩不放

运行：python -B -m unittest discover -s tests -v
"""

import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import batch_summary as bs  # noqa: E402

# 单图 JSON 必须包含的字段（与 app.py 既有单图导出逐项对应）
REQUIRED_PAYLOAD_KEYS = {
    "filename", "generated_at", "task", "mode", "models_used",
    "inference_time_ms", "image_size", "num_detections", "mean_confidence",
    "max_confidence", "decision_conf", "num_candidates_total", "detections",
    "low_conf_candidates", "class_counts", "hazard", "advice", "wbf_params",
    "class_mapping", "device", "note",
}


def make_result(num=2, max_decided=0.812, hazard=None, width=1280, height=1024,
                none=False, task="weed"):
    """构造一个字段齐全的推理结果 dict（结构与 weed/pest inference 返回一致）。"""
    dets = [
        {"category_name": "Obonianghao", "confidence": max_decided - i * 0.05,
         "bbox_xyxy": [10.0 + i, 20.0 + i, 40.0 + i, 60.0 + i]}
        for i in range(num)
    ]
    hazard = hazard or {
        "level": "中", "label": "中", "severity": 2, "density_per_mp": 22.5,
        "none": none, "decision_conf": 0.5,
    }
    return {
        "status": "ok",
        "mode": "fusion",
        "mode_display": "高精度融合（五模型 WBF）",
        "models_used": ["YOLO11s", "YOLO11s-box8"],
        "inference_time_ms": 321.5,
        "net_inference_ms": 300.0,
        "model_load_ms": 21.5,
        "includes_first_load": False,
        "per_model_time_ms": {"YOLO11s": 100.0},
        "width": width,
        "height": height,
        "num_detections": num,
        "num_candidates_total": num + 3,
        "decision_conf": 0.5,
        "mean_confidence": max_decided - 0.1 if num else None,
        "max_confidence": max_decided,
        "max_decided_confidence": max_decided if num else None,
        "detections": dets,
        "low_conf_candidates": [{"category_name": "Obonianghao", "confidence": 0.02,
                                 "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]}],
        "class_counts": {"Obonianghao": num},
        "hazard": hazard,
        "advice": {"advice": "建议人工复核。", "disclaimer": "原型规则，未经农学专家校准。"},
        "wbf_params": {"iou_thr": 0.65, "skip_box_thr": 0.01, "conf_type": "max",
                       "weights": [1, 1, 1, 1, 1]},
        "class_mapping": {"category_name": "Obonianghao", "note": "测试"},
        "device": "cuda:0",
    }


def ok_entry(index, filename, result=None, **kwargs):
    return {
        "index": index,
        "filename": filename,
        "size_kb": 512.0,
        "image": None,
        "status": "ok",
        "error": None,
        "result": result if result is not None else make_result(**kwargs),
    }


def bad_entry(index, filename, status="invalid", error="文件为空（0 字节）。"):
    return {
        "index": index,
        "filename": filename,
        "size_kb": 0.0,
        "image": None,
        "status": status,
        "error": error,
        "result": None,
    }


class SanitizeStemTests(unittest.TestCase):
    def test_strips_directories(self):
        self.assertEqual(bs.sanitize_stem("C:\\data\\a.jpg"), "a")
        self.assertEqual(bs.sanitize_stem("../../etc/passwd.png"), "passwd")

    def test_replaces_illegal_chars(self):
        # 中文属于 \\w（Unicode 词字符），保留；空格与 # 等替换为 _
        self.assertEqual(bs.sanitize_stem("田间 图#1.jpg"), "田间_图_1")
        self.assertEqual(bs.sanitize_stem("a b:c*d.jpg"), "a_b_c_d")

    def test_empty_falls_back(self):
        self.assertEqual(bs.sanitize_stem(""), "image")
        self.assertEqual(bs.sanitize_stem("..."), "image")

    def test_length_capped(self):
        self.assertEqual(len(bs.sanitize_stem("x" * 300 + ".jpg")), 80)


class UniqueStemsTests(unittest.TestCase):
    def test_dedupes_case_insensitively(self):
        stems = bs.unique_stems(["a.jpg", "a.png", "A.JPEG", "b.jpg"])
        self.assertEqual(stems, ["a", "a_2", "A_3", "b"])
        self.assertEqual(len({s.lower() for s in stems}), 4)

    def test_all_illegal_names_still_unique(self):
        stems = bs.unique_stems(["图.jpg", "图.png"])
        self.assertEqual(len({s.lower() for s in stems}), 2)


class SummaryTests(unittest.TestCase):
    def test_decided_max_prefers_explicit_field(self):
        r = make_result(num=2, max_decided=0.9)
        r["max_decided_confidence"] = 0.77
        self.assertEqual(bs.decided_max_confidence(r), 0.77)

    def test_decided_max_falls_back_to_detections(self):
        r = make_result(num=2, max_decided=0.8)
        r.pop("max_decided_confidence")
        self.assertEqual(bs.decided_max_confidence(r), 0.8)

    def test_decided_max_none_when_no_detection(self):
        r = make_result(num=0, none=True)
        self.assertIsNone(bs.decided_max_confidence(r))
        self.assertIsNone(bs.decided_max_confidence(None))

    def test_hazard_display_none_state(self):
        self.assertEqual(bs.hazard_display(make_result(num=0, none=True)), "未检出")
        self.assertEqual(bs.hazard_display(make_result(num=3)), "中")
        self.assertEqual(bs.hazard_display(None), "—")

    def test_summary_row_fields(self):
        rows = bs.build_summary_rows([
            ok_entry(0, "a.jpg", width=1280, height=1024, max_decided=0.812),
            bad_entry(1, "b.jpg"),
        ])
        self.assertEqual(list(rows[0].keys()),
                         ["文件名", "尺寸", "正式框数", "正式最高置信度", "危害等级", "耗时(ms)", "状态"])
        self.assertEqual(rows[0]["尺寸"], "1280×1024")
        self.assertEqual(rows[0]["正式框数"], 2)
        self.assertEqual(rows[0]["正式最高置信度"], "0.812")
        self.assertEqual(rows[0]["状态"], "完成")
        self.assertEqual(rows[1]["尺寸"], "—")
        self.assertEqual(rows[1]["正式最高置信度"], "—")
        self.assertEqual(rows[1]["状态"], "校验失败")

    def test_batch_totals(self):
        totals = bs.batch_totals([
            ok_entry(0, "a.jpg", num=2),
            ok_entry(1, "b.jpg", num=5),
            bad_entry(2, "c.jpg"),
        ])
        self.assertEqual(totals["total"], 3)
        self.assertEqual(totals["ok"], 2)
        self.assertEqual(totals["failed"], 1)
        self.assertEqual(totals["boxes"], 7)
        self.assertEqual(totals["max_boxes_file"], "b.jpg")


class PayloadTests(unittest.TestCase):
    def test_payload_has_all_required_keys(self):
        payload = bs.build_result_payload("a.jpg", make_result(), "杂草检测")
        self.assertEqual(set(payload.keys()), REQUIRED_PAYLOAD_KEYS)
        self.assertEqual(payload["task"], "杂草检测")
        self.assertEqual(payload["decision_conf"], 0.5)
        self.assertEqual(payload["image_size"], {"width": 1280, "height": 1024})
        self.assertEqual(payload["low_conf_candidates"][0]["confidence"], 0.02)

    def test_payload_json_bytes_roundtrip_keeps_chinese(self):
        raw = bs.payload_to_json_bytes(bs.build_result_payload("a.jpg", make_result(), "杂草检测"))
        text = raw.decode("utf-8")
        self.assertIn("高精度融合", text)
        self.assertEqual(json.loads(text)["num_detections"], 2)


class ZipTests(unittest.TestCase):
    def test_zip_contains_two_files_per_ok_image_and_summary(self):
        entries = [ok_entry(0, "a.jpg"), ok_entry(1, "b.jpg", num=4)]
        zip_bytes = bs.build_batch_zip(
            entries, "杂草检测", "高精度融合（五模型 WBF）",
            generated_at="2026-10-30 10:00:00",
            annotated_jpegs={0: b"jpg-a", 1: b"jpg-b"},
        )
        names = bs.zip_manifest(zip_bytes)
        self.assertEqual(sorted(names), sorted([
            "a_annotated.jpg", "a_result.json",
            "b_annotated.jpg", "b_result.json",
            "batch_summary.json",
        ]))
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            self.assertEqual(zf.read("a_annotated.jpg"), b"jpg-a")
            summary = json.loads(zf.read("batch_summary.json").decode("utf-8"))
            payload = json.loads(zf.read("b_result.json").decode("utf-8"))
        self.assertEqual(summary["counts"], {"total": 2, "ok": 2, "failed": 0,
                                             "total_detections": 6})
        self.assertEqual(summary["generated_at"], "2026-10-30 10:00:00")
        self.assertEqual(set(payload.keys()), REQUIRED_PAYLOAD_KEYS)

    def test_zip_skips_failed_files_and_records_error(self):
        entries = [ok_entry(0, "a.jpg"), bad_entry(1, "b.jpg", status="error",
                                                   error="CUDA out of memory")]
        zip_bytes = bs.build_batch_zip(entries, "杂草检测", "快速单模型（YOLO11s）",
                                       annotated_jpegs={0: b"jpg-a"})
        names = bs.zip_manifest(zip_bytes)
        self.assertNotIn("b_annotated.jpg", names)
        self.assertNotIn("b_result.json", names)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            summary = json.loads(zf.read("batch_summary.json").decode("utf-8"))
        self.assertEqual(summary["counts"]["failed"], 1)
        failed = [f for f in summary["files"] if f["status"] == "error"][0]
        self.assertEqual(failed["error"], "CUDA out of memory")
        self.assertTrue(failed["skipped"])
        self.assertEqual(failed["zip_files"], [])

    def test_zip_dedupes_same_stem(self):
        entries = [ok_entry(0, "a.jpg"), ok_entry(1, "a.png")]
        names = bs.zip_manifest(bs.build_batch_zip(entries, "杂草检测", "快速单模型（YOLO11s）"))
        self.assertIn("a_result.json", names)
        self.assertIn("a_2_result.json", names)

    def test_zip_without_annotated_bytes_still_writes_json(self):
        entries = [ok_entry(0, "a.jpg")]
        names = bs.zip_manifest(bs.build_batch_zip(entries, "杂草检测", "快速单模型（YOLO11s）"))
        self.assertIn("a_result.json", names)
        self.assertNotIn("a_annotated.jpg", names)


class ThumbnailTests(unittest.TestCase):
    def test_wide_image_resized_to_target_width(self):
        img = np.zeros((600, 1200, 3), dtype=np.uint8)
        img[:] = (0, 180, 0)
        data = bs.thumbnail_jpeg(img, max_width=320)
        self.assertTrue(data.startswith(b"\xff\xd8"))
        decoded = __import__("cv2").imdecode(np.frombuffer(data, np.uint8), 1)
        self.assertEqual(decoded.shape[1], 320)
        self.assertEqual(decoded.shape[0], 160)

    def test_small_image_not_upscaled(self):
        import cv2
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        decoded = cv2.imdecode(np.frombuffer(bs.thumbnail_jpeg(img, max_width=320), np.uint8), 1)
        self.assertEqual(decoded.shape[1], 200)


if __name__ == "__main__":
    unittest.main()
