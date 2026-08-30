# scripts/verify_deployment.py
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from deployment_adapter import (
    DeploymentError,
    export_result,
    infer_image,
)
from models import DualModelManager


def read_image_unicode_safe(path: Path):
    """读取图片（Unicode 路径安全，与 src/yolox_inference.py 同一模式）。"""
    img = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        img = np.asarray(
            Image.open(path).convert("RGB")
        )[:, :, ::-1]

    return img


def require_image(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} 不存在：{path}"
        )

    img = read_image_unicode_safe(path)

    if img is None:
        raise RuntimeError(
            f"{label} 无法解码：{path}"
        )

    return img


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def print_result(label: str, result: dict):
    print(
        f"[ OK ] {label}: "
        f"detections={result['num_detections']}, "
        f"device={result['device']}, "
        f"net={result['net_inference_ms']} ms, "
        f"total={result['inference_time_ms']} ms"
    )


def verify_weed_fusion(
    img,
    cfg: dict,
    manager: DualModelManager,
):
    manager.switch_task("weed")

    result = infer_image(
        img,
        cfg,
        manager,
        task="weed",
        mode="fusion",
    )

    assert_true(
        result["detections"],
        "杂草高精度融合应至少检出一个目标。",
    )

    params = result["wbf_params"]
    assert_true(
        isinstance(params, dict),
        "杂草融合必须返回 wbf_params。",
    )
    assert_true(
        params.get("iou_thr") == 0.65,
        f"杂草 WBF iou_thr 错误：{params.get('iou_thr')}",
    )
    assert_true(
        params.get("skip_box_thr") == 0.01,
        f"杂草 WBF skip_box_thr 错误：{params.get('skip_box_thr')}",
    )
    assert_true(
        params.get("conf_type") == "max",
        f"杂草 WBF conf_type 错误：{params.get('conf_type')}",
    )
    assert_true(
        params.get("weights") == [1.0, 1.0, 1.0],
        f"杂草 WBF weights 错误：{params.get('weights')}",
    )

    print_result(
        "weed fusion",
        result,
    )
    return result


def verify_weed_fast(
    img,
    cfg: dict,
    manager: DualModelManager,
):
    manager.switch_task("weed")

    result = infer_image(
        img,
        cfg,
        manager,
        task="weed",
        mode="yolo11",
    )

    assert_true(
        isinstance(result["detections"], list),
        "杂草快速模式 detections 必须为 list。",
    )
    assert_true(
        result["wbf_params"] is None,
        "杂草快速模式 wbf_params 应为 None。",
    )

    print_result(
        "weed fast",
        result,
    )
    return result


def verify_pest_fusion(
    img,
    cfg: dict,
    manager: DualModelManager,
):
    manager.switch_task("pest")

    result = infer_image(
        img,
        cfg,
        manager,
        task="pest",
        mode="fusion",
    )

    assert_true(
        isinstance(result.get("class_counts"), dict),
        "害虫融合必须存在 class_counts。",
    )
    assert_true(
        result["wbf_params"].get("type") == "classwise_WBF",
        "害虫融合必须使用 classwise_WBF。",
    )

    print_result(
        "pest fusion",
        result,
    )
    return result


def verify_pest_fast(
    img,
    cfg: dict,
    manager: DualModelManager,
):
    manager.switch_task("pest")

    result = infer_image(
        img,
        cfg,
        manager,
        task="pest",
        mode="fast",
    )

    assert_true(
        isinstance(result.get("class_counts"), dict),
        "害虫快速模式必须存在 class_counts。",
    )
    assert_true(
        result["wbf_params"] is None,
        "害虫快速模式 wbf_params 应为 None。",
    )

    print_result(
        "pest fast",
        result,
    )
    return result


def verify_missing_weight(
    img,
    cfg: dict,
    manager: DualModelManager,
):
    broken_cfg = copy.deepcopy(cfg)
    broken_cfg["weed"]["paths"]["yolo11_weights"] = str(
        ROOT / "models" / "this_file_must_not_exist.pt"
    )

    try:
        infer_image(
            img,
            broken_cfg,
            manager,
            task="weed",
            mode="yolo11",
        )
    except DeploymentError as exc:
        assert_true(
            "模型未配置" in str(exc),
            f"缺权重异常文案错误：{exc}",
        )
        print("[ OK ] missing weight: 正确抛出「模型未配置」")
        return

    raise AssertionError(
        "缺失权重场景没有抛出 DeploymentError。"
    )


def verify_export(
    img,
    result: dict,
    output_dir: Path,
):
    image_path, json_path = export_result(
        img,
        result,
        task="weed",
        mode=result["mode"],
        filename="deployment_smoke.jpg",
        output_dir=output_dir,
    )

    assert_true(
        image_path.is_file(),
        "导出的 JPG 文件不存在。",
    )
    assert_true(
        json_path.is_file(),
        "导出的 JSON 文件不存在。",
    )

    exported_image = read_image_unicode_safe(
        image_path
    )
    assert_true(
        exported_image is not None,
        "导出的 JPG 无法解码。",
    )

    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert_true(
        "note" in payload,
        "JSON 缺少 note 原型声明。",
    )
    assert_true(
        "detections" in payload,
        "JSON 缺少 detections。",
    )
    assert_true(
        "hazard" in payload,
        "JSON 缺少 hazard。",
    )
    assert_true(
        "advice" in payload,
        "JSON 缺少 advice。",
    )

    print(
        f"[ OK ] export: "
        f"{image_path.name}, "
        f"{json_path.name}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WheatFieldAI deployment smoke verification"
    )
    parser.add_argument(
        "--weed-image",
        required=True,
        help="真实杂草测试图片路径",
    )
    parser.add_argument(
        "--pest-image",
        required=True,
        help="真实害虫测试图片路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "deployment_smoke"),
        help="导出冒烟结果目录",
    )
    args = parser.parse_args()

    weed_path = Path(args.weed_image).resolve()
    pest_path = Path(args.pest_image).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 70)
    print("WheatFieldAI deployment smoke verification")
    print("=" * 70)
    print(f"weed image: {weed_path}")
    print(f"pest image: {pest_path}")
    print(f"output: {output_dir}")

    cfg = load_config(ROOT / "config.yaml")
    weed_img = require_image(
        weed_path,
        "杂草测试图片",
    )
    pest_img = require_image(
        pest_path,
        "害虫测试图片",
    )

    manager = DualModelManager(cfg)

    try:
        weed_fusion = verify_weed_fusion(
            weed_img,
            cfg,
            manager,
        )
        verify_weed_fast(
            weed_img,
            cfg,
            manager,
        )
        verify_pest_fusion(
            pest_img,
            cfg,
            manager,
        )
        verify_pest_fast(
            pest_img,
            cfg,
            manager,
        )
        verify_missing_weight(
            weed_img,
            cfg,
            manager,
        )
        verify_export(
            weed_img,
            weed_fusion,
            output_dir,
        )
    finally:
        try:
            manager.free_gpu()
        except Exception:
            pass

    print("-" * 70)
    print("[RESULT] deployment smoke verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())