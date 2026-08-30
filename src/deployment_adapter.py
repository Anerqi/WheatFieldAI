# src/deployment_adapter.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import DualModelManager
from pest_inference import infer_pest_single_image
from weed_inference import infer_weed_single_image
from drawing import draw_annotations, encode_image_to_bytes


class DeploymentError(RuntimeError):
    """部署层统一异常。"""


def infer_image(
    img_bgr,
    cfg: dict,
    manager: DualModelManager,
    task: str,
    mode: str,
) -> dict:
    """统一封装真实推理入口。不实现任何模型算法，只调用现有已验证 inference 模块。"""
    if task == "weed":
        if mode not in {"fusion", "yolo11"}:
            raise DeploymentError(f"非法杂草模式: {mode}")
        missing = missing_weights(cfg, task, mode)
        if missing:
            raise DeploymentError(
                "模型未配置：以下权重文件不存在，无法运行当前模式：\n- "
                + "\n".join(missing)
                + "\n请按 README 将权重放入 models/ 目录，或用环境变量覆盖路径。"
            )
        result = infer_weed_single_image(
            img_bgr,
            cfg,
            manager,
            mode=mode,
        )
    elif task == "pest":
        if mode not in {"fusion", "fast"}:
            raise DeploymentError(f"非法害虫模式: {mode}")
        missing = missing_weights(cfg, task, mode)
        if missing:
            raise DeploymentError(
                "模型未配置：以下权重文件不存在，无法运行当前模式：\n- "
                + "\n".join(missing)
                + "\n请按 README 将权重放入 models/ 目录，或用环境变量覆盖路径。"
            )
        result = infer_pest_single_image(
            img_bgr,
            cfg,
            manager._pest,
            mode=mode,
        )
    else:
        raise DeploymentError(f"非法任务: {task}")

    validate_result(result, task, mode)
    return result


def validate_result(
    result: dict,
    task: str,
    mode: str,
) -> None:
    """只校验结构，不修改推理结果。"""
    if not isinstance(result, dict):
        raise DeploymentError("推理函数返回值不是 dict")

    if result.get("status") != "ok":
        raise DeploymentError(
            f"模型推理未成功: status={result.get('status')!r}"
        )

    required = {
        "status",
        "mode",
        "mode_display",
        "models_used",
        "inference_time_ms",
        "net_inference_ms",
        "model_load_ms",
        "includes_first_load",
        "per_model_time_ms",
        "width",
        "height",
        "num_detections",
        "mean_confidence",
        "max_confidence",
        "detections",
        "hazard",
        "advice",
        "wbf_params",
        "class_mapping",
        "device",
    }

    missing = sorted(required - result.keys())
    if missing:
        raise DeploymentError(
            f"推理结果缺少字段: {missing}"
        )

    if result["mode"] != mode:
        raise DeploymentError(
            f"推理模式不一致: requested={mode}, returned={result['mode']}"
        )

    if task == "pest":
        for key in ("class_counts", "major_class_count"):
            if key not in result:
                raise DeploymentError(
                    f"害虫结果缺少字段: {key}"
                )

    if not isinstance(result["detections"], list):
        raise DeploymentError("detections 必须为 list")

    for index, det in enumerate(result["detections"]):
        if not isinstance(det, dict):
            raise DeploymentError(
                f"detections[{index}] 不是 dict"
            )

        required_detection = {
            "category_name",
            "confidence",
            "bbox_xyxy",
        }

        if not required_detection.issubset(det):
            raise DeploymentError(
                f"detections[{index}] 缺少必要字段"
            )

        if task == "weed":
            for key in (
                "yolo_class_id",
                "coco_category_id",
            ):
                if key not in det:
                    raise DeploymentError(
                        f"detections[{index}] 缺少杂草字段: {key}"
                    )

        if task == "pest":
            if "class_id" not in det:
                raise DeploymentError(
                    f"detections[{index}] 缺少害虫字段: class_id"
                )


def export_result(
    img_bgr,
    result: dict,
    task: str,
    mode: str,
    filename: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    导出真实推理结果：
    1. 带标注 JPG
    2. 结构化 JSON
    """
    validate_result(result, task, mode)

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = Path(filename).stem
    safe_stem = "".join(
        char
        if char.isalnum() or char in "._-"
        else "_"
        for char in stem
    )[:120]

    image_path = output_path / f"{safe_stem}_annotated.jpg"
    json_path = output_path / f"{safe_stem}_result.json"

    annotated = draw_annotations(
        img_bgr,
        result["detections"],
    )

    image_bytes = encode_image_to_bytes(
        annotated,
        ".jpg",
        quality=90,
    )

    if not image_bytes:
        raise DeploymentError("标注 JPG 编码失败")

    image_path.write_bytes(image_bytes)

    payload: dict[str, Any] = {
        "filename": filename,
        "task": task,
        "mode": result["mode_display"],
        "models_used": result["models_used"],
        "inference_time_ms": result["inference_time_ms"],
        "net_inference_ms": result["net_inference_ms"],
        "model_load_ms": result["model_load_ms"],
        "includes_first_load": result["includes_first_load"],
        "per_model_time_ms": result["per_model_time_ms"],
        "device": result["device"],
        "image_size": {
            "width": result["width"],
            "height": result["height"],
        },
        "num_detections": result["num_detections"],
        "mean_confidence": result["mean_confidence"],
        "max_confidence": result["max_confidence"],
        "detections": result["detections"],
        "hazard": result["hazard"],
        "advice": result["advice"],
        "wbf_params": result["wbf_params"],
        "class_mapping": result["class_mapping"],
        "note": (
            "本 JSON 为原型系统输出，"
            "危害等级与防治建议需农学专家审核。"
        ),
    }

    if task == "pest":
        payload["class_counts"] = result.get(
            "class_counts",
            {},
        )
        payload["major_class_count"] = result.get(
            "major_class_count",
            0,
        )

    json_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return image_path, json_path


def get_required_weights(
    cfg: dict,
    task: str,
    mode: str,
) -> list[str]:
    """返回当前任务/模式真正需要的权重路径。"""
    if task == "weed":
        paths = cfg["weed"]["paths"]
        if mode == "yolo11":
            return [paths["yolo11_weights"]]
        if mode == "fusion":
            return [
                paths["yolo11_weights"],
                paths["yolox_small_weights"],
                paths["yolox_base_weights"],
            ]

    elif task == "pest":
        paths = cfg["pest"]["paths"]
        if mode == "fast":
            return [paths["yolo11m_weights"]]
        if mode == "fusion":
            return [
                paths["yolo11m_weights"],
                paths["yolo11l_weights"],
                paths["yolo11s_weights"],
            ]

    raise DeploymentError(
        f"无法确定任务/模式所需权重: task={task}, mode={mode}"
    )


def missing_weights(
    cfg: dict,
    task: str,
    mode: str,
) -> list[str]:
    """只检查当前模式真正需要的权重。"""
    return [
        str(path)
        for path in get_required_weights(
            cfg,
            task,
            mode,
        )
        if not Path(path).is_file()
    ]
