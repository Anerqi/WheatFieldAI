# -*- coding: utf-8 -*-
"""
pest_inference.py
=================
害虫推理编排：
- 高精度融合：YOLO11m + YOLO11l + YOLO11s + refined classwise WBF
- 快速单模型：YOLO11m（与提交包 fast_model 配置一致）
- 推理参数 conf=0.001, iou=0.7, imgsz=960（与提交包 export_ensemble_zip.py 默认值一致）
- 类别映射：32 类，严格从 dataset.yaml 加载（唯一数据源）
"""

import time

import numpy as np

from pest_models import PestModelManager, PEST_NUM_CLASSES
from pest_wbf import load_class_configs, classwise_wbf
from hazard import compute_hazard_level
from advice import get_advice


def run_yolo_detections(model, img_bgr, device, conf, iou, img_size):
    """对单张 BGR 图执行 YOLO 推理，返回 (boxes (N,4) 像素 xyxy, classes (N,), scores (N,))。

    与提交包 export_ensemble_zip.py 的 predict 函数等价，区别：
    - 提交包把框归一化到 [0,1]（最终输出 YOLO 标签格式）；
    - 本函数直接返回原图像素坐标（Web 界面需要）。WBF 融合对坐标空间尺度不变。
    """
    from ultralytics import YOLO  # noqa: F401
    res = model.predict(img_bgr, imgsz=img_size, conf=conf, iou=iou, device=device,
                        batch=1, verbose=False, augment=False)[0]
    box = res.boxes
    if box is not None and len(box) > 0:
        return (box.xyxy.cpu().numpy().astype(float),
                box.cls.cpu().numpy().astype(int),
                box.conf.cpu().numpy().astype(float))
    return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))


def infer_pest_single_image(img_bgr, cfg, pest_manager, mode="fusion"):
    """对单张 BGR 图执行害虫推理。

    Args:
        img_bgr: BGR ndarray
        cfg: 完整配置 dict
        pest_manager: PestModelManager 实例
        mode: "fusion"（三模型融合）| "fast"（YOLO11m 单模型）

    Returns:
        dict: 结构化结果（含 detections、hazard、advice、timing、元信息）
    """
    h, w = img_bgr.shape[:2]
    pest_cfg = cfg["pest"]
    inf = pest_cfg["inference"]
    device = pest_manager.device

    t_start = time.time()
    load_times = {}
    per_model_times = {}

    def _time_get(fn, name):
        t0 = time.time()
        out = fn()
        load_times[name] = round((time.time() - t0) * 1000, 1)
        return out

    class_names = pest_manager.class_names()

    if mode == "fusion":
        # ---- 三模型推理 ----
        m11m = _time_get(lambda: pest_manager.get("yolo11m"), "YOLO11m_load")
        m11l = _time_get(lambda: pest_manager.get("yolo11l"), "YOLO11l_load")
        m11s = _time_get(lambda: pest_manager.get("yolo11s"), "YOLO11s_load")

        preds = []
        for key, m in [("YOLO11m", m11m), ("YOLO11l", m11l), ("YOLO11s", m11s)]:
            t0 = time.time()
            b, c, s = run_yolo_detections(m, img_bgr, device, inf["conf"], inf["iou"], inf["imgsz"])
            per_model_times[key] = round((time.time() - t0) * 1000, 1)
            preds.append((b, c, s))

        # ---- classwise WBF 融合 ----
        class_configs = load_class_configs(pest_cfg.get("classwise_config"))
        if class_configs is None:
            raise RuntimeError("害虫高精度融合模式需要 classwise_config 配置（refined classwise WBF JSON）")
        t0 = time.time()
        fb, fc, fs = classwise_wbf(preds, class_configs, num_classes=PEST_NUM_CLASSES)
        per_model_times["classwise_WBF"] = round((time.time() - t0) * 1000, 1)
        models_used = ["YOLO11m", "YOLO11l", "YOLO11s"]
        wbf_params = {"type": "classwise_WBF", "config_source": pest_cfg.get("classwise_config", "")}
    else:  # fast
        m11m = _time_get(lambda: pest_manager.get("yolo11m"), "YOLO11m_load")
        t0 = time.time()
        fb, fc, fs = run_yolo_detections(m11m, img_bgr, device, inf["conf"], inf["iou"], inf["imgsz"])
        per_model_times["YOLO11m"] = round((time.time() - t0) * 1000, 1)
        models_used = ["YOLO11m"]
        wbf_params = None

    pure_infer_ms = round((time.time() - t_start) * 1000, 1)
    model_load_ms = round(sum(load_times.values()), 1)
    net_inference_ms = round(max(pure_infer_ms - model_load_ms, 0.0), 1)
    total_ms = round(pure_infer_ms, 1)

    # ---- 结构化检测结果 ----
    detections = []
    for k in range(len(fb)):
        x1, y1, x2, y2 = fb[k]
        cls_id = int(fc[k])
        cname = class_names[cls_id]
        detections.append({
            "class_id": cls_id,
            "category_name": cname,
            "confidence": round(float(fs[k]), 6),
            "bbox_xyxy": [round(float(x1), 2), round(float(y1), 2),
                          round(float(x2), 2), round(float(y2), 2)],
        })
    # 按置信度降序排序
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ---- 统计 ----
    confs = np.array([d["confidence"] for d in detections], dtype=float) if detections else np.array([])
    # 主要类别数（检出类别去重）
    major_class_ids = set(d["class_id"] for d in detections) if detections else set()
    major_class_count = len(major_class_ids)

    # ---- 危害等级与防治建议 ----
    hazard = compute_hazard_level(len(detections), w, h, cfg["hazard"], task="pest",
                                  major_class_count=major_class_count)
    advice = get_advice(hazard["level"], cfg["advice"], task="pest", level_str=hazard["label"])

    # ---- 每类检测数量统计 ----
    class_counts = {}
    if detections:
        for d in detections:
            cn = d["category_name"]
            class_counts[cn] = class_counts.get(cn, 0) + 1
        # 按数量降序排列
        class_counts = dict(sorted(class_counts.items(), key=lambda x: -x[1]))

    return {
        "status": "ok",
        "mode": mode,
        "mode_display": "高精度融合（三模型 classwise WBF）" if mode == "fusion" else "快速单模型（YOLO11m）",
        "models_used": models_used,
        "inference_time_ms": total_ms,
        "net_inference_ms": net_inference_ms,
        "model_load_ms": model_load_ms,
        "includes_first_load": bool(model_load_ms > 800),
        "per_model_time_ms": per_model_times,
        "width": int(w),
        "height": int(h),
        "num_detections": len(detections),
        "mean_confidence": round(float(confs.mean()), 4) if len(confs) else None,
        "max_confidence": round(float(confs.max()), 4) if len(confs) else None,
        "detections": detections,
        "class_counts": class_counts,
        "major_class_count": major_class_count,
        "hazard": hazard,
        "advice": advice,
        "wbf_params": wbf_params,
        "class_mapping": {
            "num_classes": PEST_NUM_CLASSES,
            "source": pest_cfg.get("dataset_yaml", ""),
            "note": "32 类英文名称严格来自提交包 configs/dataset.yaml（唯一数据源），未重排 ID 或编造中文名",
        },
        "device": device,
    }