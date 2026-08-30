# -*- coding: utf-8 -*-
"""
weed_inference.py
=================
杂草单图推理编排（任务 08 双任务系统的杂草侧，逻辑与任务 07 一致）。

== 来源记录（Provenance）==
本文件逻辑源自任务 07 src/inference.py，未重新实现算法：
仅将配置键从顶层（inference / wbf / class_mapping）适配为本系统
双任务结构的 cfg["weed"] 子节点；危害等级与防治建议改为双任务
hazard / advice 模块并传入 task="weed"。
- YOLO11s : conf=0.001, iou=0.70, imgsz=960
- YOLOX   : conf=0.01, nms=0.65, imgsz=640
- WBF     : iou_thr=0.65, skip_box_thr=0.01, conf_type=max, weights=[1,1,1]
类别映射：YOLO 0 -> COCO 1 -> Obonianghao（常量定义：src/yolox_inference.py；与 config.yaml weed.class_mapping 一致）
== 来源记录结束 ==
"""

import time

import numpy as np

from yolox_inference import (
    run_yolox_model, read_image_bytes,
    YOLO_CLASS_ID, COCO_CATEGORY_ID, CATEGORY_NAME,
)
from wbf import wbf_fuse_boxes
from hazard import compute_hazard_level
from advice import get_advice


def run_yolo11_single(model, img_bgr, device, conf, iou, img_size):
    """YOLO11s 单图推理（与任务 03 run_yolo11 单图等价）。返回 (boxes (N,4), scores (N,))。"""
    from ultralytics import YOLO  # noqa: F401
    res = model.predict(img_bgr, imgsz=img_size, conf=conf, iou=iou, device=device,
                        batch=1, verbose=False, augment=False)[0]
    box = res.boxes
    if box is not None and len(box) > 0:
        xyxy = box.xyxy.cpu().numpy()
        confs = box.conf.cpu().numpy()
        return xyxy.astype(float), confs.astype(float)
    return np.zeros((0, 4)), np.zeros((0,))


def infer_weed_single_image(img_bgr, cfg, weed_manager, mode="fusion"):
    """对单张 BGR 图执行杂草推理，返回结构化结果 dict（与任务 07 返回值结构一致）。

    Args:
        img_bgr: BGR ndarray
        cfg: 配置 dict（双任务结构）
        weed_manager: WeedModelManager（任务 08 双任务模型管理器中的杂草侧）
        mode: "fusion"（高精度融合）| "yolo11"（快速单模型）
    """
    h, w = img_bgr.shape[:2]
    weed_cfg = cfg["weed"]
    inf_cfg = weed_cfg["inference"]
    wbf_cfg = weed_cfg["wbf"]
    device = weed_manager.device_str()

    t_start = time.time()
    per_model_times = {}
    load_times = {}

    def _time_get(fn, name):
        t0 = time.time()
        out = fn()
        load_times[name] = round((time.time() - t0) * 1000, 1)
        return out

    if mode == "fusion":
        # ---- YOLO11s ----
        m11 = _time_get(weed_manager.get_yolo11, "YOLO11s_load")
        t0 = time.time()
        b11, s11 = run_yolo11_single(m11, img_bgr, device,
                                     inf_cfg["yolo11_conf"], inf_cfg["yolo11_iou"],
                                     inf_cfg["yolo11_imgsz"])
        per_model_times["YOLO11s"] = round((time.time() - t0) * 1000, 1)

        # ---- YOLOX Small / Base ----
        boxes_list, scores_list = [], []
        for size in ("small", "base"):
            m = _time_get(lambda s=size: weed_manager.get_yolox(s), f"YOLOX-{size.title()}_load")
            t0 = time.time()
            b, s = run_yolox_model(m, img_bgr, device,
                                   conf_thre=inf_cfg["yolox_conf"],
                                   nms_thre=inf_cfg["yolox_nms"],
                                   img_size=inf_cfg["yolox_imgsz"])
            per_model_times[f"YOLOX-Dinov3 {size.title()}"] = round((time.time() - t0) * 1000, 1)
            boxes_list.append(b)
            scores_list.append(s)

        # ---- WBF 融合（任务 05 固定参数）----
        t0 = time.time()
        fused_boxes, fused_scores = wbf_fuse_boxes(
            [b11] + boxes_list, [s11] + scores_list,
            list(wbf_cfg["weights"]),
            iou_thr=wbf_cfg["iou_thr"],
            skip_box_thr=wbf_cfg["skip_box_thr"],
            conf_type=wbf_cfg["conf_type"],
        )
        per_model_times["WBF"] = round((time.time() - t0) * 1000, 1)
        boxes, scores = fused_boxes, fused_scores
        models_used = ["YOLO11s", "YOLOX-Dinov3 Small", "YOLOX-Dinov3 Base"]
        wbf_params = {
            "iou_thr": wbf_cfg["iou_thr"],
            "skip_box_thr": wbf_cfg["skip_box_thr"],
            "conf_type": wbf_cfg["conf_type"],
            "weights": list(wbf_cfg["weights"]),
        }
    else:  # 快速单模型
        m11 = _time_get(weed_manager.get_yolo11, "YOLO11s_load")
        t0 = time.time()
        boxes, scores = run_yolo11_single(m11, img_bgr, device,
                                          inf_cfg["yolo11_conf"],
                                          inf_cfg["yolo11_iou"],
                                          inf_cfg["yolo11_imgsz"])
        per_model_times["YOLO11s"] = round((time.time() - t0) * 1000, 1)
        models_used = ["YOLO11s"]
        wbf_params = None

    pure_infer_ms = round((time.time() - t_start) * 1000, 1)
    model_load_ms = round(sum(load_times.values()), 1)
    net_inference_ms = round(max(pure_infer_ms - model_load_ms, 0.0), 1)
    total_ms = round(pure_infer_ms, 1)

    # ---- 结构化检测结果 ----
    detections = []
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k]
        detections.append({
            "yolo_class_id": YOLO_CLASS_ID,
            "coco_category_id": COCO_CATEGORY_ID,
            "category_name": CATEGORY_NAME,
            "confidence": round(float(scores[k]), 6),
            "bbox_xyxy": [round(float(x1), 2), round(float(y1), 2),
                          round(float(x2), 2), round(float(y2), 2)],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    confs = np.array([d["confidence"] for d in detections], dtype=float) if detections else np.array([])

    # ---- 危害等级与防治建议 ----
    hazard = compute_hazard_level(len(detections), w, h, cfg["hazard"], task="weed")
    advice = get_advice(hazard["level"], cfg["advice"], task="weed", level_str=hazard["label"])

    return {
        "status": "ok",
        "mode": mode,
        "mode_display": "高精度融合" if mode == "fusion" else "快速单模型（YOLO11s）",
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
        "hazard": hazard,
        "advice": advice,
        "wbf_params": wbf_params,
        "class_mapping": {
            "yolo_class_id": YOLO_CLASS_ID,
            "coco_category_id": COCO_CATEGORY_ID,
            "category_name": CATEGORY_NAME,
            "note": "YOLO 内部类别 0 在 COCO 评测标注中映射为 category_id 1",
        },
        "device": device,
    }


def decode_upload(data: bytes):
    """解码上传字节为 BGR ndarray；非法/空返回 None。"""
    if not data or len(data) == 0:
        return None
    return read_image_bytes(data)
