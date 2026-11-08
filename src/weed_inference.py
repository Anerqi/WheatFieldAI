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
- YOLO11s-box8 : conf=0.001, iou=0.70, imgsz=960（任务 24 融合第 2 路）
- YOLO11s-1280 : conf=0.001, iou=0.70, imgsz=1280（任务 24 融合第 3 路）
- YOLOX   : conf=0.01, nms=0.65, imgsz=640
- D-FINE-S: ONNX, score_thr=0.05, imgsz=960（任务 30 融合第 6 路，可选）
- WBF     : 六模型 iou_thr=0.68 / 五模型退回 0.65；skip_box_thr=0.01；conf_type=avg；
            score_mode=geo_agree（beta 0.412, cap 8）；等权（见 config.yaml weed.wbf）
类别映射：YOLO 0 -> COCO 1 -> Obonianghao（唯一真源：CONTEXT.md）
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
from decision import get_decision_conf, split_by_conf, confidence_stats
from fusion_profile import resolve_fusion_profile


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
        # ---- YOLO11s 主模型（960）----
        m11 = _time_get(weed_manager.get_yolo11, "YOLO11s_load")
        t0 = time.time()
        b11, s11 = run_yolo11_single(m11, img_bgr, device,
                                     inf_cfg["yolo11_conf"], inf_cfg["yolo11_iou"],
                                     inf_cfg["yolo11_imgsz"])
        per_model_times["YOLO11s"] = round((time.time() - t0) * 1000, 1)

        # ---- YOLO11s box8 / 1280 额外分支（任务 24 融合候选）----
        m_box8 = _time_get(lambda: weed_manager.get_yolo11_variant("box8"), "YOLO11s-box8_load")
        t0 = time.time()
        b_box8, s_box8 = run_yolo11_single(m_box8, img_bgr, device,
                                           inf_cfg["yolo11_conf"], inf_cfg["yolo11_iou"],
                                           inf_cfg["yolo11_box8_imgsz"])
        per_model_times["YOLO11s-box8"] = round((time.time() - t0) * 1000, 1)

        m1280 = _time_get(lambda: weed_manager.get_yolo11_variant("1280"), "YOLO11s-1280_load")
        t0 = time.time()
        b1280, s1280 = run_yolo11_single(m1280, img_bgr, device,
                                         inf_cfg["yolo11_conf"], inf_cfg["yolo11_iou"],
                                         inf_cfg["yolo11_1280_imgsz"])
        per_model_times["YOLO11s-1280"] = round((time.time() - t0) * 1000, 1)

        # ---- YOLOX Small / Base ----
        boxes_list, scores_list = [b11, b_box8, b1280], [s11, s_box8, s1280]
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

        # ---- D-FINE-S（任务 30，第 6 支成员，ONNX 推理）----
        dfine_used = False
        _t_d = time.time()
        dfine_runner = weed_manager.get_dfine_onnx() if hasattr(weed_manager, "get_dfine_onnx") else None
        # 首次构造含 ONNX 会话创建与 CUDA 上下文初始化（约十几秒），必须计入 load 而非净推理。
        # 权重存在但 onnxruntime 缺失/会话创建失败时 _ensure() 为 False：按「未参与」退回五模型，
        # 不能把空结果当第 6 路塞进 WBF（会稀释一致性调制并错用六模型阈值）。
        dfine_ready = bool(
            dfine_runner is not None and dfine_runner.available() and dfine_runner._ensure()
        )
        load_times["D-FINE-S_load"] = round((time.time() - _t_d) * 1000, 1)
        if dfine_ready:
            t0 = time.time()
            b_df, s_df = dfine_runner.infer(img_bgr)
            per_model_times["D-FINE-S"] = round((time.time() - t0) * 1000, 1)
            if getattr(dfine_runner, "last_ok", True):
                boxes_list.append(b_df)
                scores_list.append(s_df)
                dfine_used = True

        # ---- WBF 融合（任务 24 五模型；任务 30 起为六模型）----
        weights = list(wbf_cfg["weights"])
        if len(weights) > len(boxes_list):
            # 第 6 支成员不可用时（权重缺失/ONNX 依赖异常），按前 N 个权重退回，保持既有行为
            weights = weights[: len(boxes_list)]
        if len(weights) != len(boxes_list):
            raise ValueError(
                f"WBF weights 数量 {len(weights)} 与参与模型数 {len(boxes_list)} 不一致，"
                "请检查 config.yaml 的 weed.wbf.weights"
            )
        # 六模型与五模型退回档的 iou_thr / decision_conf 分别校准，必须成套切换
        fusion_profile = resolve_fusion_profile(weed_cfg, dfine_used)
        t0 = time.time()
        fused_boxes, fused_scores = wbf_fuse_boxes(
            boxes_list, scores_list,
            weights,
            iou_thr=fusion_profile["iou_thr"],
            skip_box_thr=wbf_cfg["skip_box_thr"],
            conf_type=wbf_cfg["conf_type"],
            # 任务 26：一致性几何调制（旧配置无这些键时退回 raw，行为与任务 24 一致）
            score_mode=wbf_cfg.get("score_mode", "raw"),
            geo_beta=float(wbf_cfg.get("geo_beta", 0.0)),
            geo_cap=float(wbf_cfg.get("geo_cap", 8.0)),
        )
        per_model_times["WBF"] = round((time.time() - t0) * 1000, 1)
        boxes, scores = fused_boxes, fused_scores
        models_used = [
            "YOLO11s",
            "YOLO11s-box8",
            "YOLO11s-1280",
            "YOLOX-Dinov3 Small",
            "YOLOX-Dinov3 Base",
        ]
        if dfine_used:
            models_used.append("D-FINE-S")
        wbf_params = {
            "profile": fusion_profile["profile"],
            "iou_thr": fusion_profile["iou_thr"],
            "skip_box_thr": wbf_cfg["skip_box_thr"],
            "conf_type": wbf_cfg["conf_type"],
            "score_mode": wbf_cfg.get("score_mode", "raw"),
            "geo_beta": float(wbf_cfg.get("geo_beta", 0.0)),
            "geo_cap": float(wbf_cfg.get("geo_cap", 8.0)),
            "weights": weights,
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
        fusion_profile = None

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

    # ---- 业务口径：仅 >= decision_conf 的检测参与计数与危害等级 ----
    decision_conf = (
        fusion_profile["decision_conf"] if fusion_profile else get_decision_conf(weed_cfg)
    )
    decided, candidates = split_by_conf(detections, decision_conf)
    mean_conf, max_decided_conf = confidence_stats(decided)  # 正式检测：均值 / 最高（空列表均为 None）
    _, max_conf = confidence_stats(detections)  # 最高置信度反映模型原始输出（含低置信候选，保留向后兼容）

    # ---- 危害等级与防治建议 ----
    hazard = compute_hazard_level(len(decided), w, h, cfg["hazard"], task="weed")
    hazard["decision_conf"] = decision_conf
    advice = get_advice(hazard["level"], cfg["advice"], task="weed", level_str=hazard["label"])

    return {
        "status": "ok",
        "mode": mode,
        "mode_display": (
            ("高精度融合（六模型 WBF）" if len(models_used) >= 6 else "高精度融合（五模型 WBF）")
            if mode == "fusion" else "快速单模型（YOLO11s）"
        ),
        "models_used": models_used,
        "inference_time_ms": total_ms,
        "net_inference_ms": net_inference_ms,
        "model_load_ms": model_load_ms,
        "includes_first_load": bool(model_load_ms > 800),
        "per_model_time_ms": per_model_times,
        "width": int(w),
        "height": int(h),
        "num_detections": len(decided),
        "num_candidates_total": len(detections),
        "decision_conf": decision_conf,
        "mean_confidence": mean_conf,
        "max_confidence": max_conf,
        "max_decided_confidence": max_decided_conf,
        "detections": decided,
        "low_conf_candidates": candidates,
        "hazard": hazard,
        "advice": advice,
        "wbf_params": wbf_params,
        # D-FINE-S 已启用但未参与（权重或 onnxruntime 缺失）时为 True，页面据此提示
        "fusion_degraded": bool(fusion_profile and fusion_profile["degraded"]),
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
