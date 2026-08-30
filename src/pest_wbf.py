# -*- coding: utf-8 -*-
"""
pest_wbf.py
===========
害虫三模型 refined classwise WBF（加权框融合）实现。

== 来源记录（Provenance）==
本文件中的 single_iou / wbf / filter_class / load_class_configs / classwise_wbf
从害虫提交包**逐字节复制**并仅做适配，未重新实现融合算法：
  pest_detection_model_submission/scripts/export_ensemble_zip.py
  （该文件与 scripts/ensemble_eval.py 的 WBF 逻辑保持一致）
复制目的：Web 系统害虫高精度模式的融合逻辑与提交包评测/导出完全一致。
== 来源记录结束 ==

注意（与提交包 export 脚本的差异，仅坐标空间）：
- 提交包在 YOLO 标签输出前把框归一化到 [0,1]；本系统直接使用原图像素坐标。
  WBF 的 IoU 与加权平均融合对坐标空间是尺度不变的，融合结果等价。

类别数：32（唯一真源：configs/dataset.yaml 的 names 段）
score_mode 取值：agreement / sqrt_agreement / mean / max
"""

import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 32


def single_iou(a, b):
    """两个 xyxy 框的 IoU。"""
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def wbf(preds_list, weights, iou_thr, skip=0.0, score_mode="agreement"):
    """单类别 WBF 融合。

    preds_list: list over models of (boxes (N,4), classes (N,), scores (N,))
    weights: 各模型权重 list
    返回 (fused_boxes (M,4), fused_classes (M,), fused_scores (M,))
    """
    entries = []
    for mi, (b, c, s) in enumerate(preds_list):
        for j in range(len(b)):
            if s[j] < skip:
                continue
            entries.append((b[j], int(c[j]), float(s[j]), weights[mi], mi))
    if not entries:
        return np.zeros((0, 4)), np.array([]), np.array([])
    clusters = []
    for box, cls, score, wt, model_idx in sorted(entries, key=lambda e: -e[2]):
        placed = False
        for cl in clusters:
            if cl['cls'] != cls:
                continue
            if single_iou(cl['fbox'], box) > iou_thr:
                cl['boxes'].append(box)
                cl['scores'].append(score)
                cl['wts'].append(wt)
                cl['models'].add(model_idx)
                bs = np.array(cl['boxes'])
                sc = np.array(cl['scores']) * np.array(cl['wts'])
                cl['fbox'] = (bs * sc[:, None]).sum(0) / sc.sum()
                placed = True
                break
        if not placed:
            clusters.append({'cls': cls, 'fbox': box.copy(), 'boxes': [box],
                             'scores': [score], 'wts': [wt], 'models': {model_idx}})
    fb, fc, fs = [], [], []
    for cl in clusters:
        sc = np.array(cl['scores'])
        wt = np.array(cl['wts'])
        conf = (sc * wt).sum() / max(sum(wt), 1e-9)
        agreement = len(cl['models']) / len(weights)
        if score_mode == "agreement":
            conf = conf * agreement
        elif score_mode == "sqrt_agreement":
            conf = conf * np.sqrt(agreement)
        elif score_mode == "mean":
            pass
        elif score_mode == "max":
            conf = float(sc.max())
        else:
            raise ValueError(f"unknown score_mode: {score_mode}")
        fb.append(cl['fbox'])
        fc.append(cl['cls'])
        fs.append(conf)
    if not fb:
        return np.zeros((0, 4)), np.array([], dtype=int), np.array([])
    return np.array(fb).reshape(-1, 4), np.array(fc, dtype=int), np.array(fs)


def filter_class(pred, cls_id):
    """把某个模型的预测过滤到指定类别。pred = (boxes, classes, scores)。"""
    boxes, classes, scores = pred
    if len(classes) == 0:
        return boxes, classes, scores
    keep = classes == cls_id
    return boxes[keep], classes[keep], scores[keep]


def load_class_configs(path):
    """读取 refined classwise WBF 配置 JSON（提交包 outputs/eval/...refined_current.json）。

    返回 {class_id: {"model_weights": [...], "wbf_iou": float, "skip": float, "score_mode": str}}
    """
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = {}
    for key, cfg in data.get("class_configs", {}).items():
        configs[int(key)] = {
            "model_weights": [float(x) for x in cfg["model_weights"]],
            "wbf_iou": float(cfg["wbf_iou"]),
            "skip": float(cfg.get("skip", 0.0)),
            "score_mode": cfg.get("score_mode", "agreement"),
        }
    return configs


def classwise_wbf(preds_list, class_configs, num_classes=NUM_CLASSES):
    """逐类别运行 refined classwise WBF 融合。

    preds_list: list over models of (boxes (N,4) 像素坐标, classes (N,), scores (N,))
    class_configs: load_class_configs() 的返回；顺序与 preds_list 的模型顺序一致
    返回 (fused_boxes, fused_classes, fused_scores)
    """
    if class_configs is None:
        raise ValueError("classwise_wbf 需要 class_configs（refined classwise WBF 配置），不能为空")
    all_boxes, all_classes, all_scores = [], [], []
    for cls_id in range(num_classes):
        cfg = class_configs[cls_id]
        filtered = [filter_class(pred, cls_id) for pred in preds_list]
        boxes, classes, scores = wbf(
            filtered,
            cfg["model_weights"],
            cfg["wbf_iou"],
            skip=cfg["skip"],
            score_mode=cfg["score_mode"],
        )
        if len(boxes):
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)
    if not all_boxes:
        return np.zeros((0, 4)), np.array([], dtype=int), np.array([])
    return (np.concatenate(all_boxes, 0),
            np.concatenate(all_classes, 0),
            np.concatenate(all_scores, 0))
