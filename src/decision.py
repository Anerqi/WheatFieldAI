# -*- coding: utf-8 -*-
"""
decision.py
===========
业务决策阈值（decision_conf）：把"评测口径"与"业务口径"分开。

- 推理阈值（weed.inference.*_conf / pest.inference.conf ≈ 0.001~0.01）沿用评测口径，
  保证 mAP 计算与提交包一致，不改动已验证的推理/融合算法；
- 但这些极低置信候选框不能直接用于计数、类别统计与危害等级——
  否则无目标图像也会因大量噪声候选框被判为"重"。
- 本模块只做过滤与统计，不改变任何检测框坐标或置信度。
"""

import numpy as np

DEFAULT_DECISION_CONF = 0.25


def get_decision_conf(task_cfg):
    """读取 cfg["weed"] / cfg["pest"] 下的 decision_conf，缺省 0.25，并校验范围。"""
    value = float(task_cfg.get("decision_conf", DEFAULT_DECISION_CONF))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"decision_conf 必须在 [0, 1] 内，当前为 {value}")
    return value


def split_by_conf(detections, decision_conf):
    """按 decision_conf 拆分检测：返回 (decided, candidates)，两者均保持原有顺序。"""
    decided = [d for d in detections if d["confidence"] >= decision_conf]
    candidates = [d for d in detections if d["confidence"] < decision_conf]
    return decided, candidates


def confidence_stats(detections):
    """返回 (mean, max)；空列表返回 (None, None)。"""
    if not detections:
        return None, None
    confs = np.array([d["confidence"] for d in detections], dtype=float)
    return round(float(confs.mean()), 4), round(float(confs.max()), 4)


def class_counts(detections):
    """按类别名统计数量，按数量降序。"""
    counts = {}
    for d in detections:
        counts[d["category_name"]] = counts.get(d["category_name"], 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
