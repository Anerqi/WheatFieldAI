# -*- coding: utf-8 -*-
"""
wbf.py
======
标准 Weighted Boxes Fusion（WBF）实现，严格遵循 Solovyev, Wang, Gabruseva (2021)
"Weighted Boxes Fusion" 论文的算法定义。

== 来源记录（Provenance）==
本文件中的 iou_xyxy / wbf_fuse_boxes 函数从任务 05 run_wbf_experiment.py 逐字节复制：
  C:\\Users\\anerqi\\Desktop\\人工智能大赛\\任务输出\\05_异构WBF融合精度优化\\run_wbf_experiment.py
（行 260-340 附近）。未做任何修改。复制目的：Web 系统 WBF 与任务 05 评测 WBF 逻辑完全一致。
== 来源记录结束 ==

WBF 公式（与任务 05 一致）：
1. 按置信度降序排列所有框。
2. 对每个框，与已有簇的当前融合框计算 IoU；若 IoU >= iou_thr 则并入簇，否则新建簇。
3. 融合坐标 = Σ(coord_i * score_i * model_weight_i) / Σ(score_i * model_weight_i)
4. 融合置信度：avg -> Σ(score_i * w_i) / Σ(w_i)；max -> max(score_i)
5. 融合置信度 >= skip_box_thr 的簇才保留。
"""

import numpy as np


def iou_xyxy(box, cluster_box):
    """两个 xyxy 框的 IoU。"""
    ix1 = max(box[0], cluster_box[0])
    iy1 = max(box[1], cluster_box[1])
    ix2 = min(box[2], cluster_box[2])
    iy2 = min(box[3], cluster_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = max(0.0, cluster_box[2] - cluster_box[0]) * max(0.0, cluster_box[3] - cluster_box[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def wbf_fuse_boxes(boxes_list, scores_list, model_weights,
                   iou_thr=0.55, skip_box_thr=0.0, conf_type="avg",
                   score_mode="raw", geo_beta=0.0, geo_cap=8.0):
    """WBF 融合单图单类框。返回 (fused_boxes (M,4), fused_scores (M,))。
    
    Args:
        boxes_list: list of (N_i, 4) [x1, y1, x2, y2] per model
        scores_list: list of (N_i,) per model
        model_weights: list of float (same order as boxes_list/scores_list)
        iou_thr: 融合 IoU 阈值
        skip_box_thr: 预过滤阈值（score > skip 才参与融合；融合后 score >= skip 才输出）
        conf_type: "avg" 或 "max"
        score_mode: 融合分数的调制方式（任务 26 新增，默认 "raw" 与原实现完全一致）
            "raw"       不做调制
            "geo_agree" 有界几何调制：s^(1-g) * (min(n,cap)/cap)^g
                        其中 n 为该簇的框数（一致数），g=geo_beta，cap=geo_cap。
                        用于把「多模型重叠一致」这一质量信号并入排序；
                        值域仍为 [0,1]，不破坏置信度的百分比语义。
        geo_beta: 几何调制强度 g，建议 0.412（等价于线性 n^0.7）；0 表示不调制
        geo_cap: 一致数饱和点，簇内框数超过 cap 视为同等一致

    == 任务 26 变更说明 ==
    簇构建、坐标聚合、conf_type 语义均未改动。新增的只有最终分数的可选调制，
    默认参数下与任务 05 原实现逐值等价（tests/test_wbf_score_mode.py 有断言）。
    动机与验证见 任务输出/26_融合框策略与后处理优化/01_result.md。
    """
    n_models = len(boxes_list)
    if n_models == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes = np.concatenate(boxes_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    box_model_w = np.concatenate([np.full(len(s), w) for s, w in zip(scores_list, model_weights)])
    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    # 融合前预过滤：score > skip_box_thr 才参与融合
    keep = scores > skip_box_thr
    boxes = boxes[keep]; scores = scores[keep]; box_model_w = box_model_w[keep]
    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    # 按置信度降序
    order = scores.argsort()[::-1]
    boxes = boxes[order]; scores = scores[order]; box_model_w = box_model_w[order]

    cluster_coord_sum = []   # Σ coord*s*w
    cluster_coord_w = []     # Σ s*w
    cluster_score_num = []   # Σ s*w
    cluster_score_den = []   # Σ w
    cluster_max_score = []
    cluster_boxes = []       # 当前融合框（用于 IoU 比较）
    cluster_count = []

    for i in range(len(boxes)):
        bx = boxes[i]; sc = scores[i]; w = box_model_w[i]
        best_iou = 0.0
        best_c = -1
        for c in range(len(cluster_boxes)):
            iou_val = iou_xyxy(bx, cluster_boxes[c])
            if iou_val > best_iou:
                best_iou = iou_val
                best_c = c
        if best_iou >= iou_thr:
            c = best_c
            cluster_coord_sum[c] = cluster_coord_sum[c] + bx * sc * w
            cluster_coord_w[c] += sc * w
            cluster_score_num[c] += sc * w
            cluster_score_den[c] += w
            cluster_max_score[c] = max(cluster_max_score[c], sc)
            cluster_count[c] += 1
            if cluster_coord_w[c] > 0:
                cluster_boxes[c] = cluster_coord_sum[c] / cluster_coord_w[c]
        else:
            cluster_coord_sum.append(bx * sc * w)
            cluster_coord_w.append(sc * w)
            cluster_score_num.append(sc * w)
            cluster_score_den.append(w)
            cluster_max_score.append(sc)
            cluster_count.append(1)
            cluster_boxes.append(bx.copy())

    fused_boxes = []
    fused_scores = []
    for c in range(len(cluster_boxes)):
        if cluster_coord_w[c] <= 0:
            continue
        fused_box = cluster_coord_sum[c] / cluster_coord_w[c]
        if conf_type == "avg":
            fused_score = (cluster_score_num[c] / cluster_score_den[c]
                           if cluster_score_den[c] > 0 else cluster_max_score[c])
        elif conf_type == "max":
            fused_score = cluster_max_score[c]
        else:
            raise ValueError(f"unknown conf_type: {conf_type}")

        # 任务 26：可选的一致性调制（默认 raw 时不改变任何数值）
        if score_mode == "geo_agree":
            ratio = min(cluster_count[c], geo_cap) / geo_cap
            fused_score = (fused_score ** (1.0 - geo_beta)) * (ratio ** geo_beta)
        elif score_mode != "raw":
            raise ValueError(f"unknown score_mode: {score_mode}")

        if fused_score >= skip_box_thr:
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
    if not fused_boxes:
        return np.zeros((0, 4)), np.zeros((0,))
    return np.array(fused_boxes, dtype=float), np.array(fused_scores, dtype=float)  # noqa: E999