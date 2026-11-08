# -*- coding: utf-8 -*-
"""
fusion_profile.py
=================
杂草高精度融合的「参数档位」选择：六模型（含 D-FINE-S）与五模型退回档。

背景：iou_thr 0.68 与 decision_conf 0.44 是在「六模型 + D-FINE-S」的分数尺度上
校准的；D-FINE-S 权重或 onnxruntime 缺失时，融合会退回五模型，此时必须同时
换回任务 26 在五模型 geo_agree 尺度上校准的 iou_thr 0.65 / decision_conf 0.40，
否则阈值与分数尺度错配。

本模块只做纯配置解析，不依赖 torch / ultralytics / onnxruntime，便于免权重单测。
"""

from decision import get_decision_conf

# 任务 26 五模型 geo_agree 尺度的校准值（config 缺 five_model_fallback 段时使用）
DEFAULT_FIVE_MODEL_FALLBACK = {"iou_thr": 0.65, "decision_conf": 0.40}


def resolve_fusion_profile(weed_cfg, dfine_used):
    """返回本次融合实际使用的参数档位。

    Args:
        weed_cfg: cfg["weed"]
        dfine_used: 本次推理 D-FINE-S 是否真正参与了融合

    Returns:
        dict: profile ("six_model" | "five_model_fallback")、iou_thr、decision_conf、
        degraded（D-FINE 已启用但未参与时为 True，用于页面提示）。
    """
    wbf_cfg = weed_cfg["wbf"]
    dfine_enabled = bool((weed_cfg.get("dfine") or {}).get("enabled"))

    if dfine_used:
        return {
            "profile": "six_model",
            "iou_thr": float(wbf_cfg["iou_thr"]),
            "decision_conf": get_decision_conf(weed_cfg),
            "degraded": False,
        }

    fb = dict(DEFAULT_FIVE_MODEL_FALLBACK)
    fb.update(wbf_cfg.get("five_model_fallback") or {})
    decision_conf = float(fb["decision_conf"])
    if not 0.0 <= decision_conf <= 1.0:
        raise ValueError(
            f"five_model_fallback.decision_conf 必须在 [0, 1] 内，当前为 {decision_conf}"
        )
    return {
        "profile": "five_model_fallback",
        "iou_thr": float(fb["iou_thr"]),
        "decision_conf": decision_conf,
        "degraded": dfine_enabled,
    }
