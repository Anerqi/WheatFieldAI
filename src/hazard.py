# -*- coding: utf-8 -*-
"""
hazard.py
=========
危害等级计算（双任务，杂草与害虫使用分开的配置节点与阈值）。

- 杂草（复用任务 07 逻辑）：基于检测密度（检测框数 / 图像面积 × 1e6）
  - 阈值：config.yaml hazard.weed（light_max_per_mp / medium_max_per_mp）
- 害虫（任务 08 新增原型规则）：基于每百万像素检测密度 + 主要类别数量
  - 阈值：config.yaml hazard.pest（light_max_per_mp / medium_max_per_mp /
    major_classes_light_max / major_classes_medium_max）
  - 页面标注：尚未按虫种、诱捕时间和田间面积进行农学校准
"""

import math

from advice import get_advice


def compute_hazard_level(num_detections, width, height, hazard_cfg, task="weed",
                         major_class_count=0):
    """计算危害等级。

    Args:
        num_detections: 检测框数
        width, height: 图像像素宽高
        hazard_cfg: config.yaml 中 hazard 段的配置 dict（含 weed / pest 两个子节点）
        task: "weed" | "pest"
        major_class_count: 检出主要类别数（仅害虫使用）

    Returns:
        dict: { level, label, description, severity, density_per_mp, none, ... }
        ``none`` 为 True 表示未检出任何目标（level/label 仍保持"轻"/"轻（未检出）"），
        供表现层直接渲染中性"未检出"状态，无需解析中文 label。
    """
    node = hazard_cfg.get(task, hazard_cfg)
    area_mp = (width * height) / 1_000_000  # 兆像素
    density = num_detections / area_mp if area_mp > 0 else 0.0
    density = round(density, 2)

    light_max = float(node.get("light_max_per_mp", 10.0))
    medium_max = float(node.get("medium_max_per_mp", 40.0))

    if task == "pest":
        major_light_max = int(node.get("major_classes_light_max", 2))
        major_medium_max = int(node.get("major_classes_medium_max", 5))
        if num_detections == 0:
            level, label, severity = "轻", "轻（未检出）", 0
            description = "未检出任何目标，建议保持常规田间巡查。"
        elif density >= medium_max or major_class_count > major_medium_max:
            level, label, severity = "重", "重", 3
            description = (
                f"检测密度高（{density:.1f} 个/百万像素）或检出类别数多（{major_class_count} 类），"
                "建议尽快人工复核与田间虫情调查。"
            )
        elif density >= light_max or major_class_count > major_light_max:
            level, label, severity = "中", "中", 2
            description = (
                f"检测密度中等（{density:.1f} 个/百万像素）或出现多个类别（{major_class_count} 类），"
                "建议加强巡查并安排人工复核。"
            )
        else:
            level, label, severity = "轻", "轻", 1
            description = "检测密度低、类别数少，建议常规巡查与人工复核。"
        note = node.get(
            "prototype_note",
            "害虫原型规则：当前阈值基于每百万像素检测密度与主要类别数量估算，"
            "尚未按虫种、诱捕时间和田间面积进行农学校准，不得作为正式农业决策依据。",
        )
        return {
            "level": level,
            "label": label,
            "description": description,
            "severity": severity,
            "density_per_mp": density,
            "major_class_count": major_class_count,
            "num_detections": num_detections,
            "image_area_mp": round(area_mp, 4),
            "none": num_detections == 0,
            "thresholds": {
                "light_max_per_mp": light_max,
                "medium_max_per_mp": medium_max,
                "major_classes_light_max": major_light_max,
                "major_classes_medium_max": major_medium_max,
            },
            "note": note,
        }

    # ---- 杂草（任务 07 逻辑，保持阈值行为一致）----
    if num_detections == 0:
        level = "轻"
        label = "轻（未检出）"
        description = "未检出任何目标，建议保持常规田间巡查。"
        severity = 0
    elif density < light_max:
        level = "轻"
        label = "轻"
        description = "检测密度低，建议常规巡查与人工复核。"
        severity = 1
    elif density < medium_max:
        level = "中"
        label = "中"
        description = "检测密度中等，建议加强巡查并安排人工/机械除草。"
        severity = 2
    else:
        level = "重"
        label = "重"
        description = "检测密度高，建议尽快人工复核并安排除草作业。"
        severity = 3

    return {
        "level": level,
        "label": label,
        "description": description,
        "severity": severity,
        "density_per_mp": density,
        "num_detections": num_detections,
        "image_area_mp": round(area_mp, 4),
        "none": num_detections == 0,
        "thresholds": {
            "light_max_per_mp": light_max,
            "medium_max_per_mp": medium_max,
        },
        "note": node.get(
            "prototype_note",
            "杂草原型规则：当前阈值基于检测密度估算，需农学专家校准。",
        ),
    }


# ----------------------------------------------------------------------
# 按实际拍摄面积分级（占位演示口径，未经农学专家校准，不构成防治处方）
#
# 设计要点：compute_hazard_level 的旧像素口径与返回结构一字不改，
# 旧字段键（density_per_mp / image_area_mp / thresholds 的 per_mp 键等）
# 全部保留，供 report_html / batch_summary / 导出等下游硬下标消费。
# ----------------------------------------------------------------------

_NO_AREA_NOTE = (
    "未填写拍摄区域面积，不显示危害等级与防治建议；"
    "阈值为原型规则，未经农学专家校准。"
)


def _positive_finite(value):
    """归一化为有限正数 float，否则返回 None（拒绝 bool / 字符串 / NaN / inf / <= 0）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def to_square_meters(value, unit, hazard_cfg):
    """把用户输入的拍摄面积换算为平方米。

    Args:
        value: 面积数值（必须是有限正数；None/0/负数/NaN/inf/字符串一律无效）
        unit: "m2"（平方米，原值）| "mu"（亩，乘 field_area.mu_in_m2）
        hazard_cfg: config.yaml 中 hazard 段配置 dict（读取 field_area.mu_in_m2）

    Returns:
        float | None: 换算后的平方米数；单位非法或数值无效时返回 None。
    """
    if unit not in ("m2", "mu"):
        return None
    area = _positive_finite(value)
    if area is None:
        return None
    if unit == "m2":
        return area
    fa = (hazard_cfg or {}).get("field_area") or {}
    mu_in_m2 = _positive_finite(fa.get("mu_in_m2", 666.6667))
    if mu_in_m2 is None:
        mu_in_m2 = 666.6667
    return area * mu_in_m2


def compute_field_hazard(num_detections, field_area_m2, hazard_cfg, task="weed",
                         major_class_count=0):
    """按实际拍摄面积估算的检出密度分级片段（占位演示口径）。

    Returns:
        dict: 含 displayable / field_area_m2 / density_per_m2 / level /
        label / severity / description / none / thresholds / note / reason /
        num_detections / assessment_basis。field_area_m2 无效时返回
        displayable=False、level=None 的不可展示片段。
    """
    hazard_cfg = hazard_cfg or {}
    fa = hazard_cfg.get("field_area") or {}
    node = fa.get(task, fa)
    light_max = float(node.get("light_max_per_m2", 0.5))
    medium_max = float(node.get("medium_max_per_m2", 2.0))
    note = fa.get("prototype_note", "")

    thresholds = {
        "light_max_per_m2": light_max,
        "medium_max_per_m2": medium_max,
        "basis": "field_area",
    }
    if task == "pest":
        thresholds["major_classes_light_max"] = int(
            node.get("major_classes_light_max", 2)
        )
        thresholds["major_classes_medium_max"] = int(
            node.get("major_classes_medium_max", 5)
        )

    area = _positive_finite(field_area_m2)
    if area is None:
        return {
            "displayable": False,
            "field_area_m2": None,
            "density_per_m2": None,
            "level": None,
            "label": "未评估（未填写拍摄面积）",
            "severity": 0,
            "description": "未填写拍摄区域面积，不显示危害等级与防治建议。",
            "none": False,
            "thresholds": thresholds,
            "note": note,
            "reason": "field_area_not_provided",
            "num_detections": num_detections,
            "assessment_basis": "not_available",
        }

    density = round(num_detections / area, 2)
    major_light_max = int(node.get("major_classes_light_max", 2))
    major_medium_max = int(node.get("major_classes_medium_max", 5))

    if num_detections == 0:
        level, label, severity = "轻", "轻（未检出）", 0
    elif task == "pest":
        if density >= medium_max or major_class_count > major_medium_max:
            level, label, severity = "重", "重", 3
        elif density >= light_max or major_class_count > major_light_max:
            level, label, severity = "中", "中", 2
        else:
            level, label, severity = "轻", "轻", 1
    elif density < light_max:
        level, label, severity = "轻", "轻", 1
    elif density < medium_max:
        level, label, severity = "中", "中", 2
    else:
        level, label, severity = "重", "重", 3

    description = (
        f"按拍摄面积 {area:.2f} ㎡ 估算，检出密度 {density:.2f} 个/㎡"
        "（原型规则，未经农学校准）。"
    )
    if task == "pest" and major_class_count > major_light_max:
        description += f"（检出 {major_class_count} 类，触发类别数升级）"

    return {
        "displayable": True,
        "field_area_m2": area,
        "density_per_m2": density,
        "level": level,
        "label": label,
        "severity": severity,
        "description": description,
        "none": num_detections == 0,
        "thresholds": thresholds,
        "note": note,
        "reason": "ok",
        "num_detections": num_detections,
        "assessment_basis": "field_area",
    }


def apply_field_area(result, cfg, task="weed", field_area_value=None,
                     field_area_unit="m2"):
    """把「实际拍摄面积」口径追加/覆盖到单图推理结果（幂等、不修改传入 result）。

    每次都从 result 的 num_detections/width/height/major_class_count 与 cfg
    重新计算，不依赖 result["hazard"] 已有值，因此缓存命中后反复改面积也一致。

    - 无有效面积：hazard 保留旧结构（仅追加 displayable=False 等标记），
      advice 标记 computed=False 并保留原有 level/advice/disclaimer 文本。
    - 有有效面积：覆盖 level/label/severity/description 为新口径，但保留
      全部旧字段键（含 density_per_mp / image_area_mp / thresholds 的 per_mp 键）。
    """
    cfg = cfg or {}
    hazard_cfg = cfg.get("hazard") or {}
    advice_cfg = cfg.get("advice") or {}

    n = result.get("num_detections") or 0
    w = result.get("width") or 0
    h = result.get("height") or 0
    mcc = (
        result.get("major_class_count")
        or (result.get("hazard") or {}).get("major_class_count")
        or 0
    )

    # 完整旧结构（含 decision_conf），保证下游硬下标消费不崩
    base = compute_hazard_level(n, w, h, hazard_cfg, task=task,
                                major_class_count=mcc)
    base["decision_conf"] = result.get("decision_conf")

    # 缺 field_area 段 -> 优雅降级为「无面积」，不抛异常
    fa_cfg = hazard_cfg.get("field_area")
    if not isinstance(fa_cfg, dict):
        area_m2 = None
    else:
        area_m2 = to_square_meters(field_area_value, field_area_unit, hazard_cfg)

    if area_m2 is None:
        hz = dict(base)
        hz["displayable"] = False
        hz["assessment_basis"] = "not_available"
        hz["field_area_value"] = None
        hz["field_area_unit"] = None
        hz["field_area_m2"] = None
        hz["density_per_m2"] = None
        hz["reason"] = "field_area_not_provided"
        hz["note"] = _NO_AREA_NOTE
        adv = dict(result.get("advice") or {})
        adv["computed"] = False
        adv["reason"] = "field_area_not_provided"
    else:
        field_part = compute_field_hazard(
            n, area_m2, hazard_cfg, task=task, major_class_count=mcc
        )
        # thresholds 合并：保留 base 的 per_mp 键，追加 per_m2 / basis
        merged_thresholds = dict(base.get("thresholds") or {})
        merged_thresholds.update(field_part.get("thresholds") or {})
        hz = dict(base)
        hz.update(field_part)
        hz["thresholds"] = merged_thresholds
        hz["field_area_value"] = field_area_value
        hz["field_area_unit"] = field_area_unit
        hz["field_area_m2"] = area_m2
        hz["assessment_basis"] = "field_area"

        adv = get_advice(hz["level"], advice_cfg, task=task,
                         level_str=hz["label"])
        adv["computed"] = True
        adv["reason"] = "ok"

    out = dict(result)
    out["hazard"] = hz
    out["advice"] = adv
    return out
