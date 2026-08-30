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
        dict: { level, label, description, severity, density_per_mp, ... }
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
        elif density >= medium_max or major_class_count >= major_medium_max:
            level, label, severity = "重", "重", 3
            description = (
                f"检测密度高（{density:.1f} 个/百万像素）或检出类别数多（{major_class_count} 类），"
                "建议尽快人工复核与田间虫情调查。"
            )
        elif density >= light_max or major_class_count >= major_light_max:
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
        "thresholds": {
            "light_max_per_mp": light_max,
            "medium_max_per_mp": medium_max,
        },
        "note": node.get(
            "prototype_note",
            "杂草原型规则：当前阈值基于检测密度估算，需农学专家校准。",
        ),
    }
