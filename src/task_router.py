# -*- coding: utf-8 -*-
"""
task_router.py
==============
双任务路由与模式配置。

提供任务/模式信息、模型组合说明、类别范围描述等纯数据函数。
"""


def task_info(task, mode_key):
    """返回当前任务+模式的模型组合、类别范围、原型规则边界等描述信息。

    Args:
        task: "weed" | "pest"
        mode_key: "fusion" | "fast"（或 weed 的 "yolo11"）

    Returns:
        dict: { models, class_scope, prototype_note, mode_display }
    """
    if task == "weed":
        if mode_key == "fusion" or mode_key == "高精度融合":
            return {
                "models": "YOLO11s + YOLOX-Dinov3 Small + YOLOX-Dinov3 Base + WBF（固定参数）",
                "class_scope": "1 类：Obonianghao（牛鞭草）",
                "prototype_note": "杂草危害等级基于检测密度估算，需农学专家校准。",
                "mode_display": "高精度融合",
            }
        else:
            return {
                "models": "仅 YOLO11s（教师模型）",
                "class_scope": "1 类：Obonianghao（牛鞭草）",
                "prototype_note": "杂草危害等级基于检测密度估算，需农学专家校准。",
                "mode_display": "快速单模型（YOLO11s）",
            }
    else:  # pest
        if mode_key == "fusion" or mode_key == "高精度融合":
            return {
                "models": "YOLO11m + YOLO11l + YOLO11s + refined classwise WBF（逐类别配置）",
                "class_scope": "32 类（英文标准名，来自提交包 dataset.yaml）",
                "prototype_note": "害虫危害等级基于检测密度与主要类别数估算，尚未按虫种、诱捕时间和田间面积进行农学校准。",
                "mode_display": "高精度融合（三模型 classwise WBF）",
            }
        else:
            return {
                "models": "仅 YOLO11m",
                "class_scope": "32 类（英文标准名，来自提交包 dataset.yaml）",
                "prototype_note": "害虫危害等级基于检测密度与主要类别数估算，尚未按虫种、诱捕时间和田间面积进行农学校准。",
                "mode_display": "快速单模型（YOLO11m）",
            }


def task_label(task):
    if task == "weed":
        return "杂草检测"
    return "害虫检测"


def mode_label(mode_key):
    if mode_key == "fusion":
        return "高精度融合"
    return "快速单模型"