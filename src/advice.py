# -*- coding: utf-8 -*-
"""
advice.py
=========
防治建议生成（双任务，杂草与害虫使用分开的配置节点）。

- 杂草：复用任务 07 逻辑（conservative 农事管理表述，无具体农药）。
- 害虫：按总体危害等级生成巡查 / 人工复核 / 诱捕设备检查 / 农技人员咨询建议；
  只显示检出类别清单，不据此生成具体药剂、剂量、浓度、施用次数或安全间隔期；
  重度结果必须建议农技人员或植保专家复核；明确声明不构成农业处方。
所有建议文本从 config.yaml 的 advice 段读取（weed / pest 两个子节点）。
"""


def get_advice(hazard_level, advice_cfg, task="weed", level_str=None):
    """根据危害等级生成防治建议。

    Args:
        hazard_level: "轻" | "中" | "重" 或 "轻（未检出）"
        advice_cfg: config.yaml 中 advice 段的配置 dict（含 weed / pest 两个子节点）
        task: "weed" | "pest"
        level_str: 等级显示字符串（兼容旧调用）

    Returns:
        dict: { level, advice, disclaimer }
    """
    node = advice_cfg.get(task, advice_cfg)
    label = level_str or hazard_level

    # 规范化等级：提取 "轻" "中" "重"
    level_key = "light"
    if "未检出" in label:
        level_key = "light"
    elif label in ("轻", "中", "重"):
        level_key = {"轻": "light", "中": "medium", "重": "heavy"}[label]

    advice_text = node.get(level_key, node.get("light", ""))
    disclaimer = node.get("disclaimer", "")

    return {
        "level": label,
        "advice": advice_text,
        "disclaimer": disclaimer,
    }
