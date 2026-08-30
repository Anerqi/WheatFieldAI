# -*- coding: utf-8 -*-
"""
config.py
=========
配置加载（任务 08 双任务系统）：YAML 文件 + 环境变量覆盖。
- 杂草路径：WHEATWEED_YOLO11_WEIGHTS / WHEATWEED_YOLOX_SMALL_WEIGHTS / WHEATWEED_YOLOX_BASE_WEIGHTS
- 害虫路径：PEST_YOLO11M_WEIGHTS / PEST_YOLO11L_WEIGHTS / PEST_YOLO11S_WEIGHTS
- 害虫类别/融合配置：PEST_DATASET_YAML / PEST_CLASSWISE_CONFIG
- 设备：WHEATWEED_DEVICE（auto / cuda / cpu）
- 封装适配：相对路径以 config.yaml 所在目录为基准解析为绝对路径，
  保证从任意工作目录启动 streamlit 都能找到模型与配置资产。
"""

import os
import yaml
from pathlib import Path


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# 需要解析为绝对路径的配置键（与 config.yaml 结构一一对应）
_REL_PATH_KEYS = [
    ("weed", "paths", "yolo11_weights"),
    ("weed", "paths", "yolox_small_weights"),
    ("weed", "paths", "yolox_base_weights"),
    ("pest", "paths", "yolo11m_weights"),
    ("pest", "paths", "yolo11l_weights"),
    ("pest", "paths", "yolo11s_weights"),
    ("pest", "dataset_yaml"),
    ("pest", "classwise_config"),
]


def load_config(config_path=None):
    """加载配置文件，支持环境变量覆盖权重路径；相对路径解析为绝对路径。"""
    path = Path(config_path or _DEFAULT_CONFIG_PATH).resolve()

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 环境变量覆盖
    env_map = [
        ("WHEATWEED_YOLO11_WEIGHTS", ("weed", "paths", "yolo11_weights")),
        ("WHEATWEED_YOLOX_SMALL_WEIGHTS", ("weed", "paths", "yolox_small_weights")),
        ("WHEATWEED_YOLOX_BASE_WEIGHTS", ("weed", "paths", "yolox_base_weights")),
        ("PEST_YOLO11M_WEIGHTS", ("pest", "paths", "yolo11m_weights")),
        ("PEST_YOLO11L_WEIGHTS", ("pest", "paths", "yolo11l_weights")),
        ("PEST_YOLO11S_WEIGHTS", ("pest", "paths", "yolo11s_weights")),
        ("PEST_DATASET_YAML", ("pest", "dataset_yaml")),
        ("PEST_CLASSWISE_CONFIG", ("pest", "classwise_config")),
        ("WHEATWEED_DEVICE", ("device",)),
    ]

    for env_key, keys in env_map:
        if env_key not in os.environ or not os.environ[env_key]:
            continue

        target = cfg
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]

        target[keys[-1]] = os.environ[env_key]

    # 封装适配：相对路径以 config.yaml 所在目录为基准解析为绝对路径
    # （占位符 <...> 与绝对路径保持不变，保证任意工作目录启动均可运行）
    base_dir = path.parent

    for keys in _REL_PATH_KEYS:
        node = cfg
        for key in keys[:-1]:
            node = node[key]

        value = node[keys[-1]]

        if (
            isinstance(value, str)
            and value
            and not value.startswith("<")
            and not os.path.isabs(value)
        ):
            node[keys[-1]] = str(
                (base_dir / value).resolve()
            )

    # 解析设备
    device = cfg.get("device", "auto")

    if device == "auto":
        import torch

        cfg["device"] = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return cfg
