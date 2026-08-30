# -*- coding: utf-8 -*-
"""
pest_models.py
==============
害虫模型加载与 32 类类别映射。

- 三套权重：YOLO11m / YOLO11l / YOLO11s（Ultralytics，引用提交包权重，不复制大文件）
- 类别映射唯一数据源：configs/dataset.yaml 的 names 段（不得重排类别 ID）
- 中文别名：仅当本地存在可靠中文别名文件时才显示；当前本地无此文件，只显示英文标准名。
"""

import os

import yaml

# 禁止 Ultralytics 自动安装/下载可选依赖（本任务不允许外部网络）
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

PEST_NUM_CLASSES = 32


def load_pest_class_names(dataset_yaml_path):
    """从 dataset.yaml 解析 32 类名称（唯一数据源）。

    返回按 class_id 索引的 list：["Ostrinia furnacalis", ...]（长度 32）。
    - 只读取 names 段，不重排、不改写类别 ID。
    - 解析失败 / 数量不足 32 / ID 非 0..31 连续 → 抛错（宁可失败也不编造）。
    """
    if not dataset_yaml_path or not os.path.isfile(dataset_yaml_path):
        raise FileNotFoundError(
            f"害虫类别数据源 dataset.yaml 不存在：{dataset_yaml_path}\n"
            "类别名称唯一真源为提交包 configs/dataset.yaml，请检查配置。"
        )
    with open(dataset_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not isinstance(names, dict):
        raise ValueError("dataset.yaml 缺少 names 段或格式不正确")

    # 要求 ID 0..31 连续且无缺漏
    expected = list(range(PEST_NUM_CLASSES))
    got = sorted(int(k) for k in names.keys())
    if got != expected:
        raise ValueError(
            f"dataset.yaml 类别 ID 与预期不符：期望 {expected}，实际 {got}；"
            "类别 ID 顺序不得重排。"
        )
    ordered = [str(names[k]).strip() for k in expected]
    if any(not n for n in ordered):
        raise ValueError("dataset.yaml 存在空类别名")
    return ordered


class PestModelManager:
    """管理害虫三套模型，按需惰性加载并缓存。"""

    MODEL_KEYS = ("yolo11m", "yolo11l", "yolo11s")
    MODEL_DISPLAY = {"yolo11m": "YOLO11m", "yolo11l": "YOLO11l", "yolo11s": "YOLO11s"}

    def __init__(self, cfg, device):
        self.cfg = cfg  # cfg["pest"]
        self.device = device
        self._models = {}
        self._class_names = None

    # ------------------------------------------------------------------
    # 类别映射
    # ------------------------------------------------------------------
    def class_names(self):
        if self._class_names is None:
            self._class_names = load_pest_class_names(self.cfg.get("dataset_yaml"))
        return self._class_names

    def class_name(self, cls_id):
        return self.class_names()[int(cls_id)]

    # ------------------------------------------------------------------
    # 权重路径存在性检查
    # ------------------------------------------------------------------
    def check_weights(self, mode="fusion"):
        """检查所需权重是否存在；返回 (missing: list[str], ok: bool)。"""
        paths = self.cfg["paths"]
        required = [("yolo11m_weights", paths["yolo11m_weights"])]
        if mode == "fusion":
            required.append(("yolo11l_weights", paths["yolo11l_weights"]))
            required.append(("yolo11s_weights", paths["yolo11s_weights"]))
        missing = [f"{name}: {p}" for name, p in required if not os.path.isfile(p)]
        return missing, not missing

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def get(self, key):
        """按 key（yolo11m / yolo11l / yolo11s）惰性加载模型。"""
        if key not in self.MODEL_KEYS:
            raise KeyError(f"未知害虫模型 key: {key}（可选 {self.MODEL_KEYS}）")
        if key not in self._models:
            from ultralytics import YOLO
            weight = self.cfg["paths"][f"{key}_weights"]
            model = YOLO(weight)
            model.to(self.device)
            model.eval()
            self._models[key] = model
        return self._models[key]

    def release(self):
        """释放害虫模型（用于任务切换/显存不足）。"""
        self._models = {}

    def models_in_use(self, mode):
        if mode == "fusion":
            return ["YOLO11m", "YOLO11l", "YOLO11s"]
        return ["YOLO11m"]
