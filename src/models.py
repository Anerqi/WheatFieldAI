# -*- coding: utf-8 -*-
"""
models.py
=========
双任务模型管理器（任务 08）。

- 杂草侧：YOLO11s + YOLOX-Dinov3 Small + YOLOX-Dinov3 Base（任务 07 逻辑）
- 害虫侧：YOLO11m + YOLO11l + YOLO11s（提交包权重）

显存策略（记录在案）：
- 本机 RTX 5070 Ti 12GB 显存不足以同时容纳两套任务的模型；
  任务切换（杂草 <-> 害虫）时释放另一任务的全部模型并调用
  torch.cuda.empty_cache()，避免显存叠加导致 OOM。
- 模型缓存：DualModelManager 由 Streamlit @st.cache_resource 持有，
  避免每次上传图片重新加载。

== 来源记录（Provenance）==
杂草侧的 get_yolo11 / get_yolox / is_oom_error 逻辑与任务 07 src/models.py 一致；
害虫侧为任务 08 新增（PestModelManager 见 pest_models.py）。
== 来源记录结束 ==
"""

import os
import threading

import torch

# 禁止 Ultralytics 自动安装/下载可选依赖（本任务不允许外部网络）
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

from pest_models import PestModelManager  # noqa: E402


class DualModelManager:
    """管理杂草 + 害虫两套模型；任务切换时释放另一任务模型并清理 CUDA 缓存。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.get("device", "cpu")
        self._active_task = None
        # R17：@st.cache_resource 单例被所有会话共享；RLock 串行化任务切换/释放/推理，
        # 防止会话 A 切换任务时卸载会话 B 正在使用的模型（RLock 允许嵌套调用）
        self._lock = threading.RLock()
        # 杂草模型
        self._yolo11 = None
        self._yolox = {}  # size -> model
        # 害虫模型
        self._pest = PestModelManager(cfg.get("pest", {}), self.device)

    @property
    def lock(self):
        """共享推理锁：app 层在调用推理函数期间持有，避免与任务切换/释放竞态。"""
        return self._lock

    # ------------------------------------------------------------------
    # 任务切换
    # ------------------------------------------------------------------
    def switch_task(self, task):
        """切换检测任务。任务变化时释放另一任务模型并清理 CUDA 缓存。

        策略记录：12GB 显存不足以同时驻留两套任务模型，切换即卸载 + empty_cache。
        """
        with self._lock:
            if task == self._active_task:
                return
            self.free_gpu()
            self._active_task = task

    # ------------------------------------------------------------------
    # 权重路径存在性检查
    # ------------------------------------------------------------------
    def check_weights(self, task, mode="fusion"):
        """检查当前任务所需权重是否存在；返回 (missing: list[str], ok: bool)。"""
        if task == "pest":
            return self._pest.check_weights(mode="fusion" if mode == "fusion" else "fast")
        weed_paths = self.cfg["weed"]["paths"]
        required = [("yolo11_weights", weed_paths["yolo11_weights"])]
        if mode == "fusion":
            required.append(("yolox_small_weights", weed_paths["yolox_small_weights"]))
            required.append(("yolox_base_weights", weed_paths["yolox_base_weights"]))
        missing = [f"{name}: {p}" for name, p in required if not os.path.isfile(p)]
        return missing, not missing

    # ------------------------------------------------------------------
    # 杂草模型加载
    # ------------------------------------------------------------------
    def get_yolo11(self):
        if self._yolo11 is None:
            from ultralytics import YOLO
            weight = self.cfg["weed"]["paths"]["yolo11_weights"]
            model = YOLO(weight)
            model.to(self.device)
            model.eval()
            self._yolo11 = model
        return self._yolo11

    def get_yolox(self, size):
        if size not in self._yolox:
            try:
                from model_config import build_model
            except ImportError:
                from src.model_config import build_model
            weight = self.cfg["weed"]["paths"][f"yolox_{size}_weights"]
            model, _meta = build_model(weight, map_location=self.device, size=size)
            model.eval()
            model.to(self.device)
            self._yolox[size] = model
        return self._yolox[size]

    # ------------------------------------------------------------------
    # 害虫模型加载
    # ------------------------------------------------------------------
    def get_pest(self, key):
        return self._pest.get(key)

    def pest_class_names(self):
        return self._pest.class_names()

    # ------------------------------------------------------------------
    # 释放与设备
    # ------------------------------------------------------------------
    def free_gpu(self):
        """释放全部缓存模型（任务切换/显存不足降级时调用）。"""
        with self._lock:
            self._yolo11 = None
            self._yolox = {}
            self._pest.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def device_str(self):
        return self.device

    def models_in_use(self, task, mode):
        if task == "pest":
            return self._pest.models_in_use(mode)
        if mode == "fusion":
            return ["YOLO11s", "YOLOX-Dinov3 Small", "YOLOX-Dinov3 Base"]
        return ["YOLO11s"]


def is_oom_error(exc):
    """判断异常是否为 CUDA 显存不足。"""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text