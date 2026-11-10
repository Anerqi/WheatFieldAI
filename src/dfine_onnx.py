# -*- coding: utf-8 -*-
"""dfine_onnx.py
============
D-FINE-S 杂草检测的 ONNX 推理封装（任务 30 新增，作为 WBF 融合的第 6 支成员）。

== 来源记录（Provenance）==
模型与权重：D-FINE-S（arXiv:2410.13842）官方仓库 Peterande/D-FINE，Apache-2.0；
  权重 dfine_s_coco.pth 经 hf-mirror 下载（SHA256 48A6C8CC…683E89，与 HF LFS oid 一致），
  在本地 3,142 图杂草数据上微调 30 epoch 后导出 ONNX（见 任务输出/29_.../dfine/）。
ONNX 导出与数值对齐证据：任务输出/29_AI模型与图像识别路线调研/dfine/STEP1_ONNX.md
  模型级 parity：labels 全等、boxes 最大差 1/4 像素、scores 最大差 8.9e-06；
  端到端 787 图复算：PyTorch 0.444344 vs ONNX 0.444345。
== 来源记录结束 ==

设计要点：
  1. 只依赖 numpy + PIL + onnxruntime，不导入 D-FINE 仓库代码（其包名 src 与主工程冲突）；
  2. Windows 上 ORT 的 CUDA EP 需要 cuDNN 9 —— torch 的 wheel 自带 cudnn64_9.dll，
     把 torch/lib 加入 DLL 搜索路径即可，无需单独安装（CPU 0.2234 s/图 -> CUDA 0.0129 s/图）；
  3. 权重缺失/依赖缺失时 available() 返回 False，调用方自动退回五模型（不改变既有行为）。
"""

import os

import numpy as np

DEFAULT_IMGSZ = 960
DEFAULT_SCORE_THR = 0.05


class DFineOnnxWeed:
    """单类杂草 D-FINE-S ONNX 推理器。"""

    def __init__(self, onnx_path, score_thr=DEFAULT_SCORE_THR, imgsz=DEFAULT_IMGSZ,
                 device="auto", verbose=False):
        # 相对路径按「app 根目录」解析（src/ 的上一级），避免依赖启动时的 cwd
        if onnx_path and not os.path.isabs(onnx_path):
            app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(app_root, onnx_path)
            if os.path.isfile(candidate):
                onnx_path = candidate
        self.onnx_path = onnx_path
        self.score_thr = float(score_thr)
        self.imgsz = int(imgsz)
        self.device = device
        self.verbose = verbose
        self._session = None
        self._error = None
        self.last_ok = False  # 最近一次 infer() 是否真正完成（见 infer 说明）
        self.providers = []

    # ------------------------------------------------------------------
    def available(self):
        """权重与依赖是否齐备（不做真正的会话创建）。"""
        return bool(self.onnx_path) and os.path.isfile(self.onnx_path)

    @property
    def error(self):
        return self._error

    def _build_session(self):
        import onnxruntime as ort
        # cuDNN 9：优先用 torch wheel 自带的 DLL（Windows 无 pip 版 cuDNN）
        try:
            import torch
            for d in (os.path.join(os.path.dirname(torch.__file__), "lib"),
                      os.path.join(os.path.dirname(ort.__file__), "capi")):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
        except Exception:
            pass
        avail = ort.get_available_providers()
        want = [] if self.device == "cpu" else ["CUDAExecutionProvider"]
        want.append("CPUExecutionProvider")
        providers = [p for p in want if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(self.onnx_path, sess_options=so, providers=providers)
        self.providers = self._session.get_providers()

    def _ensure(self):
        if self._session is not None:
            return True
        if not self.available():
            self._error = "ONNX 权重不存在: %s" % self.onnx_path
            return False
        try:
            self._build_session()
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = repr(exc)
            self._session = None
            return False

    # ------------------------------------------------------------------
    def infer(self, img_bgr):
        """BGR ndarray -> (boxes Nx4 xyxy float, scores N)。异常时返回空结果。

        调用后 ``self.last_ok`` 标明本次是否真正完成推理：会话创建或运行失败时为 False，
        调用方据此把 D-FINE-S 视为「未参与」，而不是把空结果当成第 6 路成员参与融合。
        """
        self.last_ok = False
        if not self._ensure():
            return np.zeros((0, 4)), np.zeros((0,))
        from PIL import Image
        h0, w0 = img_bgr.shape[:2]
        rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(np.ascontiguousarray(rgb))
        resized = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        x = np.transpose(arr, (2, 0, 1))[None]
        sizes = np.array([[h0, w0]], dtype=np.int64)   # 必须是 int64（导出时的 dtype）
        try:
            labels, boxes, scores = self._session.run(
                None, {"images": x, "orig_target_sizes": sizes})
        except Exception as exc:  # noqa: BLE001
            self._error = repr(exc)
            return np.zeros((0, 4)), np.zeros((0,))
        b = np.asarray(boxes)[0], np.asarray(scores)[0]
        boxes_xyxy, sc = b[0], b[1]
        keep = sc >= self.score_thr
        self.last_ok = True
        return boxes_xyxy[keep].astype(float), sc[keep].astype(float)


def build_from_config(weed_cfg):
    """按 config 的 weed.dfine 节点构造推理器；未启用时返回 None。"""
    dfine_cfg = weed_cfg.get("dfine") or {}
    if not dfine_cfg.get("enabled", False):
        return None
    # 路径优先级：weed.paths.dfine_onnx（config.load_config 已解析为绝对路径）> dfine.onnx_path（原样）
    # 只读后者会在非项目根 cwd 下 isfile 失败 -> 静默退回五模型（实测踩过：生产复算得到五模型数字）
    path = (weed_cfg.get("paths") or {}).get("dfine_onnx") or dfine_cfg.get("onnx_path")
    if not path:
        return None
    return DFineOnnxWeed(
        path,
        score_thr=dfine_cfg.get("score_thr", DEFAULT_SCORE_THR),
        imgsz=dfine_cfg.get("imgsz", DEFAULT_IMGSZ),
        device=dfine_cfg.get("device", "auto"),
    )