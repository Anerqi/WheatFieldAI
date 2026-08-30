# -*- coding: utf-8 -*-
"""
yolox_inference.py
==================
YOLOX-Dinov3（Small / Base）单图推理所需函数。

== 来源记录（Provenance）==
本文件中的函数从既有已验证推理管线**逐字节复制**，未重新实现：
- read_image / letterbox / decode_outputs / postprocess / unletterbox_boxes / STRIDES：
  来源 `任务输出/03_YOLO11s与YOLOX统一评测/run_unified_eval.py`
  （与 `任务输出/02_YOLOX-Dinov3推理管线/run_inference.py` 完全等价）。
- 常量 YOLO_CLASS_ID / COCO_CATEGORY_ID / CATEGORY_NAME：与任务 02/03/05 一致。
复制目的：保证 Web 系统与评测管线（任务 02/03/05）推理逻辑一致，不产生第二套不一致实现。
== 来源记录结束 ==

类映射（合规，与任务 02/03/05 一致）：
- YOLO 内部类别 ID = 0（num_classes=1，唯一类别）
- COCO 评测标注 category_id = 1（categories id=1, name="Obonianghao"）
- 类别名 = "Obonianghao"
"""

import numpy as np
import torch

import cv2
from PIL import Image

STRIDES = (8, 16, 32)

# 类映射（唯一真源：CONTEXT.md + 任务 02/03/05 输出）
YOLO_CLASS_ID = 0
COCO_CATEGORY_ID = 1
CATEGORY_NAME = "Obonianghao"


def read_image(path):
    """读取图片为 BGR ndarray（兼容非 ASCII 路径）。"""
    with open(path, 'rb') as f:
        data = f.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        # 兜底：PIL（RGB）
        img = np.asarray(Image.open(path).convert('RGB'))[:, :, ::-1]
    return img


def read_image_bytes(data: bytes):
    """从内存字节读取图片为 BGR ndarray；解码失败返回 None（与 read_image 同策略）。"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            img = np.asarray(Image.open(__import__('io').BytesIO(data)).convert('RGB'))[:, :, ::-1]
        except Exception:
            return None
    return img


def letterbox(img, size=640):
    """等价 YOLOX preproc(img, (size,size), swap=(2,0,1))：
    等比缩放 + 左上角 114 padding，输出 **BGR 0-255 float32 CHW** 张量。
    返回 (tensor, (h0,w0), (scale=r, 0, 0))。"""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw = max(1, int(w0 * r))
    nh = max(1, int(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = resized
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).astype(np.float32).copy()).unsqueeze(0)  # BGR 0-255 CHW
    return tensor, (h0, w0), (r, 0, 0)


def decode_outputs(outputs, strides=STRIDES, cell_center=False):
    """把 head 输出转成 (B, N, 6) = [cx, cy, w, h, obj, cls]（640 画布像素坐标）。"""
    all_boxes = []
    for i, out in enumerate(outputs):
        B, _, H, W = out.shape
        stride = strides[i]
        o = out.flatten(2).permute(0, 2, 1)  # (B, H*W, 6)
        yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([xv, yv], -1).view(1, -1, 2).float().to(o.device)
        off = 0.5 if cell_center else 0.0
        xy = (o[..., 0:2] + grid + off) * stride      # 中心点（像素）
        wh = torch.exp(o[..., 2:4]) * stride          # 宽高（像素）
        all_boxes.append(torch.cat([xy, wh, o[..., 4:6]], -1))   # (B, H*W, 6)
    return torch.cat(all_boxes, 1)                     # (B, total_anchors, 6)


def postprocess(prediction, num_classes=1, conf_thre=0.01, nms_thre=0.65, class_agnostic=False):
    """等价 yolox/utils/boxes.postprocess（torchvision batched_nms，不做坐标裁剪）。
    输入 prediction: (N, 6) = [cx, cy, w, h, obj, cls]。返回 (K, 6) = [x1, y1, x2, y2, obj*cls, cls]。"""
    from torchvision.ops import batched_nms, nms
    num = prediction.shape[0]
    if num == 0:
        return prediction
    box_corner = prediction.new(prediction.shape)
    box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    prediction[:, :4] = box_corner[:, :4]

    class_conf, class_pred = torch.max(prediction[:, 5:5 + num_classes], 1, keepdim=True)
    conf_mask = ((prediction[:, 4] * class_conf.squeeze()) >= conf_thre).squeeze()
    detections = torch.cat((prediction[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]
    if detections.size(0) == 0:
        return detections
    if class_agnostic:
        keep = nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
    else:
        keep = batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                           detections[:, 6], nms_thre)
    return detections[keep]


def unletterbox_boxes(boxes, meta):
    """把 640 画布坐标映射回原图坐标（像素）：boxes /= scale，左上角 padding，不裁剪。"""
    scale, ox, oy = meta
    x1 = (boxes[:, 0] - ox) / scale
    y1 = (boxes[:, 1] - oy) / scale
    x2 = (boxes[:, 2] - ox) / scale
    y2 = (boxes[:, 3] - oy) / scale
    return torch.stack([x1, y1, x2, y2], dim=-1)


def run_yolox_model(model, img_bgr, device, conf_thre=0.01, nms_thre=0.65, img_size=640):
    """对单张 BGR 图执行 YOLOX 前向 + 后处理（与任务 02/03/05 run_yolox 单图等价）。
    返回 (boxes_xyxy (N,4) 原图像素, scores (N,))，N 可能为 0。
    """
    orig_wh = img_bgr.shape[:2]
    tensor, _orig, meta_lb = letterbox(img_bgr, img_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        outs = model(tensor)
        decoded = decode_outputs(outs, cell_center=False)[0]   # (N,6)=[cx,cy,w,h,obj,cls]
    keep = postprocess(decoded, num_classes=1, conf_thre=conf_thre, nms_thre=nms_thre)  # (K,7)
    if keep.numel() > 0:
        xyxy = unletterbox_boxes(keep[:, :4], meta_lb)
        scores = keep[:, 4] * keep[:, 5]        # obj * cls
        keep = torch.cat([xyxy, scores.unsqueeze(-1)], -1).cpu().numpy()
    else:
        keep = np.zeros((0, 5))
    if keep.shape[0] > 0:
        return keep[:, :4].astype(float), keep[:, 4].astype(float)
    return np.zeros((0, 4)), np.zeros((0,))
