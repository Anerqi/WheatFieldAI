# -*- coding: utf-8 -*-
"""
drawing.py
==========
通用标注图绘制（支持杂草 1 类和害虫 32 类）。

- 杂草检测框：绿色（统一颜色）
- 害虫检测框：按 class_id 分配 32 种高区分度颜色（HSV 色相黄金角采样）
- 标签：类别名 + 置信度
- 支持密集图限制绘制框数（max_boxes）
"""

import cv2
import numpy as np


def _generate_pest_palette(n=32, saturation=0.70, value=0.85):
    """生成 n 种高区分度 BGR 颜色（HSV 色相黄金角采样）。"""
    golden_angle = 0.618033988749895  # 黄金比例倒数
    hues = [(i * golden_angle) % 1.0 for i in range(n)]
    palette = []
    for h in hues:
        # HSV -> BGR
        hsv = np.uint8([[[h * 179, saturation * 255, value * 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


# 32 类害虫色板（全局缓存避免重复计算）
_PEST_PALETTE = _generate_pest_palette(32)


def _detection_color(detection):
    """根据检测框内容返回 BGR 颜色元组。"""
    if "class_id" in detection:
        idx = int(detection["class_id"]) % 32
        return _PEST_PALETTE[idx]
    # 杂草：绿色
    return (0, 180, 0)


def _detection_label(detection):
    """生成框标签文本。"""
    conf = detection["confidence"]
    name = detection.get("category_name", "unknown")
    return f"{name} {conf:.2f}"


def draw_annotations(img_bgr, detections, max_boxes=200):
    """在 BGR 图像上绘制检测框+标签。

    Args:
        img_bgr: BGR ndarray
        detections: list of dict
            - 杂草格式：{'yolo_class_id', 'coco_category_id', 'category_name', 'confidence', 'bbox_xyxy'}
            - 害虫格式：{'class_id', 'category_name', 'confidence', 'bbox_xyxy'}
        max_boxes: 最大绘制框数（防止超密集图面过载）

    Returns:
        BGR ndarray（标注后）
    """
    img = img_bgr.copy()
    for k, det in enumerate(detections[:max_boxes]):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        conf = det["confidence"]
        label = _detection_label(det)
        color = _detection_color(det)

        # 框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_y1 = max(0, y1 - th - 6)
        box_y2 = y1
        cv2.rectangle(img, (x1, box_y1), (x1 + tw + 4, box_y2), color, -1)
        # 标签文字（白色）
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def encode_image_to_bytes(img_bgr, fmt=".jpg", quality=90):
    """将 BGR ndarray 编码为 JPEG/PNG 字节。"""
    params = []
    if fmt == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ok, buf = cv2.imencode(fmt, img_bgr, params)
    if not ok:
        raise RuntimeError(f"编码图片失败: {fmt}")
    return buf.tobytes()