# -*- coding: utf-8 -*-
"""
report_data.py
==============
可视化汇报页（pages/2_汇报.py）的数据层：纯函数，不 import streamlit。

职责
----
1. 把真实推理结果（weed_inference / pest_inference 返回的 dict）整理成可导出 /
   可渲染的报告 payload（单图）；
2. 计算置信度直方图、框尺寸分布、类别分布、危害刻度所需的数据；
3. 汇总批量推理（≤10 张）的逐图行、危害分布与类别分布；
4. 合成页面显示图与导出图（显示图可叠加低置信候选，导出图只含正式检测）；
5. 组装批量导出 ZIP（复用 batch_summary.unique_stems 解重名）；
6. 读取现有真实评测 JSON（任务 24 / 任务 22 / classwise refined 配置），
   文件缺失或解析失败时返回 available=False，由页面显示“数据未找到”。

边界
----
- 不实现第二套推理 / WBF / 危害 / 建议算法：危害等级、防治建议、decision_conf
  全部直接取自推理结果（其源头是 config.yaml 与已验证模块）。
- 不产生任何模拟指标。缺文件就标记缺失，绝不用估计值填充。
- 不向页面暴露 classwise JSON 的 source / model_weights 私有绝对路径字段
  （AGENTS.md 3.5 隐私边界），只使用逐类 AP 数值。

== 来源记录（Provenance）==
阈值语义与 hazard.py / decision.py 一致：密度 = 框数 / (宽×高) × 1e6；
框尺寸分桶采用 COCO 32/96 像素（√面积）阈值，与评测 AP_medium / AP_large 口径同源。
== 来源记录结束 ==
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

REPORT_SCHEMA = "wheatfieldai.report.v1"

# 置信度直方图默认 10 个等宽分箱（0.0–1.0）
DEFAULT_CONF_EDGES = tuple(round(i / 10.0, 2) for i in range(11))

# 框尺寸分桶：√(宽×高) 像素，与 COCO 32/96 阈值一致
SIZE_BUCKETS = (
    ("small", "小（√面积 < 32 px）", 0.0, 32.0),
    ("medium", "中（32–96 px）", 32.0, 96.0),
    ("large", "大（≥ 96 px）", 96.0, math.inf),
)

HAZARD_ORDER = ("未检出", "轻", "中", "重")

VAL_V2_DISCLAIMER = (
    "本地验证集口径；val_v2 为模型辅助补标（多模型一致 + 高置信）得到的诊断口径，"
    "不是人工金标准，也不是独立测试集。"
)

NO_INDEPENDENT_TEST_SET = "所有指标均为本地验证集口径，项目无独立测试集。"


# ----------------------------------------------------------------------
# 路径解析
# ----------------------------------------------------------------------
def project_root() -> Path:
    """WheatFieldAI 主工程根目录。"""
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    """工作区根目录（包含 任务输出/）。"""
    return project_root().parent


def _first_existing(candidates):
    """返回第一个存在的文件路径；都不存在时返回第一个候选（便于报错定位）。"""
    candidates = [Path(c) for c in candidates]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def default_eval_paths(root=None) -> dict:
    """现有真实评测文件路径表（按实际存在调整；缺失不影响其它模块）。"""
    root = Path(root) if root is not None else workspace_root()
    task24 = root / "任务输出" / "24_杂草精度误差分析与优化" / "outputs"
    task22 = root / "任务输出" / "22_decision_conf阈值校准" / "results"
    app_root = root / "WheatFieldAI"
    return {
        "six_model_app": _first_existing([
            app_root / "outputs" / "app_six_model" / "summary.json",
            root / "任务输出" / "30_自研检测器变体_WFA-Det" / "outputs" / "app_six_model" / "summary.json",
        ]),
        "five_model_app": _first_existing([
            task24 / "app_five_model" / "summary.json",
            app_root / "outputs" / "app_five_model" / "summary.json",
        ]),
        "fusion_candidates": _first_existing([
            task24 / "fusion_candidates_eval.json",
        ]),
        "val_v2_eval": _first_existing([
            task24 / "val_v2_eval.json",
        ]),
        "decision_conf": _first_existing([
            task22 / "summary.json",
        ]),
        "classwise": _first_existing([
            app_root / "configs" / "classwise_ensemble_11m11l11s_960_refined_current.json",
        ]),
        "dataset_yaml": _first_existing([
            app_root / "configs" / "dataset.yaml",
        ]),
    }


def _read_json(path):
    """读取 JSON；返回 (data, error)。失败时 data=None。"""
    p = Path(path)
    if not p.is_file():
        return None, "文件不存在"
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:  # noqa: BLE001 - 页面需要展示任意解析错误
        return None, f"{type(exc).__name__}: {exc}"


def load_class_names(dataset_yaml):
    """从 dataset.yaml 的 names 段读取 {class_id: name}；失败返回空 dict。"""
    p = Path(dataset_yaml)
    if not p.is_file():
        return {}
    try:
        import yaml

        with p.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("names") or {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
    except Exception:  # noqa: BLE001 - 类别名缺失时页面降级为 class_<id>
        return {}
    return {}


# ----------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------
def _to_float(value):
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_POSIX_ABS = re.compile(r"^/(?:[^/\s]+/)+[^/\s]+$")


def _safe_path_text(text):
    """把绝对路径字符串压缩为文件名，避免报告泄露本机路径（AGENTS.md 3.5）。

    只识别真正的绝对路径（Windows 盘符路径 / POSIX 绝对路径），普通描述性文本
    （即使含斜杠）原样保留，避免把说明文字误判成路径。
    """
    value = str(text)
    if _WINDOWS_ABS.match(value) or _POSIX_ABS.match(value):
        # 用字符串切分而非 Path.name：Windows 反斜杠路径在 Linux CI 上也要能取到文件名
        return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return value


def sanitize_paths(value):
    """递归清洗 dict/list 中的绝对路径字符串（只影响展示层副本，不改推理结果）。"""
    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str):
        return _safe_path_text(value)
    return value


def density_per_mp(num_detections, width, height):
    """每百万像素检测框数；与 hazard.py 同公式。"""
    if not width or not height:
        return 0.0
    area_mp = (float(width) * float(height)) / 1_000_000.0
    if area_mp <= 0:
        return 0.0
    return round(float(num_detections) / area_mp, 2)


def decided_max_confidence(result, decided=None):
    """正式检测（≥ decision_conf）的最高置信度；无正式检测返回 None。

    优先取推理侧字段 ``max_decided_confidence``（weed/pest inference 均已返回），
    缺省时回退到 ``detections`` 的 max。``max_confidence`` 含低置信候选，只作溯源，
    不能用来代表“正式最高置信度”。
    """
    if not isinstance(result, dict):
        return None
    value = _to_float(result.get("max_decided_confidence"))
    if value is not None:
        return value
    decided = decided if decided is not None else (result.get("detections") or [])
    confs = [_to_float(d.get("confidence")) for d in decided if isinstance(d, dict)]
    confs = [c for c in confs if c is not None]
    return round(max(confs), 6) if confs else None


def relative_path_text(path, root):
    """把绝对路径转成相对 root 的展示文本；不在 root 下时只保留文件名。

    页面提示与表格统一走这里，避免把 ``C:\\Users\\...`` 本机路径写进界面或导出文件。
    """
    # 统一分隔符后再比较：Windows 风格路径在 POSIX 环境（CI）下也要能正确取相对段/文件名
    raw = str(path).replace("\\", "/")
    root_raw = str(root).replace("\\", "/").rstrip("/")
    if root_raw and raw.startswith(root_raw + "/"):
        return raw[len(root_raw) + 1:]
    return raw.rstrip("/").rsplit("/", 1)[-1]


# ----------------------------------------------------------------------
# 置信度直方图
# ----------------------------------------------------------------------
def confidence_histogram(decided, candidates, edges=None):
    """正式检测 vs 低置信候选的置信度直方图。

    Args:
        decided: 置信度 ≥ decision_conf 的检测（list[dict] 或 list[float]）
        candidates: 置信度 < decision_conf 的检测
        edges: 分箱边界，默认 0.0–1.0 等宽 10 箱

    Returns:
        list[dict]: [{index, lo, hi, label, decided, candidates, total}, ...]
    """
    edges = tuple(edges) if edges else DEFAULT_CONF_EDGES
    if len(edges) < 2:
        raise ValueError("分箱边界至少需要 2 个值")

    n_bins = len(edges) - 1
    d_counts = [0] * n_bins
    c_counts = [0] * n_bins

    def _bin_index(value):
        v = _to_float(value)
        if v is None:
            return None
        if v < edges[0] or v > edges[-1]:
            return None
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                if lo <= v <= hi:
                    return i
            elif lo <= v < hi:
                return i
        return None

    for bucket, target in ((decided or (), d_counts), (candidates or (), c_counts)):
        for item in bucket:
            conf = item.get("confidence") if isinstance(item, dict) else item
            idx = _bin_index(conf)
            if idx is not None:
                target[idx] += 1

    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        rows.append({
            "index": i,
            "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "label": f"{lo:.1f}–{hi:.1f}",
            "decided": d_counts[i],
            "candidates": c_counts[i],
            "total": d_counts[i] + c_counts[i],
        })
    return rows


# ----------------------------------------------------------------------
# 框尺寸分布
# ----------------------------------------------------------------------
def _box_side(detection):
    """返回 (√面积, 宽/高)；坐标非法时返回 (None, None)。"""
    box = detection.get("bbox_xyxy") if isinstance(detection, dict) else None
    if not box or len(box) != 4:
        return None, None
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None, None
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    side = math.sqrt(w * h)
    aspect = (w / h) if h > 0 else None
    return side, aspect


def box_size_distribution(detections):
    """按 √(宽×高) 分桶统计框尺寸，并给出中位/均值等描述统计。

    Returns:
        dict: {buckets:[{key,label,count,share}], total, stats:{...}, note}
    """
    detections = list(detections or [])
    counts = {key: 0 for key, _label, _lo, _hi in SIZE_BUCKETS}
    sides = []
    aspects = []

    for det in detections:
        side, aspect = _box_side(det)
        if side is None:
            continue
        sides.append(side)
        if aspect is not None:
            aspects.append(aspect)
        for key, _label, lo, hi in SIZE_BUCKETS:
            if lo <= side < hi:
                counts[key] += 1
                break

    total = len(sides)
    buckets = []
    for key, label, _lo, _hi in SIZE_BUCKETS:
        count = counts[key]
        buckets.append({
            "key": key,
            "label": label,
            "count": count,
            "share": round(count / total, 4) if total else 0.0,
        })

    if total:
        ordered = sorted(sides)
        mid = total // 2
        median = ordered[mid] if total % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        stats = {
            "mean_side_px": round(sum(sides) / total, 1),
            "median_side_px": round(median, 1),
            "min_side_px": round(ordered[0], 1),
            "max_side_px": round(ordered[-1], 1),
            "mean_aspect": round(sum(aspects) / len(aspects), 2) if aspects else None,
        }
    else:
        stats = {
            "mean_side_px": None,
            "median_side_px": None,
            "min_side_px": None,
            "max_side_px": None,
            "mean_aspect": None,
        }

    return {
        "buckets": buckets,
        "total": total,
        "stats": stats,
        "note": "分桶阈值 √(宽×高) = 32 / 96 px，与评测 AP_medium / AP_large 同源；"
                "仅统计 ≥ decision_conf 的正式检测。",
    }


# ----------------------------------------------------------------------
# 类别分布
# ----------------------------------------------------------------------
def class_distribution(detections):
    """按类别名聚合检测数量，按数量降序（同数量按名称排序保证稳定）。"""
    counts = {}
    for det in detections or []:
        name = (det.get("category_name") if isinstance(det, dict) else None) or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


# ----------------------------------------------------------------------
# 标注图合成（页面显示 vs 导出）
# ----------------------------------------------------------------------
def compose_annotation_image(img_bgr, detections, candidates=None, show_low=False,
                             max_candidates=50):
    """两遍绘制标注图，返回 ``(display_bgr, export_bgr)``。

    - 第一遍：正式检测（``draw_annotations`` 默认样式：彩色实线框 + 类别名 + 置信度）；
    - 第二遍（仅 ``show_low=True``）：低置信候选，``low_conf=True`` 样式（灰色细虚线 +
      “候选 0.xx”），数量上限 ``max_candidates``，避免密集图面过载。

    ``export_bgr`` **始终只含正式检测**：页面可以勾选叠加候选辅助查看，但导出的
    HTML / ZIP 标注图必须与计数、危害等级和“仅正式检测”的图注一致。

    绘制实现来自 ``src/drawing.py``（唯一实现），本函数只决定“画什么、画几遍”。
    """
    from drawing import draw_annotations  # 只读复用已验证绘制实现，不另起一套

    export_bgr = draw_annotations(img_bgr, list(detections or []))
    if not show_low:
        return export_bgr, export_bgr
    display_bgr = draw_annotations(
        export_bgr, list(candidates or [])[:max_candidates], low_conf=True
    )
    return display_bgr, export_bgr


# ----------------------------------------------------------------------
# 单图报告 payload
# ----------------------------------------------------------------------
def _inference_thresholds(cfg, task):
    if not isinstance(cfg, dict):
        return None
    node = (cfg.get(task) or {}).get("inference")
    if not isinstance(node, dict):
        return None
    return {str(k): v for k, v in node.items() if isinstance(v, (int, float, str))}


def build_single_report(result, task, file_name, generated_at=None, cfg=None):
    """把单图真实推理结果整理成报告 payload（JSON 可直接序列化）。

    Args:
        result: infer_weed_single_image / infer_pest_single_image 的返回值
        task: "weed" | "pest"
        file_name: 原始文件名（仅展示用）
        generated_at: ISO 时间字符串；缺省取当前时间
        cfg: 完整配置 dict（用于补充推理阈值等溯源信息，可为 None）

    Returns:
        dict: 报告 payload（含 detections / low_conf_candidates 原始数据）
    """
    if not isinstance(result, dict):
        raise TypeError("result 必须是推理函数返回的 dict")

    generated_at = generated_at or now_iso()
    decided = list(result.get("detections") or [])
    candidates = list(result.get("low_conf_candidates") or [])

    width = int(_to_float(result.get("width")) or 0)
    height = int(_to_float(result.get("height")) or 0)
    area_mp = round((width * height) / 1_000_000.0, 3) if width and height else 0.0

    hazard = dict(result.get("hazard") or {})
    advice = dict(result.get("advice") or {})

    max_conf = _to_float(result.get("max_confidence"))
    max_decided = decided_max_confidence(result, decided)
    mean_conf = _to_float(result.get("mean_confidence"))
    density = _to_float(hazard.get("density_per_mp"))
    if density is None:
        density = density_per_mp(len(decided), width, height)

    task_label = "杂草检测" if task == "weed" else "害虫检测"
    decision_conf = _to_float(result.get("decision_conf"))

    if task == "pest":
        counts = result.get("class_counts")
        if isinstance(counts, dict) and counts:
            dist = [{"name": str(k), "count": int(v)} for k, v in counts.items()]
        else:
            dist = class_distribution(decided)
    else:
        dist = class_distribution(decided)

    payload = {
        "schema": REPORT_SCHEMA,
        "generated_at": generated_at,
        "task": task,
        "task_label": task_label,
        "mode": result.get("mode"),
        "mode_display": result.get("mode_display"),
        "file_name": str(file_name),
        "image": {
            "width": width,
            "height": height,
            "megapixels": area_mp,
            "aspect": round(width / height, 3) if height else None,
        },
        "models_used": list(result.get("models_used") or []),
        "device": result.get("device"),
        "decision_conf": decision_conf,
        "metrics": {
            "num_detections": len(decided),
            "num_candidates": len(candidates),
            "num_predictions": len(decided) + len(candidates),
            # 正式最高置信度（仅 ≥ decision_conf）；max_confidence 含候选，只作溯源
            "max_decided_confidence": max_decided,
            "max_confidence": max_conf,
            "mean_confidence": mean_conf,
            "density_per_mp": density,
            "inference_time_ms": _to_float(result.get("inference_time_ms")),
            "net_inference_ms": _to_float(result.get("net_inference_ms")),
            "model_load_ms": _to_float(result.get("model_load_ms")),
            "major_class_count": result.get("major_class_count"),
        },
        "hazard": hazard,
        "advice": advice,
        "confidence_histogram": confidence_histogram(decided, candidates),
        "box_sizes": box_size_distribution(decided),
        "class_distribution": dist,
        "provenance": {
            "file_name": str(file_name),
            "generated_at": generated_at,
            "task": task,
            "task_label": task_label,
            "mode": result.get("mode"),
            "mode_display": result.get("mode_display"),
            "models_used": list(result.get("models_used") or []),
            "decision_conf": decision_conf,
            "inference_thresholds": _inference_thresholds(cfg, task),
            # 溯源：模型原始最高分（含低置信候选）与正式最高分分开保留，避免口径混淆
            "max_confidence": max_conf,
            "max_decided_confidence": max_decided,
            "wbf_params": sanitize_paths(result.get("wbf_params")),
            "device": result.get("device"),
            "image_size": f"{width}×{height} px" if width and height else "—",
            "class_mapping": sanitize_paths(result.get("class_mapping")),
            "per_model_time_ms": result.get("per_model_time_ms"),
        },
        "detections": decided,
        "low_conf_candidates": candidates,
        "limitations": [
            "危害等级与防治建议为原型规则，未经农学专家校准，不构成农业处方或正式决策依据。",
            NO_INDEPENDENT_TEST_SET,
            "decision_conf 仅决定哪些框进入计数与危害等级，不改变模型原始输出。",
        ],
    }
    return payload


# ----------------------------------------------------------------------
# 批量汇总
# ----------------------------------------------------------------------
def _hazard_bucket(result):
    """把单图危害结果归一为 未检出 / 轻 / 中 / 重。"""
    hazard = result.get("hazard") or {}
    num = int(_to_float(result.get("num_detections")) or 0)
    if num == 0 or hazard.get("none"):
        return "未检出"
    level = hazard.get("level")
    return level if level in HAZARD_ORDER else "轻"


def hazard_label(result):
    """危害等级展示文本：0 检出统一为「未检出」。

    复用 ``batch_summary.hazard_display`` 的判定（正式框数 0 或 hazard.none），
    避免报告页与主工作台出现两套“未检出 / 轻（未检出）”口径。
    ``displayable`` 显式为 False（未填写拍摄面积）时统一输出中性「未评估」；
    字段缺失时保持旧行为，避免旧结果与单测夹具回归。
    """
    if not isinstance(result, dict):
        return "—"
    hazard = result.get("hazard") or {}
    # 0 检出的「未检出」是中性状态、不是严重程度，优先保留（R10）
    if int(_to_float(result.get("num_detections")) or 0) == 0 or hazard.get("none"):
        return "未检出"
    if hazard.get("displayable") is False:
        return "未评估"
    from batch_summary import hazard_display  # 只读复用，不修改该模块

    label = hazard_display(result)
    if label == "—":
        label = str(hazard.get("label") or hazard.get("level") or "—")
    return label


def hazard_badge(result):
    """缩略图徽章：``{"badge": none|light|medium|heavy, "text": str}``。

    未检出 / 失败 / 未评估使用中性 ``none``，不显示误导性等级。
    """
    label = hazard_label(result)
    if not isinstance(result, dict) or label in ("未检出", "—", "未评估"):
        return {"badge": "none", "text": label}
    level = (result.get("hazard") or {}).get("level")
    badge = {"轻": "light", "中": "medium", "重": "heavy"}.get(level or label, "none")
    return {"badge": badge, "text": label}


def build_batch_summary(items, task, mode_display, generated_at=None, decision_conf=None):
    """汇总批量推理结果。

    Args:
        items: list[dict]，每项 {file_name, status, error, result}
        task: "weed" | "pest"
        mode_display: 模式显示名（来自推理结果，缺省传 "—"）
        generated_at: ISO 时间字符串
        decision_conf: 业务阈值（仅用于展示）

    Returns:
        dict: {generated_at, task, task_label, mode_display, decision_conf,
               totals, hazard_distribution, class_distribution, rows}
    """
    generated_at = generated_at or now_iso()
    rows = []
    ok = failed = 0
    total_det = 0
    density_sum = 0.0
    elapsed_sum = 0.0
    hazard_counts = {key: 0 for key in HAZARD_ORDER}
    class_totals = {}

    for item in items or []:
        result = item.get("result") or {}
        status = item.get("status")
        file_name = str(item.get("file_name") or "—")

        if status != "ok" or not result:
            failed += 1
            rows.append({
                "file_name": file_name,
                "status": "失败",
                "error": item.get("error") or "推理失败",
                "width": None,
                "height": None,
                "num_detections": None,
                "max_decided_confidence": None,
                "max_confidence": None,
                "density_per_mp": None,
                "hazard_label": "—",
                "hazard_level": None,
                "top_class": "—",
            })
            continue

        ok += 1
        num_det = len(result.get("detections") or [])
        total_det += num_det
        density = _to_float((result.get("hazard") or {}).get("density_per_mp"))
        if density is None:
            density = density_per_mp(num_det, result.get("width"), result.get("height"))
        density_sum += density
        elapsed_sum += _to_float(result.get("inference_time_ms")) or 0.0

        bucket = _hazard_bucket(result)
        hazard_counts[bucket] = hazard_counts.get(bucket, 0) + 1

        counts = result.get("class_counts")
        if not isinstance(counts, dict) or not counts:
            counts = {}
            for det in result.get("detections") or []:
                name = det.get("category_name") or "unknown"
                counts[name] = counts.get(name, 0) + 1
        for name, count in counts.items():
            class_totals[name] = class_totals.get(name, 0) + int(count)

        top_class = "—"
        if counts:
            top_class = max(counts.items(), key=lambda kv: kv[1])[0]

        hazard = result.get("hazard") or {}
        rows.append({
            "file_name": file_name,
            "status": "成功",
            "error": None,
            "width": int(_to_float(result.get("width")) or 0) or None,
            "height": int(_to_float(result.get("height")) or 0) or None,
            "num_detections": num_det,
            # 正式最高置信度（仅 ≥ decision_conf）；与主工作台口径一致
            "max_decided_confidence": decided_max_confidence(result, result.get("detections") or []),
            "max_confidence": _to_float(result.get("max_confidence")),
            "density_per_mp": density,
            "hazard_label": hazard_label(result),
            "hazard_level": hazard.get("level"),
            "top_class": top_class,
        })

    return {
        "schema": REPORT_SCHEMA,
        "generated_at": generated_at,
        "task": task,
        "task_label": "杂草检测" if task == "weed" else "害虫检测",
        "mode_display": mode_display or "—",
        "decision_conf": decision_conf,
        "totals": {
            "images": len(rows),
            "ok": ok,
            "failed": failed,
            "detections": total_det,
            "mean_density_per_mp": round(density_sum / ok, 2) if ok else None,
            "elapsed_ms": round(elapsed_sum, 1),
        },
        "hazard_distribution": [
            {"level": key, "count": hazard_counts.get(key, 0)} for key in HAZARD_ORDER
        ],
        "class_distribution": [
            {"name": name, "count": count}
            for name, count in sorted(class_totals.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "rows": rows,
        "limitations": [
            "危害等级为原型规则，未经农学专家校准。",
            NO_INDEPENDENT_TEST_SET,
            "批量页仅做汇总展示，逐图完整报告请在“单图报告”页生成。",
        ],
    }


def build_batch_zip(payload, items, cfg=None, summary_csv=b"", summary_html="",
                    generated_at=None):
    """组装批量导出 ZIP（内存中完成，不落临时文件）。

    内容：``summary.csv`` + ``summary.html`` + ``batch_summary.json`` + 每张成功图片的
    ``json/<stem>.json`` 与 ``annotated/<stem>.jpg``。

    - 主干用 ``batch_summary.unique_stems`` 解重名：``a.jpg`` 与 ``a.png`` 不会互相覆盖
      （ZIP 内条目名唯一，避免 zipfile 的 Duplicate name 警告与静默覆盖）；
    - 每图 JSON 由 ``build_single_report`` 现场生成，与单图页导出的 JSON 同构；
    - 失败图不写 json / annotated，只在 ``batch_summary.json`` 里保留状态与错误；
    - 行数与 CSV / HTML 完全一致（都来自同一个 payload["rows"] 与 items 列表）。

    Args:
        payload: ``build_batch_summary`` 的返回值
        items: 批量推理条目（含 file_name / status / result / annotated_jpeg）
        cfg: 完整配置 dict（传给 ``build_single_report`` 补推理阈值溯源，可为 None）
        summary_csv: CSV 字节（UTF-8-SIG）
        summary_html: HTML 文本或字节
        generated_at: 生成时间；缺省取 payload 内的时间

    Returns:
        bytes: ZIP 文件内容
    """
    from batch_summary import unique_stems  # 只读复用已验证的去重实现

    payload = payload or {}
    items = list(items or [])
    stems = unique_stems([item.get("file_name") for item in items])
    generated_at = generated_at or payload.get("generated_at") or now_iso()
    task = payload.get("task") or "weed"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if summary_csv:
            zf.writestr("summary.csv", summary_csv)
        if summary_html:
            data = summary_html.encode("utf-8") if isinstance(summary_html, str) else summary_html
            zf.writestr("summary.html", data)

        written = {}
        for item, stem in zip(items, stems):
            if item.get("status") != "ok" or not item.get("result"):
                continue
            single = build_single_report(
                item["result"], task, item.get("file_name"),
                generated_at=generated_at, cfg=cfg,
            )
            names = [f"json/{stem}.json"]
            zf.writestr(names[0], json.dumps(single, ensure_ascii=False, indent=2))
            jpg = item.get("annotated_jpeg")
            if jpg:
                names.append(f"annotated/{stem}.jpg")
                zf.writestr(names[-1], jpg)
            written[id(item)] = names

        manifest = {
            "generated_at": generated_at,
            "task": payload.get("task"),
            "task_label": payload.get("task_label"),
            "mode_display": payload.get("mode_display"),
            "decision_conf": payload.get("decision_conf"),
            "totals": payload.get("totals"),
            "files": [
                {
                    "file_name": item.get("file_name"),
                    "stem": stem,
                    "status": item.get("status"),
                    "zip_files": written.get(id(item), []),
                }
                for item, stem in zip(items, stems)
            ],
            "note": "标注 JPG 与 JSON 仅含正式检测（≥ decision_conf）；失败图不写入文件。",
        }
        zf.writestr("batch_summary.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2))

    return buffer.getvalue()


# ----------------------------------------------------------------------
# 模型性能（读取现有真实评测 JSON）
# ----------------------------------------------------------------------
def _metric_block(node):
    if not isinstance(node, dict):
        return None
    return {
        "mAP50": _to_float(node.get("mAP50")),
        "mAP50_95": _to_float(node.get("mAP50-95")),
        "AP75": _to_float(node.get("AP75")),
        "AP_medium": _to_float(node.get("AP_medium")),
        "AP_large": _to_float(node.get("AP_large")),
        "AR100": _to_float(node.get("AR@100")),
    }


def _pick_block(node):
    if not isinstance(node, dict):
        return None
    return {
        "threshold": _to_float(node.get("threshold")),
        "precision": _to_float(node.get("precision")),
        "recall": _to_float(node.get("recall")),
        "f1": _to_float(node.get("f1")),
        "tp": node.get("tp"),
        "fp": node.get("fp"),
        "fn": node.get("fn"),
        "selected": node.get("selected"),
        "num_gt": node.get("num_gt"),
        "n_images": node.get("n_images"),
    }


def _weed_models(six_app, five_app, fusion_candidates, val_v2):
    """杂草模型指标表（融合 vs 单模型），每行注明来源。

    six_app 为现行六模型（任务 30，含 D-FINE-S）的 app 真实管线复算；five_app 为
    任务 24 五模型复算，保留为历史对照行。两者都存在时六模型排在前。
    """
    rows = []

    if isinstance(six_app, dict):
        rows.append({
            "key": "six_model_app",
            "label": "六模型融合（app 真实管线）",
            "group": "融合",
            "original": _metric_block(six_app),
            "v2": None,
            "n_images": six_app.get("images"),
            "source": "outputs/app_six_model/summary.json",
            "note": (
                f"任务 30 现行口径；{six_app.get('images')} 图全量复算，"
                f"失败 {len(six_app.get('failures') or [])} 张，"
                f"{six_app.get('seconds_per_image')} s/图，峰值显存 {six_app.get('peak_reserved_gb')} GB"
            ),
        })

    if isinstance(five_app, dict):
        rows.append({
            "key": "five_model_app",
            "label": "五模型融合（app 真实管线）",
            "group": "融合",
            "original": _metric_block(five_app),
            "v2": None,
            "n_images": five_app.get("images"),
            "source": "outputs/app_five_model/summary.json",
            "note": (
                f"历史口径（任务 24，五模型）；{five_app.get('images')} 图全量复算，"
                f"失败 {len(five_app.get('failures') or [])} 张，"
                f"{five_app.get('seconds_per_image')} s/图，峰值显存 {five_app.get('peak_reserved_gb')} GB"
            ),
        })

    five_members = None
    if isinstance(fusion_candidates, dict):
        node = fusion_candidates.get("base3+1280+box8")
        if isinstance(node, dict):
            five_members = node.get("members")
            rows.append({
                "key": "five_model_offline",
                "label": "五模型融合（离线 WBF 复算）",
                "group": "融合",
                "original": _metric_block(node.get("original")),
                "v2": _metric_block(node.get("v2")),
                "source": "outputs/fusion_candidates_eval.json#base3+1280+box8",
                "note": "与 app 五模型同成员；val_v2 列为离线 WBF 结果",
            })

    if isinstance(val_v2, dict):
        base3 = val_v2.get("fusion")
        if isinstance(base3, dict):
            rows.append({
                "key": "fusion_base3",
                "label": "三模型融合（YOLO11s + YOLOX S/B）",
                "group": "融合",
                "original": _metric_block(base3.get("original")),
                "v2": _metric_block(base3.get("v2")),
                "source": "outputs/val_v2_eval.json#fusion",
                "note": "任务 05 基线融合（三成员）",
            })

        singles = (
            ("yolo11_baseline", "YOLO11s 单模型（快速模式）", "YOLO11s"),
            ("yolo11_tta", "YOLO11s + TTA", "YOLO11s"),
            ("yolox_small", "YOLOX-Dinov3 Small", "YOLOX"),
            ("yolox_base", "YOLOX-Dinov3 Base", "YOLOX"),
        )
        for key, label, group in singles:
            node = val_v2.get(key)
            if isinstance(node, dict):
                rows.append({
                    "key": key,
                    "label": label,
                    "group": group,
                    "original": _metric_block(node.get("original")),
                    "v2": _metric_block(node.get("v2")),
                    "source": f"outputs/val_v2_eval.json#{key}",
                    "note": "单模型基线（同一验证集）",
                })

    return rows, five_members


def _fusion_candidate_rows(fusion_candidates):
    rows = []
    if not isinstance(fusion_candidates, dict):
        return rows
    for key, node in fusion_candidates.items():
        if not isinstance(node, dict):
            continue
        rows.append({
            "key": key,
            "members": list(node.get("members") or []),
            "original": _metric_block(node.get("original")),
            "v2": _metric_block(node.get("v2")),
            "n_predictions": node.get("n_predictions"),
        })
    return rows


def _pest_section(classwise, names):
    if not isinstance(classwise, dict):
        return {"available": False, "reason": "classwise refined 配置缺失或无法解析"}

    per_class_raw = classwise.get("per_class")
    if not isinstance(per_class_raw, dict) or not per_class_raw:
        return {"available": False, "reason": "classwise 配置中没有 per_class 字段"}

    per_class = []
    for key, value in per_class_raw.items():
        try:
            cid = int(key)
        except (TypeError, ValueError):
            continue
        per_class.append({
            "class_id": cid,
            "name": names.get(cid, f"class_{cid}"),
            "ap": _to_float(value),
        })
    per_class.sort(key=lambda row: (-(row["ap"] or 0.0), row["class_id"]))

    return {
        "available": True,
        "map50": _to_float(classwise.get("map50")),
        "map50_95": _to_float(classwise.get("map50_95")),
        "n_images": classwise.get("n_images"),
        "imgsz": classwise.get("imgsz"),
        "n_models": len(classwise.get("weights") or []),
        "n_classes": len(per_class),
        "per_class": per_class,
        "mean_ap": round(sum(r["ap"] or 0.0 for r in per_class) / len(per_class), 4) if per_class else None,
    }


def _decision_conf_section(summary):
    if not isinstance(summary, dict):
        return {"available": False, "picks": {}}
    picks = summary.get("picks") or {}
    out = {}
    for key in ("weed_fusion", "weed_fast", "pest_fusion", "pest_fast"):
        node = picks.get(key)
        if isinstance(node, dict):
            out[key] = {
                "best_f1": _pick_block(node.get("best_f1")),
                "at_0_25": _pick_block(node.get("at_0_25")),
            }
    return {
        "available": bool(out),
        "generated_at": summary.get("generated_at"),
        "iou_thr": summary.get("iou_thr"),
        "datasets": summary.get("datasets"),
        "picks": out,
    }


def load_model_performance(root=None, paths=None):
    """读取现有真实评测文件，返回结构化性能数据。

    任何文件缺失 / 解析失败都只标记 available=False 与原因，不填任何估计值。
    """
    root = Path(root) if root is not None else workspace_root()
    paths = paths or default_eval_paths(root)

    sources = {}
    data = {}
    for key, path in paths.items():
        path = Path(path)
        sources[key] = {
            "key": key,
            "path": str(path),
            "exists": path.is_file(),
            "error": None,
        }
        if key in ("dataset_yaml",):
            continue
        obj, err = _read_json(path)
        sources[key]["error"] = err
        data[key] = obj

    names = load_class_names(paths.get("dataset_yaml", ""))

    weed_rows, five_members = _weed_models(
        data.get("six_model_app"),
        data.get("five_model_app"),
        data.get("fusion_candidates"),
        data.get("val_v2_eval"),
    )

    six_app = data.get("six_model_app") if isinstance(data.get("six_model_app"), dict) else None
    five_app = data.get("five_model_app") if isinstance(data.get("five_model_app"), dict) else None
    # 现行 app 复算：优先六模型（任务 30），缺数据时退回五模型（任务 24）
    app_app = six_app or five_app
    app_run = {
        "images": app_app.get("images"),
        "failures": len(app_app.get("failures") or []) if app_app else None,
        "seconds_per_image": app_app.get("seconds_per_image"),
        "peak_reserved_gb": app_app.get("peak_reserved_gb"),
        "decision_conf": app_app.get("decision_conf"),
    } if app_app else None
    app_members = (app_app.get("models_used") if app_app else None) or five_members
    weed = {
        "available": bool(weed_rows),
        "models": weed_rows,
        "app_variant": "six" if six_app else ("five" if five_app else None),
        "app_members": app_members,
        "app_run": app_run,
        # 兼容旧字段名（页面与外部脚本曾按 five_model_* 取用）
        "five_model_members": app_members,
        "five_model_run": app_run,
        "fusion_candidates": _fusion_candidate_rows(data.get("fusion_candidates")),
    }

    pest = _pest_section(data.get("classwise"), names)
    decision_conf = _decision_conf_section(data.get("decision_conf"))

    return {
        "schema": REPORT_SCHEMA,
        "generated_at": now_iso(),
        "root": str(root),
        "sources": sources,
        "weed": weed,
        "pest": pest,
        "decision_conf": decision_conf,
        "class_names_available": bool(names),
        # 说明文字全部由已读取的 JSON 生成，取不到就省略；页面不再硬编码任何指标/规模
        "notes": _performance_notes(app_app, decision_conf),
    }


def _fmt4(value):
    """四位小数；缺失返回“—”（仅用于说明文字）。"""
    value = _to_float(value)
    return "—" if value is None else f"{value:.4f}"


def _performance_notes(app_app, decision_conf):
    """模型性能页脚注：只根据真实读取到的 JSON 生成，缺数据就省略对应句子。"""
    notes = [NO_INDEPENDENT_TEST_SET, VAL_V2_DISCLAIMER]

    if isinstance(app_app, dict):
        metrics = _metric_block(app_app) or {}
        failures = app_app.get("failures")
        fail_text = f"，失败 {len(failures)} 张" if isinstance(failures, list) else ""
        images = app_app.get("images")
        scale_text = f" {images} 图复算" if images else " 真实管线复算"
        members = app_app.get("models_used") or []
        variant = f"{len(members)} 模型" if members else "融合模型"
        notes.append(
            f"{variant}指标 {_fmt4(metrics.get('mAP50'))} / {_fmt4(metrics.get('mAP50_95'))} / "
            f"{_fmt4(metrics.get('AP75'))}（mAP50 / mAP50-95 / AP75）来自 app 真实管线"
            f"{scale_text}{fail_text}。"
        )

    datasets = (decision_conf or {}).get("datasets") or {}
    parts = []
    weed_images = datasets.get("weed_images")
    weed_boxes = datasets.get("weed_gt_boxes")
    if weed_images and weed_boxes:
        parts.append(f"杂草验证集 {weed_images} 图 / {weed_boxes:,} 框")
    pest_images = datasets.get("pest_images")
    pest_boxes = datasets.get("pest_gt_boxes")
    if pest_images and pest_boxes:
        parts.append(f"害虫验证集 {pest_images} 图 / {pest_boxes:,} 框")
    if parts:
        notes.append("；".join(parts) + "（任务 22 校准数据集口径）。")

    notes.append(
        "害虫逐类 AP 来自提交包 refined classwise 配置，逐类参数按类单独调优，不是单一全局阈值。"
    )
    return notes
