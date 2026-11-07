# -*- coding: utf-8 -*-
"""
batch_summary.py
================
批量检测（多图）的纯数据层：结果摘要、单图 JSON payload、缩略图、ZIP 打包。

设计边界（重要）
----------------
- 本模块**不依赖 Streamlit、不加载模型、不做推理**，只处理已有结果 dict，
  因此可以在单元测试里直接调用（tests/test_batch_summary.py）。
- 推理契约不变：JSON 字段集合与单图导出的 ``_result.json`` 完全一致
  （由 ``build_result_payload`` 统一构造，app.py 的单图下载与批量 ZIP 共用同一函数）。
- 不产生临时文件：ZIP 在内存中通过 ``io.BytesIO`` 组装。

批量条目（entry）约定
---------------------
app.py 与测试共用的 entry dict 形状::

    {
      "index": 0,                  # 在上传列表中的位置（用于 Streamlit key）
      "filename": "a.jpg",         # 安全化后的文件名（safe_basename）
      "size_kb": 512.0,
      "image": ndarray | None,     # BGR 原图（仅供渲染/缩略图，不参与摘要计算）
      "status": "ok" | "invalid" | "error",
      "error": "..." | None,
      "result": dict | None,       # 推理结果（status == "ok" 时非空）
    }

``status`` 语义：
- ``ok``      推理成功，result 可用；
- ``invalid`` 上传校验失败（空文件/格式/过大/无法解码），未进入推理；
- ``error``   推理阶段失败（OOM/其他异常）。

来源记录：ZIP 字段与文件名规则为本任务（任务 25 交互优化）新增，未改动任何推理算法。
"""

import io
import json
import os
import re
import zipfile

import cv2

# 缩略图默认宽度（批量总览网格，宽度 ~320px 足够，避免 base64 体积膨胀）
DEFAULT_THUMB_WIDTH = 320

STATUS_LABELS = {
    "ok": "完成",
    "invalid": "校验失败",
    "error": "推理失败",
}


# --------------------------------------------------------------------------
# 文件名安全化
# --------------------------------------------------------------------------
def sanitize_stem(name, fallback="image"):
    """把上传文件名转成安全的 ZIP 内文件名主干（去目录、替换非法字符、限长）。

    与 app.py 的 ``safe_basename`` 同一规则（``[^\\w.\\-]`` -> ``_``），
    这里额外去掉首尾的 ``._``，避免生成隐藏文件或以点开头的条目。
    """
    base = str(name or "").replace("\\", "/").split("/")[-1]
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"[^\w.\-]", "_", stem)
    stem = stem.strip("._")
    return stem[:80] or fallback


def unique_stems(names, fallback="image"):
    """为一组文件名生成互不冲突的主干（大小写不敏感去重，冲突时追加 _2/_3…）。

    批量 ZIP 中同名文件（如 ``a.jpg`` 与 ``a.png``）会互相覆盖，必须先解冲突。
    """
    seen = set()
    stems = []
    for name in names:
        base = sanitize_stem(name, fallback=fallback)
        candidate = base
        n = 2
        while candidate.lower() in seen:
            candidate = f"{base}_{n}"
            n += 1
        seen.add(candidate.lower())
        stems.append(candidate)
    return stems


# --------------------------------------------------------------------------
# 单图结果摘要
# --------------------------------------------------------------------------
def decided_max_confidence(result):
    """正式检测（≥ decision_conf）的最高置信度；无正式检测返回 None。

    优先使用推理侧字段 ``max_decided_confidence``；缺省时从 ``detections`` 取 max，
    与 app.py 原「结果概览」卡口径一致。
    """
    if not result:
        return None
    value = result.get("max_decided_confidence")
    if value is None:
        confs = [d["confidence"] for d in result.get("detections") or []]
        value = max(confs) if confs else None
    return None if value is None else round(float(value), 4)


def is_no_detection(result):
    """未检出判定：正式框数为 0，或 hazard 标记了 none（与危害卡口径一致）。"""
    if not result:
        return True
    hazard = result.get("hazard") or {}
    return int(result.get("num_detections") or 0) == 0 or bool(hazard.get("none"))


def hazard_visible(result):
    """是否展示严重程度：仅当已按实际拍摄面积完成重算（displayable=True）时为 True。

    ``displayable`` 字段缺失（旧结果、独立推理入口、单测夹具）一律按 False 处理，
    对应「默认不填写拍摄面积 → 默认不显示危害等级与防治建议」。
    """
    return (result.get("hazard") or {}).get("displayable") is True


def hazard_display(result):
    """危害等级展示文本：未检出时统一显示「未检出」，否则显示规则标签。

    兼容边界：只在 ``displayable`` **显式为 False** 时返回「—」；字段缺失时保持
    原有行为，避免旧结果与既有单测夹具出现回归（与 hazard_visible 的严格判断分开）。
    """
    if not result:
        return "—"
    hazard = result.get("hazard") or {}
    if is_no_detection(result):
        # 0 检出的「未检出」是中性状态、不是严重程度，优先保留（R10）
        return "未检出"
    if hazard.get("displayable") is False:
        return "—"
    return str(hazard.get("label") or "—")


def summarize_entry(entry, task_key="weed"):
    """把一个 entry 压成总览需要的字段（不修改 entry，不抛异常）。"""
    result = entry.get("result")
    status = entry.get("status", "error")
    width = height = None
    if result:
        width, height = result.get("width"), result.get("height")
    elif entry.get("image") is not None:
        height, width = entry["image"].shape[:2]
    max_conf = decided_max_confidence(result)
    num_det = int(result.get("num_detections") or 0) if result else None
    return {
        "index": entry.get("index"),
        "filename": entry.get("filename") or "",
        "status": status,
        "status_label": STATUS_LABELS.get(status, "失败"),
        "error": entry.get("error"),
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "num_detections": num_det,
        "max_decided_confidence": max_conf,
        "hazard": hazard_display(result),
        "inference_time_ms": result.get("inference_time_ms") if result else None,
        "decision_conf": result.get("decision_conf") if result else None,
    }


def format_size(width, height):
    """``1280×1024`` / ``—``。"""
    if not width or not height:
        return "—"
    return f"{width}×{height}"


def format_confidence(value):
    """``0.823`` / ``—``。"""
    return "—" if value is None else f"{float(value):.3f}"


def build_summary_rows(entries, task_key="weed"):
    """生成 ``st.dataframe`` 用的汇总表行（中文列名，顺序固定）。

    数值列保持 int/float/None（便于表格排序，None 显示为空单元格）；
    置信度用三位小数文本，未检出/失败显示 ``—``。
    """
    rows = []
    for entry in entries:
        s = summarize_entry(entry, task_key=task_key)
        rows.append({
            "文件名": s["filename"],
            "尺寸": format_size(s["width"], s["height"]),
            "正式框数": s["num_detections"],
            "正式最高置信度": format_confidence(s["max_decided_confidence"]),
            "危害等级": s["hazard"],
            "耗时(ms)": s["inference_time_ms"],
            "状态": s["status_label"],
        })
    return rows


def batch_totals(entries):
    """批量合计：总张数 / 成功 / 失败 / 正式框总数 / 正式框最多的一张。"""
    ok = [e for e in entries if e.get("status") == "ok" and e.get("result")]
    failed = [e for e in entries if e.get("status") != "ok"]
    boxes = sum(int(e["result"].get("num_detections") or 0) for e in ok)
    worst = None
    for entry in ok:
        s = summarize_entry(entry)
        if worst is None or (s["num_detections"] or 0) > (worst["num_detections"] or 0):
            worst = s
    return {
        "total": len(entries),
        "ok": len(ok),
        "failed": len(failed),
        "boxes": boxes,
        "max_boxes_file": worst["filename"] if worst else None,
        "max_boxes": worst["num_detections"] if worst else 0,
    }


# --------------------------------------------------------------------------
# 单图 JSON payload（单图下载与批量 ZIP 共用）
# --------------------------------------------------------------------------
def build_result_payload(filename, result, task_label):
    """构造单图结果 JSON（字段与既有单图导出完全一致）。

    字段顺序保持与原 app.py 内联构造一致，便于人工比对；不新增/删除字段，
    保证批量 ZIP 内的 JSON 与单图下载的 JSON 可互换使用。
    """
    return {
        "filename": filename,
        "generated_at": _now_str(),
        "task": task_label,
        "mode": result["mode_display"],
        "models_used": result["models_used"],
        "inference_time_ms": result["inference_time_ms"],
        "image_size": {"width": result["width"], "height": result["height"]},
        "num_detections": result["num_detections"],
        "mean_confidence": result["mean_confidence"],
        "max_confidence": result["max_confidence"],
        "decision_conf": result["decision_conf"],
        "num_candidates_total": result["num_candidates_total"],
        "detections": result["detections"],
        "low_conf_candidates": result.get("low_conf_candidates", []),
        "class_counts": result.get("class_counts", {}),
        "hazard": result["hazard"],
        "advice": result["advice"],
        "wbf_params": result["wbf_params"],
        "class_mapping": result["class_mapping"],
        "device": result["device"],
        "note": (
            "本 JSON 为原型系统输出；未填写拍摄区域面积，危害等级与防治建议未计算，相关字段保留为兼容占位。"
            if (result.get("hazard") or {}).get("displayable") is False
            else "本 JSON 为原型系统输出，危害等级与防治建议需农学专家审核。"
        ),
    }


def payload_to_json_bytes(payload):
    """payload -> UTF-8 JSON bytes（indent=2、保留中文）。"""
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _now_str():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 缩略图
# --------------------------------------------------------------------------
def thumbnail_jpeg(img_bgr, max_width=DEFAULT_THUMB_WIDTH, quality=80):
    """把 BGR 图等比缩到宽 <= max_width 并编码为 JPEG bytes。

    只缩小不放大（小图保持原尺寸），用 INTER_AREA 减少摩尔纹。
    """
    if img_bgr is None:
        raise ValueError("thumbnail_jpeg 需要有效的 BGR 图像")
    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img_bgr = cv2.resize(img_bgr, (max_width, max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("缩略图编码失败")
    return buf.tobytes()


# --------------------------------------------------------------------------
# ZIP 打包
# --------------------------------------------------------------------------
def build_batch_zip(entries, task_label, mode_display, generated_at=None,
                    annotated_jpegs=None, payloads=None):
    """在内存中组装批量 ZIP：每张成功图片 = 标注 JPG + 结果 JSON。

    Args:
        entries: entry 列表（见模块 docstring）
        task_label: "杂草检测" / "害虫检测"
        mode_display: 推理模式展示名（写入 summary）
        generated_at: 生成时间字符串（默认当前时间；测试可固定）
        annotated_jpegs: {index: jpg bytes}，缺省时跳过该图的 JPG（不伪造）
        payloads: {index: dict}，缺省时按 entry["result"] 现场构造

    Returns:
        bytes: ZIP 文件内容

    失败/校验失败的文件不写入图片与 JSON，只在 ``batch_summary.json`` 中记录错误，
    并在返回值的 ``summary`` 中保留状态（调用方可据此提示用户）。
    """
    annotated_jpegs = annotated_jpegs or {}
    payloads = payloads or {}
    ok_entries = [e for e in entries if e.get("status") == "ok" and e.get("result")]
    stems = unique_stems([e.get("filename") for e in entries])
    stem_by_index = {e.get("index"): stems[i] for i, e in enumerate(entries)}

    buffer = io.BytesIO()
    files_summary = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            index = entry.get("index")
            stem = stem_by_index.get(index) or sanitize_stem(entry.get("filename"))
            s = summarize_entry(entry)
            record = {
                "filename": s["filename"],
                "status": s["status"],
                "status_label": s["status_label"],
                "image_size": {"width": s["width"], "height": s["height"]},
                "num_detections": s["num_detections"],
                "max_decided_confidence": s["max_decided_confidence"],
                "hazard": s["hazard"],
                "inference_time_ms": s["inference_time_ms"],
                "decision_conf": s["decision_conf"],
                "error": s["error"],
                "zip_files": [],
            }
            if entry.get("status") != "ok" or not entry.get("result"):
                record["skipped"] = True
                files_summary.append(record)
                continue

            jpg = annotated_jpegs.get(index)
            if jpg:
                name = f"{stem}_annotated.jpg"
                zf.writestr(name, jpg)
                record["zip_files"].append(name)

            payload = payloads.get(index)
            if payload is None:
                payload = build_result_payload(s["filename"], entry["result"], task_label)
            name = f"{stem}_result.json"
            zf.writestr(name, payload_to_json_bytes(payload))
            record["zip_files"].append(name)
            files_summary.append(record)

        totals = batch_totals(entries)
        summary = {
            "generated_at": generated_at or _now_str(),
            "task": task_label,
            "mode": mode_display,
            "counts": {
                "total": totals["total"],
                "ok": totals["ok"],
                "failed": totals["failed"],
                "total_detections": totals["boxes"],
            },
            "files": files_summary,
            "note": "本 ZIP 由原型系统生成；危害等级与防治建议为规则模板输出，需农学专家审核。",
        }
        zf.writestr("batch_summary.json",
                    json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))

    return buffer.getvalue()


def zip_manifest(zip_bytes):
    """读取 ZIP 字节的条目名列表（测试/自检用，不写磁盘）。"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return zf.namelist()
