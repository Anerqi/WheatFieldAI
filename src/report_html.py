# -*- coding: utf-8 -*-
"""
report_html.py
==============
可视化汇报页的离线 HTML 报告渲染（纯字符串拼接，不 import streamlit）。

- 单图报告：图片 base64 内嵌，不依赖任何外部 CDN / 字体 / JS；
- 批量汇总报告：缩略图网格 + 汇总表 + 分布条形图；
- 视觉：暖纸色编辑风（奶油 #f8f5f0 / 墨 #3c3c3a / 珊瑚 #cc785c / 衬线标题），
  与 app.py 的 Claude 设计语言同源，但本模块独立成型，可单独发给他人打开。

约束：所有数值直接来自 report_data 产出的 payload；本模块不做任何指标推算。
"""

from __future__ import annotations

import base64
from html import escape

# ----------------------------------------------------------------------
# 基础格式化
# ----------------------------------------------------------------------
def fmt_num(value, digits=3):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_int(value):
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value, digits=1):
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _esc(value):
    return escape("" if value is None else str(value))


def _b64(data):
    if not data:
        return ""
    if isinstance(data, str):
        return data
    return base64.b64encode(bytes(data)).decode("ascii")


def _img_data_uri(data, mime="image/jpeg"):
    encoded = _b64(data)
    return f"data:{mime};base64,{encoded}" if encoded else ""


def _bar_width(count, max_count):
    if not max_count or max_count <= 0 or not count:
        return 0.0
    return max(min(float(count) / float(max_count) * 100.0, 100.0), 1.2)


# ----------------------------------------------------------------------
# 组件
# ----------------------------------------------------------------------
def hazard_scale_html(hazard, task="weed", decision_conf=None):
    """危害刻度条：轻/中/重阈值取自 config（hazard.thresholds），指针为当前密度。

    未检出（severity == 0）时使用中性状态：灰化轨道 + 空心指针，不判定等级。
    """
    hazard = hazard or {}
    thresholds = hazard.get("thresholds") or {}
    light_max = hazard_scale_value(thresholds.get("light_max_per_mp"), 10.0)
    medium_max = hazard_scale_value(thresholds.get("medium_max_per_mp"), 40.0)
    if medium_max <= light_max:
        medium_max = light_max * 2.0

    density = hazard_scale_value(hazard.get("density_per_mp"), 0.0)
    severity = int(hazard_scale_value(hazard.get("severity"), 0))
    neutral = severity == 0 or not hazard

    scale_max = max(medium_max * 1.5, density * 1.15, medium_max + 1e-6)
    light_w = max(light_max / scale_max * 100.0, 0.0)
    medium_w = max((medium_max - light_max) / scale_max * 100.0, 0.0)
    heavy_w = max(100.0 - light_w - medium_w, 0.0)
    marker_left = min(max(density / scale_max * 100.0, 0.0), 99.0)

    unit = "个/百万像素"
    caption = (
        f'<div class="scale-caption">当前密度 <b>{density:.2f}</b> {unit}'
        + (f" · 业务阈值 decision_conf = {fmt_num(decision_conf, 2)}" if decision_conf is not None else "")
        + "</div>"
    )
    if task == "pest":
        major = hazard.get("major_class_count")
        if major is not None:
            caption += (
                f'<div class="scale-caption">检出类别数 <b>{_esc(major)}</b>'
                f'{_major_classes_range_text(thresholds)}</div>'
            )

    if neutral:
        marker = (
            '<div class="marker neutral" style="left:0%">'
            '<span class="marker-dot"></span></div>'
        )
        state = '<div class="scale-state neutral">未检出：中性状态，不判定危害等级</div>'
    else:
        marker = f'<div class="marker" style="left:{marker_left:.1f}%"><span class="marker-dot"></span></div>'
        state = (
            f'<div class="scale-state">危害等级 <b>{_esc(hazard.get("label") or hazard.get("level") or "—")}</b>'
            f' · 严重度 {severity}/3</div>'
        )

    return (
        f'<div class="scale{" neutral" if neutral else ""}">'
        f'<div class="scale-track">'
        f'<div class="seg light" style="width:{light_w:.2f}%">轻</div>'
        f'<div class="seg medium" style="width:{medium_w:.2f}%">中</div>'
        f'<div class="seg heavy" style="width:{heavy_w:.2f}%">重</div>'
        f'{marker}'
        f'</div>'
        f'<div class="scale-ticks"><span>0</span><span>轻/中 {light_max:.1f}</span>'
        f'<span>中/重 {medium_max:.1f}</span><span>{scale_max:.1f}</span></div>'
        f'{state}{caption}'
        f'</div>'
    )


def hazard_scale_value(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _major_classes_range_text(thresholds):
    """害虫类别数分档文案，必须与 hazard.py 的比较符一致。

    hazard.py 用 ``major_class_count > light_max`` 判中、``> medium_max`` 判重，
    即 轻 ≤ light_max、中 (light_max, medium_max]、重 > medium_max；
    因此“重”的下界是 medium_max + 1，而不是 medium_max。
    """
    try:
        light_max = int(thresholds.get("major_classes_light_max"))
        medium_max = int(thresholds.get("major_classes_medium_max"))
    except (TypeError, ValueError):
        return ""
    return (f"（轻 ≤ {light_max} 类，中 {light_max + 1}–{medium_max} 类，"
            f"重 ≥ {medium_max + 1} 类）")


def histogram_html(bins, decision_conf=None):
    """置信度直方图（正式 vs 低置信候选），纯 CSS 柱状图。"""
    bins = list(bins or [])
    if not bins:
        return '<div class="empty">无置信度数据</div>'

    max_count = max((b.get("decided", 0) + b.get("candidates", 0)) for b in bins) or 1
    cols = []
    for item in bins:
        decided = int(item.get("decided") or 0)
        cand = int(item.get("candidates") or 0)
        h_d = decided / max_count * 100.0
        h_c = cand / max_count * 100.0
        cols.append(
            '<div class="hcol">'
            '<div class="hbars">'
            f'<span class="hbar cand" style="height:{h_c:.2f}%" title="低置信候选 {cand}"></span>'
            f'<span class="hbar decided" style="height:{h_d:.2f}%" title="正式检测 {decided}"></span>'
            '</div>'
            f'<div class="hlabel">{_esc(item.get("label"))}</div>'
            f'<div class="hcount">{decided + cand}</div>'
            '</div>'
        )

    total_decided = sum(int(b.get("decided") or 0) for b in bins)
    total_cand = sum(int(b.get("candidates") or 0) for b in bins)
    threshold = f"≥ {fmt_num(decision_conf, 2)}" if decision_conf is not None else "正式"
    return (
        '<div class="legend">'
        f'<span><i style="background:var(--coral)"></i>正式检测（{threshold}）{total_decided}</span>'
        f'<span><i style="background:#cfc8bb"></i>低置信候选 {total_cand}</span>'
        '</div>'
        f'<div class="hist">{"".join(cols)}</div>'
        f'<div class="hist-note">纵轴峰值 {max_count} 框；横轴为置信度区间</div>'
    )


def bar_list_html(items, value_key="count", label_key="label", value_fmt=None,
                  color="var(--coral)", empty_text="无数据"):
    """水平条形列表：适合框尺寸分桶 / 类别分布 / 危害分布。"""
    items = list(items or [])
    if not items:
        return f'<div class="empty">{_esc(empty_text)}</div>'

    max_value = max((item.get(value_key) or 0) for item in items) or 1
    rows = []
    for item in items:
        value = item.get(value_key) or 0
        label = item.get(label_key) or "—"
        shown = value_fmt(value) if value_fmt else fmt_int(value)
        width = _bar_width(value, max_value)
        rows.append(
            '<div class="bar-row">'
            f'<div class="bar-label" title="{_esc(label)}">{_esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{color}"></div></div>'
            f'<div class="bar-val">{_esc(shown)}</div>'
            '</div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'


def _metric_cell(label, value, unit=""):
    unit_html = f'<div class="u">{_esc(unit)}</div>' if unit else ""
    return (
        f'<div class="metric"><div class="k">{_esc(label)}</div>'
        f'<div class="v">{_esc(value)}</div>{unit_html}</div>'
    )


def _kv_rows(pairs):
    rows = []
    for key, value in pairs:
        if value is None or value == "":
            continue
        rows.append(
            f'<div class="kv-k">{_esc(key)}</div><div class="kv-v">{value}</div>'
        )
    return "".join(rows)


def component_css():
    """组件级样式（刻度条 / 直方图 / 条形列表 / 徽章 / 表格）。

    可单独注入 Streamlit 页面复用（pages/2_汇报.py），保证页面与导出报告同源。
    """
    return """
:root{--paper:#f8f5f0;--card:#fdfbf6;--ink:#3c3c3a;--body:#5f5c56;--muted:#8a867e;
--hair:#e2ded6;--hair-soft:#eee9e0;--coral:#cc785c;--coral-deep:#a9583e;
--ok:#35714b;--warn:#8a6510;--danger:#b85c48}
.scale{font-size:12px;color:var(--body)}
.scale-track{position:relative;display:flex;height:26px;border-radius:6px;overflow:hidden;
border:1px solid var(--hair);background:#efeae1}
.seg{display:flex;align-items:center;justify-content:center;font-size:11px;letter-spacing:.1em;
color:#fff;overflow:hidden;white-space:nowrap}
.seg.light{background:#8fae94}.seg.medium{background:#d3a862}.seg.heavy{background:#c07a63}
.scale.neutral .seg{background:#dcd7cd;color:#8a867e}
.marker{position:absolute;top:-5px;bottom:-5px;width:2px;background:var(--ink);z-index:2}
.marker-dot{position:absolute;top:-5px;left:-5px;width:12px;height:12px;border-radius:50%;
background:var(--ink);border:2px solid var(--paper);display:block}
.marker.neutral{background:var(--muted)}
.marker.neutral .marker-dot{background:var(--paper);border:2px solid var(--muted)}
.scale-ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:7px}
.scale-state{margin-top:8px;font-size:13px;color:var(--ink)}
.scale-state.neutral{color:var(--muted)}
.scale-caption{font-size:11.5px;color:var(--muted);margin-top:4px;line-height:1.55}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--body);margin-bottom:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.hist{display:flex;align-items:flex-end;gap:9px;height:150px;border-bottom:1px solid var(--hair);padding-top:6px}
.hcol{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%;min-width:0}
.hbars{display:flex;align-items:flex-end;gap:2px;height:100%}
.hbar{flex:1;border-radius:2px 2px 0 0;min-height:0;display:block}
.hbar.decided{background:var(--coral)}.hbar.cand{background:#cfc8bb}
.hlabel{font-size:9.5px;color:var(--muted);text-align:center;margin-top:6px;white-space:nowrap}
.hcount{font-size:10px;color:var(--ink);text-align:center;font-variant-numeric:tabular-nums}
.hist-note{font-size:11px;color:var(--muted);margin-top:8px}
.bars{display:grid;gap:8px}
.bar-row{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr) 62px;gap:10px;align-items:center}
.bar-label{font-size:12px;color:var(--body);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{background:#efeae1;border-radius:4px;height:13px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}
.bar-val{font-size:12px;font-variant-numeric:tabular-nums;text-align:right;color:var(--ink)}
.note{background:var(--paper);border:1px solid var(--hair);border-radius:8px;padding:11px 14px;
font-size:12.5px;color:var(--body);line-height:1.65;margin-top:14px}
.tag{display:inline-block;border:1px solid var(--hair);border-radius:999px;padding:2px 9px;
font-size:11px;color:var(--body);background:var(--paper);margin:0 6px 6px 0}
.badge{display:inline-block;border-radius:4px;padding:1px 7px;font-size:11px;border:1px solid var(--hair)}
.badge.light{color:var(--ok);border-color:#bcd0c1;background:#f0f5f1}
.badge.medium{color:var(--warn);border-color:#e0cfa6;background:#f8f2e4}
.badge.heavy{color:var(--danger);border-color:#e3c3b9;background:#faeeea}
.badge.none{color:var(--muted);border-color:var(--hair);background:var(--paper)}
table.tbl{width:100%;border-collapse:collapse;font-size:12px}
table.tbl th{text-align:left;font-weight:600;color:var(--muted);font-size:11px;letter-spacing:.06em;
border-bottom:1px solid var(--hair);padding:7px 8px;white-space:nowrap}
table.tbl td{padding:7px 8px;border-bottom:1px solid var(--hair-soft);color:var(--body);vertical-align:top}
table.tbl td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
table.tbl tr:last-child td{border-bottom:none}
.empty{font-size:13px;color:var(--muted);padding:8px 0}
"""


def page_css():
    """整页样式（独立 HTML 报告使用）。"""
    return """
:root{--paper:#f8f5f0;--card:#fdfbf6;--ink:#3c3c3a;--body:#5f5c56;--muted:#8a867e;
--hair:#e2ded6;--hair-soft:#eee9e0;--coral:#cc785c;--coral-deep:#a9583e;
--ok:#35714b;--warn:#8a6510;--danger:#b85c48}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,-apple-system,sans-serif;
line-height:1.6;-webkit-font-smoothing:antialiased}
.page{max-width:1040px;margin:0 auto;padding:44px 32px 64px}
h1,h2,h3{font-family:Georgia,"Times New Roman","Songti SC","Noto Serif SC","Source Han Serif SC",serif;
font-weight:400;letter-spacing:-.01em;margin:0;color:var(--ink)}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--coral-deep);font-weight:600}
.rule{width:56px;height:2px;background:var(--coral);margin:14px 0 16px}
.masthead{border-bottom:1px solid var(--hair);padding-bottom:22px;margin-bottom:26px}
.masthead h1{font-size:32px;line-height:1.22;margin-bottom:10px;overflow-wrap:anywhere}
.meta{font-size:13px;color:var(--muted)}
.meta em{font-style:normal;color:var(--hair);margin:0 8px}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:22px}
.metric{background:var(--card);border:1px solid var(--hair);border-radius:10px;padding:14px 16px}
.metric .k{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.metric .v{font-size:27px;line-height:1.15;margin-top:4px;font-variant-numeric:tabular-nums}
.metric .u{font-size:11px;color:var(--muted);margin-top:2px}
figure.plate{margin:0 0 22px;background:var(--card);border:1px solid var(--hair);
border-radius:12px;padding:12px}
figure.plate img{width:100%;height:auto;display:block;border-radius:6px;background:#efeae1}
figcaption{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.5}
.panel{background:var(--card);border:1px solid var(--hair);border-radius:12px;padding:20px 22px;margin-bottom:18px}
.panel h2{font-size:19px;margin-bottom:2px}
.panel .sub{font-size:12px;color:var(--muted);margin-bottom:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}
.grid2 .panel{margin-bottom:0}
.advice{white-space:pre-line;font-size:14px;line-height:1.78;color:var(--body)}
.advice-head{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.advice-level{font-family:Georgia,"Songti SC",serif;font-size:17px}
.kv{display:grid;grid-template-columns:170px minmax(0,1fr);gap:8px 16px;font-size:12.5px}
.kv-k{color:var(--muted)}
.kv-v{color:var(--ink);overflow-wrap:anywhere}
.thumbs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.thumb{margin:0;background:var(--paper);border:1px solid var(--hair);border-radius:10px;padding:8px}
.thumb img{width:100%;height:150px;object-fit:cover;display:block;border-radius:6px;background:#efeae1}
.thumb-ph{height:150px;display:flex;align-items:center;justify-content:center;
background:#efeae1;border-radius:6px;color:var(--muted);font-size:12px;text-align:center;padding:8px}
.thumb figcaption{font-size:11.5px;margin-top:8px;line-height:1.5}
.thumb .fname{display:block;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.thumb .tmeta{display:block;color:var(--muted);font-size:11px;margin-top:3px}
.foot{border-top:1px solid var(--hair);margin-top:26px;padding-top:16px;font-size:11.5px;
color:var(--muted);line-height:1.75}
.foot b{color:var(--body);font-weight:600}
.foot ul{margin:6px 0 0;padding-left:18px}
@media (max-width:760px){.page{padding:28px 16px 44px}.metrics{grid-template-columns:1fr 1fr}
.grid2{grid-template-columns:1fr}.thumbs{grid-template-columns:1fr 1fr}.masthead h1{font-size:24px}}
@media print{body{background:#fff}.page{padding:0;max-width:none}.panel,figure.plate{break-inside:avoid}}
"""


def _css():
    return page_css() + component_css()


# ----------------------------------------------------------------------
# 单图报告
# ----------------------------------------------------------------------
def render_single_report_html(payload, annotated_jpeg=None, annotated_alt=None):
    """生成单图独立 HTML 报告（图片 base64 内嵌，无外部依赖）。"""
    payload = payload or {}
    task = payload.get("task") or "weed"
    metrics = payload.get("metrics") or {}
    hazard = payload.get("hazard") or {}
    advice = payload.get("advice") or {}
    provenance = payload.get("provenance") or {}
    image = payload.get("image") or {}
    box_sizes = payload.get("box_sizes") or {}
    bins = payload.get("confidence_histogram") or []
    classes = payload.get("class_distribution") or []
    decision_conf = payload.get("decision_conf")

    file_name = payload.get("file_name") or "—"
    title = f"{payload.get('task_label') or '检测'}报告 · {file_name}"

    meta_parts = [
        _esc(payload.get("task_label") or "—"),
        _esc(payload.get("mode_display") or "—"),
        _esc(image.get("width") and f'{image.get("width")}×{image.get("height")} px' or "—"),
        _esc(payload.get("generated_at") or "—"),
    ]
    meta_html = "<em>·</em>".join(meta_parts)

    max_decided = metrics.get("max_decided_confidence")
    if max_decided is None and "max_decided_confidence" not in metrics:
        max_decided = metrics.get("max_confidence")  # 兼容旧 payload（无该字段）
    metrics_html = "".join([
        _metric_cell("正式检测框", fmt_int(metrics.get("num_detections")), "≥ decision_conf"),
        _metric_cell("正式最高置信度", fmt_num(max_decided, 3), "仅正式检测"),
        _metric_cell("推理耗时", fmt_num(metrics.get("inference_time_ms"), 0), "ms（含模型加载）"),
        _metric_cell("检测密度", fmt_num(metrics.get("density_per_mp"), 2), "个/百万像素"),
    ])

    if annotated_jpeg:
        img_html = (
            f'<figure class="plate"><img src="{_img_data_uri(annotated_jpeg)}" '
            f'alt="{_esc(annotated_alt or (file_name + " 标注图"))}"/>'
            f'<figcaption>{_esc(file_name)} · 标注框仅包含置信度 ≥ '
            f'{fmt_num(decision_conf, 2)} 的正式检测；低置信候选见直方图。</figcaption></figure>'
        )
    else:
        img_html = '<div class="panel"><div class="empty">未提供标注图</div></div>'

    size_items = [
        {"label": b.get("label"), "count": b.get("count")} for b in (box_sizes.get("buckets") or [])
    ]
    size_stats = box_sizes.get("stats") or {}
    size_stats_html = (
        f'<div class="hist-note">中位 √面积 {fmt_num(size_stats.get("median_side_px"), 1)} px · '
        f'均值 {fmt_num(size_stats.get("mean_side_px"), 1)} px · '
        f'范围 {fmt_num(size_stats.get("min_side_px"), 1)}–{fmt_num(size_stats.get("max_side_px"), 1)} px · '
        f'平均宽高比 {fmt_num(size_stats.get("mean_aspect"), 2)}</div>'
    )

    class_items = [
        {"label": f'{c.get("name")} · {fmt_int(c.get("count"))}', "count": c.get("count")}
        for c in classes
    ]
    if task == "weed" and classes:
        class_html = (
            f'<div class="kv"><div class="kv-k">类别</div><div class="kv-v">'
            f'{_esc(classes[0].get("name"))}（单类任务）</div>'
            f'<div class="kv-k">正式检测</div><div class="kv-v">{fmt_int(classes[0].get("count"))} 个</div></div>'
        )
    else:
        class_html = bar_list_html(
            class_items, value_fmt=fmt_int, empty_text="未检出任何目标"
        )

    threshold_pairs = []
    inference_thresholds = provenance.get("inference_thresholds") or {}
    for key, value in inference_thresholds.items():
        threshold_pairs.append((f"推理阈值 {key}", _esc(value)))

    wbf = provenance.get("wbf_params")
    wbf_text = "—"
    if isinstance(wbf, dict):
        wbf_text = " · ".join(f"{_esc(k)}={_esc(v)}" for k, v in wbf.items())
    elif wbf:
        wbf_text = _esc(wbf)

    class_mapping = provenance.get("class_mapping") or {}
    mapping_text = ""
    if isinstance(class_mapping, dict) and class_mapping:
        mapping_text = " · ".join(f"{_esc(k)}={_esc(v)}" for k, v in class_mapping.items())

    prov_rows = _kv_rows([
        ("源文件", _esc(file_name)),
        ("图像尺寸", _esc(provenance.get("image_size") or "—")),
        ("检测任务 / 模式", f'{_esc(payload.get("task_label"))} · {_esc(payload.get("mode_display"))}'),
        ("使用模型", _esc("、".join(payload.get("models_used") or []) or "—")),
        ("业务阈值 decision_conf", fmt_num(decision_conf, 2)),
        ("模型原始最高置信度（含低置信候选）", fmt_num(metrics.get("max_confidence"), 3)),
        ("推理设备", _esc(payload.get("device") or "—")),
        ("WBF / 融合参数", wbf_text),
        ("推理耗时明细", (
            f'总 {fmt_num(metrics.get("inference_time_ms"), 0)} ms · '
            f'纯推理 {fmt_num(metrics.get("net_inference_ms"), 0)} ms · '
            f'模型加载 {fmt_num(metrics.get("model_load_ms"), 0)} ms'
        )),
        ("类别映射", mapping_text),
        ("生成时间", _esc(payload.get("generated_at") or "—")),
    ] + threshold_pairs)

    limitations = payload.get("limitations") or []
    limitations_html = "".join(f"<li>{_esc(item)}</li>" for item in limitations)

    advice_text = (advice.get("advice") or "").strip()
    disclaimer = (advice.get("disclaimer") or "").strip()

    # 0 检出：显示“未检出 / 严重度 —”，不再显示“轻（未检出）· 严重度 0/3”
    no_detection = (
        bool(hazard.get("none"))
        or int(hazard_scale_value(hazard.get("severity"), 0)) == 0
        or int(hazard_scale_value(metrics.get("num_detections"), 0)) == 0
    )
    if no_detection:
        hazard_level_text = "未检出"
        hazard_severity_text = "严重度 —"
        hazard_badge_class = "badge none"
    else:
        hazard_level_text = hazard.get("label") or "—"
        hazard_severity_text = f'严重度 {fmt_int(hazard.get("severity"))}/3'
        hazard_badge_class = f'badge {_badge_class(hazard.get("level"))}'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <div class="kicker">WheatFieldAI · 单图检测报告</div>
    <div class="rule"></div>
    <h1>{_esc(file_name)}</h1>
    <div class="meta">{meta_html}</div>
  </header>

  <section class="metrics">{metrics_html}</section>

  {img_html}

  <div class="grid2">
    <section class="panel">
      <h2>危害刻度</h2>
      <div class="sub">阈值来自 config.yaml 的 hazard 段（原型规则，未经农学专家校准）</div>
      {hazard_scale_html(hazard, task=task, decision_conf=decision_conf)}
    </section>
    <section class="panel">
      <h2>置信度分布</h2>
      <div class="sub">正式检测与低置信候选分开统计（推理阈值保持评测口径）</div>
      {histogram_html(bins, decision_conf)}
    </section>
  </div>

  <div class="grid2">
    <section class="panel">
      <h2>框尺寸分布</h2>
      <div class="sub">按 √(宽×高) 分桶，仅统计正式检测</div>
      {bar_list_html(size_items, empty_text="无正式检测框")}
      {size_stats_html}
    </section>
    <section class="panel">
      <h2>{'害虫类别分布' if task == 'pest' else '类别分布'}</h2>
      <div class="sub">{'按正式检测计数，类别名来自 dataset.yaml' if task == 'pest' else '单类任务：播娘蒿（Obonianghao）'}</div>
      {class_html}
    </section>
  </div>

  <section class="panel">
    <h2>防治建议</h2>
    <div class="advice-head"><span class="advice-level">危害等级：{_esc(hazard_level_text)}</span>
    <span class="{hazard_badge_class}">{_esc(hazard_severity_text)}</span></div>
    <div class="advice">{_esc(advice_text) if advice_text else "无建议文本（配置缺失）"}</div>
    <div class="note"><b>原型声明：</b>{_esc(disclaimer) if disclaimer else "危害等级与防治建议为规则模板生成，未经农学专家校准，不构成农业处方。"}</div>
  </section>

  <section class="panel">
    <h2>溯源</h2>
    <div class="sub">报告中的每个数字都来自本次真实推理输出</div>
    <div class="kv">{prov_rows}</div>
  </section>

  <footer class="foot">
    <b>使用边界</b>
    <ul>{limitations_html}</ul>
    <div style="margin-top:10px">WheatFieldAI · 西北农林科技大学首届人工智能大赛 · 赛道二（AI+农林行业应用）· 本地 Streamlit 工作台导出</div>
  </footer>
</div>
</body>
</html>
"""


def _badge_class(level):
    return {"轻": "light", "中": "medium", "重": "heavy"}.get(level or "", "none")


# ----------------------------------------------------------------------
# 批量汇总报告
# ----------------------------------------------------------------------
def render_batch_report_html(payload, thumbnails=None):
    """生成批量汇总 HTML（缩略图网格 + 分布 + 逐图表格）。"""
    payload = payload or {}
    totals = payload.get("totals") or {}
    rows = payload.get("rows") or []
    thumbnails = thumbnails or []

    metrics_html = "".join([
        _metric_cell("图片总数", fmt_int(totals.get("images")), "本批上传"),
        _metric_cell("成功 / 失败", f'{fmt_int(totals.get("ok"))} / {fmt_int(totals.get("failed"))}', "逐图独立推理"),
        _metric_cell("正式检测框", fmt_int(totals.get("detections")), "≥ decision_conf"),
        _metric_cell("平均密度", fmt_num(totals.get("mean_density_per_mp"), 2), "个/百万像素"),
    ])

    hazard_items = [
        {"label": h.get("level"), "count": h.get("count")}
        for h in (payload.get("hazard_distribution") or [])
    ]
    class_items = [
        {"label": c.get("name"), "count": c.get("count")}
        for c in (payload.get("class_distribution") or [])
    ]

    thumbs_html = []
    for thumb in thumbnails:
        badge = thumb.get("badge") or "none"
        if thumb.get("jpeg"):
            img_tag = (
                f'<img src="{_img_data_uri(thumb.get("jpeg"))}" '
                f'alt="{_esc(thumb.get("file_name"))}"/>'
            )
        else:
            # 失败/无缩略图：用占位块，绝不输出 <img src="">（会触发浏览器重复请求）
            img_tag = '<div class="thumb-ph">无缩略图</div>'
        thumbs_html.append(
            '<figure class="thumb">'
            f'{img_tag}'
            '<figcaption>'
            f'<span class="fname">{_esc(thumb.get("file_name"))}</span>'
            f'<span class="tmeta"><span class="badge {_esc(badge)}">{_esc(thumb.get("badge_text") or "—")}</span> '
            f'{_esc(thumb.get("caption") or "")}</span>'
            '</figcaption></figure>'
        )
    thumbs_block = (
        f'<div class="thumbs">{"".join(thumbs_html)}</div>'
        if thumbs_html else '<div class="empty">未提供缩略图</div>'
    )

    table_rows = []
    for row in rows:
        status = row.get("status") or "—"
        badge = "none" if status != "成功" else _badge_class(row.get("hazard_level"))
        width, height = row.get("width"), row.get("height")
        size_text = f"{fmt_int(width)}×{fmt_int(height)}" if width and height else "—"
        max_decided = row.get("max_decided_confidence")
        if max_decided is None and "max_decided_confidence" not in row:
            max_decided = row.get("max_confidence")  # 兼容旧 payload
        table_rows.append(
            "<tr>"
            f'<td>{_esc(row.get("file_name"))}</td>'
            f'<td>{_esc(status)}</td>'
            f'<td class="num">{size_text}</td>'
            f'<td class="num">{fmt_int(row.get("num_detections"))}</td>'
            f'<td class="num">{fmt_num(max_decided, 3)}</td>'
            f'<td class="num">{fmt_num(row.get("density_per_mp"), 2)}</td>'
            f'<td><span class="badge {badge}">{_esc(row.get("hazard_label") or "—")}</span></td>'
            f'<td>{_esc(row.get("top_class") or "—")}</td>'
            "</tr>"
        )

    limitations = payload.get("limitations") or []
    limitations_html = "".join(f"<li>{_esc(item)}</li>" for item in limitations)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>WheatFieldAI · 批量汇总报告</title>
<style>{_css()}</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <div class="kicker">WheatFieldAI · 批量汇总报告</div>
    <div class="rule"></div>
    <h1>{_esc(payload.get("task_label") or "批量检测")} · {fmt_int(totals.get("images"))} 张图片</h1>
    <div class="meta">{_esc(payload.get("mode_display") or "—")}<em>·</em>decision_conf {fmt_num(payload.get("decision_conf"), 2)}<em>·</em>{_esc(payload.get("generated_at") or "—")}</div>
  </header>

  <section class="metrics">{metrics_html}</section>

  <section class="panel">
    <h2>缩略图</h2>
    <div class="sub">标注框仅含正式检测（≥ decision_conf）；每张图片独立推理</div>
    {thumbs_block}
  </section>

  <div class="grid2">
    <section class="panel">
      <h2>危害分布</h2>
      <div class="sub">按单图危害等级计数（未检出单独统计）</div>
      {bar_list_html(hazard_items, empty_text="无图片")}
    </section>
    <section class="panel">
      <h2>类别分布</h2>
      <div class="sub">全部图片正式检测的类别合计</div>
      {bar_list_html(class_items, empty_text="未检出任何目标")}
    </section>
  </div>

  <section class="panel">
    <h2>逐图汇总</h2>
    <div class="sub">失败图片保留行并标注原因，不影响其它图片</div>
    <table class="tbl">
      <thead><tr><th>文件</th><th>状态</th><th>尺寸</th><th>框数</th><th>正式最高置信度</th>
      <th>密度 /MP</th><th>危害</th><th>主要类别</th></tr></thead>
      <tbody>{"".join(table_rows) or '<tr><td colspan="8">无数据</td></tr>'}</tbody>
    </table>
  </section>

  <footer class="foot">
    <b>使用边界</b>
    <ul>{limitations_html}</ul>
    <div style="margin-top:10px">WheatFieldAI · 批量汇总由本地 Streamlit 工作台导出</div>
  </footer>
</div>
</body>
</html>
"""
