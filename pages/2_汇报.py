# -*- coding: utf-8 -*-
"""
pages/2_汇报.py
===============
可视化汇报页（Streamlit 多页入口，主 app.py 不改）。

三个 tab：
01 单图报告 —— 真实推理（与 app.py 同一调用链）→ 标注图 / 指标 / 危害刻度 /
              置信度直方图 / 框尺寸分布 / 类别分布 / 防治建议 / 溯源 → HTML + JSON 导出；
02 批量汇总 —— ≤10 张顺序推理 + st.progress，缩略图网格 / 危害分布 / 类别分布 /
              逐图表格 → CSV + HTML + ZIP 导出；
03 模型性能 —— 读取现有真实评测 JSON（任务 24 / 任务 22 / classwise refined 配置），
              缺失时显示“数据未找到”，不编造数字。

工程约束（AGENTS.md）：
- 不 import app.py（会执行整页）；模型管理器用 st.cache_resource；
- 推理/融合/危害/建议全部复用 src 中已验证模块，不另起一套；
- 展示层与算法分离：本页只整理与呈现结果，不改变推理契约；
- 所有图表数据来自真实推理结果或真实 JSON；合成数据仅用于 tests/test_report_data.py。
"""

import base64
import hashlib
import json
import os
import re
import sys
from html import escape
from pathlib import Path

import streamlit as st

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import altair as alt  # noqa: E402
import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from config import load_config  # noqa: E402
from models import DualModelManager, is_oom_error  # noqa: E402
from weed_inference import infer_weed_single_image, decode_upload  # noqa: E402
from pest_inference import infer_pest_single_image  # noqa: E402
from drawing import draw_annotations  # noqa: E402
from task_router import task_info  # noqa: E402
import report_data as rd  # noqa: E402
import report_html as rh  # noqa: E402

_ICON = _ROOT / "static" / "wheat-icon-128.png"
_LOGO = _ROOT / "static" / "wheat-icon-64.png"

st.set_page_config(
    page_title="可视化汇报 · WheatFieldAI",
    page_icon=str(_ICON) if _ICON.is_file() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# 视觉常量与页面样式（暖纸色编辑风，与导出报告同源）
# ----------------------------------------------------------------------
CORAL = "#cc785c"
CORAL_DEEP = "#a9583e"
INK = "#3c3c3a"
MUTED = "#8a867e"
HAIR = "#e2ded6"

PAGE_CSS = """
:root{--canvas:#fdfdfd;--surface:#f8f5f0;--card:#fdfbf6;--ink:#3c3c3a;--body:#6b6b68;
--muted:#6f6c66;--hair:#e2ded6;--hair-soft:#eee9e0;--coral:#cc785c;--coral-deep:#a9583e;
--ok:#35714b;--warn:#8a6510;--danger:#b85c48;
--serif:"SmileySans","Songti SC","Source Han Serif SC",Georgia,"Times New Roman",serif;
--sans:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--canvas);
color:var(--ink);font-family:var(--sans)}
.block-container{max-width:1380px;padding-top:30px;padding-bottom:44px}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border:1px solid var(--hair);
border-radius:12px;padding:16px}
[data-testid="stHorizontalBlock"]{gap:16px}
.rp-bar{background:var(--surface);border:1px solid var(--hair);border-radius:12px;
padding:18px 22px;margin-bottom:16px}
.rp-bar-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.rp-logo{width:36px;height:36px;border-radius:8px;object-fit:cover;border:1px solid var(--hair)}
.rp-brand{font-family:var(--serif);font-size:18px;letter-spacing:-.2px}
.rp-sep{width:1px;height:16px;background:var(--hair)}
.rp-title{font-family:var(--serif);font-size:22px;letter-spacing:-.2px;line-height:1.3}
.rp-bar-sub{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-top:9px}
.rp-kicker{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.rp-h{font-family:var(--serif);font-size:19px;line-height:1.3;margin:2px 0 0;color:var(--ink)}
.rp-sub{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.55}
.rp-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;
margin-bottom:12px;flex-wrap:wrap}
.rp-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.rp-metrics.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.rp-metric{background:var(--canvas);border:1px solid var(--hair);border-radius:8px;padding:12px 13px}
.rp-metric .k{font-size:11px;letter-spacing:.06em;color:var(--muted);text-transform:uppercase}
.rp-metric .v{font-size:26px;line-height:1.15;margin-top:5px;font-variant-numeric:tabular-nums;
color:var(--ink);white-space:nowrap}
.rp-metric .u{font-size:11px;color:var(--muted);margin-top:3px}
.rp-note{background:var(--surface);border:1px solid var(--hair);border-radius:8px;
padding:12px 14px;font-size:12.5px;color:var(--body);line-height:1.65}
.rp-empty{background:var(--surface);border:1px dashed var(--hair);border-radius:12px;
padding:28px 24px;text-align:center}
.rp-empty .t{font-family:var(--serif);font-size:18px;margin-bottom:6px}
.rp-empty .d{font-size:13px;color:var(--body);line-height:1.7;max-width:620px;margin:0 auto}
.rp-advice{white-space:pre-line;font-size:14px;line-height:1.78;color:var(--body)}
.rp-kv{display:grid;grid-template-columns:180px minmax(0,1fr);gap:8px 16px;font-size:12.5px}
.rp-kv .k{color:var(--muted)}
.rp-kv .v{color:var(--ink);overflow-wrap:anywhere}
.thumb-ph{height:150px;display:flex;align-items:center;justify-content:center;
background:#efeae1;border-radius:6px;color:var(--muted);font-size:12px;text-align:center;padding:8px}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--hair)}
[data-testid="stTabs"] [data-baseweb="tab"]{font-family:var(--serif);font-size:15px;
color:var(--muted);padding:9px 14px;background:transparent}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--coral-deep)}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background-color:var(--coral)}
.stDownloadButton button{width:100%;min-height:42px;border-radius:8px;font-weight:600;
border:1px solid var(--coral);color:#fff;background:var(--coral)}
.stDownloadButton button:hover{background:var(--coral-deep);border-color:var(--coral-deep);color:#fff}
.stButton button{width:100%;min-height:42px;border-radius:8px;border:1px solid var(--hair);
background:var(--canvas);color:var(--ink)}
.stButton button:hover{border-color:var(--coral);color:var(--coral-deep)}
[data-testid="stSegmentedControl"] button[aria-checked="true"]{background:var(--card)!important;
color:var(--ink)!important;border:1px solid var(--hair)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--canvas);border:1px dashed var(--hair);
border-radius:8px}
[data-testid="stDataFrame"]{border:1px solid var(--hair);border-radius:8px}
@media(max-width:900px){.rp-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}
.block-container{padding-left:12px;padding-right:12px}}
"""

st.markdown(f"<style>{PAGE_CSS}{rh.component_css()}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 缓存与工具
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_manager(cfg):
    """进程内缓存模型管理器（与 app.py 同一类，任务切换时释放另一侧模型）。"""
    return DualModelManager(cfg)


@st.cache_data(show_spinner=False, ttl=600)
def load_performance(root_str):
    """读取评测 JSON（10 分钟缓存；文件改动后刷新页面即可）。"""
    return rd.load_model_performance(Path(root_str))


def esc(value):
    return escape("" if value is None else str(value))


def safe_stem(name, max_len=60):
    stem = Path(str(name)).stem
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("_")
    return (stem or "report")[:max_len]


def mode_key_of(task_key, mode_label):
    if mode_label == "高精度融合":
        return "fusion"
    return "yolo11" if task_key == "weed" else "fast"


def validate_upload(uploaded, cfg):
    """与 app.py 同口径的上传校验；返回问题列表。"""
    problems = []
    upload_cfg = cfg.get("upload", {})
    allowed = set(upload_cfg.get("allowed_extensions") or [".jpg", ".jpeg", ".png"])
    max_bytes = float(upload_cfg.get("max_file_size_mb", 20)) * 1024 * 1024
    ext = os.path.splitext(uploaded.name)[1].lower()
    if uploaded.size == 0:
        problems.append("文件为空（0 字节）。")
    elif ext not in allowed:
        problems.append(f"不支持的文件格式 {ext}，仅支持 JPG / JPEG / PNG。")
    elif uploaded.size > max_bytes:
        problems.append(
            f"文件过大（{uploaded.size / 1024 / 1024:.1f} MB），超过限制 "
            f"{upload_cfg.get('max_file_size_mb', 20)} MB。"
        )
    return problems


def run_inference(manager, cfg, task_key, mode_key, img_bgr):
    """真实推理入口：与 app.py 完全相同的调用链，不改变任何参数。

    任务切换只在真正推理前发生（``switch_task`` 对同一任务幂等）：渲染阶段不切换，
    否则单图 tab 与批量 tab 每次 rerun 各切一次，会反复卸载/重载模型。
    """
    with manager.lock:
        manager.switch_task(task_key)
        if task_key == "weed":
            return infer_weed_single_image(img_bgr, cfg, manager, mode=mode_key)
        return infer_pest_single_image(img_bgr, cfg, manager._pest, mode=mode_key)


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def jpeg_bytes(img_bgr, quality=88):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def thumb_jpeg(img_bgr, width=520):
    h, w = img_bgr.shape[:2]
    if w > width:
        scale = width / float(w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    return jpeg_bytes(img_bgr, quality=78)


def card_head(kicker, title, sub=""):
    sub_html = f'<div class="rp-sub">{esc(sub)}</div>' if sub else ""
    return (f'<div class="rp-head"><div><div class="rp-kicker">{esc(kicker)}</div>'
            f'<div class="rp-h">{esc(title)}</div>{sub_html}</div></div>')


def metric_cells(items, two=False):
    cells = []
    for label, value, unit in items:
        unit_html = f'<div class="u">{esc(unit)}</div>' if unit else ""
        cells.append(
            f'<div class="rp-metric"><div class="k">{esc(label)}</div>'
            f'<div class="v">{esc(value)}</div>{unit_html}</div>'
        )
    cls = "rp-metrics two" if two else "rp-metrics"
    return f'<div class="{cls}">{"".join(cells)}</div>'


def kv_html(pairs):
    rows = []
    for key, value in pairs:
        if value is None or value == "":
            continue
        rows.append(f'<div class="k">{esc(key)}</div><div class="v">{value}</div>')
    return f'<div class="rp-kv">{"".join(rows)}</div>'


def empty_state(title, desc):
    return (f'<div class="rp-empty"><div class="t">{esc(title)}</div>'
            f'<div class="d">{desc}</div></div>')


def alt_theme(chart, height=240):
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(
            labelColor=INK, titleColor=MUTED, gridColor="#eee9e0",
            domainColor=HAIR, tickColor=HAIR, labelFontSize=11, titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="top",
                          direction="horizontal", labelFontSize=11)
    )


def confidence_chart(bins, decision_conf):
    rows = []
    order = []
    for item in bins:
        order.append(item["label"])
        rows.append({"区间": item["label"], "序列": "正式检测", "框数": item["decided"]})
        rows.append({"区间": item["label"], "序列": "低置信候选", "框数": item["candidates"]})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    series = ["正式检测", "低置信候选"]
    y_max = max([row["框数"] for row in rows] + [1])
    chart = alt.Chart(df).mark_bar(size=11).encode(
        x=alt.X("区间:N", sort=order, axis=alt.Axis(labelAngle=0, labelFontSize=10)),
        xOffset=alt.XOffset("序列:N", sort=series),
        y=alt.Y("框数:Q", stack=None,
                scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(tickMinStep=1)),
        color=alt.Color(
            "序列:N", sort=series,
            scale=alt.Scale(domain=series, range=[CORAL, "#cfc8bb"]),
            legend=alt.Legend(title=None),
        ),
        tooltip=[alt.Tooltip("区间:N"), alt.Tooltip("序列:N"), alt.Tooltip("框数:Q")],
    )
    return alt_theme(chart, 250)


def size_chart(box_sizes):
    buckets = (box_sizes or {}).get("buckets") or []
    if not buckets:
        return None
    order = [b["label"] for b in buckets]
    df = pd.DataFrame([{"分桶": b["label"], "框数": b["count"]} for b in buckets])
    x_max = max([b["count"] for b in buckets] + [1])
    x_scale = alt.Scale(domain=[0, x_max])
    base = alt.Chart(df).mark_bar(color=CORAL, cornerRadius=2).encode(
        y=alt.Y("分桶:N", sort=order, title=None, axis=alt.Axis(labelFontSize=10)),
        x=alt.X("框数:Q", stack=None, scale=x_scale, axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("分桶:N"), alt.Tooltip("框数:Q")],
    )
    text = alt.Chart(df).mark_text(align="left", dx=4, color=INK, fontSize=11).encode(
        y=alt.Y("分桶:N", sort=order),
        x=alt.X("框数:Q", stack=None, scale=x_scale),
        text=alt.Text("框数:Q"),
    )
    return alt_theme(base + text, 200)


def class_chart(dist):
    if not dist:
        return None
    df = pd.DataFrame([{"类别": c["name"], "框数": c["count"]} for c in dist])
    x_max = max([c["count"] for c in dist] + [1])
    chart = alt.Chart(df).mark_bar(color=CORAL, cornerRadius=2).encode(
        y=alt.Y("类别:N", sort="-x", title=None,
                axis=alt.Axis(labelFontSize=10, labelLimit=240)),
        x=alt.X("框数:Q", stack=None, scale=alt.Scale(domain=[0, x_max]),
                axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("类别:N"), alt.Tooltip("框数:Q")],
    )
    return alt_theme(chart, max(220, 24 * len(df)))


def hazard_chart(dist):
    if not dist:
        return None
    order = ["未检出", "轻", "中", "重"]
    colors = ["#cfc8bb", "#8fae94", "#d3a862", "#c07a63"]
    df = pd.DataFrame([{"危害": h["level"], "图片数": h["count"]} for h in dist])
    y_max = max([h["count"] for h in dist] + [1])
    chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(
        x=alt.X("危害:N", sort=order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("图片数:Q", stack=None, scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("危害:N", sort=order,
                        scale=alt.Scale(domain=order, range=colors), legend=None),
        tooltip=[alt.Tooltip("危害:N"), alt.Tooltip("图片数:Q")],
    )
    return alt_theme(chart, 220)


def model_compare_chart(models):
    rows = []
    order = []
    for row in models:
        order.append(row["label"])
        original = row.get("original") or {}
        if original.get("mAP50_95") is not None:
            rows.append({"模型": row["label"], "口径": "原始 val", "mAP50-95": original["mAP50_95"]})
        v2 = row.get("v2") or {}
        if v2.get("mAP50_95") is not None:
            rows.append({"模型": row["label"], "口径": "val_v2（诊断）", "mAP50-95": v2["mAP50_95"]})
    if not rows:
        return None
    series = ["原始 val", "val_v2（诊断）"]
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("模型:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=11, labelLimit=280)),
        yOffset=alt.YOffset("口径:N", sort=series),
        x=alt.X("mAP50-95:Q", stack=None, scale=alt.Scale(domain=[0, 0.55]),
                axis=alt.Axis(format=".1f")),
        color=alt.Color("口径:N", sort=series,
                        scale=alt.Scale(domain=series, range=[CORAL, "#cfc8bb"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("模型:N"), alt.Tooltip("口径:N"),
                 alt.Tooltip("mAP50-95:Q", format=".4f")],
    )
    return alt_theme(chart, max(240, 42 * len(order)))


def pest_ap_chart(per_class):
    if not per_class:
        return None
    df = pd.DataFrame([
        {"类别": f'{c["name"]}', "AP": c["ap"], "class_id": c["class_id"]}
        for c in per_class
    ])
    chart = alt.Chart(df).mark_bar(color=CORAL, cornerRadius=2).encode(
        y=alt.Y("类别:N", sort="-x", title=None,
                axis=alt.Axis(labelFontSize=10, labelLimit=240)),
        x=alt.X("AP:Q", stack=None, scale=alt.Scale(domain=[0, 0.8]),
                title="AP50-95（逐类）", axis=alt.Axis(format=".1f")),
        tooltip=[alt.Tooltip("类别:N"), alt.Tooltip("class_id:Q"),
                 alt.Tooltip("AP:Q", format=".4f", title="AP50-95（逐类）")],
    )
    return alt_theme(chart, max(360, 19 * len(df)))


# ----------------------------------------------------------------------
# 页眉
# ----------------------------------------------------------------------
_cfg = load_config()
_manager = get_model_manager(_cfg)

logo_html = ""
if _LOGO.is_file():
    _logo_b64 = base64.b64encode(_LOGO.read_bytes()).decode("ascii")
    logo_html = f'<img class="rp-logo" src="data:image/png;base64,{_logo_b64}" alt=""/>'

st.markdown(
    f'<div class="rp-bar"><div class="rp-bar-row">{logo_html}'
    f'<span class="rp-brand">Anerqi</span><span class="rp-sep"></span>'
    f'<span class="rp-title">可视化汇报</span></div>'
    f'<div class="rp-bar-sub">WheatFieldAI · 单图报告 / 批量汇总 / 模型性能 · 数据全部来自真实推理与现有评测文件</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**可视化汇报**")
    st.caption("独立页；主工作台在「app」页。")
    st.markdown(
        f'<div class="rp-note">推理设备：{esc(_manager.device_str())}<br>'
        f'业务阈值 decision_conf：杂草 {esc(_cfg["weed"].get("decision_conf"))} · '
        f'害虫 {esc(_cfg["pest"].get("decision_conf"))}</div>',
        unsafe_allow_html=True,
    )
    st.caption("本页与主工作台各自持有模型管理器；显存紧张时先释放本页缓存再切回主工作台。")
    if st.button("释放本页模型缓存", key="rp_free_gpu"):
        _manager.free_gpu()
        st.session_state.pop("rp_single_cache", None)
        st.session_state.pop("rp_batch_state", None)
        st.success("已释放本页模型缓存。")

tab_single, tab_batch, tab_perf = st.tabs(["01 · 单图报告", "02 · 批量汇总", "03 · 模型性能"])


# ======================================================================
# 01 单图报告
# ======================================================================
def render_single_tab():
    c1, c2, c3 = st.columns([2.3, 2.3, 5.4], gap="medium")
    with c1:
        st.session_state.setdefault("rp_task", "杂草检测")
        task_label = st.segmented_control("检测对象", ["杂草检测", "害虫检测"], key="rp_task")
        task_label = task_label or "杂草检测"
        task_key = "pest" if task_label == "害虫检测" else "weed"
    with c2:
        st.session_state.setdefault("rp_mode", "高精度融合")
        mode_label = st.segmented_control("推理模式", ["高精度融合", "快速单模型"], key="rp_mode")
        mode_label = mode_label or "高精度融合"
        mode_key = mode_key_of(task_key, mode_label)
    with c3:
        uploaded = st.file_uploader(
            "上传单张田间图片",
            type=["jpg", "jpeg", "png"],
            key="rp_single_upload",
            help="支持 JPG / JPEG / PNG；单张 ≤20 MB。推理参数与主工作台完全一致。",
        )

    # 不在渲染阶段切换任务：switch_task 由 run_inference 在持锁推理前调用（幂等）
    missing, weights_ok = _manager.check_weights(task_key, mode_key)
    info = task_info(task_key, mode_key)

    if not weights_ok:
        st.markdown(
            f'<div class="rp-note"><b>当前模式无法运行：模型未配置。</b><br>'
            f'以下权重不存在，本页不会生成任何模拟结果：<br>'
            + "<br>".join(esc(item) for item in missing) + "</div>",
            unsafe_allow_html=True,
        )
        return

    if uploaded is None:
        st.markdown(
            empty_state(
                "上传一张图片开始生成报告",
                "选择检测对象与推理模式后上传田间图片。本页将展示标注图、正式检测框数、"
                "最高置信度、耗时、密度、危害刻度、置信度直方图、框尺寸分布、类别分布、"
                "防治建议与溯源信息，并可导出独立 HTML / JSON。",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="rp-note" style="margin-top:12px">当前模式：{esc(info["mode_display"])} · '
            f'{esc(info["models"])} · {esc(info["class_scope"])}</div>',
            unsafe_allow_html=True,
        )
        return

    problems = validate_upload(uploaded, _cfg)
    if problems:
        st.markdown(
            f'<div class="rp-note"><b>文件校验失败：</b>{esc(problems[0])}<br>'
            f'该文件未进入推理。</div>',
            unsafe_allow_html=True,
        )
        return

    img_bgr = decode_upload(uploaded.getvalue())
    if img_bgr is None:
        st.markdown(
            '<div class="rp-note"><b>图片无法解码</b>（可能已损坏或并非真实图片），未进入推理。</div>',
            unsafe_allow_html=True,
        )
        return

    file_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()[:20]
    cache_key = f"{task_key}|{mode_key}|{file_hash}"
    cache = st.session_state.setdefault("rp_single_cache", {})
    entry = cache.get(cache_key)

    if entry is None:
        with st.spinner("正在推理…"):
            try:
                result = run_inference(_manager, _cfg, task_key, mode_key, img_bgr)
                entry = {"result": result, "generated_at": rd.now_iso()}
                cache[cache_key] = entry
                while len(cache) > 4:
                    cache.pop(next(iter(cache)))
            except Exception as exc:  # noqa: BLE001 - 需要把真实异常展示给用户
                if is_oom_error(exc):
                    _manager.free_gpu()
                    st.markdown(
                        '<div class="rp-note"><b>GPU 显存不足（CUDA out of memory）。</b><br>'
                        '已释放本页模型缓存。请关闭其他占用显存的程序，或切换到「快速单模型」后重试。'
                        '本页不会用降级结果冒充真实推理。</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("释放显存并重试", key="rp_oom_retry"):
                        cache.pop(cache_key, None)
                        st.rerun()
                else:
                    st.markdown(
                        f'<div class="rp-note"><b>推理失败：</b>{esc(f"{type(exc).__name__}: {exc}")}</div>',
                        unsafe_allow_html=True,
                    )
                return

    result = entry["result"]
    payload = rd.build_single_report(
        result, task_key, uploaded.name, generated_at=entry["generated_at"], cfg=_cfg
    )
    metrics = payload["metrics"]
    hazard = payload["hazard"]

    show_low = st.checkbox(
        "标注图叠加低置信候选框",
        value=False,
        key="rp_show_low",
        help=f"默认只画置信度 ≥ {result['decision_conf']:.2f} 的正式检测；"
             "勾选后额外叠加灰色虚线候选框（仅页面显示，不参与计数）。导出报告始终只含正式检测。",
    )
    # 两遍绘制（与 app.py 一致）：先正式检测（彩色实线），再可选低置信候选（灰色虚线）。
    # export_bgr 始终只含正式检测，导出 HTML/JSON 与图注“仅正式检测”保持一致。
    display_bgr, export_bgr = rd.compose_annotation_image(
        img_bgr,
        result["detections"],
        result.get("low_conf_candidates"),
        show_low=show_low,
    )

    left, right = st.columns([7, 5], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown(
                card_head("01 · 视觉结果", "标注图",
                          f"{uploaded.name} · {payload['image']['width']}×{payload['image']['height']} px"
                          f" · 文件 {uploaded.size / 1024:.0f} KB"),
                unsafe_allow_html=True,
            )
            st.image(to_rgb(display_bgr), width="stretch")
            hidden = len(result.get("low_conf_candidates") or [])
            if hidden and show_low:
                st.caption(
                    f"灰色细虚线框为 {min(hidden, 50)} 个置信度 < {result['decision_conf']:.2f} 的"
                    f"低置信候选（最多显示 50 个），仅用于辅助查看，不参与计数与危害等级；"
                    f"导出报告只含正式检测。"
                )
            elif hidden:
                st.caption(
                    f"已隐藏 {hidden} 个置信度 < {result['decision_conf']:.2f} 的低置信候选框；"
                    f"计数与危害等级基于其余 {len(result['detections'])} 个正式检测。"
                )
    with right:
        with st.container(border=True):
            st.markdown(card_head("02 · 关键指标", "结果概览"), unsafe_allow_html=True)
            max_conf = metrics["max_decided_confidence"]
            st.markdown(
                metric_cells([
                    ("正式检测框", f'{metrics["num_detections"]}', "≥ decision_conf"),
                    ("正式最高置信度", rh.fmt_num(max_conf, 3) if max_conf is not None else "—",
                     "仅正式检测"),
                    ("耗时", rh.fmt_num(metrics["inference_time_ms"], 0), "ms（含模型加载）"),
                    ("密度", rh.fmt_num(metrics["density_per_mp"], 2), "个/百万像素"),
                ], two=True),
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="rp-kicker">03 · 危害刻度</div>'
                '<div class="rp-sub">阈值来自 config.yaml 的 hazard 段（原型规则）</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                rh.hazard_scale_html(hazard, task=task_key, decision_conf=result["decision_conf"]),
                unsafe_allow_html=True,
            )

    col_hist, col_size = st.columns(2, gap="medium")
    with col_hist:
        with st.container(border=True):
            st.markdown(
                card_head("04 · 置信度分布", "正式检测 vs 低置信候选",
                          f"decision_conf = {result['decision_conf']:.2f}；推理阈值保持评测口径"),
                unsafe_allow_html=True,
            )
            chart = confidence_chart(payload["confidence_histogram"], result["decision_conf"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            st.caption(
                f"正式检测 {metrics['num_detections']} 个 · 低置信候选 {metrics['num_candidates']} 个 · "
                f"平均置信度 {rh.fmt_num(metrics['mean_confidence'], 3)}"
            )
    with col_size:
        with st.container(border=True):
            st.markdown(
                card_head("05 · 框尺寸分布", "按 √(宽×高) 分桶",
                          "32 / 96 px 阈值与评测 AP_medium / AP_large 同源"),
                unsafe_allow_html=True,
            )
            chart = size_chart(payload["box_sizes"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            stats = payload["box_sizes"]["stats"]
            st.caption(
                f"中位 √面积 {rh.fmt_num(stats['median_side_px'], 1)} px · "
                f"均值 {rh.fmt_num(stats['mean_side_px'], 1)} px · "
                f"范围 {rh.fmt_num(stats['min_side_px'], 1)}–{rh.fmt_num(stats['max_side_px'], 1)} px · "
                f"平均宽高比 {rh.fmt_num(stats['mean_aspect'], 2)}"
            )

    with st.container(border=True):
        st.markdown(
            card_head("06 · 类别分布", "害虫类别分布" if task_key == "pest" else "类别分布",
                      "按正式检测计数，类别名来自 dataset.yaml" if task_key == "pest"
                      else "单类任务：播娘蒿（Obonianghao）"),
            unsafe_allow_html=True,
        )
        if task_key == "pest":
            chart = class_chart(payload["class_distribution"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            else:
                st.markdown('<div class="rp-note">未检出任何目标。</div>', unsafe_allow_html=True)
        else:
            count = metrics["num_detections"]
            st.markdown(
                f'<div class="rp-note">类别：<b>Obonianghao（播娘蒿）</b> · 正式检测 <b>{count}</b> 个'
                f'（置信度 ≥ {result["decision_conf"]:.2f}）。</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown(card_head("07 · 防治建议", "规则模板输出"), unsafe_allow_html=True)
        advice = payload["advice"]
        # 0 检出：显示“未检出 / —”，不显示“轻（未检出）（严重度 0/3）”
        no_detection = (
            bool(hazard.get("none"))
            or int(hazard.get("severity") or 0) == 0
            or int(metrics["num_detections"]) == 0
        )
        hazard_text = "未检出" if no_detection else esc(hazard.get("label") or "—")
        severity_text = "—" if no_detection else f'{esc(hazard.get("severity"))}/3'
        st.markdown(
            f'<div class="rp-kv" style="grid-template-columns:120px minmax(0,1fr)">'
            f'<div class="k">危害等级</div><div class="v">{hazard_text}'
            f'（严重度 {severity_text}）</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="rp-advice" style="margin-top:10px">{esc((advice.get("advice") or "").strip())}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="rp-note" style="margin-top:12px"><b>原型声明：</b>'
            f'{esc((advice.get("disclaimer") or "").strip())}</div>',
            unsafe_allow_html=True,
        )

    col_prov, col_export = st.columns([7, 5], gap="medium")
    with col_prov:
        with st.container(border=True):
            st.markdown(card_head("08 · 溯源", "本次报告的可核查信息"), unsafe_allow_html=True)
            prov = payload["provenance"]
            wbf = prov.get("wbf_params")
            wbf_text = "—"
            if isinstance(wbf, dict):
                wbf_text = " · ".join(f"{k}={v}" for k, v in wbf.items())
            elif wbf:
                wbf_text = esc(wbf)
            mapping = prov.get("class_mapping") or {}
            mapping_text = (" · ".join(f"{k}={v}" for k, v in mapping.items())
                            if isinstance(mapping, dict) and mapping else "")
            thresholds = prov.get("inference_thresholds") or {}
            threshold_text = " · ".join(f"{k}={v}" for k, v in thresholds.items()) or "—"
            st.markdown(
                kv_html([
                    ("源文件", esc(payload["file_name"])),
                    ("图像尺寸", esc(prov.get("image_size"))),
                    ("任务 / 模式", esc(f'{payload["task_label"]} · {payload["mode_display"]}')),
                    ("使用模型", esc("、".join(payload["models_used"]) or "—")),
                    ("业务阈值 decision_conf", rh.fmt_num(result["decision_conf"], 2)),
                    ("模型原始最高置信度（含候选）", rh.fmt_num(metrics["max_confidence"], 3)),
                    ("推理阈值（评测口径）", esc(threshold_text)),
                    ("WBF / 融合参数", wbf_text),
                    ("推理设备", esc(payload.get("device") or "—")),
                    ("耗时明细", esc(
                        f'总 {rh.fmt_num(metrics["inference_time_ms"], 0)} ms · '
                        f'纯推理 {rh.fmt_num(metrics["net_inference_ms"], 0)} ms · '
                        f'模型加载 {rh.fmt_num(metrics["model_load_ms"], 0)} ms'
                    )),
                    ("类别映射", esc(mapping_text) or "—"),
                    ("生成时间", esc(payload["generated_at"])),
                ]),
                unsafe_allow_html=True,
            )
    with col_export:
        with st.container(border=True):
            st.markdown(card_head("09 · 导出", "独立文件，离线可打开"), unsafe_allow_html=True)
            # 导出图来自 compose_annotation_image 的 export_bgr：永远只含正式检测，
            # 不受“叠加低置信候选”开关影响，与 HTML 图注一致。
            export_resized = export_bgr
            h, w = export_resized.shape[:2]
            if w > 1600:
                export_resized = cv2.resize(export_resized, (1600, int(h * 1600 / w)),
                                            interpolation=cv2.INTER_AREA)
            annotated_jpeg = jpeg_bytes(export_resized, quality=88)
            html_report = rh.render_single_report_html(
                payload, annotated_jpeg, annotated_alt=f'{payload["file_name"]} 标注图'
            )
            json_report = json.dumps(payload, ensure_ascii=False, indent=2)
            stem = safe_stem(uploaded.name)
            st.download_button(
                "下载 HTML 报告",
                data=html_report.encode("utf-8"),
                file_name=f"report_{stem}.html",
                mime="text/html",
                key="rp_dl_html",
            )
            st.download_button(
                "下载 JSON 结果",
                data=json_report.encode("utf-8"),
                file_name=f"report_{stem}.json",
                mime="application/json",
                key="rp_dl_json",
            )
            st.caption(
                "HTML 报告内嵌标注图（base64，仅正式检测），不依赖外部 CDN；"
                "JSON 含全部检测框坐标与溯源字段。"
            )


# ======================================================================
# 02 批量汇总
# ======================================================================
def render_batch_tab():
    c1, c2 = st.columns([3, 7], gap="medium")
    with c1:
        st.session_state.setdefault("rp_batch_task", "杂草检测")
        task_label = st.segmented_control("检测对象", ["杂草检测", "害虫检测"], key="rp_batch_task")
        task_label = task_label or "杂草检测"
        task_key = "pest" if task_label == "害虫检测" else "weed"
        st.session_state.setdefault("rp_batch_mode", "快速单模型")
        mode_label = st.segmented_control("推理模式", ["高精度融合", "快速单模型"], key="rp_batch_mode")
        mode_label = mode_label or "快速单模型"
        mode_key = mode_key_of(task_key, mode_label)
    with c2:
        max_files = int(_cfg.get("upload", {}).get("max_files_per_batch", 10))
        uploads = st.file_uploader(
            f"上传图片（最多 {max_files} 张，顺序推理）",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="rp_batch_upload",
            help="逐张独立推理；单张失败不影响其它图片。批量页只做汇总，逐图完整报告请用单图报告页。",
        )

    # 同单图 tab：任务切换只发生在 run_inference 内，渲染阶段不切换
    missing, weights_ok = _manager.check_weights(task_key, mode_key)
    if not weights_ok:
        st.markdown(
            f'<div class="rp-note"><b>当前模式无法运行：模型未配置。</b><br>'
            + "<br>".join(esc(item) for item in missing) + "</div>",
            unsafe_allow_html=True,
        )
        return

    if not uploads:
        st.markdown(
            empty_state(
                f"上传最多 {max_files} 张图片开始批量汇总",
                "逐张顺序推理，进度条显示当前图片；完成后展示缩略图网格、每图框数 / 危害 / 状态徽章、"
                "危害分布、类别分布与逐图表格，并可导出 CSV / HTML / ZIP。",
            ),
            unsafe_allow_html=True,
        )
        return

    if len(uploads) > max_files:
        st.warning(f"单批最多 {max_files} 张，超出部分已忽略。")
        uploads = uploads[:max_files]

    signature = hashlib.sha256(
        (f"{task_key}|{mode_key}|" + "|".join(
            hashlib.sha256(uploaded.getvalue()).hexdigest() for uploaded in uploads
        )).encode("utf-8")
    ).hexdigest()[:20]

    state = st.session_state.get("rp_batch_state")
    run_now = st.button("开始批量推理", key="rp_batch_run", type="primary")
    if state and state.get("sig") != signature and not run_now:
        st.info("检测到文件或模式已变化，上次批量结果已过期；请重新点击「开始批量推理」。")
        state = None

    if run_now:
        items = []
        thumbs = []
        total = len(uploads)
        progress = st.progress(0.0, text="准备推理…")
        for index, uploaded in enumerate(uploads):
            progress.progress(index / total, text=f"正在推理 {index + 1}/{total}：{uploaded.name}")
            problems = validate_upload(uploaded, _cfg)
            if problems:
                items.append({"file_name": uploaded.name, "status": "error", "error": problems[0]})
                thumbs.append({"file_name": uploaded.name, "jpeg": None, "badge": "none",
                               "badge_text": "失败", "caption": problems[0]})
                continue
            img_bgr = decode_upload(uploaded.getvalue())
            if img_bgr is None:
                items.append({"file_name": uploaded.name, "status": "error",
                              "error": "图片无法解码"})
                thumbs.append({"file_name": uploaded.name, "jpeg": None, "badge": "none",
                               "badge_text": "失败", "caption": "图片无法解码"})
                continue
            try:
                result = run_inference(_manager, _cfg, task_key, mode_key, img_bgr)
            except Exception as exc:  # noqa: BLE001
                if is_oom_error(exc):
                    _manager.free_gpu()
                    error_text = "GPU 显存不足（CUDA out of memory），已释放本页模型缓存；建议改用快速单模型"
                else:
                    error_text = f"{type(exc).__name__}: {exc}"
                items.append({"file_name": uploaded.name, "status": "error", "error": error_text})
                thumbs.append({"file_name": uploaded.name, "jpeg": None, "badge": "none",
                               "badge_text": "失败", "caption": error_text})
                continue

            annotated = draw_annotations(img_bgr, result["detections"])
            annotated_jpeg = jpeg_bytes(annotated, quality=86)
            items.append({
                "file_name": uploaded.name,
                "status": "ok",
                "result": result,
                "annotated_jpeg": annotated_jpeg,
            })
            hazard = result.get("hazard") or {}
            # 未检出统一中性徽章「未检出」，不显示“轻（未检出）”
            badge = rd.hazard_badge(result)
            thumbs.append({
                "file_name": uploaded.name,
                "jpeg": thumb_jpeg(annotated),
                "badge": badge["badge"],
                "badge_text": badge["text"],
                "caption": (f'{len(result["detections"])} 框 · '
                            f'{rh.fmt_num((hazard or {}).get("density_per_mp"), 2)} /MP'),
            })
        progress.progress(1.0, text="推理完成")
        progress.empty()

        first_ok = next((item["result"] for item in items if item.get("status") == "ok"), None)
        state = {
            "sig": signature,
            "task": task_key,
            "mode_key": mode_key,
            "items": items,
            "thumbs": thumbs,
            "generated_at": rd.now_iso(),
            "mode_display": (first_ok or {}).get("mode_display") or task_info(task_key, mode_key)["mode_display"],
            "decision_conf": (first_ok or {}).get("decision_conf"),
        }
        st.session_state["rp_batch_state"] = state

    if not state:
        return

    items = state["items"]
    payload = rd.build_batch_summary(
        items,
        state["task"],
        state["mode_display"],
        generated_at=state["generated_at"],
        decision_conf=state["decision_conf"],
    )
    totals = payload["totals"]

    st.markdown(
        metric_cells([
            ("图片总数", f'{totals["images"]}', f'单批上限 {max_files} 张'),
            ("成功 / 失败", f'{totals["ok"]} / {totals["failed"]}', "逐图独立推理"),
            ("正式检测框", f'{totals["detections"]}', "≥ decision_conf"),
            ("平均密度", rh.fmt_num(totals["mean_density_per_mp"], 2), "个/百万像素"),
        ]),
        unsafe_allow_html=True,
    )
    if totals["failed"]:
        failed_names = "、".join(row["file_name"] for row in payload["rows"] if row["status"] == "失败")
        st.warning(f"有 {totals['failed']} 张图片失败：{failed_names}。失败行保留在表格中，不影响其它图片。")

    with st.container(border=True):
        st.markdown(
            card_head("01 · 缩略图", "逐图结果",
                      f'{state["mode_display"]} · decision_conf '
                      f'{rh.fmt_num(state["decision_conf"], 2)} · {state["generated_at"]}'),
            unsafe_allow_html=True,
        )
        thumb_html = []
        for thumb in state["thumbs"]:
            if thumb.get("jpeg"):
                img_tag = (f'<img src="data:image/jpeg;base64,'
                           f'{base64.b64encode(thumb["jpeg"]).decode("ascii")}" '
                           f'alt="{esc(thumb["file_name"])}"/>')
            else:
                img_tag = '<div class="thumb-ph">推理失败</div>'
            thumb_html.append(
                f'<figure class="thumb">{img_tag}<figcaption>'
                f'<span class="fname">{esc(thumb["file_name"])}</span>'
                f'<span class="tmeta"><span class="badge {esc(thumb["badge"])}">'
                f'{esc(thumb["badge_text"])}</span> {esc(thumb["caption"])}</span>'
                f'</figcaption></figure>'
            )
        st.markdown(f'<div class="thumbs">{"".join(thumb_html)}</div>', unsafe_allow_html=True)

    col_h, col_c = st.columns(2, gap="medium")
    with col_h:
        with st.container(border=True):
            st.markdown(card_head("02 · 危害分布", "按单图危害等级"), unsafe_allow_html=True)
            chart = hazard_chart(payload["hazard_distribution"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
    with col_c:
        with st.container(border=True):
            st.markdown(card_head("03 · 类别分布", "全部图片正式检测合计"), unsafe_allow_html=True)
            chart = class_chart(payload["class_distribution"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            else:
                st.markdown('<div class="rp-note">未检出任何目标。</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(card_head("04 · 逐图汇总", "失败图片保留行并标注原因"), unsafe_allow_html=True)
        df = pd.DataFrame([
            {
                "文件": row["file_name"],
                "状态": row["status"],
                "宽×高": (f'{row["width"]}×{row["height"]}' if row["width"] else "—"),
                "框数": row["num_detections"],
                "正式最高置信度": row["max_decided_confidence"],
                "密度 /MP": row["density_per_mp"],
                "危害": row["hazard_label"],
                "主要类别": row["top_class"],
                "错误": row["error"] or "",
            }
            for row in payload["rows"]
        ])
        st.dataframe(df, hide_index=True, width="stretch")

    with st.container(border=True):
        st.markdown(card_head("05 · 导出", "CSV / HTML / ZIP"), unsafe_allow_html=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        html_text = rh.render_batch_report_html(payload, state["thumbs"])
        # ZIP 由 report_data 组装：主干用 batch_summary.unique_stems 解重名
        # （a.jpg / a.png 不再互相覆盖），并写入 batch_summary.json。
        zip_bytes = rd.build_batch_zip(
            payload,
            items,
            cfg=_cfg,
            summary_csv=csv_bytes,
            summary_html=html_text,
            generated_at=state["generated_at"],
        )
        stamp = re.sub(r"[^0-9]", "", state["generated_at"])[:14]
        d1, d2, d3 = st.columns(3, gap="small")
        with d1:
            st.download_button("下载 CSV", data=csv_bytes, file_name=f"batch_{stamp}.csv",
                               mime="text/csv", key="rp_batch_csv")
        with d2:
            st.download_button("下载 HTML", data=html_text.encode("utf-8"),
                               file_name=f"batch_{stamp}.html", mime="text/html",
                               key="rp_batch_html")
        with d3:
            st.download_button("下载 ZIP", data=zip_bytes,
                               file_name=f"batch_{stamp}.zip", mime="application/zip",
                               key="rp_batch_zip")
        st.caption(
            "ZIP 内含 summary.csv / summary.html / batch_summary.json / "
            "每图 JSON（完整检测框与溯源）/ 每图标注 JPG（仅正式检测）；重名文件自动加后缀。"
        )


# ======================================================================
# 03 模型性能
# ======================================================================
def render_perf_tab():
    perf = load_performance(str(rd.workspace_root()))

    st.markdown(
        '<div class="rp-note"><b>口径声明：</b>本页所有指标均为<b>本地验证集</b>口径，项目无独立测试集；'
        '<b>val_v2 仅为模型辅助诊断口径</b>（多模型一致 + 高置信补标），不是人工金标准，'
        '也不是官方测试集。缺失文件显示“数据未找到”，本页不生成任何估计值。</div>',
        unsafe_allow_html=True,
    )

    missing = [src for src in perf["sources"].values() if not src["exists"]]
    if missing:
        # 只显示相对路径/文件名，不把本机绝对路径写进界面（AGENTS.md 3.5）
        st.warning(
            "数据未找到：\n"
            + "\n".join(
                f'- {rd.relative_path_text(src["path"], perf["root"])}' for src in missing
            )
        )

    weed = perf["weed"]
    pest = perf["pest"]
    dc = perf["decision_conf"]
    datasets = dc.get("datasets") or {}

    def dataset_scale(task):
        """验证集规模文案：只从任务 22 summary 的 datasets 取，取不到就省略数字。"""
        if task == "pest":
            images, boxes = datasets.get("pest_images"), datasets.get("pest_gt_boxes")
        else:
            images, boxes = datasets.get("weed_images"), datasets.get("weed_gt_boxes")
        if images and boxes:
            return f"本地验证集 {images} 图 / {boxes:,} 框"
        return "本地验证集"

    with st.container(border=True):
        st.markdown(
            card_head("01 · 杂草", "五模型融合 vs 单模型",
                      f"mAP50-95 / mAP50 / AP75；同一{dataset_scale('weed')}"),
            unsafe_allow_html=True,
        )
        if not weed["available"]:
            st.markdown('<div class="rp-note">数据未找到：杂草评测 JSON（任务 24 outputs）。本节不显示任何数字。</div>',
                        unsafe_allow_html=True)
        else:
            five = next((row for row in weed["models"] if row["key"] == "five_model_app"), None)
            run = weed.get("five_model_run") or {}
            if five and five.get("original"):
                original = five["original"]
                st.markdown(
                    metric_cells([
                        ("五模型 mAP50", rh.fmt_num(original.get("mAP50"), 4), "本地验证集"),
                        ("五模型 mAP50-95", rh.fmt_num(original.get("mAP50_95"), 4), "本地验证集"),
                        ("五模型 AP75", rh.fmt_num(original.get("AP75"), 4), "本地验证集"),
                        ("复算规模", f'{run.get("images") or "—"} 图',
                         f'失败 {run.get("failures") if run.get("failures") is not None else "—"} 张 · '
                         f'{rh.fmt_num(run.get("seconds_per_image"), 2)} s/图'),
                    ]),
                    unsafe_allow_html=True,
                )
                run_scale = f'{run.get("images")} 图' if run.get("images") else "真实管线"
                st.caption(
                    f'五模型成员：{"、".join(weed.get("five_model_members") or [])}。'
                    f'复算来自 app 真实管线 {run_scale}，峰值显存 {run.get("peak_reserved_gb")} GB。'
                )
            chart = model_compare_chart(weed["models"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")

            table_rows = []
            for row in weed["models"]:
                original = row.get("original") or {}
                v2 = row.get("v2") or {}
                table_rows.append({
                    "模型 / 组合": row["label"],
                    "分组": row["group"],
                    "mAP50（原始）": original.get("mAP50"),
                    "mAP50-95（原始）": original.get("mAP50_95"),
                    "AP75（原始）": original.get("AP75"),
                    "mAP50-95（val_v2）": v2.get("mAP50_95"),
                    "AP75（val_v2）": v2.get("AP75"),
                    "说明": row.get("note") or "",
                })
            st.dataframe(
                pd.DataFrame(table_rows).round(4),
                hide_index=True,
                width="stretch",
            )
            st.caption("val_v2 列来自模型辅助补标的诊断口径，不能与原始 val 列混用比较结论。")

            if weed["fusion_candidates"]:
                with st.expander("融合候选对比（任务 24 候选实验）", expanded=False):
                    cand_rows = []
                    for cand in weed["fusion_candidates"]:
                        original = cand.get("original") or {}
                        v2 = cand.get("v2") or {}
                        cand_rows.append({
                            "候选": cand["key"],
                            "成员": "、".join(cand.get("members") or []),
                            "mAP50-95（原始）": original.get("mAP50_95"),
                            "AP75（原始）": original.get("AP75"),
                            "mAP50-95（val_v2）": v2.get("mAP50_95"),
                            "预测框数": cand.get("n_predictions"),
                        })
                    st.dataframe(pd.DataFrame(cand_rows).round(4), hide_index=True,
                                 width="stretch")

    with st.container(border=True):
        st.markdown(
            card_head("02 · 害虫", "逐类 AP（refined classwise 融合）",
                      "类别名来自 dataset.yaml；逐类参数按类单独调优"),
            unsafe_allow_html=True,
        )
        if not pest.get("available"):
            st.markdown(
                f'<div class="rp-note">数据未找到：{esc(pest.get("reason") or "classwise refined 配置")}。'
                f'本节不显示任何数字。</div>',
                unsafe_allow_html=True,
            )
        else:
            pest_boxes = datasets.get("pest_gt_boxes")
            scale_unit = (
                f"{pest_boxes:,} 框（任务 22 口径）" if pest_boxes else "任务 22 口径"
            )
            st.markdown(
                metric_cells([
                    ("mAP50", rh.fmt_num(pest.get("map50"), 4), "本地验证集"),
                    ("mAP50-95", rh.fmt_num(pest.get("map50_95"), 4), "本地验证集"),
                    ("类别数", f'{pest.get("n_classes")}', f'融合 {pest.get("n_models")} 个模型'),
                    ("验证规模", f'{pest.get("n_images")} 图', scale_unit),
                ]),
                unsafe_allow_html=True,
            )
            chart = pest_ap_chart(pest["per_class"])
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            with st.expander(f'逐类 AP50-95 明细（{pest.get("n_classes")} 类）', expanded=False):
                st.dataframe(
                    pd.DataFrame([
                        {"class_id": row["class_id"], "类别（拉丁名）": row["name"],
                         "AP50-95（逐类）": row["ap"]}
                        for row in pest["per_class"]
                    ]).round(4),
                    hide_index=True,
                    width="stretch",
                )
            per_class = pest.get("per_class") or []
            if per_class:
                st.caption(
                    f'{len(per_class)} 类平均 AP50-95（逐类）{rh.fmt_num(pest.get("mean_ap"), 4)}；'
                    f'最高 {esc(per_class[0]["name"])} {rh.fmt_num(per_class[0]["ap"], 4)}；'
                    f'最低 {esc(per_class[-1]["name"])} {rh.fmt_num(per_class[-1]["ap"], 4)}。'
                )

    with st.container(border=True):
        st.markdown(
            card_head("03 · 模型对比", "业务阈值校准（decision_conf）",
                      "任务 22：本地验证集上按 F1 逐阈值扫描；"
                      f'当前 config.yaml 取杂草 {_cfg["weed"].get("decision_conf")} / '
                      f'害虫 {_cfg["pest"].get("decision_conf")}'),
            unsafe_allow_html=True,
        )
        if not dc.get("available"):
            st.markdown('<div class="rp-note">数据未找到：任务 22 decision_conf 校准 summary.json。</div>',
                        unsafe_allow_html=True)
        else:
            labels = {
                "weed_fusion": "杂草 · 高精度融合",
                "weed_fast": "杂草 · 快速单模型",
                "pest_fusion": "害虫 · 高精度融合",
                "pest_fast": "害虫 · 快速单模型",
            }
            rows = []
            for key, label in labels.items():
                node = dc["picks"].get(key)
                if not node:
                    continue
                best = node.get("best_f1") or {}
                at_025 = node.get("at_0_25") or {}
                rows.append({
                    "组合": label,
                    "最佳 F1 阈值": best.get("threshold"),
                    "Precision": best.get("precision"),
                    "Recall": best.get("recall"),
                    "F1": best.get("f1"),
                    "图片数": best.get("n_images"),
                    "GT 框数": best.get("num_gt"),
                    "阈值 0.25 时 F1": at_025.get("f1"),
                })
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
            st.caption(
                f'校准数据集：杂草 {datasets.get("weed_images", "—")} 图 / '
                f'{datasets.get("weed_gt_boxes", "—")} 框；害虫 {datasets.get("pest_images", "—")} 图 / '
                f'{datasets.get("pest_gt_boxes", "—")} 框。校准结果偏乐观（验证集曾用于调参）；'
                f'config.yaml 实际取杂草 {_cfg["weed"].get("decision_conf")} / '
                f'害虫 {_cfg["pest"].get("decision_conf")}，表中 best-F1 阈值仅作参考。'
            )

    with st.container(border=True):
        st.markdown(card_head("04 · 数据源", "本页读取的真实评测文件"), unsafe_allow_html=True)
        source_rows = []
        for key, src in perf["sources"].items():
            source_rows.append({
                "键": key,
                "文件": rd.relative_path_text(src["path"], perf["root"]),
                "状态": "已读取" if src["exists"] and not src["error"] else (
                    "解析失败" if src["exists"] else "数据未找到"
                ),
                "错误": src["error"] or "",
            })
        st.dataframe(pd.DataFrame(source_rows), hide_index=True, width="stretch")
        st.markdown(
            "<div class='rp-note' style='margin-top:10px'>" +
            "<br>".join(esc(note) for note in perf.get("notes") or []) +
            "</div>",
            unsafe_allow_html=True,
        )


with tab_single:
    render_single_tab()
with tab_batch:
    render_batch_tab()
with tab_perf:
    render_perf_tab()
