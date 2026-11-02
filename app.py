# ./app.py
# -*- coding: utf-8 -*-
"""小麦田间杂草与害虫智能识别及辅助决策系统 · Streamlit 工作台（Bento-Box Grid · Claude 版）。

栅格规范（12 列）：行1 配置(4)+状态(8)；行2 原图(6)+标注(6)；行3 摘要(4)+类别统计(4)+危害(4)；
行4 检测明细(12,可折叠)；行5 防治建议(6)+导出(6)；行6 推理参数(12,可折叠)。
卡片=单一业务职责；全局统一 gap=16px / 卡片内边距 16px；
视觉语言采用 DESIGN-claude.md（Claude 设计语言）：暖奶油画布 #faf9f5 + 珊瑚 #cc785c 系 + 浅色附录卡 + 衬线标题；品牌水印「Anerqi」纯字标（无图形标志）；卡片不用彩色侧边条，行内卡片等高对齐。
业务口径：仅置信度 ≥ decision_conf（config.yaml）的检测参与计数/类别统计/危害等级/JSON 主列表；低于该值的候选框单独保留，可在页面勾选查看。
模块化说明：各卡片为独立渲染函数，未来模块（摄像头流/批量推理/历史记录/报告导出）按同样
的「一个业务一张卡」方式接入；不提供未实现功能的假开关。
"""
import base64
import hashlib
import json
import os
import re
import sys
import time
from html import escape
from pathlib import Path

import streamlit as st

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "True")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_THIS_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_THIS_DIR / "src"))

from config import load_config
from models import DualModelManager, is_oom_error
from weed_inference import infer_weed_single_image, decode_upload
from pest_inference import infer_pest_single_image
from drawing import draw_annotations, encode_image_to_bytes
from task_router import task_info, task_label
import batch_summary  # 任务25：批量总览的纯数据层（摘要/JSON payload/缩略图/ZIP）

st.set_page_config(
    page_title="小麦田间杂草与害虫双任务识别工作台",
    page_icon="static/wheat-icon-128.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(r"""
<style>
@font-face{font-family:"SmileySans";src:url("/app/static/SmileySans-Oblique.ttf.woff2") format("woff2"),url("/app/static/SmileySans-Oblique.otf") format("opentype");font-weight:400;font-style:normal;font-display:swap}
:root{--canvas:#fdfdfd;--surface:#f8f5f0;--surface-card:#f8f5f0;--ink:#3c3c3a;--body:#6b6b68;--muted:#6f6c66;--muted-soft:#9a978f;--hairline:#e2ded6;--hairline-soft:#ebe7e0;--coral:#cc785c;--coral-btn:#b25f41;--coral-active:#a9583e;--coral-text:#9c4a2f;--ok:#35714b;--warn:#8a6510;--danger:#b85c48;--danger-text:#ab4f3e;--teal:#2a7263;--amber:#9c6a1f;--radius-sm:6px;--radius:8px;--radius-lg:12px;--serif:"SmileySans","Smiley Sans","得意黑","Source Han Serif SC","Songti SC",Garamond,"Times New Roman",serif;--sans:"SmileySans","Smiley Sans","得意黑","Source Sans",Inter,"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;--song:"SimSun","宋体","Songti SC","Noto Serif SC","Source Han Serif SC",serif}
html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{font-family:var(--sans);color:var(--ink)}
/* 全站字体收敛为两种：得意黑（标题/正文，var(--serif)/var(--sans)）+ 宋体（标签/数值/元数据等次要文字，var(--song)） */
[data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"],
[data-testid="stAppViewContainer"] button,[data-testid="stAppViewContainer"] [role="button"],
[data-testid="stAppViewContainer"] label,[data-testid="stAppViewContainer"] [data-baseweb="input"] input,
[data-testid="stAppViewContainer"] [data-testid="stFileUploader"]{font-family:var(--sans)}
body,.stApp{background:var(--canvas)}
.block-container{max-width:1440px;padding-top:64px;padding-bottom:32px}
*,*::before,*::after{box-sizing:border-box}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden}
:focus-visible{outline:2px solid var(--coral-active)!important;outline-offset:2px;border-radius:4px}
button:focus-visible,[role="button"]:focus-visible,input:focus-visible,textarea:focus-visible{box-shadow:0 0 0 2px var(--coral-active)!important}
button,[role="button"],input[type="file"]{min-height:44px}
button,[role="button"]{transition:border-color .16s ease-out,color .16s ease-out,background-color .16s ease-out}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}}
[data-testid="stHorizontalBlock"]{gap:16px}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius-lg);padding:16px}
/* 列容器：列盒拉伸到行高（等高）；卡片内撑拉伸，让同一行各卡片底部边框对齐 */
[data-testid="stColumn"]{display:flex;flex-direction:column}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]{flex:1;display:flex;flex-direction:column}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]>[data-testid="stLayoutWrapper"]{flex:1;display:flex;flex-direction:column}
[data-testid="stColumn"]>[data-testid="stVerticalBlock"]>[data-testid="stLayoutWrapper"]>[data-testid="stVerticalBlock"]{flex:1;display:flex;flex-direction:column}
/* 压缩顶部控件垂直占用，缓解“控制区过高、状态区过空”的顶部失衡 */
[data-testid="stWidgetLabel"]{padding-top:1px;padding-bottom:1px}
[data-testid="stFileUploaderDropzone"]{padding-top:8px;padding-bottom:8px}
[data-testid="stFileUploaderDropzone"] button{min-height:36px;font-size:13px}
/* 页眉（Claude 顶部奶油导航式） */
.workbench-bar{background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--radius-lg);padding:16px 20px;margin-bottom:16px}
.bar-logo{width:40px;height:40px;border-radius:9px;object-fit:cover;flex:none;border:1px solid var(--hairline)}
.bar-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.brand-word{font-family:var(--serif);font-size:20px;font-weight:400;letter-spacing:-0.2px;color:var(--ink)}
.brand-sep{width:1px;height:18px;background:var(--hairline);flex:none}
.brand-title{font-family:var(--serif);font-size:15px;font-weight:400;line-height:1.3;color:var(--ink)}
.brand-sub{font-family:var(--song);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.bento-panel{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius-lg);padding:16px;height:100%}
.bento-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.bento-head .head-main{min-width:0}
.panel-title{font-family:var(--serif);font-size:18px;font-weight:400;letter-spacing:-0.2px;line-height:1.3;margin:0;color:var(--ink)}
.panel-kicker{font-family:var(--sans);font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.helper{font-size:13px;color:var(--body);line-height:1.5}.small{font-size:12px;color:var(--muted);line-height:1.5}
.status-row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.status-item{display:inline-flex;align-items:center;gap:7px;font-family:var(--song);font-size:12px;font-weight:500;color:var(--ink)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--teal);flex:none}
.dot.warn{background:var(--amber)}
.dot.danger{background:var(--danger)}
.dot.loading{background:var(--amber)}
.prototype-note{background:var(--surface-card);border:1px solid var(--hairline);border-radius:var(--radius);padding:12px 14px;margin:12px 0 0;color:var(--body);font-size:13px;line-height:1.55}
.mode-note{background:var(--surface-card);border:1px solid var(--hairline);border-radius:var(--radius);padding:11px 13px;color:var(--body);font-size:13px;line-height:1.55;margin:10px 0 14px}
.divider{height:1px;background:var(--hairline-soft);margin:9px 0}
.file-name{font-size:14px;font-weight:600;overflow-wrap:anywhere;color:var(--ink)}.file-meta{font-family:var(--song);font-size:11px;color:var(--muted);font-weight:400}
.file-status{display:inline-flex;align-items:center;gap:7px;font-family:var(--song);font-size:12px;font-weight:600}.file-status.running{color:var(--warn)}.file-status.error{color:var(--danger-text)}
.result-frame{background:var(--surface-card);border:1px solid var(--hairline);border-radius:var(--radius);padding:10px}
.result-image{width:100%;flex:1;min-height:220px;max-height:420px;display:flex;align-items:center;justify-content:center;overflow:hidden;background:var(--surface-card);border-radius:var(--radius)}
.result-image img{width:100%;height:100%;object-fit:contain}
.result-meta{font-family:var(--song);font-size:11px;color:var(--muted);line-height:1.5}
.metric-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.metric-box{background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--radius);padding:12px}
.metric-label{font-family:var(--song);font-size:11px;letter-spacing:.06em;color:var(--muted);margin-bottom:6px}
.metric-value{font-family:var(--song);font-size:30px;font-weight:600;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap}
.metric-unit{font-size:11px;color:var(--muted);margin-top:4px}
.hazard{background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--radius);padding:14px}
.hazard.light .hazard-line{color:var(--ok)}
.hazard.medium .hazard-line{color:var(--warn)}
.hazard.heavy .hazard-line{color:var(--danger)}
.hazard.none .hazard-line{color:var(--ink)}
.hazard-line{display:flex;align-items:center;gap:9px;font-family:var(--serif);font-weight:400;font-size:22px;letter-spacing:-0.2px;color:var(--ink)}
.hazard-detail{font-size:13px;line-height:1.5;margin-top:6px;color:var(--body)}.hazard-reminder{font-weight:700;margin-top:7px;color:var(--danger-text)}
.advice{font-size:15px;line-height:1.7;color:var(--ink)}
.tech-details{font-family:var(--song);font-size:11px;color:var(--muted);line-height:1.6}
.kv-list{display:grid;gap:8px}
.kv-card{border:1px solid var(--hairline);border-radius:var(--radius);padding:10px 12px;font-size:13px;line-height:1.5;overflow-wrap:anywhere;background:var(--canvas)}
.kv-card .kv-conf{font-weight:700;color:var(--ink)}
.error-box{background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--radius);padding:12px 14px;color:var(--danger-text);font-size:14px;line-height:1.5}
/* 按钮：下载=珊瑚主按钮；其余=奶油次按钮 */
.stDownloadButton button{width:100%;min-height:44px;border-radius:var(--radius);font-weight:600;border:1px solid var(--coral-btn);color:#FFFFFF;background:var(--coral-btn)}
.stDownloadButton button:hover{border-color:var(--coral-active);background:var(--coral-active)}
.stButton button{width:100%;min-height:44px;border-radius:var(--radius);font-weight:600;border:1px solid var(--hairline);color:var(--ink);background:var(--canvas)}
.stButton button:hover{border-color:var(--coral);color:var(--coral-text)}
/* 分段控件：选中=奶油卡底+墨字（Claude category-tab-active） */
[data-testid="stSegmentedControl"] button[aria-checked="true"],[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],[role="radiogroup"] button[aria-checked="true"]{background:var(--surface-card)!important;color:var(--ink)!important;border:1px solid var(--hairline)!important}
[data-testid="stSegmentedControl"] [role="radiogroup"] button,[role="radiogroup"] button{min-height:44px}
[data-testid="stFileUploaderDropzone"]{background:var(--canvas);border:1px dashed var(--hairline);border-radius:var(--radius)}
[data-testid="stExpander"]{border:1px solid var(--hairline);border-radius:var(--radius);background:var(--canvas)}
[data-testid="stExpander"] summary{font-family:var(--serif);font-size:15px}
[data-testid="stDataFrame"]{border:1px solid var(--hairline);border-radius:var(--radius)}
/* 附录卡（浅色，与面板同语言） */
.site-footer{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius-lg);padding:24px 28px;margin-top:16px}
.footer-brand{font-family:var(--serif);font-size:18px;letter-spacing:-0.2px;color:var(--ink);margin-bottom:14px}
.footer-title{font-family:var(--serif);font-size:18px;font-weight:400;letter-spacing:-0.2px;color:var(--ink);margin:0 0 8px}
.site-footer .panel-kicker{color:var(--muted)}
.footer-body{font-size:13px;line-height:1.7;color:var(--body)}
.footer-body strong{color:var(--ink)}
/* —— 任务16：信息条 / 空状态引导 / 结果强调 —— */
.info-note{background:var(--surface-card);border:1px solid var(--hairline);border-radius:var(--radius);padding:10px 12px;color:var(--body);font-size:13px;line-height:1.55;overflow-wrap:anywhere;margin-top:12px}
.info-note strong{color:var(--ink)}
.info-note svg{vertical-align:-0.18em;color:var(--coral-text)}
.info-tag{flex:none;font-family:var(--song);font-size:11px;font-weight:600;color:var(--coral-text);border:1px solid var(--hairline);border-radius:var(--radius-sm);background:var(--canvas);padding:1px 6px;white-space:nowrap;margin-right:8px}
.empty-card{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius-lg);padding:14px 16px 16px;margin-top:12px;display:flex;flex-direction:column;justify-content:center}
.empty-kicker{font-family:var(--song);font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.empty-title{font-family:var(--serif);font-size:19px;font-weight:400;letter-spacing:-0.2px;color:var(--ink);margin:0 0 6px}
.empty-desc{font-size:13px;color:var(--body);line-height:1.5;margin-bottom:10px}
.empty-steps{display:grid;gap:6px}
.step{display:flex;align-items:flex-start;gap:10px;font-size:13px;color:var(--body);line-height:1.45}
.step-num{flex:none;width:22px;height:22px;border-radius:50%;background:var(--canvas);border:1px solid var(--hairline);display:inline-flex;align-items:center;justify-content:center;font-family:var(--song);font-size:11px;font-weight:600;color:var(--coral-text)}
/* 长文本 / 长文件名截断与换行，避免窄屏撑破 */
.file-name,.info-note,.mode-note,.error-box{overflow-wrap:anywhere}
[data-testid="stDataFrame"]{overflow-x:auto}
/* —— 任务16·第二轮：卡片等高后的内部空间消化（边框对齐修复）；R17 改顶部对齐 —— */
/* R17：短卡片被超高同行卡（如 14 类全量统计表）拉伸时，space-between 会把标题与内容块撕到
   顶/底两端形成大面积空白；统一改 flex-start 顶部紧贴，余量留在卡片底部 */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .status-row){justify-content:flex-start}
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > .empty-card){flex:1;display:flex;flex-direction:column}
/* 图像帧在卡内拉伸，吸收输入/标注卡高度差（图像 object-fit:contain 居中，不裁切） */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"] > .result-frame){flex:1;display:flex;flex-direction:column}
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"] > .result-frame) [data-testid="stMarkdown"],
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"] > .result-frame) [data-testid="stMarkdown"] > div,
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"] > .result-frame) [data-testid="stMarkdownContainer"]{flex:1;display:flex;flex-direction:column}
/* 短卡片（摘要/危害/导出）内容顶部对齐，避免被超高同行卡拉伸后撕裂（R17） */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .metric-strip){justify-content:flex-start}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hazard){justify-content:flex-start}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stDownloadButton"]){justify-content:flex-start}
/* 统计表封顶，控制行3高度，避免长表格把整行撑太高 */
[data-testid="stDataFrame"]{max-height:320px;overflow-y:auto}

/* —— R16：类别统计表全量展示（撤销 R15 的 420px 滚动容器：Windows 覆盖式滚动条自动隐藏，14 类时 3 行被静默裁剪且无可发现性；sticky 表头在无滚动容器时会吸附视口顶部，一并移除） —— */
.class-table{border:1px solid var(--hairline);border-radius:var(--radius)}
.class-table table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:13px}
.class-table th{background:#f0ebe2;font-family:var(--sans);font-style:normal;font-size:11px;font-weight:600;letter-spacing:.08em;color:#6b5d4f;text-align:left;padding:8px 12px;border-bottom:2px solid #d9c9a8;white-space:nowrap}
.class-table th.num,.class-table td.num{width:72px;text-align:right}
.class-table td{padding:7px 12px;border-bottom:1px solid var(--hairline);color:var(--body)}
.class-table td:first-child{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.class-table td.num{font-family:var(--sans);font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums}
.class-table tr:last-child td{border-bottom:none}


/* —— 任务25：首屏紧凑摘要条（KPI）与批量总览网格 —— */
.kpi-strip{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 12px}
.kpi-chip{flex:1 1 160px;min-width:150px;background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius);padding:9px 12px}
.kpi-chip .kpi-k{font-family:var(--song);font-size:11px;letter-spacing:.06em;color:var(--muted);margin-bottom:3px}
.kpi-chip .kpi-v{font-family:var(--song);font-size:24px;font-weight:600;line-height:1.05;font-variant-numeric:tabular-nums;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi-chip .kpi-v.sm{font-size:13px}
.kpi-chip .kpi-s{font-size:11px;color:var(--muted);margin-top:2px}
.kpi-chip.ok .kpi-v{color:var(--ok)}
.kpi-chip.warn .kpi-v{color:var(--warn)}
.kpi-chip.danger .kpi-v{color:var(--danger)}
.kpi-chip.none .kpi-v{color:var(--ink)}
.batch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:2px}
.batch-thumb{background:var(--canvas);border:1px solid var(--hairline);border-radius:var(--radius);overflow:hidden;display:flex;flex-direction:column}
.batch-thumb img{width:100%;height:116px;object-fit:cover;display:block;background:#efeae2}
.bt-placeholder{height:116px;display:flex;align-items:center;justify-content:center;background:var(--surface-card);color:var(--muted);font-size:12px}
.bt-body{padding:8px 10px}
.bt-name{font-size:12px;font-weight:600;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bt-meta{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:5px;font-family:var(--song);font-size:11px;color:var(--muted)}
.badge{font-family:var(--song);font-size:10px;font-weight:600;border-radius:var(--radius-sm);padding:1px 6px;border:1px solid var(--hairline);background:var(--surface);color:var(--ink)}
.badge.ok{color:var(--ok);border-color:#bcd6c5}
.badge.fail{color:var(--danger-text);border-color:#e3c2ba}
.bt-hz.ok{color:var(--ok)}.bt-hz.warn{color:var(--warn)}.bt-hz.danger{color:var(--danger-text)}.bt-hz.none{color:var(--muted)}
/* 批量汇总表：固定列宽（百分比），保证 7 列在 1440 宽下全部可见 */
.batch-table table{table-layout:fixed}
.batch-table th.num,.batch-table td.num{width:auto;text-align:right;white-space:nowrap}
.batch-table td:first-child{text-align:left}
.batch-table td .badge{display:inline-block}

@media(max-width:768px){.block-container{padding-left:12px;padding-right:12px}.workbench-bar{padding:14px 16px}.brand-title{font-size:14px}.brand-word{font-size:18px}.metric-strip{grid-template-columns:1fr}[data-testid="stVerticalBlockBorderWrapper"],.bento-panel{padding:14px}}
@media(max-width:520px){.brand-word{font-size:17px}.brand-title{font-size:13px}.bar-row{gap:8px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model_manager(cfg):
    return DualModelManager(cfg)

def safe_basename(name):
    name=name.replace("\\","/").split("/")[-1]
    name=re.sub(r"[^\w.\-]","_",name)
    return name[:120]

def _bgr_jpg_b64(img_bgr):
    import cv2
    ok,buf=cv2.imencode(".jpg",img_bgr,[cv2.IMWRITE_JPEG_QUALITY,88])
    return base64.b64encode(buf.tobytes()).decode("ascii")

def lucide(kind,size=16):
    paths={
        "wheat":'<path d="M12 14V2"/><path d="M7 8c0-2 1-4 3-5 1 2 1 4 0 5"/><path d="M17 8c0-2-1-4-3-5-1 2-1 4 0 5"/><path d="M7 12c0-2 1-4 3-5 1 2 1 4 0 5"/><path d="M17 12c0-2-1-4-3-5-1 2-1 4 0 5"/><path d="M2 22c2-5 6-8 10-8s8 3 10 8"/>',
        "alert":'<path d="m21 12-8.5-8.5a2.12 2.12 0 0 0-3 0L3 10.9a2.12 2.12 0 0 0 0 3L9.5 20a2.12 2.12 0 0 0 3 0L21 14.1a2.12 2.12 0 0 0 0-2.1Z"/><path d="M12 8v5"/><path d="M12 16h.01"/>',
        "check":'<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
        "x":'<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
        "upload":'<path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 20h16"/>',
        "cpu":'<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/>',
        "clock":'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "activity":'<path d="M3 12h4l3-8 4 16 3-8h4"/>',
        "image":'<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>'
    }
    body=paths.get(kind,paths["upload"])
    return f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-0.15em;flex:none">{body}</svg>'

def hazard_semantic(level,label):
    if "未检出" in label:return "○ 轻（未检出）"
    if level=="轻":return "● 轻"
    if level=="中":return "▲ 中"
    return "■ 重"

def render_status_item(kind,text):
    return f'<span class="status-item"><span class="dot {kind}"></span><span>{escape(text)}</span></span>'

def card_head(kicker,title,right_html=""):
    right=f'<div>{right_html}</div>' if right_html else ""
    return f'<div class="bento-head"><div class="head-main"><div class="panel-kicker">{kicker}</div><div class="panel-title">{title}</div></div>{right}</div>'

def image_frame(img_bgr,alt):
    return f'<div class="result-frame"><div class="result-image"><img alt="{escape(alt)}" src="data:image/jpeg;base64,{_bgr_jpg_b64(img_bgr)}"/></div></div>'

# ================= 行1：配置盒(4) + 系统状态盒(8) =================
cfg=load_config()
# R17：mode_control 默认值改由 session_state 初始化（组件不再传 default，消除
# default 与 Session State API 并存的告警；OOM 降级路径的显式赋值语义不变）
st.session_state.setdefault("mode_control","高精度融合")
if st.session_state.get("force_fast_mode"):
    st.session_state["mode_control"]="快速单模型"
    st.session_state["force_fast_mode"]=False

st.markdown(f'<div class="workbench-bar"><div class="bar-row"><img src="/app/static/wheat-icon-64.png" alt="" aria-hidden="true" class="bar-logo"/><span class="brand-word">Anerqi</span><span class="brand-sep"></span><span class="brand-title">小麦田间杂草与害虫双任务识别工作台</span></div><div class="brand-sub">Weed &amp; Pest Recognition Workbench · 小麦田间杂草与害虫智能识别及辅助决策系统</div></div>',unsafe_allow_html=True)

col_c,col_s=st.columns([4,8],gap="medium")
with col_c:
    with st.container(border=True):
        st.markdown(card_head("01 · 工作台控制","检测任务"),unsafe_allow_html=True)
        task=st.segmented_control("检测对象",["杂草检测","害虫检测"],default="杂草检测",key="task_control",help="请选择检测对象；系统不会自动猜测图片属于哪个任务。")
        task_key="weed" if task=="杂草检测" else "pest"
        mode=st.segmented_control("推理模式",["高精度融合","快速单模型"],key="mode_control",help="高精度融合使用五模型 WBF（YOLO11s 960/box8/1280 + YOLOX Small/Base）；快速单模型用于快速复核或显存不足场景。")
        mode_key="fusion" if mode=="高精度融合" else ("yolo11" if task_key=="weed" else "fast")
        uploaded_files=st.file_uploader("上传田间图片",type=["jpg","jpeg","png"],accept_multiple_files=True,key=f"upload_{task_key}",help="支持 JPG/JPEG/PNG；单张不超过 20 MB；单批最多 10 张。切换检测对象会清空已上传图片。")
        st.markdown('<div class="helper">JPG / JPEG / PNG · 单张 ≤20MB · 单批 ≤10张<br>切换检测对象会清空已上传图片</div>',unsafe_allow_html=True)
with col_s:
    with st.spinner("正在加载模型…"):
        manager=get_model_manager(cfg)
    manager.switch_task(task_key)
    missing_weights,weights_ok=manager.check_weights(task_key,mode_key)
    if not weights_ok:chip=render_status_item("danger","权重缺失")
    elif st.session_state.get("last_oom"):chip=render_status_item("warn","显存不足降级")
    else:chip=render_status_item("ok","权重已找到（首次推理加载）")
    info=task_info(task_key,mode_key)
    with st.container(border=True):
        st.markdown(card_head("02 · 状态监控","运行状态"),unsafe_allow_html=True)
        st.markdown(f'<div class="status-row">{chip}{render_status_item("ok",task_label(task_key))}{render_status_item("ok",info["mode_display"])}</div>',unsafe_allow_html=True)
        st.markdown('<div class="result-meta" style="margin-top:8px">首次推理会加载模型，约 1–3 秒。</div>',unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="small"><strong>类别范围：</strong>{escape(info["class_scope"])}<br><strong>规则边界：</strong>{escape(info["prototype_note"])}</div>',unsafe_allow_html=True)
        # 状态卡内：模式说明 → 原型声明 → 批量/空状态（随卡片等高，底部边框对齐）
        st.markdown(f'<div class="info-note"><span class="info-tag">{escape(info["mode_display"])}</span>{escape(info["models"])} · {escape(info["class_scope"])}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="prototype-note">{lucide("alert",16)} <strong>原型系统声明：</strong>本系统为演示原型，危害等级与防治建议由规则模板生成，未经农学专家校准，不构成防治处方或正式农业决策依据。</div>',unsafe_allow_html=True)
        if uploaded_files:
            st.markdown(f'<div class="info-note batch-note">{lucide("image",16)} <strong>批量规则：</strong>每张图片独立校验与推理，单张失败不影响其余图片；单批最多 {escape(str(cfg["upload"]["max_files_per_batch"]))} 张，单张不超过 {escape(str(cfg["upload"]["max_file_size_mb"]))} MB。</div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="empty-card" role="region" aria-label="准备开始检测"><div class="empty-kicker">00 · 待命</div><div class="empty-title">准备开始检测</div><div class="empty-desc">选择检测对象与推理模式后，上传田间图片即可启动真实模型推理。</div><div class="empty-steps"><div class="step"><span class="step-num" aria-hidden="true">1</span><span>选择检测对象与推理模式</span></div><div class="step"><span class="step-num" aria-hidden="true">2</span><span>上传田间图片（JPG / JPEG / PNG，单张 ≤20MB）</span></div><div class="step"><span class="step-num" aria-hidden="true">3</span><span>查看标注结果、危害等级与防治建议</span></div></div></div>',unsafe_allow_html=True)
    if not weights_ok:
        st.markdown(f'<div class="error-box" style="margin-top:12px">{lucide("x",16)} <strong>当前模式无法运行</strong><br>以下模型权重不存在：<br>{"<br>".join(escape(m) for m in missing_weights)}<br><span class="small">请检查 config.yaml 或环境变量中的权重路径。</span></div>',unsafe_allow_html=True)

# ================= 结果区：批量总览 + 单图完整结果 =================
# 任务25（交互优化）：多图上传不再纵向堆叠 N 套完整结果卡。
# - 有效上传 >1：先对全部文件做校验与推理（复用 results_cache，key=task|mode|file_hash），
#   再渲染「批量总览」卡（缩略图网格 + 汇总表 + 选图 + 展开开关 + 批量 ZIP），
#   默认只渲染选中图片的完整结果；展开开关打开时保留原来的逐图完整结果。
# - 有效上传 =1：沿用原有逐图卡片结构，只多一条首屏紧凑摘要条（KPI）。
# 推理契约、结果键、下载 key、decision_conf 语义均未改动。

def _hazard_chip(result):
    """危害等级 chip：(css 类, 展示文本)。未检出走中性态，其余按轻/中/重上色。"""
    if batch_summary.is_no_detection(result):
        return "none", "未检出"
    level = (result.get("hazard") or {}).get("level")
    return {"轻": "ok", "中": "warn", "重": "danger"}.get(level, "ok"), batch_summary.hazard_display(result)

def kpi_strip_html(entry, task_key):
    """紧凑摘要条：正式框数 / 正式最高置信度 / 危害等级 / 耗时。

    放在图片行之前，让 1440×900 首屏在滚动前就能读到业务口径关键指标。
    数值口径与「结果概览」「危害等级」卡完全一致（仅正式检测，≥ decision_conf）。
    """
    result = entry["result"]
    hz_cls, hz_text = _hazard_chip(result)
    max_conf = batch_summary.decided_max_confidence(result)
    conf_text = "—" if max_conf is None else f"{max_conf:.3f}"
    chips = [
        f'<div class="kpi-chip"><div class="kpi-k">当前图片</div><div class="kpi-v sm" title="{escape(entry["filename"])}">{escape(entry["filename"])}</div><div class="kpi-s">{entry["size_kb"]:.0f} KB</div></div>',
        f'<div class="kpi-chip"><div class="kpi-k">正式检测框数</div><div class="kpi-v">{result["num_detections"]}</div><div class="kpi-s">框 · 置信度 ≥ {result["decision_conf"]:.2f}</div></div>',
        f'<div class="kpi-chip"><div class="kpi-k">正式最高置信度</div><div class="kpi-v">{conf_text}</div><div class="kpi-s">仅正式检测</div></div>',
        f'<div class="kpi-chip {hz_cls}"><div class="kpi-k">危害等级</div><div class="kpi-v">{escape(hz_text)}</div><div class="kpi-s">原型规则 · 需农学校准</div></div>',
        f'<div class="kpi-chip"><div class="kpi-k">处理耗时</div><div class="kpi-v">{result["inference_time_ms"]}</div><div class="kpi-s">ms</div></div>',
    ]
    return f'<div class="kpi-strip">{"".join(chips)}</div>'

def _annot_cache(entry):
    """标注图缓存（按 task|mode|file_hash）：{jpg: bytes, thumb: base64}。

    缩略图与下载按钮共用同一次绘制，避免为缩略图重复推理/重复绘制。
    """
    key = entry.get("cache_key") or f"annot_{entry['index']}"
    cache = st.session_state.setdefault("annot_cache", {})
    item = cache.get(key)
    if item is None:
        annotated = draw_annotations(entry["image"], entry["result"]["detections"])
        item = {
            "jpg": encode_image_to_bytes(annotated, ".jpg"),
            "thumb": base64.b64encode(batch_summary.thumbnail_jpeg(annotated)).decode("ascii"),
        }
        cache[key] = item
        while len(cache) > 12:
            cache.pop(next(iter(cache)))
    return item

def batch_grid_html(entries):
    """缩略图网格：标注图缩略 + 文件名 + 框数 + 危害等级 + 状态徽章。"""
    cards = []
    for entry in entries:
        s = batch_summary.summarize_entry(entry)
        name = escape(s["filename"])
        if entry["status"] == "ok":
            b64 = _annot_cache(entry)["thumb"]
            img_html = f'<img alt="{name} 标注缩略图" src="data:image/jpeg;base64,{b64}"/>'
            hz_cls, hz_text = _hazard_chip(entry["result"])
            meta = f'<span class="bt-boxes">{s["num_detections"]} 框</span><span class="bt-hz {hz_cls}">{escape(hz_text)}</span>'
            badge = '<span class="badge ok">完成</span>'
        else:
            img_html = '<div class="bt-placeholder">无预览</div>'
            meta = ""
            badge = f'<span class="badge fail">{escape(s["status_label"])}</span>'
        cards.append(
            f'<div class="batch-thumb">{img_html}<div class="bt-body">'
            f'<div class="bt-name" title="{name}">{name}</div>'
            f'<div class="bt-meta">{badge}{meta}</div></div></div>'
        )
    return f'<div class="batch-grid">{"".join(cards)}</div>'

def prepare_upload(up, upload_cfg):
    """上传校验（空文件/扩展名/大小/可解码）。返回 (fname, problems, img_bgr)。"""
    fname = safe_basename(up.name)
    ext = os.path.splitext(up.name)[1].lower()
    allowed_exts = set(upload_cfg["allowed_extensions"])
    max_bytes = upload_cfg["max_file_size_mb"] * 1024 * 1024
    problems = []
    img_bgr = None
    if up.size == 0:
        problems.append("文件为空（0 字节）。")
    elif ext not in allowed_exts:
        problems.append(f"不支持的文件格式 `{ext}`，仅支持 JPG / JPEG / PNG。")
    elif up.size > max_bytes:
        problems.append(f"文件过大（{up.size/1024/1024:.1f} MB），超过限制 {upload_cfg['max_file_size_mb']} MB。")
    if not problems:
        img_bgr = decode_upload(up.getvalue())
        if img_bgr is None:
            problems.append("文件扩展名合法，但图片内容无法解码（可能损坏或并非真实图片）。")
    return fname, problems, img_bgr

def new_entry(idx, up, fname, img_bgr):
    return {
        "index": idx,
        "filename": fname,
        "size_kb": up.size / 1024,
        "image": img_bgr,
        "result": None,
        "status": "ok",
        "error": None,
        "oom": False,
        "cache_key": None,
    }

def infer_entry(entry, task_key, mode_key, manager, cfg, up):
    """对已校验的 entry 做推理（命中 results_cache 时零成本）。返回 entry。"""
    # R17：Streamlit 任何组件交互都会整脚本重跑，以 任务|模式|文件内容哈希 为键缓存成功结果
    results_cache = st.session_state.setdefault("results_cache", {})
    file_hash = hashlib.sha256(up.getvalue()).hexdigest()[:20]
    cache_key = f"{task_key}|{mode_key}|{file_hash}"
    entry["cache_key"] = cache_key
    result = results_cache.get(cache_key)
    if result is not None:
        entry["result"] = result
        entry["status"] = "ok"
        return entry
    st.session_state["last_oom"] = False
    try:
        with manager.lock:  # 串行化共享单例上的推理，防止其他会话切换任务时卸载本会话正在使用的模型
            if task_key == "weed":
                result = infer_weed_single_image(entry["image"], cfg, manager, mode=mode_key)
            else:
                result = infer_pest_single_image(entry["image"], cfg, manager._pest, mode=mode_key)
    except Exception as exc:
        if is_oom_error(exc):
            manager.free_gpu()
            st.session_state["last_oom"] = True
            entry["oom"] = True
            entry["error"] = "GPU 显存不足（CUDA out of memory）。已释放模型缓存。请关闭其他占用显存的程序，或切换到「快速单模型」模式。"
        else:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["status"] = "error"
        return entry
    # 仅缓存成功结果（失败不缓存，OOM 重试仍会真实重跑）；FIFO 上限 12 条防内存膨胀
    results_cache[cache_key] = result
    while len(results_cache) > 12:
        results_cache.pop(next(iter(results_cache)))
    st.session_state["last_oom"] = False
    entry["result"] = result
    entry["status"] = "ok"
    return entry

def render_error_block(entry, task_key):
    """校验失败 / 推理失败卡片（与原有错误卡一致）。"""
    idx = entry["index"]
    fname = entry["filename"]
    if entry["status"] == "invalid":
        with st.container(border=True):
            st.markdown(card_head("文件状态", escape(fname), render_status_item("danger", "校验失败")), unsafe_allow_html=True)
            st.markdown(f'<div class="error-box">{lucide("x",16)} <strong>{escape(entry["error"] or "文件校验未通过。")}</strong><br><span class="small">该文件未进入推理；请修正后重试，其余图片不受影响。</span></div>', unsafe_allow_html=True)
        return
    with st.container(border=True):
        st.markdown(card_head("文件状态", escape(fname), render_status_item("danger", "推理失败")), unsafe_allow_html=True)
        st.markdown(f'<div class="error-box">{lucide("x",16)} <strong>无法完成当前文件推理</strong><br>{escape(entry["error"] or "未知错误")}<br><span class="small">本次推理失败不影响其他图片的处理。</span></div>', unsafe_allow_html=True)
        if entry.get("oom") and st.button("切换到快速单模型模式并重试", key=f"oom_fallback_{idx}", help="将当前模式切换为快速单模型，然后重新运行页面。"):
            st.session_state["force_fast_mode"] = True
            st.rerun()

def render_result_block(entry, task_key, mode_key, show_kpi=True):
    """单张图片的完整结果（原有逐图卡片的渲染体，键名保持 idx 维度不变）。

    show_kpi=False 用于批量总览的“所选图片”视图：批量卡顶部已有同口径 KPI 条，
    避免同一屏重复两条。
    """
    idx = entry["index"]
    fname = entry["filename"]
    img_bgr = entry["image"]
    result = entry["result"]
    size_kb = entry["size_kb"]

    # ---- 首屏紧凑摘要条（正式框数 / 正式最高置信度 / 危害等级 / 耗时）----
    if show_kpi:
        st.markdown(kpi_strip_html(entry, task_key), unsafe_allow_html=True)

    # ---- 行2：原图盒(6) + 标注盒(6) ----
    c_i1,c_i2=st.columns([6,6],gap="medium")
    with c_i1:
        with st.container(border=True):
            st.markdown(card_head("03 · 输入","输入原图",render_status_item("ok","检测完成")),unsafe_allow_html=True)
            st.markdown(f'<div class="file-name" style="margin-bottom:8px">{escape(fname)} <span class="file-meta">{size_kb:.0f} KB · {img_bgr.shape[1]}×{img_bgr.shape[0]} px</span></div>',unsafe_allow_html=True)
            st.markdown(image_frame(img_bgr,f"{fname} 原图"),unsafe_allow_html=True)
    with c_i2:
        with st.container(border=True):
            hd,cc=st.columns([5,2],gap="small")
            with hd:
                st.markdown(card_head("04 · 视觉分析","标注结果",render_status_item("ok","检测完成")),unsafe_allow_html=True)
            with cc:
                show_low_img=st.checkbox("显示低置信候选框",value=False,key=f"lowconf_{task_key}_{idx}",help=f"默认仅显示置信度 ≥ {result['decision_conf']:.2f} 的检测框（与危害等级口径一致）；勾选后叠加显示低置信候选框（最多 50 个）。")
            dets=result["detections"];hidden=result.get("low_conf_candidates",[])
            if show_low_img:
                # 先画正式检测（实线彩色），再叠加候选（灰色细虚线）；候选按置信度降序最多 50 个
                candidates_shown=sorted(hidden,key=lambda d:d["confidence"],reverse=True)[:50]
                annotated=draw_annotations(img_bgr,dets)
                annotated=draw_annotations(annotated,candidates_shown,low_conf=True)
            else:
                annotated=draw_annotations(img_bgr,dets)
            st.markdown(image_frame(annotated,f"{fname} 标注图"),unsafe_allow_html=True)
            if show_low_img:
                st.markdown(f'<div class="result-meta" style="margin-top:8px">图例：实线=正式检测，虚线=低置信候选（置信度 &lt; {result["decision_conf"]:.2f}）。</div>',unsafe_allow_html=True)
                if len(hidden)>50:
                    st.markdown(f'<div class="result-meta">候选共 {len(hidden)} 个，仅显示前 50 个候选。</div>',unsafe_allow_html=True)
            elif hidden:
                st.markdown(f'<div class="result-meta" style="margin-top:8px">已隐藏 {len(hidden)} 个置信度 &lt; {result["decision_conf"]:.2f} 的低置信候选框；危害等级与统计基于其余 {len(dets)} 个检测。</div>',unsafe_allow_html=True)

    # ---- 行3：摘要(4) + 类别统计(4) + 危害(4) ----
    c_k,c_st,c_h=st.columns([4,4,4],gap="medium")
    with c_k:
        with st.container(border=True):
            # 最高置信度只看正式检测（≥ decision_conf）；优先用推理侧字段，缺省时从 detections 取 max
            max_decided=batch_summary.decided_max_confidence(result)
            max_conf=f'{float(max_decided):.3f}' if max_decided is not None else "—"
            st.markdown(card_head("05 · 检测摘要","结果概览"),unsafe_allow_html=True)
            st.markdown(f'<div class="metric-strip"><div class="metric-box"><div class="metric-label">正式检测框数</div><div class="metric-value">{result["num_detections"]}</div><div class="metric-unit">框</div></div><div class="metric-box"><div class="metric-label">正式最高置信度</div><div class="metric-value">{max_conf}</div><div class="metric-unit">仅正式检测</div></div><div class="metric-box"><div class="metric-label">处理耗时</div><div class="metric-value">{result["inference_time_ms"]}</div><div class="metric-unit">ms</div></div></div>',unsafe_allow_html=True)
            st.markdown('<div class="result-meta" style="margin-top:10px">框数≠株数；密集/重叠目标可能重复计数。</div>',unsafe_allow_html=True)
            load_note=f'首次含模型加载 {result["model_load_ms"]:.0f} ms；后续图片纯推理约 {result["net_inference_ms"]:.0f} ms' if result.get("includes_first_load") else f'纯推理约 {result["net_inference_ms"]:.0f} ms'
            st.markdown(f'<div class="result-meta" style="margin-top:6px">{load_note}</div>',unsafe_allow_html=True)
    with c_st:
        with st.container(border=True):
            st.markdown(card_head("06 · 分类统计","按类别统计"),unsafe_allow_html=True)
            if task_key=="pest":
                if result.get("class_counts"):
                    _rows="".join(f'<tr><td title="{escape(k)}">{escape(k)}</td><td class="num">{v}</td></tr>' for k,v in result["class_counts"].items())
                    st.markdown(f'<div class="class-table"><table><thead><tr><th>类别（拉丁学名）</th><th class="num">数量</th></tr></thead><tbody>{_rows}</tbody></table></div>',unsafe_allow_html=True)
                    st.markdown(f'<div class="small" style="margin-top:6px">共 {len(result["class_counts"])} 类 · {result["num_detections"]} 框（置信度 ≥ {result["decision_conf"]:.2f}）。</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="helper">未检出害虫（置信度 ≥ {result["decision_conf"]:.2f}）。</div>',unsafe_allow_html=True)
            else:
                # 任务25：口径统一为「框」，与害虫侧「N 框」一致（原为「N 个目标」）
                st.markdown(f'**Obonianghao（播娘蒿）** · {result["num_detections"]} 个框',unsafe_allow_html=True)
    with c_h:
        with st.container(border=True):
            hazard=result["hazard"];level=hazard["level"];label=hazard["label"]
            no_detection=result["num_detections"]==0 or bool(hazard.get("none"))
            if no_detection:
                hazard_cls="none";glyph="○";hazard_text="未检出目标"
            else:
                hazard_cls={"轻":"light","中":"medium","重":"heavy"}.get(level,"light")
                glyph=hazard_semantic(level,label);hazard_text=f"危害等级：{label}"
            pest_detail=f'，检出类别 {result.get("major_class_count",0)} 类' if task_key=="pest" and not no_detection else ""
            reminder='<div class="hazard-reminder">需要人工复核，请结合田间情况判断。</div>' if level=="重" and not no_detection else ""
            st.markdown(card_head("07 · 风险告警","危害等级"),unsafe_allow_html=True)
            st.markdown(f'<div class="hazard {hazard_cls}"><div class="hazard-line">{escape(glyph)} <span>{escape(hazard_text)}</span></div><div class="hazard-detail">检测密度 {hazard["density_per_mp"]:.2f} 个/百万像素{pest_detail}</div>{reminder}<div class="small" style="margin-top:7px">原型规则，基于检测密度估算，需农学专家校准。</div></div>',unsafe_allow_html=True)

    # ---- 行4：检测明细(12,可折叠) ----
    with st.expander("检测框明细（科研）",expanded=False):
        c1,c2,c3=st.columns([1,1,1],gap="small")
        show_low=c1.checkbox("显示低置信行",value=False,key=f"tbl_low_{idx}",help=f"默认仅显示置信度 ≥ {result['decision_conf']:.2f} 的行。")
        show_xy=c2.checkbox("显示坐标列",value=False,key=f"tbl_xy_{idx}",help="展开 x1/y1/x2/y2/宽/高 列。")
        view=c3.selectbox("视图",["表格","键值卡片"],key=f"tbl_view_{idx}",label_visibility="collapsed")
        dets=result["detections"]+result.get("low_conf_candidates",[])
        if not dets:
            st.markdown("未检出任何目标。")
        else:
            shown=[d for d in dets if show_low or d["confidence"]>=result["decision_conf"]]
            hidden_n=len(dets)-len(shown)
            if view=="键值卡片":
                rows="".join(f'<div class="kv-card"><span class="kv-conf">#{i+1}</span> · {escape(d["category_name"] if task_key=="pest" else "Obonianghao（播娘蒿）")} · 置信度 <span class="kv-conf">{d["confidence"]:.4f}</span>'+(f' · ({int(d["bbox_xyxy"][0])},{int(d["bbox_xyxy"][1])}) → ({int(d["bbox_xyxy"][2])},{int(d["bbox_xyxy"][3])})' if show_xy else '')+'</div>' for i,d in enumerate(shown))
                st.markdown(f'<div class="kv-list">{rows}</div>',unsafe_allow_html=True)
            else:
                rows=[]
                for i,d in enumerate(shown):
                    r={"序号":i+1,"类别":d["category_name"] if task_key=="pest" else "Obonianghao（播娘蒿）","置信度":f'{d["confidence"]:.4f}'}
                    if show_xy:
                        r.update({"x1":int(d["bbox_xyxy"][0]),"y1":int(d["bbox_xyxy"][1]),"x2":int(d["bbox_xyxy"][2]),"y2":int(d["bbox_xyxy"][3]),"宽":int(d["bbox_xyxy"][2]-d["bbox_xyxy"][0]),"高":int(d["bbox_xyxy"][3]-d["bbox_xyxy"][1])})
                    rows.append(r)
                rows.sort(key=lambda r:(r["类别"],-float(r["置信度"])))
                st.dataframe(rows,width="stretch",hide_index=True)
            note=f"已隐藏 {hidden_n} 个置信度 < {result['decision_conf']:.2f} 的行。" if hidden_n and not show_low else ""
            st.markdown(f'<div class="small" style="margin-top:6px">显示 {len(shown)} / {len(dets)} 条。{note} 危害等级与统计仅基于置信度 ≥ {result["decision_conf"]:.2f} 的检测。</div>',unsafe_allow_html=True)

    # ---- 行5：防治建议(6) + 导出(6) ----
    c_a,c_e=st.columns([6,6],gap="medium")
    with c_a:
        with st.container(border=True):
            st.markdown(card_head("09 · 农事决策","防治建议（需农学专家审核）"),unsafe_allow_html=True)
            st.markdown(f'<div class="advice">{escape(result["advice"]["advice"]).replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="small" style="margin-top:10px">{escape(result["advice"]["disclaimer"]).replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
    with c_e:
        with st.container(border=True):
            st.markdown(card_head("10 · 结果输出","结果文件"),unsafe_allow_html=True)
            annotated_bytes=_annot_cache(entry)["jpg"]
            stem=os.path.splitext(fname)[0]
            st.download_button(f"下载 {fname} 标注图 JPG",data=annotated_bytes,file_name=f"{stem}_annotated.jpg",mime="image/jpeg",key=f"dl_img_{task_key}_{idx}",help="下载当前文件的带标注 JPG 图像（含置信度 ≥ 业务阈值的检测框）")
            # JSON 由 batch_summary.build_result_payload 统一构造：单图下载与批量 ZIP 字段完全一致
            json_payload=batch_summary.build_result_payload(fname,result,task_label(task_key))
            st.download_button(f"下载 {fname} 检测结果 JSON",data=json.dumps(json_payload,ensure_ascii=False,indent=2),file_name=f"{stem}_result.json",mime="application/json",key=f"dl_json_{task_key}_{idx}",help="下载当前文件的结构化检测结果 JSON（detections 为业务口径检测；低置信候选框见 low_conf_candidates）")

    # ---- 行6：推理参数(12,可折叠) ----
    with st.expander("推理参数与模型详情",expanded=False):
        meta=[f'模型：{"、".join(result["models_used"])}',f'设备：{result["device"]}',f'纯推理耗时：{result["net_inference_ms"]} ms',f'本次总耗时：{result["inference_time_ms"]} ms',f'每模型耗时：{result["per_model_time_ms"]}']
        if result.get("includes_first_load"):meta.append(f'首次模型加载：{result["model_load_ms"]:.0f} ms')
        wp=result.get("wbf_params")
        if wp:meta.append(f'WBF：iou={wp["iou_thr"]}, skip={wp["skip_box_thr"]}, conf={wp["conf_type"]}, weights={wp["weights"]}' if "iou_thr" in wp else f'融合：{wp["type"]}（配置来源：{wp.get("config_source","")}）')
        st.markdown('<div class="tech-details">'+"<br>".join(escape(x) for x in meta)+"</div>",unsafe_allow_html=True)

def batch_table_html(rows):
    """汇总表（HTML 表格，7 列固定比例，避免 st.dataframe 横向溢出把「状态」列挤出视口）。

    数据量 ≤10 行，用 HTML 表可保证全部列在一屏宽度内可见，并与既有 class-table 视觉一致。
    """
    widths = ["34%", "10%", "9%", "13%", "9%", "9%", "9%"]
    heads = ["文件名", "尺寸", "正式框数", "正式最高置信度", "危害等级", "耗时(ms)", "状态"]
    head = f'<th style="width:{widths[0]}">{escape(heads[0])}</th>' + "".join(
        f'<th class="num" style="width:{w}">{escape(h)}</th>' for h, w in zip(heads[1:], widths[1:])
    )
    body = []
    for r in rows:
        ok = r["状态"] == "完成"
        badge = f'<span class="badge {"ok" if ok else "fail"}">{escape(str(r["状态"]))}</span>'
        boxes = "" if r["正式框数"] is None else r["正式框数"]
        ms = "" if r["耗时(ms)"] is None else r["耗时(ms)"]
        body.append(
            f'<tr><td title="{escape(str(r["文件名"]))}">{escape(str(r["文件名"]))}</td>'
            f'<td class="num">{escape(str(r["尺寸"]))}</td>'
            f'<td class="num">{boxes}</td>'
            f'<td class="num">{escape(str(r["正式最高置信度"]))}</td>'
            f'<td class="num">{escape(str(r["危害等级"]))}</td>'
            f'<td class="num">{ms}</td>'
            f'<td class="num">{badge}</td></tr>'
        )
    return f'<div class="class-table batch-table"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

def batch_zip_bytes(entries, task_key, ok_entries):
    """组装批量 ZIP（内存中完成，不写临时文件）；按结果签名缓存，避免每次重绘。"""
    signature = f"{task_key}|" + "|".join(
        f'{e["index"]}:{e.get("cache_key") or e["status"]}' for e in entries
    )
    cache = st.session_state.setdefault("batch_zip_cache", {})
    if signature in cache:
        return cache[signature]
    jpegs = {}
    payloads = {}
    for entry in ok_entries:
        jpegs[entry["index"]] = _annot_cache(entry)["jpg"]
        payloads[entry["index"]] = batch_summary.build_result_payload(entry["filename"], entry["result"], task_label(task_key))
    data = batch_summary.build_batch_zip(
        entries, task_label(task_key), ok_entries[0]["result"]["mode_display"],
        annotated_jpegs=jpegs, payloads=payloads,
    )
    cache[signature] = data
    while len(cache) > 2:
        cache.pop(next(iter(cache)))
    return data

def render_batch_overview(entries, task_key, mode_key):
    """批量总览卡 + 下方完整结果（默认只渲染选中图；可展开全部）。"""
    ok_entries = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] != "ok"]
    totals = batch_summary.batch_totals(entries)
    status_chip = render_status_item("ok" if not failed else "warn", f"{totals['ok']}/{totals['total']} 张成功")
    # 选图控件的 key 绑定本批文件签名：更换上传集合时重置选择，避免旧选中项不在新 options 中
    batch_sig = hashlib.sha1("|".join(f'{e["filename"]}:{e["size_kb"]:.0f}' for e in entries).encode("utf-8")).hexdigest()[:8]
    selected = None
    expand_all = False
    with st.container(border=True):
        st.markdown(card_head("11 · 批量总览", f"批量检测总览 · {totals['total']} 张", status_chip), unsafe_allow_html=True)
        if not ok_entries:
            st.markdown(f'<div class="error-box">{lucide("x",16)} <strong>本批图片全部未完成推理</strong><br><span class="small">逐张原因见下方错误卡片；修正后可重新上传。</span></div>', unsafe_allow_html=True)
        else:
            options = [e["index"] for e in ok_entries]
            labels = {e["index"]: f'{e["filename"]} · {batch_summary.summarize_entry(e)["num_detections"]} 框' for e in ok_entries}
            pick_key = f"batch_pick_{task_key}_{mode_key}_{batch_sig}"
            # 先按上一次选择渲染摘要条（首屏可见），再渲染控件本身，避免“摘要条被控件挤到首屏之外”
            prev_pick = st.session_state.get(pick_key)
            picked = prev_pick if prev_pick in options else options[0]
            selected = next(e for e in ok_entries if e["index"] == picked)
            st.markdown(kpi_strip_html(selected, task_key), unsafe_allow_html=True)
            c_pick, c_tog = st.columns([7, 5], gap="medium")
            with c_pick:
                picked = st.selectbox("查看哪张图的完整结果", options=options, format_func=lambda i: labels.get(i, str(i)), index=options.index(picked), key=pick_key, help="切换后下方只渲染所选图片的完整结果；推理结果已缓存，不会重复推理。")
            with c_tog:
                expand_all = st.toggle("展开全部结果", value=False, key=f"batch_expand_{task_key}_{mode_key}", help="关闭时只渲染选中图片的完整结果；打开后按上传顺序渲染每张图片的完整卡片（页面会很长）。")
            selected = next(e for e in ok_entries if e["index"] == picked)
            st.markdown(batch_grid_html(entries), unsafe_allow_html=True)
            st.markdown('<div class="panel-kicker" style="margin-top:16px">汇总表 · 业务口径（正式检测框 ≥ decision_conf）</div>', unsafe_allow_html=True)
            st.markdown(batch_table_html(batch_summary.build_summary_rows(entries)), unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            zip_bytes = batch_zip_bytes(entries, task_key, ok_entries)
            st.download_button(
                f"下载全部结果 ZIP（{totals['ok']} 张成功 · 标注 JPG + JSON）",
                data=zip_bytes,
                file_name=f"batch_{task_key}_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                key=f"dl_batch_zip_{task_key}_{mode_key}",
                help="每张成功图片包含标注 JPG 与结果 JSON；失败/校验不通过的文件跳过，并在 batch_summary.json 中记录原因。",
            )
            st.markdown(f'<div class="small" style="margin-top:6px">ZIP 内含 {totals["ok"]} 组结果（每组标注 JPG + JSON）与 batch_summary.json；失败图片不写入结果，只在 summary 记录错误。合计正式检测框 {totals["boxes"]} 个。</div>', unsafe_allow_html=True)
            if failed:
                detail = "；".join(f'{escape(e["filename"])}：{escape(e["error"] or "未知错误")}' for e in failed)
                st.markdown(f'<div class="info-note"><span class="info-tag">失败 {len(failed)} 张</span>{detail}</div>', unsafe_allow_html=True)

    if selected is None:
        for entry in failed:
            render_error_block(entry, task_key)
    elif expand_all:
        for entry in entries:
            if entry["status"] == "ok":
                render_result_block(entry, task_key, mode_key)
            else:
                render_error_block(entry, task_key)
    else:
        render_result_block(selected, task_key, mode_key, show_kpi=False)
        for entry in failed:
            render_error_block(entry, task_key)

if uploaded_files:
    upload_cfg=cfg["upload"];max_files=upload_cfg["max_files_per_batch"]
    if len(uploaded_files)>max_files:
        st.warning(f"单批最多处理 {max_files} 张图片，超出部分已忽略。")
    uploaded_files=uploaded_files[:max_files]
    is_batch=len(uploaded_files)>1
    if is_batch:
        # 批量：先全部校验+推理（不渲染完整结果占位），再渲染总览
        entries=[]
        progress_box=st.empty()
        for idx,up in enumerate(uploaded_files):
            progress_box.markdown(f'<div class="file-status running" role="status" aria-live="polite"><span class="dot loading"></span><span>正在批量推理 {idx+1}/{len(uploaded_files)} · {escape(safe_basename(up.name))}</span></div>',unsafe_allow_html=True)
            fname,problems,img_bgr=prepare_upload(up,upload_cfg)
            entry=new_entry(idx,up,fname,img_bgr)
            if problems:
                entry["status"]="invalid";entry["error"]=problems[0]
            else:
                infer_entry(entry,task_key,mode_key,manager,cfg,up)
            entries.append(entry)
        progress_box.empty()
        render_batch_overview(entries,task_key,mode_key)
    else:
        up=uploaded_files[0]
        fname,problems,img_bgr=prepare_upload(up,upload_cfg)
        entry=new_entry(0,up,fname,img_bgr)
        if problems:
            entry["status"]="invalid";entry["error"]=problems[0]
            render_error_block(entry,task_key)
        else:
            group=st.empty()
            with group.container(border=True):
                c1,c2=st.columns([6,6],gap="medium")
                with c1:
                    with st.container(border=True):
                        st.markdown(card_head("03 · 输入","输入原图",render_status_item("loading","等待推理")),unsafe_allow_html=True)
                        st.markdown(f'<div class="file-name" style="margin-bottom:8px">{escape(fname)} <span class="file-meta">{up.size/1024:.0f} KB</span></div>',unsafe_allow_html=True)
                        st.markdown('<div class="result-frame"><div class="result-image"><div class="helper">等待模型输出</div></div></div>',unsafe_allow_html=True)
                with c2:
                    with st.container(border=True):
                        st.markdown(card_head("04 · 视觉分析","标注结果",render_status_item("loading","正在推理…")),unsafe_allow_html=True)
                        st.markdown('<div class="result-frame"><div class="result-image"><div class="helper">正在推理…</div></div></div>',unsafe_allow_html=True)
            status_line=st.empty()
            status_line.markdown(f'<div class="file-status running" role="status" aria-live="polite"><span class="dot loading"></span><span>正在推理…</span></div>',unsafe_allow_html=True)
            with st.spinner("正在推理…"):
                infer_entry(entry,task_key,mode_key,manager,cfg,up)
            group.empty();status_line.empty()
            if entry["status"]=="ok":
                render_result_block(entry,task_key,mode_key)
            else:
                render_error_block(entry,task_key)

st.markdown(f'<div class="site-footer"><div class="footer-brand">Anerqi</div><div class="panel-kicker">附录 · 关于本系统</div><div class="footer-title">验证集指标与模型说明</div><div class="footer-body"><strong>杂草侧</strong>：YOLO11s（960）+ YOLO11s-box8（960）+ YOLO11s-1280 + YOLOX-Dinov3 Small/Base 五模型 WBF；本地验证集 COCO 评测 mAP50=0.8232、mAP50-95=0.4525、AP75=0.4357，基于 787 张验证集图片、2596 个标注框。该结果不是独立测试集指标，不代表对任意上传图片的精度承诺。<br><strong>害虫侧</strong>：YOLO11m + YOLO11l + YOLO11s refined classwise WBF；本地验证集 mAP50=0.80524、mAP50-95=0.52410。平台测试集没有公开真值，平台实际分数以正式评测为准。<br><strong>类别</strong>：杂草 1 类：Obonianghao（播娘蒿）；害虫 32 类英文标准名（无可靠中文别名文件，不自行翻译），运行时从 dataset.yaml names 段加载并严格保持 ID 0–31 顺序。指标均为验证集口径。</div></div>',unsafe_allow_html=True)
