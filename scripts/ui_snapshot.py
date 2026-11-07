#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""WheatFieldAI 工作台 UI 快照与几何校验工具（只读，不修改被测应用）。

作用
----
对一个正在运行的 Streamlit 页面做可复现的 UI 快照：
每个宽度输出 视口截图 / 整页截图 / 「01 工作台控制」卡片裁剪 / 「拍摄区域面积」
控件行裁剪，并写出 metrics.json（元素位置尺寸字体 + 5 条几何校验结果）。

依赖
----
- Python 3.9+（建议用 `python -B` 运行，避免写入 __pycache__）
- playwright。若 Playwright 自带 chromium 未安装，可用本机 Chrome：
  p.chromium.launch(channel="chrome")
- Pillow（可选）：有则用 Pillow 从整页图裁剪；无则退回 Playwright 的 clip

用法
----
  python -B WheatFieldAI/scripts/ui_snapshot.py --widths 1440 --label before
  python -B WheatFieldAI/scripts/ui_snapshot.py --url http://127.0.0.1:8501 \
      --out <输出目录> \
      --widths "1440,1280,768" --label shot --wait 7

输出（--out 目录）
------------------
  {label}_{width}_viewport.png   视口截图（device_scale_factor=1）
  {label}_{width}_full.png       整页截图
  {label}_{width}_card.png       「01 工作台控制」卡片裁剪
  {label}_{width}_area_row.png   「拍摄区域面积」控件行裁剪
  metrics.json                   规范结果（每次运行覆盖，始终是最近一次）
  {label}_metrics.json           同内容存档，便于 before/after 对比

校验项（checks，不抛异常，只记录 pass/fail）
--------------------------------------------
  C1 单位开关底边  与 输入框容器底边 差 <= 2px
  C2 单位开关高度  与 输入框容器高度 差 <= 1px
  C3 标签左边缘    与 输入框容器左边缘 差 <= 1px
  C4 输入框 font-size >= 14px
  C5 页面内最小可见字号 >= 12px

退出码
------
  0  全部 checks 通过
  1  至少一条 check 未通过（metrics.json 已正常写出）
  2  工具自身失败（浏览器启动失败 / 页面加载超时 / 目录或文件不可写等）

作为模块复用
------------
  from ui_snapshot import run
  result = run(url="http://127.0.0.1:8501", out_dir=r"...", widths=[1440], label="before")
  result["summary"]["all_pass"]   # bool
  result["metrics_path"]          # metrics.json 绝对路径
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime

TOOL_NAME = "ui_snapshot.py"
TOOL_VERSION = "1.0.0"

DEFAULT_URL = "http://127.0.0.1:8501"
DEFAULT_OUT = os.path.join(tempfile.gettempdir(), "wf_ui", "shots")
DEFAULT_WIDTHS = "1440,1280,768"
DEFAULT_LABEL = "shot"
DEFAULT_WAIT = 7.0
DEFAULT_HEIGHT = 1000
DEFAULT_CHANNEL = "chrome"
LOAD_TITLE = "检测任务"
LOAD_TIMEOUT_MS = 30000

EXIT_PASS = 0
EXIT_CHECK_FAILED = 1
EXIT_TOOL_ERROR = 2


class FatalCaptureError(RuntimeError):
    """页面根本没有渲染成功（导航失败 / 标题超时）。

    这种失败意味着「什么都没测到」，不能等同于「UI 校验不通过」，
    因此最终退出码为 2（工具级失败），但仍会写出 metrics.json 便于排查。
    """

# 校验阈值
THRESHOLD_BOTTOM_PX = 2.0
THRESHOLD_HEIGHT_PX = 1.0
THRESHOLD_ALIGN_PX = 1.0
THRESHOLD_INPUT_FONT_PX = 14.0
THRESHOLD_MIN_FONT_PX = 12.0

# ---------------------------------------------------------------------------
# 浏览器内测量脚本
# ---------------------------------------------------------------------------
_MEASURE_JS = r"""() => {
  const doc = document;
  const sx = window.scrollX || 0;
  const sy = window.scrollY || 0;
  const r2 = (v) => Math.round(v * 100) / 100;

  const tagOf = (el) => {
    if (!el) return null;
    const tid = el.getAttribute && el.getAttribute('data-testid');
    if (tid) return '[' + tid + ']';
    const cls = (el.className || '').toString().split(/\s+/).filter(Boolean)
      .filter((c) => !/^st-emotion-cache/.test(c)).slice(0, 2);
    return (el.tagName || '').toLowerCase() + (cls.length ? '.' + cls.join('.') : '');
  };

  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const raw = (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')
      ? (el.value || el.placeholder || '') : (el.innerText || '');
    return {
      selector: tagOf(el),
      x: r2(r.left + sx),
      y: r2(r.top + sy),
      right: r2(r.right + sx),
      bottom: r2(r.bottom + sy),
      width: r2(r.width),
      height: r2(r.height),
      viewport_x: r2(r.left),
      viewport_y: r2(r.top),
      font_size: parseFloat(cs.fontSize) || null,
      font_size_raw: cs.fontSize,
      font_weight: cs.fontWeight,
      line_height: cs.lineHeight,
      color: cs.color,
      visible: r.width > 0 && r.height > 0
        && cs.visibility !== 'hidden' && cs.display !== 'none',
      text: raw.replace(/\s+/g, ' ').trim().slice(0, 60),
    };
  };

  const out = {};

  // ---- 元素定位 -----------------------------------------------------------
  const num = doc.querySelector('[data-testid="stNumberInput"]');
  const inputContainer = num ? num.querySelector('[data-testid="stNumberInputContainer"]') : null;
  const inputField = num ? num.querySelector('[data-testid="stNumberInputField"]') : null;
  const label = num ? num.querySelector('[data-testid="stWidgetLabel"]') : null;

  // 单位分段控件：约定取 [data-testid="stButtonGroup"] 内最后一个 [role="radiogroup"]
  let unitGroup = null;
  const unitInfo = { group_count: 0, group_index: null, radiogroup_count_in_group: 0, fallback: null };
  const bgs = doc.querySelectorAll('[data-testid="stButtonGroup"]');
  unitInfo.group_count = bgs.length;
  for (let i = bgs.length - 1; i >= 0; i--) {
    const rgs = bgs[i].querySelectorAll('[role="radiogroup"]');
    if (rgs.length) {
      unitGroup = rgs[rgs.length - 1];
      unitInfo.group_index = i;
      unitInfo.radiogroup_count_in_group = rgs.length;
      break;
    }
  }
  if (!unitGroup) {
    const all = doc.querySelectorAll('[role="radiogroup"]');
    if (all.length) {
      unitGroup = all[all.length - 1];
      unitInfo.fallback = 'global-last-radiogroup';
    }
  }

  // 卡片：优先 stVerticalBlockBorderWrapper（部分 Streamlit 版本）；
  // 1.59.x 的 st.container(border=True) 渲染成带 1px 边框的 stVerticalBlock，
  // 因此退回「最近的、有可见边框的祖先元素」。
  let card = null;
  let cardHow = 'not-found';
  if (num) {
    card = num.closest('[data-testid="stVerticalBlockBorderWrapper"]');
    if (card) cardHow = 'stVerticalBlockBorderWrapper';
  }
  if (!card && num) {
    let e = num.parentElement;
    while (e && e !== doc.body) {
      const cs = getComputedStyle(e);
      const bw = parseFloat(cs.borderTopWidth) || 0;
      const bc = cs.borderTopColor || '';
      if (bw > 0 && cs.borderTopStyle !== 'none'
          && bc !== 'rgba(0, 0, 0, 0)' && bc !== 'transparent') {
        const rr = e.getBoundingClientRect();
        if (rr.width > 0 && rr.height > 0) {
          card = e;
          cardHow = 'bordered-ancestor-fallback';
          break;
        }
      }
      e = e.parentElement;
    }
  }

  let hint = null;
  if (card) hint = card.querySelector('.area-hint');
  if (!hint) hint = doc.querySelector('.area-hint');

  // ---- 输出 ---------------------------------------------------------------
  out.elements = {
    label: box(label),
    inputContainer: box(inputContainer),
    inputField: box(inputField),
    unitGroup: box(unitGroup),
    areaHint: box(hint),
    card: box(card),
  };
  out.unitGroupSelection = unitInfo;
  out.cardSelection = {
    locator: cardHow,
    selector: card ? tagOf(card) : null,
    kicker: card && card.querySelector('.panel-kicker')
      ? (card.querySelector('.panel-kicker').innerText || '').trim() : null,
    title: card && card.querySelector('.panel-title')
      ? (card.querySelector('.panel-title').innerText || '').trim() : null,
  };
  out.found = {
    numberInput: !!num,
    inputContainer: !!inputContainer,
    inputField: !!inputField,
    widgetLabel: !!label,
    unitGroup: !!unitGroup,
    areaHint: !!hint,
    card: !!card,
  };
  out.numberInputCount = doc.querySelectorAll('[data-testid="stNumberInput"]').length;

  // ---- 页面尺寸 -----------------------------------------------------------
  const de = doc.documentElement;
  out.document = {
    width: de.scrollWidth,
    height: de.scrollHeight,
    client_width: de.clientWidth,
    client_height: de.clientHeight,
    device_pixel_ratio: window.devicePixelRatio,
    scroll_x: sx,
    scroll_y: sy,
  };

  // ---- 页面最小可见字号 ---------------------------------------------------
  const root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
  const SKIP = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1, TITLE: 1, HEAD: 1, META: 1 };
  const small = [];
  let minFs = null;
  let textCount = 0;
  const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  let node;
  while ((node = walker.nextNode())) {
    const t = (node.textContent || '').trim();
    if (!t) continue;
    const el = node.parentElement;
    if (!el || SKIP[el.tagName]) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (parseFloat(cs.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const fs = parseFloat(cs.fontSize);
    if (!isFinite(fs) || fs <= 0) continue;
    textCount++;
    if (minFs === null || fs < minFs) minFs = fs;
    if (fs < 12.5 && small.length < 40) {
      small.push({ font_size: fs, selector: tagOf(el), text: t.slice(0, 32) });
    }
  }
  small.sort((a, b) => a.font_size - b.font_size);
  out.fonts = {
    page_min_px: minFs,
    text_nodes_considered: textCount,
    below_12px_count: small.filter((s) => s.font_size < 12).length,
    smallest_samples: small.slice(0, 6),
    label_px: out.elements.label ? out.elements.label.font_size : null,
    input_container_px: out.elements.inputContainer ? out.elements.inputContainer.font_size : null,
    input_field_px: out.elements.inputField ? out.elements.inputField.font_size : null,
    unit_group_px: out.elements.unitGroup ? out.elements.unitGroup.font_size : null,
    area_hint_px: out.elements.areaHint ? out.elements.areaHint.font_size : null,
  };

  return out;
}"""

_TITLE_JS = """() => !!(document.body && document.body.innerText
  && document.body.innerText.indexOf('%s') !== -1)""" % LOAD_TITLE

_SPINNER_GONE_JS = """() => document.querySelectorAll('[data-testid="stSpinner"]').length === 0"""

_NUMINPUT_PRESENT_JS = """() => !!document.querySelector('[data-testid="stNumberInput"]')"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _eprint(msg):
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _setup_stdio():
    """Windows 控制台默认 GBK，统一改成 UTF-8，避免中文报错变成乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return ("%." + str(nd) + "f") % v
    return str(v)


def _diff(a, b):
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _union(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return {
        "x": min(b["x"] for b in boxes),
        "y": min(b["y"] for b in boxes),
        "right": max(b["right"] for b in boxes),
        "bottom": max(b["bottom"] for b in boxes),
        "width": max(b["right"] for b in boxes) - min(b["x"] for b in boxes),
        "height": max(b["bottom"] for b in boxes) - min(b["y"] for b in boxes),
    }


def _mk_check(cid, name, title, passed, threshold, detail, values):
    return {
        "id": cid,
        "name": name,
        "title": title,
        "status": "pass" if passed else "fail",
        "threshold": threshold,
        "detail": detail,
        "values": values,
    }


def _check_missing(cid, name, title, threshold, missing, values=None):
    return _mk_check(
        cid, name, title, False, threshold,
        "未找到元素：%s" % "、".join(missing), values or {},
    )


def _parse_widths(raw):
    widths = []
    for part in str(raw).replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            w = int(float(part))
        except ValueError:
            raise ValueError(
                "无法解析 --widths 中的宽度 %r；请用逗号分隔的整数，例如 \"1440,1280,768\"" % part
            )
        if w < 200 or w > 10000:
            raise ValueError("宽度 %d 超出合理范围（200-10000）" % w)
        if w not in widths:
            widths.append(w)
    if not widths:
        raise ValueError("--widths 未提供任何有效宽度")
    return widths


def _safe_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "shot"


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------
def _crop_with_pillow(src_png, dst_png, box, pad, scale):
    from PIL import Image

    with Image.open(src_png) as im:
        w, h = im.size
        left = int(round(box["x"] * scale)) - int(round(pad * scale))
        top = int(round(box["y"] * scale)) - int(round(pad * scale))
        right = int(round(box["right"] * scale)) + int(round(pad * scale))
        bottom = int(round(box["bottom"] * scale)) + int(round(pad * scale))
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        if right - left < 2 or bottom - top < 2:
            raise ValueError("裁剪区域退化：(%d,%d)-(%d,%d)" % (left, top, right, bottom))
        im.crop((left, top, right, bottom)).save(dst_png, "PNG")
        return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def _crop_with_playwright(page, dst_png, box, pad, scale):
    clip = {
        "x": max(0.0, box["x"] - pad),
        "y": max(0.0, box["y"] - pad),
        "width": box["width"] + pad * 2,
        "height": box["height"] + pad * 2,
    }
    page.screenshot(path=dst_png, clip=clip, full_page=True)
    return {"left": int(clip["x"] * scale), "top": int(clip["y"] * scale),
            "width": int(clip["width"] * scale), "height": int(clip["height"] * scale)}


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def build_checks(measure, blocked_reason=None):
    """根据一次测量结果生成 5 条 checks。任何异常都转成 fail，不向外抛。

    blocked_reason 非空表示页面没渲染成功、根本没测到东西：5 条一律记为 fail
    并标注 blocked=True，供调用方与「测到了但没通过」区分。
    """
    checks = []
    els = measure.get("elements") or {}
    fonts = measure.get("fonts") or {}
    found = measure.get("found") or {}

    label = els.get("label")
    cont = els.get("inputContainer")
    field = els.get("inputField")
    unit = els.get("unitGroup")

    # C1 单位开关底边 vs 输入框容器底边
    try:
        if not found.get("unitGroup") or not found.get("inputContainer"):
            miss = []
            if not found.get("unitGroup"):
                miss.append("单位分段控件 [role=radiogroup]")
            if not found.get("inputContainer"):
                miss.append("输入框容器 [stNumberInputContainer]")
            checks.append(_check_missing(
                "C1", "unit_bottom_matches_input_container", "单位开关底边 vs 输入框容器底边",
                "<=%gpx" % THRESHOLD_BOTTOM_PX, miss))
        else:
            d = _diff(unit["bottom"], cont["bottom"])
            ok = d <= THRESHOLD_BOTTOM_PX
            checks.append(_mk_check(
                "C1", "unit_bottom_matches_input_container", "单位开关底边 vs 输入框容器底边",
                ok, "<=%gpx" % THRESHOLD_BOTTOM_PX,
                "|%.2f - %.2f| = %.2fpx %s %gpx" % (
                    unit["bottom"], cont["bottom"], d,
                    "<=" if ok else ">", THRESHOLD_BOTTOM_PX),
                {"unitGroup.bottom": unit["bottom"], "inputContainer.bottom": cont["bottom"],
                 "delta": round(d, 2)}))
    except Exception as exc:  # noqa: BLE001
        checks.append(_mk_check("C1", "unit_bottom_matches_input_container",
                                "单位开关底边 vs 输入框容器底边", False,
                                "<=%gpx" % THRESHOLD_BOTTOM_PX,
                                "测量异常：%s" % exc, {}))

    # C2 单位开关高度 vs 输入框容器高度
    try:
        if not found.get("unitGroup") or not found.get("inputContainer"):
            checks.append(_check_missing(
                "C2", "unit_height_matches_input_container", "单位开关高度 vs 输入框容器高度",
                "<=%gpx" % THRESHOLD_HEIGHT_PX, ["单位分段控件或输入框容器"]))
        else:
            d = _diff(unit["height"], cont["height"])
            ok = d <= THRESHOLD_HEIGHT_PX
            checks.append(_mk_check(
                "C2", "unit_height_matches_input_container", "单位开关高度 vs 输入框容器高度",
                ok, "<=%gpx" % THRESHOLD_HEIGHT_PX,
                "|%.2f - %.2f| = %.2fpx %s %gpx" % (
                    unit["height"], cont["height"], d,
                    "<=" if ok else ">", THRESHOLD_HEIGHT_PX),
                {"unitGroup.height": unit["height"], "inputContainer.height": cont["height"],
                 "delta": round(d, 2)}))
    except Exception as exc:  # noqa: BLE001
        checks.append(_mk_check("C2", "unit_height_matches_input_container",
                                "单位开关高度 vs 输入框容器高度", False,
                                "<=%gpx" % THRESHOLD_HEIGHT_PX,
                                "测量异常：%s" % exc, {}))

    # C3 标签左边缘 vs 输入框容器左边缘
    try:
        if not found.get("widgetLabel") or not found.get("inputContainer"):
            checks.append(_check_missing(
                "C3", "label_left_aligns_input_container", "标签左边缘 vs 输入框容器左边缘",
                "<=%gpx" % THRESHOLD_ALIGN_PX, ["标签 [stWidgetLabel] 或输入框容器"]))
        else:
            d = _diff(label["x"], cont["x"])
            ok = d <= THRESHOLD_ALIGN_PX
            checks.append(_mk_check(
                "C3", "label_left_aligns_input_container", "标签左边缘 vs 输入框容器左边缘",
                ok, "<=%gpx" % THRESHOLD_ALIGN_PX,
                "|%.2f - %.2f| = %.2fpx %s %gpx" % (
                    label["x"], cont["x"], d,
                    "<=" if ok else ">", THRESHOLD_ALIGN_PX),
                {"label.x": label["x"], "inputContainer.x": cont["x"], "delta": round(d, 2)}))
    except Exception as exc:  # noqa: BLE001
        checks.append(_mk_check("C3", "label_left_aligns_input_container",
                                "标签左边缘 vs 输入框容器左边缘", False,
                                "<=%gpx" % THRESHOLD_ALIGN_PX,
                                "测量异常：%s" % exc, {}))

    # C4 输入框 font-size >= 14px
    try:
        src = field or cont
        if src is None or src.get("font_size") is None:
            checks.append(_check_missing(
                "C4", "input_font_size_ge_14", "输入框字号",
                ">=%gpx" % THRESHOLD_INPUT_FONT_PX,
                ["输入框 [stNumberInputField] 与容器"]))
        else:
            fs = float(src["font_size"])
            ok = fs >= THRESHOLD_INPUT_FONT_PX
            checks.append(_mk_check(
                "C4", "input_font_size_ge_14", "输入框字号",
                ok, ">=%gpx" % THRESHOLD_INPUT_FONT_PX,
                "font-size = %.2fpx（%s）%s %gpx" % (
                    fs, "input" if field else "container",
                    ">=" if ok else "<", THRESHOLD_INPUT_FONT_PX),
                {"font_size": fs, "source": "input" if field else "container",
                 "container_font_size": cont.get("font_size") if cont else None}))
    except Exception as exc:  # noqa: BLE001
        checks.append(_mk_check("C4", "input_font_size_ge_14", "输入框字号", False,
                                ">=%gpx" % THRESHOLD_INPUT_FONT_PX,
                                "测量异常：%s" % exc, {}))

    # C5 页面最小可见字号 >= 12px
    try:
        min_fs = fonts.get("page_min_px")
        if min_fs is None:
            checks.append(_check_missing(
                "C5", "page_min_font_ge_12", "页面最小可见字号",
                ">=%gpx" % THRESHOLD_MIN_FONT_PX, ["任何可见文本"]))
        else:
            fs = float(min_fs)
            ok = fs >= THRESHOLD_MIN_FONT_PX
            worst = fonts.get("smallest_samples") or []
            hint = ""
            if not ok and worst:
                hint = "；最小处：" + " / ".join(
                    "%gpx %s" % (s.get("font_size"), s.get("selector") or "?")
                    for s in worst[:3])
            checks.append(_mk_check(
                "C5", "page_min_font_ge_12", "页面最小可见字号",
                ok, ">=%gpx" % THRESHOLD_MIN_FONT_PX,
                "最小可见字号 = %.2fpx %s %gpx（<%gpx 的文本 %d 处）%s" % (
                    fs, ">=" if ok else "<", THRESHOLD_MIN_FONT_PX,
                    THRESHOLD_MIN_FONT_PX, fonts.get("below_12px_count") or 0, hint),
                {"page_min_px": fs,
                 "below_12px_count": fonts.get("below_12px_count"),
                 "samples": worst[:6]}))
    except Exception as exc:  # noqa: BLE001
        checks.append(_mk_check("C5", "page_min_font_ge_12", "页面最小可见字号", False,
                                ">=%gpx" % THRESHOLD_MIN_FONT_PX,
                                "测量异常：%s" % exc, {}))

    if blocked_reason:
        for chk in checks:
            chk["status"] = "fail"
            chk["blocked"] = True
            chk["detail"] = "页面未就绪，未测量：%s" % blocked_reason
            chk["values"] = {}
    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _launch_browser(p, channel, chrome_path):
    kwargs = {"headless": True}
    if chrome_path:
        kwargs["executable_path"] = chrome_path
    else:
        kwargs["channel"] = channel
    try:
        return p.chromium.launch(**kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        raise RuntimeError(
            "无法启动浏览器。\n"
            "  尝试参数：%s\n"
            "  原始错误：%s\n"
            "  处理建议：确认已安装 Google Chrome（本机默认 "
            r"C:\Program Files\Google\Chrome\Application\chrome.exe" "），\n"
            "           或用 --chrome-path 指向 chrome.exe；Playwright 内置 chromium 未安装时"
            "必须使用 channel=\"chrome\"。" % (kwargs, msg.splitlines()[0] if msg else "")
        ) from exc


def run(url=DEFAULT_URL, out_dir=DEFAULT_OUT, widths=None, label=DEFAULT_LABEL,
        wait_seconds=DEFAULT_WAIT, height=DEFAULT_HEIGHT, channel=DEFAULT_CHANNEL,
        chrome_path=None, load_timeout_ms=LOAD_TIMEOUT_MS, log=None):
    """执行一次快照。返回结果字典；工具级失败抛 RuntimeError（附可读信息）。"""
    log = log or (lambda *a: None)
    if widths is None:
        widths = _parse_widths(DEFAULT_WIDTHS)
    if isinstance(widths, str):
        widths = _parse_widths(widths)
    widths = [int(w) for w in widths]
    if not widths:
        raise ValueError("widths 不能为空")

    label = _safe_name(label)
    out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise RuntimeError("无法创建输出目录 %s：%s" % (out_dir, exc)) from exc

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "未安装 playwright（%s）。请先安装：python -m pip install playwright" % exc
        ) from exc

    try:
        import PIL  # noqa: F401
        crop_engine = "pillow"
    except ImportError:
        crop_engine = "playwright-clip"

    result = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "url": url,
        "label": label,
        "out_dir": out_dir,
        "wait_seconds": float(wait_seconds),
        "viewport_height": int(height),
        "device_scale_factor": 1,
        "crop_engine": crop_engine,
        "load_criteria": {"title_text": LOAD_TITLE, "timeout_ms": int(load_timeout_ms)},
        "widths": {},
        "checks": [],
        "errors": [],
        "fatal_errors": [],
    }

    with sync_playwright() as p:
        browser = _launch_browser(p, channel, chrome_path)
        try:
            try:
                result["browser_version"] = browser.version
            except Exception:
                result["browser_version"] = None

            for width in widths:
                entry = _capture_one(
                    browser=browser, url=url, width=width, height=int(height),
                    out_dir=out_dir, label=label, wait_seconds=float(wait_seconds),
                    load_timeout_ms=int(load_timeout_ms), log=log,
                )
                result["widths"][str(width)] = entry
                for chk in entry["checks"]:
                    flat = dict(chk)
                    flat["width"] = width
                    result["checks"].append(flat)
                if entry.get("error"):
                    result["errors"].append({"width": width, "error": entry["error"]})
                    if entry.get("fatal"):
                        result["fatal_errors"].append(
                            {"width": width, "error": entry["error"]})
        finally:
            try:
                browser.close()
            except Exception:
                pass

    total = len(result["checks"])
    passed = sum(1 for c in result["checks"] if c["status"] == "pass")
    failed = total - passed
    all_pass = bool(total) and failed == 0 and not result["errors"]
    result["summary"] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "all_pass": all_pass,
        "fatal": bool(result["fatal_errors"]),
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    archive_path = os.path.join(out_dir, "%s_metrics.json" % label)
    try:
        _write_json(metrics_path, result)
        _write_json(archive_path, result)
    except OSError as exc:
        raise RuntimeError("无法写出 metrics.json（%s）：%s" % (out_dir, exc)) from exc
    result["metrics_path"] = metrics_path
    result["metrics_archive_path"] = archive_path
    return result


def _capture_one(browser, url, width, height, out_dir, label, wait_seconds,
                 load_timeout_ms, log):
    entry = {
        "width": width,
        "viewport": {"width": width, "height": height},
        "files": {},
        "checks": [],
        "error": None,
        "fatal": False,
        "notes": [],
    }
    ctx = None
    page = None
    try:
        ctx = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        t0 = time.time()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=load_timeout_ms + 30000)
        except Exception as exc:  # noqa: BLE001
            raise FatalCaptureError(
                "无法打开 %s（服务是否在运行？）：%s" % (url, str(exc).splitlines()[0])
            ) from exc

        # 加载判定：等待出现「检测任务」标题，最多 load_timeout_ms
        try:
            page.wait_for_function(_TITLE_JS, timeout=load_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            raise FatalCaptureError(
                "等待「%s」标题超时（%dms）。页面可能未渲染完成或不是目标应用。URL=%s"
                % (LOAD_TITLE, load_timeout_ms, url)
            ) from exc
        entry["load"] = {"title_found": True, "title_elapsed_ms": int((time.time() - t0) * 1000)}

        # 关键控件出现（尽力而为，不致命）
        try:
            page.wait_for_function(_NUMINPUT_PRESENT_JS, timeout=15000)
        except Exception:
            entry["notes"].append("15s 内未出现 [data-testid=\"stNumberInput\"]，几何校验将判 fail")

        # 固定等待，让 Streamlit 完成二次渲染 / 字体加载
        if wait_seconds > 0:
            page.wait_for_timeout(int(wait_seconds * 1000))

        # 模型加载 spinner 消失（尽力而为）
        try:
            page.wait_for_function(_SPINNER_GONE_JS, timeout=15000)
            entry["load"]["spinner_gone"] = True
        except Exception:
            entry["load"]["spinner_gone"] = False
            entry["notes"].append("spinner 在 15s 内未消失（仅记录，不影响校验）")

        # 回到页首，保证视口截图与文档坐标一致
        try:
            page.evaluate("() => { window.scrollTo(0, 0); }")
            page.wait_for_timeout(250)
        except Exception:
            pass

        measure = page.evaluate(_MEASURE_JS)
        entry["document"] = measure.get("document")
        entry["elements"] = measure.get("elements")
        entry["fonts"] = measure.get("fonts")
        entry["found"] = measure.get("found")
        entry["unit_group_selection"] = measure.get("unitGroupSelection")
        entry["card_selection"] = measure.get("cardSelection")
        entry["number_input_count"] = measure.get("numberInputCount")

        # 截图
        vp_path = os.path.join(out_dir, "%s_%d_viewport.png" % (label, width))
        full_path = os.path.join(out_dir, "%s_%d_full.png" % (label, width))
        page.screenshot(path=vp_path, full_page=False)
        page.screenshot(path=full_path, full_page=True)
        entry["files"]["viewport"] = vp_path
        entry["files"]["full"] = full_path

        # 裁剪
        doc_w = (measure.get("document") or {}).get("width") or width
        scale = 1.0
        try:
            from PIL import Image
            with Image.open(full_path) as im:
                if doc_w:
                    scale = im.size[0] / float(doc_w)
        except Exception:
            scale = 1.0
        entry["crop_scale"] = round(scale, 4)

        els = measure.get("elements") or {}
        crop_specs = [
            ("card", els.get("card"), 10, "card"),
            ("area_row", _union([els.get("label"), els.get("inputContainer"),
                                 els.get("unitGroup")]), 12, "area_row"),
        ]
        entry["crops"] = {}
        for name, box, pad, suffix in crop_specs:
            dst = os.path.join(out_dir, "%s_%d_%s.png" % (label, width, suffix))
            if not box:
                entry["notes"].append("跳过裁剪 %s：未定位到区域" % name)
                continue
            try:
                try:
                    info = _crop_with_pillow(full_path, dst, box, pad, scale)
                    engine = "pillow"
                except ImportError:
                    info = _crop_with_playwright(page, dst, box, pad, scale)
                    engine = "playwright-clip"
                info["engine"] = engine
                info["box"] = box
                entry["crops"][name] = {"path": dst, **info}
                entry["files"][name] = dst
            except Exception as exc:  # noqa: BLE001
                entry["notes"].append("裁剪 %s 失败：%s" % (name, exc))

        entry["checks"] = build_checks(measure)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)
        entry["fatal"] = isinstance(exc, FatalCaptureError)
        entry["checks"] = build_checks({}, blocked_reason=str(exc))  # 全部 fail，不抛
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                pass
    return entry


# ---------------------------------------------------------------------------
# 人类可读摘要
# ---------------------------------------------------------------------------
def format_summary(result):
    lines = []
    add = lines.append
    add("=" * 78)
    add("UI snapshot  ·  %s  v%s" % (result["tool"], result["version"]))
    add("=" * 78)
    add("url      : %s" % result["url"])
    add("out      : %s" % result["out_dir"])
    add("label    : %s" % result["label"])
    add("widths   : %s" % ", ".join(str(k) for k in result["widths"]))
    add("wait     : %gs   viewport height: %dpx   crop: %s"
        % (result["wait_seconds"], result["viewport_height"], result["crop_engine"]))
    add("browser  : %s" % _fmt(result.get("browser_version")))
    add("")

    for wkey, e in result["widths"].items():
        doc = e.get("document") or {}
        add("-" * 78)
        add("[%s] document %sx%s  scroll=(%s,%s)"
            % (wkey, _fmt(doc.get("width"), 0), _fmt(doc.get("height"), 0),
               _fmt(doc.get("scroll_x"), 0), _fmt(doc.get("scroll_y"), 0)))
        if e.get("error"):
            add("  !! 采集失败：%s" % e["error"])
        files = e.get("files") or {}
        if files.get("viewport"):
            add("  截图  viewport=%s" % os.path.basename(files["viewport"]))
        if files.get("full"):
            add("        full    =%s" % os.path.basename(files["full"]))
        if files.get("card"):
            add("        card    =%s" % os.path.basename(files["card"]))
        if files.get("area_row"):
            add("        area_row=%s" % os.path.basename(files["area_row"]))

        card = (e.get("elements") or {}).get("card")
        sel = e.get("card_selection") or {}
        if card:
            add("  卡片  %s  x=%s y=%s w=%s h=%s   «%s / %s»"
                % (sel.get("locator"), _fmt(card["x"]), _fmt(card["y"]),
                   _fmt(card["width"]), _fmt(card["height"]),
                   _fmt(sel.get("kicker"), 0), _fmt(sel.get("title"), 0)))

        els = e.get("elements") or {}
        rows = [
            ("label", "标签        "),
            ("inputContainer", "输入框容器  "),
            ("unitGroup", "单位开关    "),
            ("areaHint", "说明文字    "),
        ]
        for key, caption in rows:
            b = els.get(key)
            if not b:
                add("  %s  (未找到)" % caption)
                continue
            add("  %s  x=%s y=%s w=%s h=%s  right=%s bottom=%s  fs=%s"
                % (caption, _fmt(b["x"]), _fmt(b["y"]), _fmt(b["width"]), _fmt(b["height"]),
                   _fmt(b["right"]), _fmt(b["bottom"]), _fmt(b.get("font_size"), 0)))
        fld = els.get("inputField")
        if fld:
            add("  输入框 input  fs=%s  h=%s" % (_fmt(fld.get("font_size"), 0), _fmt(fld["height"])))
        fonts = e.get("fonts") or {}
        add("  页面最小可见字号 = %s px（<%gpx 的文本 %s 处）"
            % (_fmt(fonts.get("page_min_px")), THRESHOLD_MIN_FONT_PX,
               _fmt(fonts.get("below_12px_count"), 0)))
        for s in (fonts.get("smallest_samples") or [])[:3]:
            add("      最小样本：%gpx  %s  «%s»" % (s["font_size"], s.get("selector"), s.get("text")))
        add("")
        add("  checks:")
        for c in e.get("checks") or []:
            add("   [%s] %s %-38s %s" % (c["status"].upper(), c["id"], c["name"], c["detail"]))
        for note in e.get("notes") or []:
            add("  note: %s" % note)
        add("")

    add("=" * 78)
    for c in result["checks"]:
        add("  %-4s w=%-5s %-4s %s" % (c["id"], c.get("width"), c["status"].upper(), c["name"]))
    add("=" * 78)
    s = result["summary"]
    add("RESULT: %d/%d checks passed  ·  failed %d  ·  all_pass=%s"
        % (s["passed"], s["total"], s["failed"], "yes" if s["all_pass"] else "no"))
    if result.get("errors"):
        add("采集错误：")
        for err in result["errors"]:
            add("  - width %s: %s" % (err.get("width"), err.get("error")))
    add("metrics : %s" % result.get("metrics_path"))
    add("archive : %s" % result.get("metrics_archive_path"))
    if result.get("fatal_errors"):
        add("exit    : %d  （页面未渲染成功，属工具级失败，不是 UI 校验结论）"
            % EXIT_TOOL_ERROR)
    else:
        add("exit    : %d" % (EXIT_PASS if s["all_pass"] else EXIT_CHECK_FAILED))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        prog="ui_snapshot.py",
        description="对运行中的 WheatFieldAI Streamlit 工作台做 UI 快照与几何/字号校验。",
        epilog=("退出码：0=全部 checks 通过；1=有 checks 未通过（metrics.json 已写出）；"
                "2=工具自身失败。"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="页面地址，默认 %(default)s")
    ap.add_argument("--out", default=DEFAULT_OUT, help="截图与 metrics.json 输出目录，默认 %(default)s")
    ap.add_argument("--widths", default=DEFAULT_WIDTHS,
                    help="逗号分隔的视口宽度，默认 \"%(default)s\"")
    ap.add_argument("--label", default=DEFAULT_LABEL, help="文件名前缀，默认 %(default)s")
    ap.add_argument("--wait", type=float, default=DEFAULT_WAIT,
                    help="加载判定通过后的固定等待秒数，默认 %(default)s")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                    help="视口高度，默认 %(default)s")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL,
                    help="Playwright 浏览器 channel，默认 %(default)s（必须用本机 Chrome）")
    ap.add_argument("--chrome-path", default=None,
                    help="直接指定 chrome.exe 绝对路径（优先于 --channel）")
    ap.add_argument("--load-timeout", type=int, default=LOAD_TIMEOUT_MS // 1000,
                    help="等待「%s」标题的最长秒数，默认 %%(default)s" % LOAD_TITLE)
    ap.add_argument("--json", action="store_true", help="额外把结果 JSON 打印到 stdout")
    ap.add_argument("--quiet", action="store_true", help="只打印最后的结果行")
    ap.add_argument("--debug", action="store_true", help="工具失败时打印完整 traceback")
    return ap


def main(argv=None):
    _setup_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        widths = _parse_widths(args.widths)
    except ValueError as exc:
        _eprint("参数错误：%s" % exc)
        return EXIT_TOOL_ERROR
    if args.wait < 0:
        _eprint("参数错误：--wait 不能为负")
        return EXIT_TOOL_ERROR
    if args.load_timeout <= 0:
        _eprint("参数错误：--load-timeout 必须为正整数秒")
        return EXIT_TOOL_ERROR

    try:
        result = run(
            url=args.url,
            out_dir=args.out,
            widths=widths,
            label=args.label,
            wait_seconds=args.wait,
            height=args.height,
            channel=args.channel,
            chrome_path=args.chrome_path,
            load_timeout_ms=args.load_timeout * 1000,
        )
    except Exception as exc:  # noqa: BLE001
        _eprint("")
        _eprint("TOOL ERROR: %s" % exc)
        if args.debug:
            _eprint(traceback.format_exc())
        else:
            _eprint("（加 --debug 查看完整 traceback）")
        return EXIT_TOOL_ERROR

    if args.quiet:
        s = result["summary"]
        sys.stdout.write("checks %d/%d passed  all_pass=%s  metrics=%s\n"
                         % (s["passed"], s["total"],
                            "yes" if s["all_pass"] else "no", result["metrics_path"]))
    else:
        sys.stdout.write(format_summary(result) + "\n")
    if args.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    try:
        sys.stdout.flush()
    except Exception:
        pass

    if result.get("fatal_errors"):
        for fe in result["fatal_errors"]:
            _eprint("TOOL ERROR (width %s): %s" % (fe.get("width"), fe.get("error")))
        _eprint("页面未渲染成功，未取得有效 UI 度量；退出码 %d。" % EXIT_TOOL_ERROR)
        return EXIT_TOOL_ERROR

    return EXIT_PASS if result["summary"]["all_pass"] else EXIT_CHECK_FAILED


if __name__ == "__main__":
    sys.exit(main())
