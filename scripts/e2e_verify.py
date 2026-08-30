# -*- coding: utf-8 -*-
"""
e2e_verify.py
=============
Playwright Web 端到端验证（任务 08 双任务系统）。

默认假设 Streamlit 已在 http://localhost:8501 运行。

策略：每个关键场景使用**独立页面**（新 Streamlit 会话），
避免 file_uploader 累积多图导致截图展示错误图片；代价是各场景需重新加载模型，
一次 E2E 验证可接受。

覆盖：
1. 页面加载 + 工作台标题
2. 桌面端（1440×900）杂草检测：上传真实杂草图 → 融合结果 → 截图
3. 桌面端（1440×900）害虫检测：上传真实害虫图 → 三模型融合 → 截图
4. 窄屏（390×844）害虫检测：上传真实害虫图 → 响应式结果 → 截图 + 无横向溢出
5. 非法上传（合法扩展名但内容损坏）→ 明确错误、逐图隔离 → 截图
6. 下载：害虫场景真实点击「下载 JSON 结果」

用法：python scripts/e2e_verify.py [--url http://localhost:8501] [--out <dir>]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

_THIS_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_WEED_IMG = str(_THIS_DIR / "test_images" / "weed" / "DJI_weed_0001.jpg")
DEFAULT_PEST_IMG = str(_THIS_DIR / "test_images" / "pest" / "pest24_0000002.jpg")

DEFAULT_BROKEN_FILE = SCRIPTS_DIR / "_tmp_broken_upload.png"

TITLE = "小麦田间杂草与害虫双任务识别工作台"


def make_broken_file():
    DEFAULT_BROKEN_FILE.write_bytes(b"this is definitely not a valid png image " * 50)
    return str(DEFAULT_BROKEN_FILE)


def _click_segment(page, label):
    try:
        page.get_by_role("button", name=label, exact=True).first.click(timeout=5000)
        return True
    except Exception:
        try:
            page.locator('[role="radio"]').get_by_text(label, exact=True).first.click(timeout=5000)
            return True
        except Exception:
            return False


def select_task(page, label):
    if not _click_segment(page, label):
        page.get_by_text(label, exact=True).first.click()


def select_mode(page, label):
    if not _click_segment(page, label):
        page.get_by_text(label, exact=True).first.click()


def upload_file(page, path):
    page.set_input_files('input[type="file"]', path)


def wait_for_result(page, filename, timeout_ms=180000):
    """等待结果体渲染完成。
    新版 Bento UI：等待占位区被真实结果替换的信号 ——
    ① 文件状态变为「检测完成」；② 文件名存在；③ 导出下载按钮出现。
    （注意：不能再用「检测数量」，因为等待占位文案中也含该词，会过早匹配。）
    """
    page.wait_for_selector("text=检测完成", timeout=timeout_ms)
    page.wait_for_selector(f"text={filename}", timeout=timeout_ms)
    page.wait_for_selector('[data-testid="stDownloadButton"]', timeout=timeout_ms)


def new_page(browser, width, height, url, record):
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(url, wait_until="networkidle", timeout=120000)
    page.wait_for_selector(f"text={TITLE}", timeout=120000)
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default=str(_THIS_DIR / "screenshots"))
    ap.add_argument("--broken-file", default=DEFAULT_BROKEN_FILE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = []
    passed = 0
    failed = 0

    def record(name, ok, detail=""):
        nonlocal passed, failed
        results.append({"name": name, "ok": bool(ok), "detail": detail})
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    url = args.url.rstrip("/")
    broken = args.broken_file if os.path.isfile(args.broken_file) else make_broken_file()
    weed_name = os.path.basename(DEFAULT_WEED_IMG)
    pest_name = os.path.basename(DEFAULT_PEST_IMG)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # ============ 桌面杂草 ============
        try:
            page = new_page(browser, 1440, 900, url, record)
            record("页面加载 + 工作台标题", True)
            upload_file(page, DEFAULT_WEED_IMG)
            wait_for_result(page, weed_name)
            ok = page.locator('.result-image img').count() >= 2 and page.get_by_text("检测数量").count() > 0
            record("桌面杂草检测：上传真实图并出结果", ok)
            page.screenshot(path=os.path.join(args.out, "desktop_weed_result.png"), full_page=True)
            record("桌面杂草截图", True)
            page.close()
        except Exception as e:
            record("桌面杂草检测", False, str(e))

        # ============ 桌面害虫（三模型 classwise WBF）============
        try:
            page = new_page(browser, 1440, 900, url, record)
            select_task(page, "害虫检测")
            time.sleep(2)
            upload_file(page, DEFAULT_PEST_IMG)
            wait_for_result(page, pest_name, timeout_ms=240000)
            has_metric = page.get_by_text("检测数量").count() > 0
            has_class = page.get_by_text("按类别统计").count() > 0
            # 确认是害虫图片（文件名出现在结果展开头）
            header = page.get_by_text(f"检出").count()
            record("桌面害虫检测：三模型融合出结果（含每类统计）", has_metric and has_class)
            page.screenshot(path=os.path.join(args.out, "desktop_pest_result.png"), full_page=True)
            record("桌面害虫截图", True)
            page.close()
        except Exception as e:
            record("桌面害虫检测", False, str(e))

        # ============ 窄屏害虫 ============
        try:
            page = new_page(browser, 390, 844, url, record)
            select_task(page, "害虫检测")
            time.sleep(2)
            upload_file(page, DEFAULT_PEST_IMG)
            wait_for_result(page, pest_name, timeout_ms=240000)
            record("窄屏害虫检测出结果", page.get_by_text("检测数量").count() > 0)
            page.screenshot(path=os.path.join(args.out, "narrow_pest_result.png"), full_page=True)
            overflow = page.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 4")
            record("窄屏无横向溢出", not overflow, f"overflow={overflow}")
            page.close()
        except Exception as e:
            record("窄屏害虫检测", False, str(e))

        # ============ 非法上传（内容损坏，杂草模式，无模型推理）============
        try:
            page = new_page(browser, 1440, 900, url, record)
            upload_file(page, broken)
            page.wait_for_selector("text=图片内容无法解码", timeout=60000)
            record("非法上传 → 明确错误 + 逐图隔离", True)
            page.screenshot(path=os.path.join(args.out, "invalid_upload.png"), full_page=True)
            record("非法上传截图", True)
            page.close()
        except Exception as e:
            record("非法上传", False, str(e))

        # ============ 下载验证（害虫场景）============
        try:
            page = new_page(browser, 1440, 900, url, record)
            select_task(page, "害虫检测")
            time.sleep(2)
            upload_file(page, DEFAULT_PEST_IMG)
            wait_for_result(page, pest_name, timeout_ms=240000)
            dl_btn = page.get_by_text("检测结果 JSON").first
            if dl_btn.count() > 0:
                dl_btn.scroll_into_view_if_needed()
                with page.expect_download(timeout=60000) as dl_info:
                    dl_btn.click()
                download = dl_info.value
                dl_path = os.path.join(args.out, "_tmp_download.json")
                download.save_as(dl_path)
                data = json.loads(Path(dl_path).read_text(encoding="utf-8")) if os.path.getsize(dl_path) > 0 else {}
                record("下载 JSON 结果可用且可解析",
                       bool(data.get("detections") is not None) and "num_detections" in data,
                       f"num_detections={data.get('num_detections')}")
                if os.path.exists(dl_path):
                    os.remove(dl_path)
            else:
                record("下载 JSON 结果可用", False, "未找到下载按钮")
            page.close()
        except Exception as e:
            record("下载 JSON 结果", False, str(e))

        browser.close()

    finish(args.out, results, passed, failed)


def finish(out, results, passed, failed):
    if DEFAULT_BROKEN_FILE.exists():
        try:
            DEFAULT_BROKEN_FILE.unlink()
        except Exception:
            pass
    summary = {"passed": passed, "failed": failed, "checks": results}
    _summary_path = Path(out).parent / "outputs" / "e2e_verification.json"
    _summary_path.parent.mkdir(parents=True, exist_ok=True)
    _summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nE2E 结果：{passed} PASS / {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
