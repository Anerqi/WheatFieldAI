# -*- coding: utf-8 -*-
"""
verify_pipeline.py
==================
CLI 端到端验证（任务 08 双任务系统）。

覆盖（按 acceptance criteria）：
1. 配置加载 + 害虫 32 类映射与 dataset.yaml 完全一致（ID 0..31 连续）
2. refined classwise WBF 配置校验（32 类逐类参数齐全）
3. 杂草高精度融合真实推理（真实杂草图片）
4. 害虫快速单模型（YOLO11m）真实推理（真实害虫图片）
5. 害虫三模型 classwise WBF 真实推理
6. 危害等级 / 防治建议 / JSON 结构字段
7. 边界：空文件、非法内容、纯色空白图（0 框 → 轻）、权重缺失报错
8. 记录模型首载耗时与缓存后纯推理耗时

用法：python scripts/verify_pipeline.py [--weed-img <path>] [--pest-img <path>] [--out <dir>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_THIS_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_THIS_DIR / "src"))

import numpy as np

from config import load_config
from models import DualModelManager, is_oom_error
from weed_inference import infer_weed_single_image, decode_upload
from pest_inference import infer_pest_single_image
from pest_models import load_pest_class_names
from pest_wbf import load_class_configs, NUM_CLASSES
from drawing import draw_annotations, encode_image_to_bytes


DEFAULT_WEED_IMG = "test_images/weed/DJI_weed_0001.jpg"
DEFAULT_PEST_IMG = "test_images/pest/pest24_0000002.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weed-img", default=DEFAULT_WEED_IMG)
    ap.add_argument("--pest-img", default=DEFAULT_PEST_IMG)
    ap.add_argument("--out", default=str(_THIS_DIR / "outputs"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {"checks": [], "passed": 0, "failed": 0}
    log = []

    def check(name, ok, detail=""):
        results["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
        log.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    # ------------------------------------------------------------------
    print("== 1. 配置与类别映射 ==")
    cfg = load_config()
    check("配置加载", isinstance(cfg, dict) and "weed" in cfg and "pest" in cfg)

    # 害虫 32 类映射（唯一数据源 dataset.yaml）
    dataset_yaml = cfg["pest"]["dataset_yaml"]
    names = load_pest_class_names(dataset_yaml)
    check("害虫类别数量 = 32", len(names) == 32, f"实际 {len(names)}")
    check("类别 ID 0..31 连续（不重排）",
          all(isinstance(n, str) and n for n in names))
    expected_first_last = ("Ostrinia furnacalis", "Melahotus")
    check("首尾类别名与 dataset.yaml 一致",
          names[0] == expected_first_last[0] and names[-1] == expected_first_last[1],
          f"{names[0]} ... {names[-1]}")

    # classwise WBF 配置校验
    cw_path = cfg["pest"]["classwise_config"]
    cw = load_class_configs(cw_path)
    check("classwise WBF 配置存在", cw is not None)
    if cw:
        keys = sorted(cw.keys())
        check("classwise 配置覆盖 32 类", keys == list(range(32)), f"缺失: {set(range(32)) - set(keys)}")
        ok_shape = all(
            len(v["model_weights"]) == 3 and 0 < v["wbf_iou"] <= 1 and v["score_mode"] in
            ("agreement", "sqrt_agreement", "mean", "max")
            for v in cw.values()
        )
        check("逐类参数完整（3 权重 / iou / score_mode）", ok_shape)

    # 权重路径存在性
    missing_w, ok_w = DualModelManager(cfg).check_weights("weed", "fusion")
    check("杂草三模型权重存在", ok_w, str(missing_w))
    missing_p, ok_p = DualModelManager(cfg).check_weights("pest", "fusion")
    check("害虫三模型权重存在", ok_p, str(missing_p))

    # ------------------------------------------------------------------
    manager = DualModelManager(cfg)

    # ============ 杂草高精度融合 ============
    print("== 2. 杂草高精度融合（真实图片）==")
    manager.switch_task("weed")
    weed_img = decode_upload(Path(args.weed_img).read_bytes())
    check("杂草样例图可解码", weed_img is not None, args.weed_img)
    t0 = time.time()
    wr = infer_weed_single_image(weed_img, cfg, manager, mode="fusion")
    weed_total_ms = round((time.time() - t0) * 1000, 1)
    check("杂草融合返回 ok", wr["status"] == "ok")
    check("杂草融合检出框 > 0", wr["num_detections"] > 0, f"{wr['num_detections']} 框")
    check("杂草类别 Obonianghao",
          all(d["category_name"] == "Obonianghao" for d in wr["detections"]))
    check("杂草危害等级合法", wr["hazard"]["level"] in ("轻", "中", "重"))
    check("杂草建议非空", bool(wr["advice"]["advice"]))
    check("杂草 WBF 参数 = 任务05 固定（iou 0.65/skip 0.01/max/1:1:1）",
          wr["wbf_params"] and wr["wbf_params"]["iou_thr"] == 0.65
          and wr["wbf_params"]["skip_box_thr"] == 0.01
          and wr["wbf_params"]["conf_type"] == "max"
          and wr["wbf_params"]["weights"] == [1.0, 1.0, 1.0])
    check("杂草融合首载 + 纯推理耗时已记录",
          wr["model_load_ms"] > 0 and wr["net_inference_ms"] > 0,
          f"load={wr['model_load_ms']}ms net={wr['net_inference_ms']}ms total={weed_total_ms}ms")

    # 缓存后第二次推理（纯推理耗时）
    t0 = time.time()
    wr2 = infer_weed_single_image(weed_img, cfg, manager, mode="fusion")
    weed_cached_ms = round((time.time() - t0) * 1000, 1)
    check("杂草缓存后纯推理耗时", wr2["model_load_ms"] < 500,
          f"cached run total={weed_cached_ms}ms, model_load={wr2['model_load_ms']}ms")

    # 保存杂草验证产物
    w_ann = draw_annotations(weed_img, wr["detections"])
    encode_image_to_bytes(w_ann, ".jpg")
    (Path(args.out) / "weed_fusion_annotated.jpg").write_bytes(encode_image_to_bytes(w_ann, ".jpg"))
    (Path(args.out) / "weed_fusion_result.json").write_text(
        json.dumps({k: v for k, v in wr.items() if k != "detections"} |
                   {"detections": wr["detections"][:50]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.append(f"杂草融合首载: load={wr['model_load_ms']}ms net={wr['net_inference_ms']}ms; "
               f"缓存后 total={weed_cached_ms}ms")

    manager.free_gpu()  # 释放杂草模型

    # ============ 害虫快速单模型 ============
    print("== 3. 害虫快速单模型（YOLO11m，真实图片）==")
    manager.switch_task("pest")
    pest_img = decode_upload(Path(args.pest_img).read_bytes())
    check("害虫样例图可解码", pest_img is not None, args.pest_img)
    t0 = time.time()
    pr_fast = infer_pest_single_image(pest_img, cfg, manager._pest, mode="fast")
    pest_fast_total_ms = round((time.time() - t0) * 1000, 1)
    check("害虫快速返回 ok", pr_fast["status"] == "ok")
    check("害虫快速使用 YOLO11m", pr_fast["models_used"] == ["YOLO11m"])
    check("害虫快速检出框 > 0", pr_fast["num_detections"] > 0, f"{pr_fast['num_detections']} 框")
    check("害虫检测含 class_id / 类别名 / 置信度 / bbox",
          all({"class_id", "category_name", "confidence", "bbox_xyxy"} <= set(d.keys())
              for d in pr_fast["detections"]))
    check("害虫类别 ID 在 0..31 内",
          all(0 <= d["class_id"] < 32 for d in pr_fast["detections"]))
    check("害虫类别名来自 dataset.yaml",
          all(d["category_name"] == names[d["class_id"]] for d in pr_fast["detections"]))
    check("害虫快速危害等级合法", pr_fast["hazard"]["level"] in ("轻", "中", "重"))
    check("害虫快速无 WBF 参数（单模型）", pr_fast["wbf_params"] is None)
    check("害虫快速耗时记录", pr_fast["model_load_ms"] > 0 and pr_fast["net_inference_ms"] > 0,
          f"load={pr_fast['model_load_ms']}ms net={pr_fast['net_inference_ms']}ms")

    # 缓存后第二次（纯推理）
    t0 = time.time()
    pr_fast2 = infer_pest_single_image(pest_img, cfg, manager._pest, mode="fast")
    pest_fast_cached_ms = round((time.time() - t0) * 1000, 1)
    check("害虫快速缓存后纯推理耗时", pr_fast2["model_load_ms"] < 500,
          f"cached total={pest_fast_cached_ms}ms")

    (Path(args.out) / "pest_fast_annotated.jpg").write_bytes(
        encode_image_to_bytes(draw_annotations(pest_img, pr_fast["detections"]), ".jpg"))
    (Path(args.out) / "pest_fast_result.json").write_text(
        json.dumps({k: v for k, v in pr_fast.items() if k != "detections"} |
                   {"detections": pr_fast["detections"][:50]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.append(f"害虫快速首载: load={pr_fast['model_load_ms']}ms net={pr_fast['net_inference_ms']}ms; "
               f"缓存后 total={pest_fast_cached_ms}ms")

    # ============ 害虫三模型 classwise WBF ============
    print("== 4. 害虫三模型 refined classwise WBF（真实图片）==")
    t0 = time.time()
    pr_fusion = infer_pest_single_image(pest_img, cfg, manager._pest, mode="fusion")
    pest_fusion_total_ms = round((time.time() - t0) * 1000, 1)
    check("害虫融合返回 ok", pr_fusion["status"] == "ok")
    check("害虫融合使用三模型",
          pr_fusion["models_used"] == ["YOLO11m", "YOLO11l", "YOLO11s"])
    check("害虫融合使用 classwise WBF 配置",
          pr_fusion["wbf_params"] and pr_fusion["wbf_params"]["type"] == "classwise_WBF")
    check("害虫融合检出框 > 0", pr_fusion["num_detections"] > 0, f"{pr_fusion['num_detections']} 框")
    check("害虫融合类别名正确", all(d["category_name"] == names[d["class_id"]]
                                    for d in pr_fusion["detections"]))
    check("害虫融合每类统计存在", isinstance(pr_fusion.get("class_counts"), dict))
    check("害虫融合危害等级合法", pr_fusion["hazard"]["level"] in ("轻", "中", "重"))
    check("害虫融合建议非空且含诱捕设备检查", 
          bool(pr_fusion["advice"]["advice"]) and "诱捕" in pr_fusion["advice"]["advice"],
          pr_fusion["advice"]["advice"][:40] + "...")
    check("害虫融合耗时记录",
          pr_fusion["model_load_ms"] > 0 and pr_fusion["net_inference_ms"] > 0,
          f"load={pr_fusion['model_load_ms']}ms net={pr_fusion['net_inference_ms']}ms "
          f"total={pest_fusion_total_ms}ms")
    log.append(f"害虫融合首载: load={pr_fusion['model_load_ms']}ms net={pr_fusion['net_inference_ms']}ms; "
               f"total={pest_fusion_total_ms}ms")

    # 缓存后第二次（纯推理）
    t0 = time.time()
    pr_fusion2 = infer_pest_single_image(pest_img, cfg, manager._pest, mode="fusion")
    pest_fusion_cached_ms = round((time.time() - t0) * 1000, 1)
    check("害虫融合缓存后纯推理耗时", pr_fusion2["model_load_ms"] < 500,
          f"cached total={pest_fusion_cached_ms}ms")

    (Path(args.out) / "pest_fusion_annotated.jpg").write_bytes(
        encode_image_to_bytes(draw_annotations(pest_img, pr_fusion["detections"]), ".jpg"))
    (Path(args.out) / "pest_fusion_result.json").write_text(
        json.dumps({k: v for k, v in pr_fusion.items() if k != "detections"} |
                   {"detections": pr_fusion["detections"][:80]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log.append(f"害虫融合缓存后 total={pest_fusion_cached_ms}ms")

    manager.free_gpu()

    # ============ 5. 边界场景 ============
    print("== 5. 边界场景 ==")
    # 5.1 空文件
    empty = decode_upload(b"")
    check("空文件 → None（拒绝）", empty is None)
    # 5.2 非法内容（合法扩展名，损坏字节）
    broken = decode_upload(b"this is not an image at all" * 10)
    check("损坏内容 → None（拒绝）", broken is None)
    # 5.3 纯色空白图 → 0 框 → 危害「轻（未检出）」
    blank = np.full((640, 640, 3), 128, np.uint8)
    manager.switch_task("pest")
    br = infer_pest_single_image(blank, cfg, manager._pest, mode="fast")
    check("纯色空白图 0 框 → 危害轻",
          br["num_detections"] == 0 and br["hazard"]["level"] == "轻"
          and "未检出" in br["hazard"]["label"])
    # 5.4 权重缺失报错
    missing_result = None
    for mode, task in [("weed", "fusion"), ("pest", "fusion")]:
        mgr = DualModelManager(cfg)
        miss, ok = mgr.check_weights(task, mode)
        check(f"权重检查接口可用（{task}/{mode}）", isinstance(miss, list) and isinstance(ok, bool))

    # 5.5 害虫危害等级单元：密度与类别数驱动
    from hazard import compute_hazard_level
    h_light = compute_hazard_level(3, 1280, 1024, cfg["hazard"], task="pest", major_class_count=1)
    h_med = compute_hazard_level(10, 1280, 1024, cfg["hazard"], task="pest", major_class_count=4)
    h_heavy_d = compute_hazard_level(50, 1280, 1024, cfg["hazard"], task="pest", major_class_count=1)
    h_heavy_c = compute_hazard_level(5, 1280, 1024, cfg["hazard"], task="pest", major_class_count=8)
    check("害虫危害单元：低密度少类 → 轻", h_light["level"] == "轻")
    check("害虫危害单元：中等 → 中", h_med["level"] == "中")
    check("害虫危害单元：高密度 → 重", h_heavy_d["level"] == "重")
    check("害虫危害单元：类别数多 → 重", h_heavy_c["level"] == "重")
    check("杂草危害单元：3 框 → 轻", compute_hazard_level(3, 1280, 1024, cfg["hazard"], task="weed")["level"] == "轻")
    check("杂草危害单元：120 框 → 重", compute_hazard_level(120, 1280, 1024, cfg["hazard"], task="weed")["level"] == "重")

    # ------------------------------------------------------------------
    results["timing"] = {
        "weed_fusion_first": {"model_load_ms": wr["model_load_ms"], "net_inference_ms": wr["net_inference_ms"]},
        "weed_fusion_cached_total_ms": weed_cached_ms,
        "pest_fast_first": {"model_load_ms": pr_fast["model_load_ms"], "net_inference_ms": pr_fast["net_inference_ms"]},
        "pest_fast_cached_total_ms": pest_fast_cached_ms,
        "pest_fusion_first": {"model_load_ms": pr_fusion["model_load_ms"], "net_inference_ms": pr_fusion["net_inference_ms"]},
        "pest_fusion_cached_total_ms": pest_fusion_cached_ms,
    }
    results["inputs"] = {
        "weed_image": args.weed_img,
        "pest_image": args.pest_img,
        "dataset_yaml": dataset_yaml,
        "classwise_config": cw_path,
    }
    results["notes"] = "\n".join(log)

    out_json = Path(args.out) / "pipeline_verification.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果：{results['passed']} PASS / {results['failed']} FAIL → {out_json}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
