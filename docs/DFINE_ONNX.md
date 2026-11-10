# D-FINE-S ONNX 权重说明

杂草高精度融合第 6 支成员 D-FINE-S 的 ONNX 权重不随代码树分发，随 GitHub Release [`assets-v3`](https://github.com/Anerqi/WheatFieldAI/releases/tag/assets-v3) 提供。本文件说明来源、格式、校验、放置与降级行为。

## 1. 来源与许可

- 模型：D-FINE-S（arXiv:2410.13842），官方实现 Peterande/D-FINE，Apache-2.0。
- 初始权重：`dfine_s_coco.pth`（SHA-256 前缀 48A6C8CC，与 Hugging Face LFS oid 一致）。
- 微调：在项目内部 3,142 张杂草训练图上微调 30 epoch，单类（`Obonianghao`）。
- 许可：推理封装与权重按 Apache-2.0 分发，不适用项目 MIT；见 `THIRD_PARTY_NOTICES.md`。

## 2. ONNX 输入输出

- 输入 `images`：`float32[1, 3, 960, 960]`，RGB，归一化 `/255`；
- 输入 `orig_target_sizes`：`int64[1, 2]`，`[h, w]`；
- 输出 `labels` / `boxes`（xyxy，按 `orig_target_sizes` 还原）/ `scores`；
- 推理阈值 `score_thr = 0.05`（`config.yaml` 的 `weed.dfine`）。

## 3. 数值对齐（导出验证）

- 模型级 parity：labels 全等；boxes 最大差 1/4 像素；scores 最大差 8.9e-06。
- 端到端 787 图复算：PyTorch 0.444344 vs ONNX 0.444345（mAP50-95，本地验证集口径）。

## 4. 下载、校验与放置

```text
# Release assets-v3
weed_dfine_s_best.onnx     40.4 MB   Apache-2.0
SHA256SUMS_v3.txt                    SHA-256 校验清单
```

- SHA-256 完整值见 `SHA256SUMS_v3.txt`（前缀 `572040a21984…`）；
- 放置到 `models/weed/dfine_s/best.onnx`；路径唯一真源为 `config.yaml` 的 `weed.paths.dfine_onnx`，也可用环境变量覆盖；
- 校验：`python -B packaging/check_models.py`。

## 5. 依赖与降级

- 推理依赖 `numpy` / `Pillow` / `onnxruntime`（见 `requirements.txt`；CUDA 环境可改用 `onnxruntime-gpu`）。
- 权重缺失、`onnxruntime` 未安装或会话创建/推理失败时，按 `src/fusion_profile.py` 成套退回五模型参数（WBF `iou_thr` 0.65、`decision_conf` 0.40），页面提示「已退回五模型」，结果 JSON 的 `wbf_params.profile` 与 `fusion_degraded` 标明降级状态。

## 6. 指标口径

本地验证集（787 图 / 2596 框，无独立测试集）：六模型 mAP50 0.8403 / mAP50-95 0.4709 / AP75 0.4683 / AR@100 0.6263；五模型历史口径 0.8399 / 0.4623 / 0.4466 / 0.6101。不是平台测试集指标，也不代表对任意上传图片的精度承诺。
