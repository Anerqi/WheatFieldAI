# WheatFieldAI · 术语与事实底稿（CONTEXT）

[返回主工程说明](../README.md) · [项目 Wiki](../wiki/README.md) · [项目总索引](../INDEX.md)

> **重建说明**：`config.yaml` 与部分历史报告引用“项目根 `CONTEXT.md` 为术语唯一真源”，但原文件在工作区中不存在。本文件于 **2026-10-28** 依据当前代码、配置与已验收任务报告重建，**2026-11-07 对齐任务 30 现行口径（六模型 + D-FINE-S、`decision_conf` 0.44、127 单测）**，作为**当前工程术语与口径的索引**，不是对已丢失原文的逐字恢复。若与代码/配置冲突，以代码与配置为准。

## 1. 项目与任务

- 项目：小麦田间杂草与害虫智能识别及辅助决策系统（WheatFieldAI）。
- 形态：本地 Streamlit Web 工作台；用户主动选择检测对象（杂草/害虫）与模式（高精度融合/快速单模型）。
- 杂草任务：1 类 `Obonianghao`；害虫任务：32 类，ID 0–31。
- 输出：检测框、类别统计、坐标明细、危害等级、防治建议、标注 JPG、结构化 JSON。
- 危害等级与防治建议是**原型规则**，未经农学专家校准，不构成农业处方。

## 2. 模式与模型组合

| 任务 | 模式 | 模型组合 | 关键参数 |
|---|---|---|---|
| 杂草 | 高精度融合 `fusion` | YOLO11s（960）+ YOLO11s-box8（960）+ YOLO11s-1280 + YOLOX-Dinov3 Small + YOLOX-Dinov3 Base + D-FINE-S（ONNX，960；任务 30 第 6 支）+ 六模型 WBF | YOLO conf 0.001 / iou 0.70；960/960/1280；YOLOX conf 0.01 / nms 0.65 / imgsz 640；D-FINE score_thr 0.05；WBF iou **0.68** / skip 0.01 / conf `avg` / score_mode `geo_agree` / 六等权 |
| 杂草 | 快速单模型 `yolo11` | YOLO11s | 同上 YOLO 参数 |
| 害虫 | 高精度融合 `fusion` | YOLO11m + YOLO11l + YOLO11s + refined classwise WBF | conf 0.001 / iou 0.70 / imgsz 960；逐类参数见 classwise JSON |
| 害虫 | 快速单模型 `fast` | YOLO11m | 同上 |

### 2.1 业务决策阈值（decision_conf）

推理阈值保持评测口径（杂草 0.001/0.01、害虫 0.001），业务展示另设阈值：

```yaml
weed:
  decision_conf: 0.44
pest:
  decision_conf: 0.30
```

- 只有置信度 ≥ `decision_conf` 的检测参与计数、类别统计、密度估算与 JSON `detections`；
- 低于阈值的检测保留在 `low_conf_candidates`，页面可勾选查看；
- 不影响 mAP 评测；杂草 **0.44** 依据任务 30 在「六模型 + `iou_thr` 0.68」新分数尺度上的 F1 校准（F1 最优 0.44，val P=0.780 / R=0.788 / F1=0.784），害虫 0.30 依据任务 22；
- 历史值：任务 26 五模型 `geo_agree` 尺度取 0.40（F1 0.774，最优 0.39），任务 24 `max` 尺度取 0.50（F1 0.763）；
- 0.44/0.30 是本地验证集初始值，正式应用前需按现场数据复核。

### 2.2 融合分数的一致性调制（任务 26）

杂草 WBF 的输出分数为 `avg^(1-g) · (min(n,cap)/cap)^g`，其中 n 为簇内框数、g=0.412、cap=8：

- 只影响排序与置信度取值，不改变坐标、簇构建与 iou/skip/weights；
- 值域仍为 [0,1]，不破坏百分比语义；
- `score_mode="raw"` 时与任务 24 的 `conf_type=max` 行为一致，可回退。

### 2.3 五模型退回档

D-FINE-S 未参与融合（未启用、权重缺失、`onnxruntime` 未安装或会话运行失败）时，`src/fusion_profile.py` 成套切换到 `weed.wbf.five_model_fallback`：`iou_thr` **0.65**、`decision_conf` **0.40**（任务 26 五模型 `geo_agree` 尺度校准值）。结果 JSON 的 `wbf_params.profile` 标明 `six_model` / `five_model_fallback`，`fusion_degraded` 为 True 时页面提示已退回。

## 3. 类别映射

### 3.1 杂草

```text
YOLO 内部类别 ID = 0
COCO 评测 category_id = 1
类别名 = "Obonianghao"
num_classes = 1
```

### 3.2 害虫

- 32 类名称唯一真源：[configs/dataset.yaml](configs/dataset.yaml) 的 `names` 段。
- 要求 ID 0–31 连续、共 32 类；代码不自行翻译、不重排、不编造类别名。
- 逐类融合参数唯一真源：[configs/classwise_ensemble_11m11l11s_960_refined_current.json](configs/classwise_ensemble_11m11l11s_960_refined_current.json)；`class_configs` 必须覆盖全部 32 类，每类含 `model_weights`、`wbf_iou`、`skip`、`score_mode`。
- `score_mode` 取值：`agreement` / `sqrt_agreement` / `mean` / `max`。

## 4. 指标与数据口径

### 4.1 数据规模

| 数据 | 规模 | 用途 |
|---|---|---|
| 杂草训练集 | 3142 图 / 8624 框 | MADA 平台训练 |
| 杂草验证集 | 787 图 / 2596 框 | 本地统一评测 / WBF；**无独立测试集** |
| 害虫训练集 | 17,304 图 / 112,809 框 | 32 类训练 |
| 害虫验证集 | 4,330 图 / 28,133 框 | 本地自研评估器 / 逐类 WBF；**无独立测试集** |

### 4.2 本地验证集指标

| 模型/融合 | mAP50 | mAP50-95 | AP75 | AR@100 |
|---|---:|---:|---:|---:|
| 杂草 YOLO11s 单模型 | 0.803197 | 0.429190 | 0.404958 | 0.596263 |
| 杂草 YOLOX-Dinov3 Small 单模型 | 0.808880 | 0.402419 | 0.352460 | 0.516564 |
| **杂草六模型 WBF + D-FINE-S + 一致性调制（任务 30，现行）** | **0.8403** | **0.4709** | **0.4683** | **0.6263** |
| 杂草五模型 WBF（任务 26，历史口径） | 0.8399 | 0.4623 | 0.4466 | 0.6101 |
| 害虫 YOLO11m 单模型 | 0.78069 | 0.50116 | — | — |
| 害虫 YOLO11l 单模型 | 0.77872 | 0.49824 | — | — |
| 害虫 YOLO11s 单模型 | 0.75735 | 0.48031 | — | — |
| **害虫 refined classwise WBF** | **0.80524** | **0.52410** | — | — |

所有数字均为**本地验证集**口径，不代表平台测试集成绩，也不保证任意上传图片的效果。

## 5. 危害等级与建议

- **口径**：按用户填写的「实际拍摄区域面积」计算，`密度 = 正式检测框数 ÷ 面积`，单位 **个/㎡**（`det_per_m2`）。面积可填 ㎡ 或亩（1 亩 = 666.6667 ㎡，内部统一换算为 ㎡）。
- **默认不填写面积 → 不显示危害等级、密度与防治建议**：主工作台隐藏 KPI 危害 chip、危害卡、防治建议卡与批量汇总的危害列。
- 杂草：密度 < 0.5 为轻；0.5–2 为中；≥ 2 为重。害虫另有主要类别数 2 / 5 的升级规则。
- **以上阈值为占位演示值，无农学依据、未经农学专家校准**，不构成防治处方或正式农业决策依据；正式应用前须按田间数据重新标定并经专家审核。
- 历史口径 `det_per_mp`（每百万像素检测框数，杂草 10 / 40、害虫 5 / 15 + 类别数 2 / 5）仅保留在导出字段 `density_per_mp` 与 `config.yaml` 的 `hazard.weed` / `hazard.pest` 中用于兼容，**不再用于页面分级**。
- 建议只提供巡查/复核/设备检查/咨询农技人员等方向性内容；**不提供农药名称、剂量、浓度、施用时间、施用次数、安全间隔期**。
- 新口径阈值与建议文本在 [config.yaml](config.yaml) 的 `hazard.field_area` / `advice` 段，算法在 [src/hazard.py](src/hazard.py)；修改需同步更新文档与测试。

## 6. 上传与运行

- 支持 `.jpg` / `.jpeg` / `.png`；单张 ≤ 20 MB；单批 ≤ 10 张。
- 权重路径支持环境变量覆盖；相对路径以 `config.yaml` 所在目录为基准。
- 无 CUDA 可启动，但建议快速单模型；融合模式 CUDA OOM 时释放缓存并支持切换快速模式。
- 任务切换时释放另一任务模型与 CUDA 缓存。

## 7. 权重资产

| 任务 | 路径 | 大小 |
|---|---|---:|
| 杂草 YOLO11s | `models/weed/yolo11s/baseline_best.pt` | 19.3 MB |
| 杂草 YOLO11s-box8 | `models/weed/yolo11s_box8/best.pt` | 19.3 MB |
| 杂草 YOLO11s-1280 | `models/weed/yolo11s_1280/best.pt` | 19.4 MB |
| 杂草 YOLOX-Dinov3 Small | `models/weed/dinov3_small/best_ckpt.pth` | 287.7 MB |
| 杂草 YOLOX-Dinov3 Base | `models/weed/dinov3_base/best_ckpt.pth` | 541 MB |
| 害虫 YOLO11m | `models/pest/yolo11m/best.pt` | 121.0 MB |
| 害虫 YOLO11l | `models/pest/yolo11l/best.pt` | 152.8 MB |
| 害虫 YOLO11s | `models/pest/yolo11s/best.pt` | 19.3 MB |
| 杂草 D-FINE-S（ONNX，第 6 支） | `models/weed/dfine_s/best.onnx` | 40.4 MB |

九套资产合计约 1,203.8 MB；`python -B packaging/check_models.py` 实测 missing=0 / invalid=0。

SHA-256 前 12 位与检查方法见 [packaging/model_assets.yaml](packaging/model_assets.yaml) 与 [packaging/check_models.py](packaging/check_models.py)。

## 8. 模块溯源

| 模块 | 来源 | 说明 |
|---|---|---|
| `src/wbf.py` | 任务 05 `run_wbf_experiment.py` | 标准 WBF，逐字节复制 |
| `src/pest_wbf.py` | 害虫提交包 `export_ensemble_zip.py` | refined classwise WBF，逐字节复制并适配坐标空间 |
| `src/yolox_inference.py` | 任务 02/03 推理与统一评测脚本 | 预处理、解码、后处理、反 letterbox |
| `src/model_config.py` | 任务 05 YOLOX-Dinov3 结构重建 | Small/Base 结构参数化 |
| `src/weed_inference.py` | 任务 07 `src/inference.py` | 仅适配双任务配置键；接入 decision_conf 业务过滤；任务 30 增加 D-FINE-S 第 6 路（缺成员时自动退回五模型） |
| `src/dfine_onnx.py` | 任务 30 新增 | D-FINE-S ONNX 推理封装（仅依赖 numpy/PIL/onnxruntime） |
| `src/decision.py` | 2026-10-28 业务阈值校准 | 决策阈值读取、检测拆分、置信度统计与类别计数 |
| `src/models.py` / `pest_models.py` | 任务 08 | 双任务模型管理与显存策略 |
| `src/hazard.py` / `advice.py` | 任务 07/08 | 规则原型，配置驱动 |
| `app.py` | 任务 16 终版 UI | 表现层；推理调用不变 |

## 9. 文档权威顺序

1. 代码与 `config.yaml` / `configs/*.yaml` / classwise JSON：当前运行事实。
2. 任务报告与 QA 证据：历史执行与验证事实。
3. 本文件：术语与口径索引。
4. `README.md` / Wiki：面向读者的解释。
5. `project-status.md`：排期状态，可能滞后于交付。

## 10. 已知边界

- 原始 `CONTEXT.md` 已缺失；本文件是重建版，不声称逐字还原。
- 数据集许可、源码/第三方许可仍未全部完成书面核验；分发前见 [license_manifest.md](../任务输出/17_最终开源准备与GitHub发布/license_manifest.md)。
- 本文件不替代真实推理验收；运行状态以实际启动与模型检查为准。
