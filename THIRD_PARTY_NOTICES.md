# THIRD_PARTY_NOTICES — 第三方组件与许可边界

本仓库以 MIT 许可证发布（见 `LICENSE`），但 **MIT 仅覆盖本项目自有的源代码、脚本、配置与文档**（范围见 `LICENSE-BOUNDARY.md`）。下列第三方组件**不受**项目 MIT 自动覆盖，按其各自许可证或权利状态处理。

## 1. 随仓库分发的组件

| 组件 | 位置 | 来源 | 许可/权利状态 | 边界 |
|---|---|---|---|---|
| 得意黑 Smiley Sans 字体 | `static/SmileySans-Oblique.ttf.woff2`、`static/SmileySans-Oblique.otf` | atelierAnchor 开源字体项目（版权与许可信息读取自字体文件内嵌 name 表） | SIL Open Font License 1.1 | 按 OFL-1.1 分发；不受项目 MIT 覆盖；详见 `static/FONT-LICENSE.md` |
| 小麦田图标 | `static/wheat-icon-128.png`、`static/wheat-icon-64.png` | 项目 UI 资源（任务 16）；作者于 2026-08-30 确认随仓库分发并按项目 MIT 授权 | 项目 MIT 范围（作者确认） | 图标创作方式与原始权利的书面来源记录：待补充；责任人：用户；下一步：补充来源记录；若来源权利状态变化，将从仓库移除 |
| YOLOX-Dinov3 衍生模型结构 | `src/model_config.py` | 衍生重建自 MADA 平台导出源码包（2026-08-30 经用户提供的导出包实测核验，SHA-256 9c860f51…）：YOLOX 框架类（Megvii 版权头；上游项目社区公开信息为 Apache-2.0）+ 平台扩展（YOLOPAFPNDinoV3 / YOLOXHeadDinoV3）+ DINOv3 ConvNeXt（Meta 版权头，DINOv3 License Agreement） | **许可证待核验**（包内无 LICENSE/NOTICE 文件，实测确认） | **不受项目 MIT 覆盖**；保留于本仓库以便完整运行；再分发授权需用户依据平台使用条款与 DINOv3 License 确认 |
| 推理适配层 | `src/yolox_inference.py` | 2026-08-30 核验：与导出包内全部候选来源表达式相似度 decode_outputs 0.0 / postprocess 0.133，letterbox/unletterbox/STRIDES 包内不存在 | 项目自研（核验确认） | 纳入项目 MIT 范围；算法遵循 YOLOX 解码定义，表达式为项目自写 |
| WBF 实现 | `src/wbf.py`、`src/pest_wbf.py` | 项目实现，遵循 Solovyev, Wang, Gabruseva (2021) "Weighted Boxes Fusion" 论文算法定义 | 算法为学术方法；实现按项目代码处理（纳入 MIT） | 注明论文出处；若实现被确认复制自第三方库，将移出 MIT 范围并按原库许可标注 |

## 2. 经包管理器安装的依赖（本仓库不分发其源码）

| 依赖 | 版本基准 | 许可证 |
|---|---|---|
| Streamlit | >=1.59 | Apache-2.0 |
| Ultralytics (YOLO11) | ==8.4.81 | AGPL-3.0 |
| OpenCV (opencv-python) | >=4.10 | Apache-2.0 |
| NumPy | >=1.26 | BSD-3-Clause |
| PyYAML | >=6.0.3 | MIT |
| Pillow | >=10.0 | MIT-CMU |
| pandas | >=2.0 | BSD-3-Clause |
| PyTorch / torchvision | 2.11.0+cu128 基准 | BSD-3-Clause（按官方渠道安装） |

依赖许可以各官方仓库最新文本为准。待补充；责任人：用户/Executor；下一步：完成书面核验。

## 3. 权重与数据集的分发方式

模型权重与数据集**不在代码树内**，以 GitHub Release 资产（tag `assets-v1`）按各自条款分发；均**不适用项目 MIT**：

| 资产 | 来源与许可 |
|---|---|
| `weed_yolo11s_baseline_best.pt`、`pest_yolo11m/l/s_best.pt` | Ultralytics YOLO11 相关模型与实现受 Ultralytics 许可条款约束；本项目公开发布所采用的 YOLO11 资产按 **AGPL-3.0** 路径处理 |
| `weed_yolox_dinov3_small_best_ckpt.pth`、`weed_yolox_dinov3_base_best_ckpt.pth` | 基于 Meta DINOv3 lvd1689m 预训练权重微调，构成 DINOv3 License 意义下的衍生作品。**2026-08-31 书面核验完成**（官方协议原文存档 `licenses/DINOv3-License.md`）：协议 §2.a 授予使用、复制、分发、创作衍生作品之权利；§2.i 要求按协议条款分发并随附协议副本 | 按 **DINOv3 License** 分发（不适用项目 MIT）；随附协议副本；使用限制（贸易管制、禁止军事等终端用途）以协议原文为准 |
| `dataset_weed_wheatweed_v1.zip` | MADA 赛题一平台免费公开数据集（DJI 田间图像与标注，train 3142 + val 787）；作者于 2026-08-31 确认可随本项目再分发 |
| `dataset_pest_train_images_v1.part1/2.zip`、`dataset_pest_labels_splits_v1.zip` | MADA 赛题二平台免费公开数据集（train 21634 图与标注；类别体系对应公开学术数据集 IP102）；作者于 2026-08-31 确认可随本项目再分发 |

代码树内仍不含任何权重、数据集、真实图片。本仓库不含任何 Token、Cookie、API Key 或个人凭证。
