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

## 3. 明确不随本仓库分发的资产

- **六套模型权重**（`models/` 下任何 `.pt` / `.pth` 等）：用户自训练资产；其中杂草 YOLOX-Dinov3 权重基于 Meta DINOv3 lvd1689m 预训练权重微调（导出包实测含该预训练 checkpoint），再分发适用 DINOv3 License 与平台条款——待用户书面核验；请自行准备你有权使用的权重。
- **数据集**：图片、标注、分割清单、压缩包一律不随仓库分发；`configs/dataset.yaml` 仅作类别名数据源。
- **真实测试图片**：`test_images/` 下仅保留说明文档，图片请自行准备。
- **真实运行输出与日志**：不随仓库分发。
- 本仓库不含任何 Token、Cookie、API Key 或个人凭证。
