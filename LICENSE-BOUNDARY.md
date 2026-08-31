# LICENSE-BOUNDARY — MIT 覆盖范围说明

`LICENSE`（MIT，版权人 Anerqi）**仅覆盖**以下本项目自有内容：

- 项目源代码：`app.py`、`src/`（除 `src/model_config.py`——衍生重建自平台导出包与上游框架，见 THIRD_PARTY_NOTICES.md）、`scripts/`、`packaging/` 的 `.py` 文件；
- 项目配置与元数据：`config.yaml`、`.streamlit/config.toml`、`.gitignore`、`configs/`、`packaging/model_assets.yaml`、`requirements.txt`；
- 项目文档：`README.md`、`packaging/JUDGE_GUIDE.md`、`test_images/README.md`、`THIRD_PARTY_NOTICES.md`、本文件；
- 项目 UI 资源：`static/wheat-icon-128.png`、`static/wheat-icon-64.png`（作者于 2026-08-30 确认随仓库分发并按项目 MIT 授权；图标创作方式与原始权利的书面来源记录待补充，见 THIRD_PARTY_NOTICES.md）。

**不适用项目 MIT**的内容：

| 内容 | 处理方式 |
|---|---|
| 得意黑 Smiley Sans 字体（`static/SmileySans-Oblique.*`） | SIL OFL 1.1，见 `static/FONT-LICENSE.md` |
| YOLOX-Dinov3 衍生模型结构（`src/model_config.py`） | 衍生重建自平台导出源码包（YOLOX/Megvii + DINOv3/Meta + 平台扩展，2026-08-30 实测核验），许可证待核验，不受 MIT 覆盖 |
| 第三方 pip 依赖（Streamlit、Ultralytics、OpenCV、NumPy、PyYAML、Pillow、pandas、PyTorch 等） | 按各自许可，见 `THIRD_PARTY_NOTICES.md` |
| 模型权重与数据集 | 不随**代码树**分发；以 Release 资产（tag `assets-v1`）按各自条款提供（YOLO11 系权重按 Ultralytics 许可框架的 AGPL-3.0 路径处理；DINOv3 衍生权重按 DINOv3 License 分发、协议副本见 `licenses/DINOv3-License.md`；数据集来自 MADA 平台免费公开数据集，作者于 2026-08-31 确认可随本项目再分发），均不适用项目 MIT |

边界状态说明：本仓库**不声称**所有模型/数据集/衍生代码许可已完成核验；待核验项清单与责任人见 `THIRD_PARTY_NOTICES.md`。
