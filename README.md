<!-- README.md -->
# 小麦田间杂草与害虫智能识别及辅助决策系统

本项目是已通过真实 CLI / Web 验证的「08_杂草害虫双任务Web系统」封装版。封装层采用整体复制基线 + 增量包装方式，不重写已验证的推理、融合与规则算法。

## 1. 能力

- 杂草 / 害虫双任务，用户主动选择检测对象。
- 高精度融合 / 快速单模型。
- JPG/JPEG/PNG，单批最多 10 张，单张最大 20 MB。
- 杂草 YOLO11s + YOLOX-Dinov3 Small/Base + 固定 WBF。
- 害虫 YOLO11m + YOLO11l + YOLO11s + refined classwise WBF。
- 检测框、类别统计、坐标明细、危害等级、防治建议。
- 标注 JPG + 结构化 JSON。
- GPU OOM 降级到快速单模型。
- CPU 环境可启动，但建议快速单模型。
- 任务切换时释放另一任务模型与 CUDA 缓存。

### 1.1 当前页面 UI（任务 16 版）

当前页面采用任务 16 版「Bento 网格」布局（珊瑚色浅底、编号卡片）：

- 顶部品牌栏（小麦田图标 + 站点标题）；行 1「01 · 工作台控制」（检测对象 / 推理模式 / 上传）+「02 · 运行状态」（模型就绪等状态点、模式信息条、原型声明、批量规则）；
- 行 2「输入原图」+「标注结果」（含显示低置信候选框开关）；行 3「结果概览 KPI」+「按类别统计」+「危害等级」；
- 行 4「检测推举明细」（可折叠）；行 5「防治建议」+「结果导出」（标注 JPG / JSON 下载）；行 6「推理参数与模型详情」（可折叠）；页脚为验证集指标与模型说明。

表现层事实：

- 全站仅两种字体：得意黑 SmileySans（标题与正文，`static/SmileySans-Oblique.ttf.woff2` + `.otf`）与宋体（标签/数值/元数据等次要文字）；
- 浏览器标签页 favicon 与品牌栏图标使用小麦田插画（`static/wheat-icon-128.png` / `static/wheat-icon-64.png`）；
- 字体与图标由 Streamlit 静态服务提供（`.streamlit/config.toml` 中 `enableStaticServing = true`），离线可用，不访问外部网络；
- 上传控件文案与服务端限制一致（`maxUploadSize = 20`，单张 ≤20 MB）；
- 无结果时显示带编号三步引导的空状态卡；展示层低置信过滤（< 0.10）仅影响标注图与明细表默认显示，危害等级 / 类别统计 / JSON 始终基于全部检测结果；
- 危害等级与防治建议仍为原型规则，需农学专家校准，不构成正式农业处方。

## 2. 项目结构

```text
WheatFieldAI/
├─ app.py
├─ config.yaml
├─ requirements.txt
├─ LICENSE / LICENSE-BOUNDARY.md / THIRD_PARTY_NOTICES.md
├─ .streamlit/config.toml     ← 主题 + 上传限制 + 静态服务开关
├─ static/                    ← 得意黑双字体 + 小麦田图标 + 字体许可（OFL-1.1）
├─ src/                       ← 推理与业务模块
├─ scripts/                   ← 验证脚本
├─ configs/                   ← 类别唯一真源与 classwise WBF 配置
├─ packaging/                 ← 启动器与检查脚本
└─ test_images/README.md      ← 真实测试图片请自行放置（图片不随仓库分发）
```

src/ 各模块沿用已验证的推理实现（溯源见各文件头注释与 THIRD_PARTY_NOTICES.md）；deployment_adapter.py 只提供封装接口，不实现新的推理算法。运行时生成的 models/、logs/、outputs/ 已被 .gitignore 排除，不随仓库分发。

## 3. 启动

推荐：

```text
.\packaging\start.bat
```

或：

```text
powershell -ExecutionPolicy Bypass -File .\packaging\start.ps1
```

手动：

```text
python -B -m streamlit run app.py --server.port 8501 --server.headless true
```

默认：

```text
http://localhost:8501
```

## 4. 配置路径

config.yaml 路径支持相对路径（以项目根为基准）与环境变量绝对路径覆盖。

环境变量：

```text
WHEATWEED_YOLO11_WEIGHTS
WHEATWEED_YOLOX_SMALL_WEIGHTS
WHEATWEED_YOLOX_BASE_WEIGHTS
PEST_YOLO11M_WEIGHTS
PEST_YOLO11L_WEIGHTS
PEST_YOLO11S_WEIGHTS
PEST_DATASET_YAML
PEST_CLASSWISE_CONFIG
WHEATWEED_DEVICE
WHEATWEED_PORT
```

相对路径由 src/config.py 解析为绝对路径，因此从任意工作目录启动均可定位模型和配置。

## 5. 环境

已验证基准：

```text
Python 3.13.9
PyTorch 2.11.0+cu128
torchvision 0.26.0+cu128
CUDA 12.8
Streamlit 1.59.1
Ultralytics 8.4.81
opencv-python 5.0+
numpy 2.4+
Pillow 12+
pandas 2.3+
```

CUDA 版 PyTorch 单独安装：

```text
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

启动器和检查脚本不会自动安装依赖，也不会联网。

## 6. 模型资产

必须自行放置六套真实权重：

```text
models/weed/yolo11s/baseline_best.pt
models/weed/dinov3_small/best_ckpt.pth
models/weed/dinov3_base/best_ckpt.pth
models/pest/yolo11m/best.pt
models/pest/yolo11l/best.pt
models/pest/yolo11s/best.pt
```

配置：

```text
configs/dataset.yaml
configs/classwise_ensemble_11m11l11s_960_refined_current.json
```

不会生成或伪造任何权重。

### 6.1 完整权重与数据集下载（Release 资产）

六套权重与训练/验证数据集以 GitHub Release 资产形式提供（tag `assets-v1`），**不随代码树分发**，且**不适用项目 MIT**（各资产按其来源条款，见 Release 说明与 `THIRD_PARTY_NOTICES.md`）：

```text
weed_yolo11s_baseline_best.pt             18 MB   Ultralytics YOLO11（AGPL-3.0）
weed_yolox_dinov3_small_best_ckpt.pth    288 MB   基于 Meta DINOv3 lvd1689m 微调（DINOv3 License，公开前待核验）
weed_yolox_dinov3_base_best_ckpt.pth     541 MB   同上
pest_yolo11m_best.pt                     115 MB   Ultralytics YOLO11（AGPL-3.0）
pest_yolo11l_best.pt                     146 MB   Ultralytics YOLO11（AGPL-3.0）
pest_yolo11s_best.pt                      18 MB   Ultralytics YOLO11（AGPL-3.0）
dataset_weed_wheatweed_v1.zip           1.05 GB   杂草 WheatWeed train(3142)+val(787) 图与标注 + 配置
dataset_pest_train_images_v1.part1.zip  ~1.0 GB   害虫训练图片（两卷，解压到同一目录）
dataset_pest_train_images_v1.part2.zip  ~1.0 GB   害虫训练图片（两卷，解压到同一目录）
dataset_pest_labels_splits_v1.zip         13 MB   害虫训练标注 + train/val 划分 + 数据集配置
SHA256SUMS.txt                                    全部资产校验和
```

下载后按 `SHA256SUMS.txt` 校验；权重放入 `models/` 对应目录（或用环境变量指定路径）；害虫图片两卷解压到同一目录后与 labels/ 组合。各资产来源与许可边界详见 `THIRD_PARTY_NOTICES.md` 与 Release 说明。

## 7. 资产检查

```text
python -B packaging/check_models.py
```

查看 SHA-256：

```text
python -B packaging/check_models.py --print-sha256
```

packaging/model_assets.yaml 保存六套权重的真实 SHA-256 前 12 位；路径唯一真源仍然是 config.yaml。

## 8. 32 类唯一真源

configs/dataset.yaml 的 names 是害虫类别唯一真源。

要求：

```text
ID 0..31 连续
共 32 类
```

代码不自行翻译、不重新排序、不编造类别名。

## 9. classwise WBF

运行时读取：

```text
class_configs[*].model_weights
class_configs[*].wbf_iou
class_configs[*].skip
class_configs[*].score_mode
```

必须覆盖全部 0..31。

不得使用默认配置，不得使用杂草 WBF 参数替代。

## 10. 导出

Web 下载入口：

```text
app.py
```

适合评委操作；JSON task 使用中文标签，并含 generated_at。

CLI / 批处理入口：

```text
src/deployment_adapter.py
export_result()
```

适合自动化；task 使用 "weed" / "pest"，不含 generated_at。

两个入口核心字段一致：

```text
detections
hazard
advice
wbf_params
class_mapping
device
note
```

note 为原型声明，必须保留。

## 11. CLI 部署冒烟

准备真实杂草图和真实害虫图后：

```text
python -B scripts/verify_deployment.py `
  --weed-image "test_images/weed/你的杂草图.jpg" `
  --pest-image "test_images/pest/你的害虫图.jpg"
```

脚本真实加载本地模型，不使用模拟检测结果。

覆盖：

```text
杂草 fusion
杂草 yolo11
害虫 fusion
害虫 fast
缺失权重
JPG + JSON 导出
```

## 12. test_images

test_images/weed/ 放任务 02 验证集真实 DJI_...jpg 类杂草图片。

test_images/pest/ 放赛题二 test_q2 真实害虫图片。

不要使用静态截图作为模型真实性验收样本。

图片与数据集本身不随本仓库分发，请使用你有权使用的真实图片。

## 13. 日志

启动器自动创建：

```text
logs/
```

日志名称：

```text
streamlit_YYYYmmdd_HHMMSS.log
```

常见关键词：

```text
KeyError: 'net_inference_ms'
CUDA out of memory
Address already in use
模型未配置
class_configs
```

对应处理：

```text
KeyError: 'net_inference_ms'
→ 清理 __pycache__，使用 python -B

CUDA out of memory
→ 释放显存，切换快速单模型

Address already in use
→ 关闭占用端口程序或修改 WHEATWEED_PORT

模型未配置
→ 检查 config.yaml、六套权重及环境变量

class_configs
→ 检查 JSON 是否完整覆盖 0..31
```

## 14. 端口

默认：

```text
8501
```

修改：

```text
$env:WHEATWEED_PORT="8502"
.\packaging\start.bat
```

启动器会先检查端口，实际启动和浏览器地址均使用同一个端口。

## 15. 指标口径

杂草：

```text
本地验证集
mAP50 = 0.828036
mAP50-95 = 0.443757
AP75 = 0.415953
```

害虫：

```text
本地验证集 refined classwise WBF
mAP50 = 0.80524
mAP50-95 = 0.52410
```

均不是平台测试集指标，也不代表任意上传图片的精度。

## 16. 防治建议

防治建议属于原型规则，未经农学专家校准。

系统不提供：

```text
具体农药名称
剂量
浓度
施用时间
施用次数
安全间隔期
```

重度结果必须建议农技人员或植保专家复核。

## 17. 无权重

缺少权重时：

```text
模型未配置
```

不会生成模拟框、随机结果或静态 JSON。

## 18. 无 CUDA

CUDA 不可用时系统可以启动，但表示为 CPU 环境。

推荐：

```text
快速单模型
```

不能将 CPU 模式描述为 GPU 性能。

## 19. Windows 现场恢复

清理陈旧字节码：

```text
Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse -Force
```

重新启动：

```text
python -B -m streamlit run app.py --server.port 8501 --server.headless true
```

GPU OOM：释放 GPU 后切快速单模型。

端口占用：关闭占用程序或设置 WHEATWEED_PORT。

## 20. 可选 PyInstaller 外壳

待本地验证。

只有路线 A 在目标评委机完成真实验证后才建议继续。

推荐结构：

```text
launcher/
├─ WheatFieldAI.spec
└─ launcher.py
```

原则：

```text
exe 只负责检查环境、定位项目目录、启动 Streamlit；
六套权重继续外置；
models/ 不嵌入 exe；
CUDA 版 PyTorch 不作为普通资源强行塞入 exe；
不尝试把约 1.18 GB 模型和数 GB CUDA/PyTorch 运行时做成单文件；
用户未本地打包、启动、真实推理验证前，不得声称 exe 已成功。
```

PyInstaller 自查：

```text
Python runtime
PyTorch DLL
CUDA DLL
OpenCV
Ultralytics
YOLOX 自定义模块
相对路径
models 外置
logs
端口
浏览器启动
目标机无 NVIDIA GPU
```

## 21. 许可证

- 本仓库的项目源代码、脚本、配置与文档以 MIT 许可证发布（见 `LICENSE`）；MIT 的覆盖范围与明确排除项见 `LICENSE-BOUNDARY.md`。
- 第三方组件按各自许可处理，不受项目 MIT 自动覆盖：依赖包与 YOLOX-Dinov3 衍生模型结构（`src/model_config.py`，核验记录见该文件）等逐项说明见 `THIRD_PARTY_NOTICES.md`。
- 得意黑（Smiley Sans）字体随仓库分发，字体本身采用 SIL Open Font License 1.1，版权与许可信息见 `static/FONT-LICENSE.md`；字体不适用项目 MIT。
- 小麦田图标（`static/wheat-icon-*.png`）为项目 UI 资源，经作者确认随仓库分发，纳入项目 MIT 范围。
- 六套模型权重**不随仓库分发**：运行前请自行准备你有权使用的权重，放入 `models/` 或用环境变量指定路径。
- 数据集与真实测试图片**不随仓库分发**：`configs/dataset.yaml` 仅作 32 类类别名数据源；`test_images/` 请放置你自行准备的真实图片。
- 真实运行需要你自行准备合规资产（权重、环境）；本仓库不生成、不伪造任何权重或检测结果。
- 当前**不声称**所有模型 / 数据集许可已完成核验；待核验项与责任人清单见 `THIRD_PARTY_NOTICES.md`。

## 22. 真实性声明

本项目不会使用模拟检测框、静态 JSON、随机结果或占位结果冒充 AI 推理。

真实推理必须依赖真实权重和满足要求的本地运行环境。

本仓库不保证在任意目标电脑上一键启动成功：启动器（start.bat / start.ps1）在干净目标机的行为尚未完成验证。
启动失败时请参考 §13 与 §19 排障，或使用 §3 的手动启动命令。
