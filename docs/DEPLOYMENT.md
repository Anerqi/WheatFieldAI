<!-- docs/DEPLOYMENT.md -->
# 部署与现场运行手册（DEPLOYMENT）

本文件收录运行期的部署细节：日志、端口、缺失资产行为、无 CUDA 环境、Windows 现场恢复与打包计划。主 README 只保留入门路径，问题排查以本文件为准。

## 1. 日志

启动器自动创建：

```text
logs/
```

日志名称：

```text
streamlit_YYYYmmdd_HHMMSS.log
```

常见关键词与处理：

```text
KeyError: 'net_inference_ms'
→ 清理 __pycache__，使用 python -B

CUDA out of memory
→ 释放显存，切换快速单模型（界面提供一键切换按钮）

Address already in use
→ 关闭占用端口程序或修改 WHEATWEED_PORT

模型未配置
→ 检查 config.yaml、六套权重及环境变量

class_configs
→ 检查 JSON 是否完整覆盖 0..31
```

## 2. 端口

默认：

```text
8501
```

修改：

```text
$env:WHEATWEED_PORT="8502"
.\packaging\start.bat
```

启动器会先检查端口，实际启动和浏览器地址均使用同一个端口；端口被占用时启动器给出处理指引后退出，不会强行启动第二个服务。

## 3. 缺失模型资产时的行为

缺少权重时系统显示：

```text
模型未配置
```

不会生成模拟框、随机结果或静态 JSON。按 README §7.2 下载资产放入 `models/`（或用环境变量指定路径）后，运行 `python -B packaging/check_models.py`（missing=0 即资产就位）。

## 4. 无 CUDA 环境

CUDA 不可用时系统可以启动并执行推理，但性能有限，建议使用快速单模型模式；不能将 CPU 模式描述为 GPU 性能。

## 5. Windows 现场恢复

清理陈旧字节码：

```text
Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse -Force
```

重新启动：

```text
python -B -m streamlit run app.py --server.port 8501 --server.headless true
```

GPU OOM：释放 GPU 后切快速单模型（界面提供一键切换按钮）。

端口占用：关闭占用程序或设置 WHEATWEED_PORT。

## 6. Windows 桌面打包计划（PyInstaller，后续路线）

Windows 桌面封装（PyInstaller 外壳）属于后续路线：目标评委机真实验证尚未完成，当前不将其视为已交付能力。

候选结构：

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
不尝试把约 1.2 GB 模型和数 GB CUDA/PyTorch 运行时做成单文件。
```

在目标机完成真实打包、启动、推理验证之前，不声称 exe 已成功。
