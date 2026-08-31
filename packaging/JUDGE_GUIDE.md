<!-- packaging/JUDGE_GUIDE.md -->
# 评委快速操作说明

## 0. 页面一览（任务 16 版 UI）

页面为本地 Streamlit 工作台（Bento 网格布局，珊瑚色浅底），非在线 AI 服务，不访问外部网络：

- 「01 · 工作台控制」卡：选择检测对象（杂草/害虫）、推理模式（高精度融合/快速单模型）、上传图片；
- 「02 · 运行状态」卡：模型就绪/权重缺失状态点、当前模式信息条、原型声明、批量规则（单批 ≤10 张、单张 ≤20 MB）；
- 结果出现后依次查看：「标注结果」（检测框可视化）→「结果概览」（数量/置信度/耗时）→「按类别统计」→「危害等级」→「检测框明细」（科研用途，可折叠）→「防治建议」→「结果导出」；
- 页面字体为得意黑（标题/正文）与宋体（标签/数值），favicon 与品牌栏为小麦田插画图标，均由本包 `static/` 本地提供；
- 无结果时右栏显示三步引导空状态卡。
- YOLO11 系权重与训练/验证数据集通过仓库 Release（tag `assets-v1`）提供；DINOv3 衍生权重未纳入公开 Release（见 README §7.2）。各资产许可见 `THIRD_PARTY_NOTICES.md`。

## 1. 启动

双击：

```text
packaging/start.bat
```

启动器依次执行：

```text
环境检查 → 模型资产检查 → 端口检查 → 启动 Streamlit → 等待就绪 → 打开浏览器
```

默认地址：

```text
http://localhost:8501
```

端口可以通过环境变量 WHEATWEED_PORT 修改。

例如：

```text
$env:WHEATWEED_PORT="8502"
.\packaging\start.bat
```

浏览器和启动器提示会使用实际端口。

## 2. 3 分钟演示

页面元素位置：检测对象与推理模式在左上「01 · 工作台控制」卡；状态点在「02 · 运行状态」卡；标注图在「标注结果」卡；数量/置信度/耗时在「结果概览」卡；下载按钮在「结果导出」卡。

```text
选择「杂草检测」。
选择「高精度融合」。
上传真实杂草田间图片。
查看检测框、数量、置信度、密度和危害等级。
查看防治建议与原型免责声明。
查看结果元信息，核对杂草 WBF 固定参数。
下载标注 JPG 与 JSON。
切换「害虫检测」。
选择「快速单模型」或「高精度融合」。
上传真实害虫图片。
展示 32 类英文标准名、类别统计、危害等级。
下载结果。
```

## 3. 导出

Web 页面中的下载按钮是第一导出入口，适合评委直接演示，JSON 含 generated_at，task 使用中文任务标签。

`src/deployment_adapter.py` 的 `export_result()` 是第二导出入口，适合 CLI、批处理或后续自动化流程，task 使用 "weed" / "pest"，不写 generated_at。

两个入口的核心结构一致，均必须保留：

```text
detections
hazard
advice
wbf_params
class_mapping
device
note
```

其中：

```text
note = 原型系统声明
```

不得删除。

## 4. 无权重

缺少权重时显示：

```text
模型未配置
```

系统不会产生模拟检测框、随机结果或静态 JSON。

## 5. 无 CUDA

如果 CUDA 不可用：

```text
当前环境使用 CPU。
```

建议选择：

```text
快速单模型
```

不能把 CPU 模式描述成 GPU 性能。

## 6. OOM

出现：

```text
CUDA out of memory
```

处理：

```text
关闭其他占用显存的软件；
回到页面切换「快速单模型」；
重新推理。
```

## 7. 查看日志

日志自动写入：

```text
logs/streamlit_YYYYmmdd_HHMMSS.log
```

启动器会打印当前日志文件路径。

常见关键词：

| 关键词 | 处理 |
|---|---|
| KeyError: 'net_inference_ms' | 清理 __pycache__，使用 python -B 重启 |
| CUDA out of memory | 释放 GPU 显存，切换快速单模型 |
| Address already in use | 检查默认 8501，关闭占用程序或设置 WHEATWEED_PORT |
| 模型未配置 | 检查六套权重及 config.yaml |
| class_configs | 检查 classwise JSON 是否完整覆盖 0..31 |

清理陈旧字节码：

```text
Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse -Force
```

然后：

```text
python -B -m streamlit run app.py --server.port 8501 --server.headless true
```

## 8. 指标口径

杂草：

```text
mAP50 = 0.828036
mAP50-95 = 0.443757
AP75 = 0.415953
```

害虫 refined classwise WBF：

```text
mAP50 = 0.80524
mAP50-95 = 0.52410
```

全部属于本地验证集口径，不是平台测试集指标。

## 9. 原型规则

危害等级和防治建议属于原型规则，未经农学专家校准，不构成正式农业决策依据。

重度结果必须建议农技人员或植保专家复核。

系统不提供具体农药名称、剂量、浓度、施用时间或安全间隔期。

## 10. 许可证

正式提交前必须独立核验：

```text
MADA 赛题数据集许可；
六套自训练权重再分发限制；
Ultralytics；
YOLOX/Dinov3；
WBF 及其他第三方代码。
```

当前项目不声称全部许可证已经核验完成。
