<!-- test_images/README.md -->
# 真实验收图片

本目录只放用户本地已有的真实图片，不生成模拟图片。

推荐：

```text
test_images/
├─ weed/
│  ├─ DJI_weed_0001.jpg        ← 真实杂草田间图（DJI 航拍）
│  └─ ...
└─ pest/
   ├─ pest24_0000002.jpg      ← 真实害虫图（赛题二 test_q2）
   └─ ...
```

> 注意：Windows 上 OpenCV 无法读取中文文件名的图片，
> 本目录内文件名请使用 ASCII 字符（字母/数字/下划线）。

## 杂草图片

使用任务 02 验证集中的真实 DJI_...jpg 类图片。

建议至少准备：

```text
1 张正常杂草田间图片；
1 张备用图片。
```

## 害虫图片

使用赛题二 test_q2 中的真实害虫图片。

建议至少准备：

```text
1 张正常害虫图片；
1 张备用图片。
```

不要把静态截图或人工绘制框的图片作为真实性验收样本。

## CLI 冒烟

示例：

```text
python -B scripts/verify_deployment.py `
  --weed-image "test_images/weed/DJI_weed_0001.jpg" `
  --pest-image "test_images/pest/pest24_0000002.jpg"
```

图片路径必须替换为实际存在的本地文件。