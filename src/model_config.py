"""
== 来源记录（Provenance）==
本文件由任务 07（07_Web识别系统）从任务 05 目录原样复制：
  任务05_异构WBF融合精度优化/model_config.py（项目内部相对来源；公开版已移除本机路径）
复制后仅新增本段头部注释；模型结构、build_model 接口与其余代码逐字节一致
（复制时 SHA-256 比对一致，QA 报告中复核）。
用途：Web 系统加载 YOLOX-Dinov3 Small / Base 权重，保持与任务 02/03/05 推理管线一致。
== 来源记录结束 ==

== 原文件 docstring 如下 ==
model_config.py (task 05 - WBF fusion experiment)
=================================================
YOLOX-Dinov3 **结构忠实重建**，支持两种规格：Small 与 Base。

来源（与任务 02 相同，逐行等价于 MADA 云端导出的 YOLOX-Dinov3 源码包
（公开版已移除本机导出路径与导出包文件名记录；平台导出包许可边界见 THIRD_PARTY_NOTICES.md），提取于任务 02 的
`src_reference/`）：

- `yolox/models/yolo_pafpn_dinvo3.py`（`class YOLOPAFPNDinoV3`）
- `yolox/models/network_blocks.py`（BaseConv / DWConv / Bottleneck / CSPLayer）
- `yolox/models/yolo_head_dinov3.py`（`class YOLOXHeadDinoV3`）
- `yolox/models/yolox.py`（`class YOLOX`）
- `dinov3/models/convnext.py`（`class ConvNeXt`）
- `dinov3/hub/backbones.py`（`dinov3_convnext_small` / `dinov3_convnext_base`）
- `yolox/data/data_augment.py`（ValTransform/preproc 输入为 BGR 0-255 / CHW / 左上角 114 padding）
- `yolox/utils/boxes.py`（postprocess = torchvision batched_nms）
- `yolox/evaluators/coco_evaluator.py`（bboxes /= scale，不裁剪）

本文件在任务 03 的 Small 重建基础上，把 ConvNeXt 的 dims、FPN 的 depth、head 的 in_channels
参数化，以支持 Small 与 Base 两种规格，用于异构 WBF 融合中对 YOLOX-Dinov3 Base 权重的本地推理。

规格差异（来自任务 04 的权重结构核验）：
- Small：ConvNeXt dims=[96,192,384,768]，FPN depth=0.67，head in_channels=[192,384,768]，inner_channel=192
- Base ：ConvNeXt dims=[128,256,512,1024]，FPN depth=1.0，head in_channels=[256,512,1024]，inner_channel=256
两规格深度均为 depth=[3,3,27,3]。

注意：本文件只做结构重建与前向，不实现 WBF；不重训模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# ConvNeXt 用 LayerNorm（channels_last / channels_first）
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - u) / torch.sqrt(s + self.eps) + self.bias


# ----------------------------------------------------------------------------
# YOLOX 网络块（network_blocks.py 等价）
# ----------------------------------------------------------------------------
class BaseConv(nn.Module):
    def __init__(self, cin, cout, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(cin, cout, ksize, stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act == 'silu' else nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, cin, cout, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(cin, cin, ksize, stride, groups=cin, act=act)
        self.pconv = BaseConv(cin, cout, 1, 1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden = int(cout * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(cin, hidden, 1, 1, act=act)
        self.conv2 = Conv(hidden, cout, 3, 1, act=act)
        self.use_add = shortcut and cin == cout

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = BaseConv(cin, hidden, 1, 1, act=act)
        self.conv2 = BaseConv(cin, hidden, 1, 1, act=act)
        self.conv3 = BaseConv(2 * hidden, cout, 1, 1, act=act)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, 1.0, depthwise, act) for _ in range(n)])

    def forward(self, x):
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), 1))


# ----------------------------------------------------------------------------
# ConvNeXt Block（dinov3/models/convnext.py Block 等价）
# ----------------------------------------------------------------------------
class CNBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)   # NHWC -> NCHW
        return inp + x


class ConvNeXt(nn.Module):
    """ConvNeXt（DINOv3），支持不同 dims/depths。Small=([3,3,27,3],[96,192,384,768])；
    Base=([3,3,27,3],[128,256,512,1024])。forward 返回 4 个 stage 特征，末级用 self.norm 归一化。"""

    def __init__(self, in_chans=3, depths=(3, 3, 27, 3), dims=(96, 192, 384, 768)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first'),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        self.stages = nn.ModuleList([
            nn.Sequential(*[CNBlock(dims[i]) for _ in range(depths[i])]) for i in range(4)
        ])
        self.embed_dim = dims[-1]
        self.embed_dims = dims
        self.n_blocks = 4
        self.norm = LayerNorm(dims[-1], eps=1e-6)
        self.norms = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity(), self.norm])

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        feats = []
        for i, feat in enumerate(outs):
            if isinstance(self.norms[i], nn.Identity):
                feats.append(feat)
            else:
                B, C, H, W = feat.shape
                patches = feat.flatten(2).transpose(1, 2)
                patches = self.norms[i](patches)
                feat = patches.transpose(1, 2).reshape(B, C, H, W).contiguous()
                feats.append(feat)
        return tuple(feats)


# ----------------------------------------------------------------------------
# YOLOPAFPNDinoV3（yolo_pafpn_dinvo3.yolo_pafpn_dinvo3 等价，可调 depth）
# ----------------------------------------------------------------------------
class YOLOPAFPNDinoV3(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), depths=(3, 3, 27, 3),
                 depth=0.67, width=1.0, act='silu'):
        super().__init__()
        self.backbone_dinov3 = ConvNeXt(depths=depths, dims=dims)
        in_channels = self.backbone_dinov3.embed_dims[1:]      # 去掉 stage0
        self.in_channels = in_channels
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1), persistent=False)

        Conv = BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, act=act)
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, act=act)
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, act=act)
        self.bu_conv_extra = nn.Identity()
        self.global_attention = nn.ModuleList([nn.Identity() for _ in range(3)])

    def forward(self, x):
        with torch.no_grad():
            x = (x[:, [2, 1, 0]].contiguous() - self.pixel_mean) / self.pixel_std
        x3, x2, x1, x0 = self.backbone_dinov3(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)
        return pan_out2, pan_out1, pan_out0


# ----------------------------------------------------------------------------
# YOLOXHeadDinoV3（yolo_head_dinov3.yolo_head_dinov3 等价，in_channels 可调）
# ----------------------------------------------------------------------------
class YOLOXHeadDinoV3(nn.Module):
    def __init__(self, num_classes=1, strides=(8, 16, 32), in_channels=(192, 384, 768), act='silu'):
        super().__init__()
        inner_channel = max(in_channels[0], in_channels[-1] // 4)
        self.num_classes = num_classes
        self.strides = list(strides)
        self.cls_convs, self.reg_convs = nn.ModuleList(), nn.ModuleList()
        self.cls_preds, self.reg_preds, self.obj_preds, self.stems = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels[i], inner_channel, 1, 1, act=act))
            self.cls_convs.append(nn.Sequential(BaseConv(inner_channel, inner_channel, 3, 1, act=act),
                                                BaseConv(inner_channel, inner_channel, 3, 1, act=act)))
            self.reg_convs.append(nn.Sequential(BaseConv(inner_channel, inner_channel, 3, 1, act=act),
                                                BaseConv(inner_channel, inner_channel, 3, 1, act=act)))
            self.cls_preds.append(nn.Conv2d(inner_channel, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(inner_channel, 4, 1))
            self.obj_preds.append(nn.Conv2d(inner_channel, 1, 1))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_x, reg_x = x, x
            cls_output = self.cls_preds[k](self.cls_convs[k](cls_x))
            reg_output = self.reg_preds[k](self.reg_convs[k](reg_x))
            obj_output = self.obj_preds[k](self.reg_convs[k](reg_x))
            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)
        return outputs


class YOLOXDinoV3(nn.Module):
    def __init__(self, size='small'):
        super().__init__()
        spec = MODEL_SPECS[size]
        self.size = size
        self.backbone = YOLOPAFPNDinoV3(dims=spec['dims'], depths=(3, 3, 27, 3),
                                        depth=spec['fpn_depth'])
        self.head = YOLOXHeadDinoV3(num_classes=1, in_channels=spec['head_in_channels'])

    def forward(self, x):
        fpn_outs = self.backbone(x)
        return self.head(fpn_outs)


# 规格表
MODEL_SPECS = {
    'small': {
        'dims': (96, 192, 384, 768),
        'fpn_depth': 0.67,
        'head_in_channels': (192, 384, 768),
        'inner_channel': 192,
        'params_expected': None,   # 由任务 02 核验为 Small（62.38M 量级）
        'source': '任务 02/03 一致的 YOLOX-Dinov3 Small',
    },
    'base': {
        'dims': (128, 256, 512, 1024),
        'fpn_depth': 1.0,
        'head_in_channels': (256, 512, 1024),
        'inner_channel': 256,
        'params_expected': 114638482,
        'source': '任务 04 核验：ConvNeXt-Base + depth=1.0 + inner_channel=256',
    },
}


def build_model(weights_path=None, map_location='cpu', size='small'):
    model = YOLOXDinoV3(size=size)
    meta = {'size': size, 'spec': MODEL_SPECS[size]}
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location=map_location, weights_only=False)
        meta['top_keys'] = list(ckpt.keys())
        meta['start_epoch'] = ckpt.get('start_epoch')
        meta['best_ap'] = ckpt.get('best_ap')
        meta['curr_ap'] = ckpt.get('curr_ap')
        sd = ckpt['model']
        res = model.load_state_dict(sd, strict=True)
        meta['load_state_dict'] = str(res)
        meta['n_rebuilt_params'] = sum(p.numel() for p in model.parameters())
        return model, meta
    return model, meta
