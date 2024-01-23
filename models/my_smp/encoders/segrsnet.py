import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import EncoderMixin
from timm.models.layers import make_divisible
from timm.models.resnet import ResNet

class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SegRSBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        radix=2,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        act_layer=nn.GELU,
        norm_layer=None,
        drop_layer=None,
        **kwargs,
    ):
        super(SegRSBlock, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor
            )
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs // radix,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs,
        )
        self.bn0 = norm_layer(mid_chs // radix) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels // radix, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2_0 = nn.Conv2d(attn_chs, out_channels // radix, 1, groups=groups)
        self.fc2_1 = nn.Conv2d(attn_chs, out_channels // radix , 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.att_norm = nn.GroupNorm(out_channels // radix // radix, out_channels // radix)
        
        self.conv2 = nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(out_channels // radix // radix, out_channels // radix)
        
        self.conv3 = nn.Conv2d(out_channels // radix, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(out_channels // radix, out_channels)
        self.act3 = act_layer()
        
        self.squeeze_projector = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.attn_conv = nn.Conv2d(out_channels, out_channels, 1)
        
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)
        
        B, RC, H, W = x.shape
        if self.radix > 1:
            x_gap = x.reshape((B * self.radix, -1, H, W))
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
        else:
            x_gap = x
            
        x_h = self.pool_h(x_gap)
        x_w = self.pool_w(x_gap).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.fc1(y)
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.fc2_0(x_h).sigmoid()
        a_w = self.fc2_1(x_w).sigmoid()
        x_attn = a_h * a_w * x_gap
        x_attn = self.att_norm(x_attn)
        
        x1 = x_attn
        x2 = self.conv2(x_attn)
        x11 = self.softmax(self.agp(x1).reshape(B * self.radix, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(B * self.radix, RC // self.radix, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(B * self.radix, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(B * self.radix, RC // self.radix, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(B * self.radix, 1, H, W).sigmoid()
        x_attn = x_attn * weights
        
        # x2 = self.gn2(x2)
        avg_attn = torch.mean(x2, dim=1, keepdim=True)
        max_attn = torch.max(x2, dim=1, keepdim=True)[0]
        aggre_attn = torch.cat([avg_attn, max_attn], dim=1)
        aggre_attn = self.squeeze_projector(aggre_attn).sigmoid()
        
        x_attn = x_attn * aggre_attn
        x_attn = x_attn.reshape((B, -1, H, W))
        x_attn = self.attn_conv(x_attn)
        if self.radix > 1:
            x = x.sum(dim=1)
        else:
            x = x.reshape((B, RC, H, W))
            
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.act3(x)

        x = x * x_attn
        return x

class SegRSNetBottleneck(nn.Module):
    """SegRSNet Bottleneck"""
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=4,
        cardinality=1,
        base_width=64,
        avd=False,
        avd_first=False,
        is_first=False,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        att_dim=128,
        reduction=16,
    ):
        super(SegRSNetBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.0)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(group_width // radix, group_width)
        self.act1 = act_layer()
        self.avd_first = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and avd_first
            else None
        )

        if self.radix >= 1:
            self.conv2 = SegRSBlock(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                radix=radix,
                norm_layer=norm_layer,
                drop_layer=drop_block,
            )
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                bias=False,
            )
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer()
        self.avd_last = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and not avd_first
            else None
        )

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_o = out + shortcut
        out_o = self.act3(out_o)
        return out_o

class SegRSEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def make_dilated(self, *args, **kwargs):
        raise ValueError("SegRSNet encoders do not support dilated mode")

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


segrsnet_weights = {
    "segrsnet-14": {
        "whu":          "https://drive.google.com/file/d/1uNvSUg75br11A7_x674MVw205cveyzjb",
        "mass":         "https://drive.google.com/file/d/1GjSaPFex555CNEhxI0PR2INU_8VsOkSG",
        "deepglobe":    "https://drive.google.com/file/d/1h-af-E7HcdBpQ9S9N5E15Ni2NZ9RN_QB"
    },
    "segrsnet-26": {
        "whu":          "https://drive.google.com/file/d/1pGzS_AfLpfll7kHha72W9_F0ofUgsvdJ",
        "mass":         "https://drive.google.com/file/d/1Mniq7KGypE6A61G9nj9AVOjkgtJaFy5H",
        "deepglobe":    "https://drive.google.com/file/d/15ZVpQfWFoIvPByaXvRXPmmmq9JSv_7KC"
    },
    "segrsnet-50": {
        "whu":          "https://drive.google.com/file/d/1ZoF-8ev5koC5Xx6oAOwFrH6Qe26Sucpk",
        "mass":         "https://drive.google.com/file/d/17ZR1uGtRHWHw3Wu4TcSuFaRtQ4Dqfneo",
        "deepglobe":    "https://drive.google.com/file/d/1HNpELPRqbT2MdCVM7wxZ3N7sGph7wshh"
    },
    "segrsnet-101": {
        "whu":          "https://drive.google.com/file/d/10ld50r1ZE-xjbaidQt_08Vtl6pIvh1m8",
        "mass":         "https://drive.google.com/file/d/1ca4rvrZd8txzq12CX-5pH0c7KSCduANd",
    },
}

segrsnet_resolutions = {
    "segrsnet-14": {
        "whu":          256,
        "mass":         256,
        "deepglobe":    1024
    },
    "segrsnet-26": {
        "whu":          256,
        "mass":         384,
        "deepglobe":    1024
    },
    "segrsnet-50": {
        "whu":          384,
        "mass":         384,
        "deepglobe":    1024
    },
    "segrsnet-101": {
        "whu":          384,
        "mass":         384,
    },
}

pretrained_settings = {}
for model_name, sources in segrsnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

segrsnet_encoders = {
    "segrsnet-14": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["segrsnet-14"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SegRSNetBottleneck,
            "layers": [1, 1, 1, 1],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 4, "avd": True, "avd_first": False},
        },
    },
    "segrsnet-26": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["segrsnet-26"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SegRSNetBottleneck,
            "layers": [2, 2, 2, 2],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "segrsnet-50": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["segrsnet-50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SegRSNetBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "segrsnet-101": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["segrsnet-101"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": SegRSNetBottleneck,
            "layers": [3, 4, 23, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
}