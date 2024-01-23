import torch
import torch.nn as nn
from efficientnet_pytorch.model import MemoryEfficientSwish


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.act_block(x)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        # self.bn = nn.BatchNorm2d(normalized_shape)
        # nn.InstanceNorm2d

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        # return self.bn(x)

class EfficientAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, group_split=[4, 4], kernel_sizes=[3], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True, final_proj=False):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, out_dim, 1, 1, 0, bias=qkv_bias) if final_proj else nn.Identity()
        # self.proj = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # self.ln = LayerNorm(dim, data_format="channels_first")
        # self.ln = nn.GroupNorm(dim // 4, dim) 
        

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        # x = self.ln(x)
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))

# block = EfficientAttention(64, 128, kernel_sizes=[3]).cuda()
# input = torch.rand(2, 64, 128, 128).cuda()
# output = block(input)
# print(input.size(), output.size())
""" ResNeSt Models

Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang
Modified for torchscript compat, and consistency with timm by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import EncoderMixin
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import make_divisible
from timm.models.registry import register_model
from timm.models.resnet import ResNet

""" Split Attention Conv2d (for ResNeSt Models)

Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl at https://github.com/zhanghang1989/ResNeSt

Modified for torchscript compat, performance, and consistency with timm by Ross Wightman
"""


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
        # self.conv = 
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
        # self.gn2 = nn.GroupNorm(out_channels // radix // radix, out_channels // radix)
        
        self.conv3 = nn.Conv2d(out_channels // radix, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(out_channels // radix, out_channels)
        self.act3 = act_layer()
        
        # self.squeeze_projector = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.attn_conv = nn.Conv2d(out_channels, out_channels, 1)
        
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x):
        x = self.conv(x)  # 3x3 Conv C = mid_chs
        # print(f'x.shape = {x.shape}')
        x = self.bn0(x)  # BN
        x = self.drop(x)
        x = self.act0(x)  # ReLU
        
        B, RC, H, W = x.shape
        if self.radix > 1:
            x_gap = x.reshape((B * self.radix, -1, H, W))
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
        else:
            x_gap = x
            
        # SDM Begin

        # Replace with x Pool y Pool
        x_h = self.pool_h(x_gap)
        x_w = self.pool_w(x_gap).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        # print(f'y.shape = {y.shape}')
        y = self.fc1(y)
        # y = self.bn1(y)
        # y = self.act1(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        # print(f'x_h.shape = {x_h.shape}')
        x_w = x_w.permute(0, 1, 3, 2)
        # print(f'x_w.shape = {x_w.shape}')

        a_h = self.fc2_0(x_h).sigmoid()
        # print(f'a_h.shape = {a_h.shape}')

        a_w = self.fc2_1(x_w).sigmoid()
        # print(f'a_w.shape = {a_w.shape}')
        
        # print(f'x_gap.shape = {x_gap.shape}')

        x_attn = a_h * a_w * x_gap
        x_attn = self.att_norm(x_attn)
        # print(f'x_attn = {x_attn.shape}')
        
        # SDM End
        
        # CRSM Begin
        
        x1 = x_attn
        
        x2 = self.conv2(x_attn)
        
        x11 = self.softmax(self.agp(x1).reshape(B * self.radix, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(B * self.radix, RC // self.radix, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(B * self.radix, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(B * self.radix, RC // self.radix, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(B * self.radix, 1, H, W).sigmoid()
        # print(f'weights.shape = {weights.shape}')
        x_attn = x_attn * weights
        
        # CRSM End
        
        # CAM Begin
        
        # x2 = self.gn2(x2)
        # avg_attn = torch.mean(x2, dim=1, keepdim=True)
        # max_attn = torch.max(x2, dim=1, keepdim=True)[0]
        # aggre_attn = torch.cat([avg_attn, max_attn], dim=1)
        # aggre_attn = self.squeeze_projector(aggre_attn).sigmoid()
        # x_attn = x_attn * aggre_attn

        # CAM End
        
        x_attn = x_attn.reshape((B, -1, H, W))
        x_attn = self.attn_conv(x_attn)
        # print(f'x_attn + aggre_attn.shape = {x_attn.shape}')
        if self.radix > 1:
            x = x.sum(dim=1)
        else:
            x = x.reshape((B, RC, H, W))
            
        # print(f'x.shape = {x.shape}')
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.act3(x)

        # print(x.max(), x.min())
        # return 
        x = x * x_attn
        return x
        # x = self.col_attn(x)
        # return x + x_colattn

def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h * w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1.0 / M / M) * torch.ones(dim, dim, device=x.device) + (
        1.0 / M
    ) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


class ResNeSCBottleneck(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
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
        super(ResNeSCBottleneck, self).__init__()
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
        # self.triplet_attention = TripletAttention(planes * 4, 16)

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

        # out = self.triplet_attention(out)
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
        raise ValueError("ResNest encoders do not support dilated mode")

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


resnest_weights = {
    "timm-resnest14d-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth",  # noqa
    },
    "timm-resnest26d-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth",  # noqa
    },
    "timm-resnest50d-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",  # noqa
    },
    "timm-resnest101e-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",  # noqa
    },
    "timm-resnest200e-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",  # noqa
    },
    "timm-resnest269e-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",  # noqa
    },
    "timm-resnest50d_4s2x40d-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth",  # noqa
    },
    "timm-resnest50d_1s4x24d-sca": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in resnest_weights.items():
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

timm_resnesc_encoders = {
    "timm-resnest14d-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest14d-sca"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [1, 1, 1, 1],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 4, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest26d-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest26d-sca"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [2, 2, 2, 2],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest50d-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d-sca"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest101e-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest101e-sca"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 23, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest200e-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest200e-sca"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 24, 36, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest269e-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest269e-sca"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 30, 48, 8],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "timm-resnest50d_4s2x40d-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d_4s2x40d-sca"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 40,
            "cardinality": 2,
            "block_args": {"radix": 4, "avd": True, "avd_first": True},
        },
    },
    "timm-resnest50d_1s4x24d-sca": {
        "encoder": SegRSEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d_1s4x24d-sca"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 24,
            "cardinality": 4,
            "block_args": {"radix": 1, "avd": True, "avd_first": True},
        },
    },
}


# for k, v in timm_resnesc_encoders.items():
#     name = f"{k}-sca"
#     smp.encoders.encoders[name] = v
#     # print(f"Added Model:\t{name}")
# print(f"Added SCA Models!!!")

# if __name__ == "__main__":
#     model = smp.Unet("timm-resnest14d-sca", encoder_weights=None, in_channels=1, classes=1)
#     a = torch.randn(1, 1, 256, 256)
#     b = model(a)
#     print(b.shape)
