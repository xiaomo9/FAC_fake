# -*- coding: utf-8 -*-
# @Time    : 2024/6/15 14:10
# @author    : XiaoMo
# @Software: PyCharm
# 大鹏一日同风起，扶摇直上九万里
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class GGCA(nn.Module):  #(Global Grouped Coordinate Attention) 全局分组坐标注意力
    def __init__(self, channel, h, w, reduction=16, num_groups=4):
        super(GGCA, self).__init__()
        self.num_groups = num_groups  # 分组数
        self.group_channels = channel // num_groups  # 每组的通道数
        self.h = h  # 高度方向的特定尺寸
        self.w = w  # 宽度方向的特定尺寸

        # 定义H方向的全局平均池化和最大池化
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))  # 输出大小为(h, 1)
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
        # 定义W方向的全局平均池化和最大池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))  # 输出大小为(1, w)
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))

        # 定义共享的卷积层，用于通道间的降维和恢复
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1))
        )
        # 定义sigmoid激活函数
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        # 确保通道数可以被分组数整除,一般分组数,要选择整数,不然不能被整除。而且是小一点.groups选择4挺好。
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        # 将输入特征图按通道数分组
        x = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        # 分别在H方向进行全局平均池化和最大池化
        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)

        # 分别在W方向进行全局平均池化和最大池化
        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用共享卷积层进行特征处理
        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, self.h, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, self.h, 1))

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, self.w))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, self.w))

        # 计算注意力权重
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, self.h, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用注意力权重
        out = x * att_h * att_w
        out = out.view(batch_size, channel, height, width)

        return out

# ====================================WTConv==============================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pywt
import pywt.data
import torch
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    # 创建一个小波对象
    w = pywt.Wavelet(wave)

    # 反转并转换小波的高通和低通滤波器到 PyTorch tensor
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # 高通滤波器
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # 低通滤波器

    # 创建小波分解滤波器
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # 低通-低通滤波器
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # 低通-高通滤波器
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # 高通-低通滤波器
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # 高通-高通滤波器
    ], dim=0)

    # 扩展滤波器通道数以匹配输入通道数
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 反转并转换小波的高通和低通滤波器到 PyTorch tensor
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  # 逆变换高通滤波器
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])  # 逆变换低通滤波器

    # 创建小波重构滤波器
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  # 低通-低通滤波器
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  # 低通-高通滤波器
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  # 高通-低通滤波器
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)  # 高通-高通滤波器
    ], dim=0)

    # 扩展滤波器通道数以匹配输出通道数
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape  # 批量大小、通道数、高度、宽度
    # 计算填充量，以保持卷积输出的尺寸
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 执行小波变换
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)  # 进行卷积操作
    # 调整形状以匹配小波变换的输出格式
    x = x.reshape(b, c, 4, h // 2, w // 2)  # 将输出重新调整为 [batch, channels, 4, h', w']
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape  # 批量大小、通道数、高度（小波变换后的）、宽度（小波变换后的）
    # 计算填充量，以保持反卷积输出的尺寸
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 将输入数据重新调整为合适的形状以进行反向卷积
    x = x.reshape(b, c * 4, h_half, w_half)  # 重新调整为 [batch, channels*4, h', w']
    # 执行小波逆变换
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)  # 进行反卷积操作
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        # 确保输入通道数和输出通道数相同
        # assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # 创建小波变换和逆变换滤波器
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 定义小波变换和逆变换函数
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 基础卷积层
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 小波卷积层和尺度模块
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        # 如果步幅大于1，定义步幅滤波器
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []  # 存储每一层的小波低频分量
        x_h_in_levels = []  # 存储每一层的小波高频分量
        shapes_in_levels = []  # 存储每一层的形状

        curr_x_ll = x  # 当前低频分量

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)  # 记录当前层的形状
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):  # 如果图像尺寸不是偶数
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)  # 计算填充量
                curr_x_ll = F.pad(curr_x_ll, curr_pads)  # 填充图像

            curr_x = self.wt_function(curr_x_ll)  # 小波变换
            curr_x_ll = curr_x[:, :, 0, :, :]  # 低频分量

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  # 变换形状
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))  # 卷积和尺度变换
            curr_x_tag = curr_x_tag.reshape(shape_x)  # 恢复原始形状

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # 保存低频分量
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 保存高频分量

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()  # 获取当前层的低频分量
            curr_x_h = x_h_in_levels.pop()  # 获取当前层的高频分量
            curr_shape = shapes_in_levels.pop()  # 获取当前层的形状

            curr_x_ll = curr_x_ll + next_x_ll  # 逐层相加

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)  # 拼接低频和高频分量
            next_x_ll = self.iwt_function(curr_x)  # 小波逆变换

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]  # 修剪形状

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))  # 基础卷积和尺度变换
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)  # 应用步幅

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)  # 初始化尺度参数
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)  # 应用尺度参数
# =========================================end==============================================

class CViT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=32),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=32),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=64),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=64),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            WTConv2d(in_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        num_patches = (7 // patch_size) ** 2
        # num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(32, 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.ggca = GGCA(512,7,7) # channel, h, w

    def forward(self, img, mask=None):
        p = self.patch_size
        # img: torch.Size([32, 3, 224, 224])
        x = self.features(img)  # torch.Size([32, 512, 7, 7])
        x1 = self.ggca(x)
        x = x + x1
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        y = self.patch_to_embedding(y) # 32, 1, 1024
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        # ======加入注意力模块==============
        # n, c, h, w = shape, 2, 32, 32
        # x = x.view(n, c, h, w)
        # x = self.ggca(x)
        # x = x.view(n, 2, 1024)
        # ================================
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])

        return self.mlp_head(x)


if __name__ == '__main__':
    x = torch.randn(32, 3, 224, 224)
    model = CViT()
    x = model(x)
    print(x.shape)

