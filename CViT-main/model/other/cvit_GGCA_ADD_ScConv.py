import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math
from torchsummary import summary
import torch.autograd

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


# ----------------------------------------GGCA------------------------------------------------------
class GGCA(nn.Module):  # (Global Grouped Coordinate Attention) 全局分组坐标注意力
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


# -----------------------------------------end----------------------------------------------------------
# -----------------------------------------ODConv----------------------------------------------------------

class GroupBatchnorm2d(nn.Module):
    """
    实现了分组批量归一化的2D层
    """

    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        """
        初始化GroupBatchnorm2d层
        :param c_num: 输入通道数
        :param group_num: 分组数
        :param eps: 防止除零的极小值
        """
        super(GroupBatchnorm2d, self).__init__()  # 调用父类的初始化方法
        assert c_num >= group_num  # 确保输入通道数大于等于分组数
        self.group_num = group_num  # 保存分组数
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))  # 偏置参数
        self.eps = eps  # 保存防止除零的极小值

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (N, C, H, W)
        :return: 归一化后的输出
        """
        N, C, H, W = x.size()  # 获取输入张量的维度
        x = x.view(N, self.group_num, -1)  # 重新调整形状以进行分组归一化
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        return x * self.weight + self.bias  # 应用权重和偏置


class SRU(nn.Module):
    """
    SRU模块（带有分组归一化和门控机制）
    """

    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        """
        初始化SRU模块
        :param oup_channels: 输出通道数
        :param group_num: 分组数
        :param gate_treshold: 门控阈值
        :param torch_gn: 是否使用PyTorch内置的GroupNorm
        """
        super().__init__()  # 调用父类的初始化方法

        # 根据参数选择使用GroupNorm还是自定义的GroupBatchnorm2d
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 保存门控阈值
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid函数

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (N, C, H, W)
        :return: 经过SRU模块处理后的输出
        """
        gn_x = self.gn(x)  # 通过归一化层
        w_gamma = self.gn.weight / sum(self.gn.weight)  # 计算权重的归一化因子
        w_gamma = w_gamma.view(1, -1, 1, 1)  # 调整形状
        reweights = self.sigmoid(gn_x * w_gamma)  # 计算重新加权的值

        # 门控操作
        w1 = torch.where(reweights > self.gate_treshold, torch.ones_like(reweights), reweights)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweights > self.gate_treshold, torch.zeros_like(reweights), reweights)  # 大于门限值的设为0，否则保留原值

        x_1 = w1 * x  # 应用第一个权重
        x_2 = w2 * x  # 应用第二个权重
        y = self.reconstruct(x_1, x_2)  # 重建输出
        return y

    def reconstruct(self, x_1, x_2):
        """
        重建输出
        :param x_1: 第一部分的输入
        :param x_2: 第二部分的输入
        :return: 经过重建后的输出
        """
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 将x_1在通道维度上分成两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 将x_2在通道维度上分成两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 合并两部分的结果


class CRU(nn.Module):
    """
    CRU模块（通道重分配模块）
    """

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        """
        初始化CRU模块
        :param op_channel: 操作通道数
        :param alpha: 上升通道比例
        :param squeeze_radio: 压缩比
        :param group_size: 分组大小
        :param group_kernel_size: 分组卷积核大小
        """
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)  # 上升通道数
        self.low_channel = low_channel = op_channel - up_channel  # 下降通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 上升通道的压缩卷积
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 下降通道的压缩卷积

        # 上升通道的卷积
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

        # 下降通道的卷积
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (N, C, H, W)
        :return: 经过CRU模块处理后的输出
        """
        # 分割
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)  # 按通道分割
        up, low = self.squeeze1(up), self.squeeze2(low)  # 压缩

        # 转换
        Y1 = self.GWC(up) + self.PWC1(up)  # 上升通道的卷积结果
        Y2 = torch.cat([self.PWC2(low), low], dim=1)  # 下降通道的卷积结果与原始低通道的连接

        # 融合
        out = torch.cat([Y1, Y2], dim=1)  # 连接上升通道和下降通道的结果
        out = F.softmax(self.advavg(out), dim=1) * out  # 自适应池化后进行Softmax操作，然后与结果相乘
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)  # 分割结果
        return out1 + out2  # 合并并返回


class ScConv(nn.Module):
    """
    ScConv模块（组合SRU和CRU）
    """

    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        """
        初始化ScConv模块
        :param op_channel: 操作通道数
        :param group_num: 分组数
        :param gate_treshold: 门控阈值
        :param alpha: 上升通道比例
        :param squeeze_radio: 压缩比
        :param group_size: 分组大小
        :param group_kernel_size: 分组卷积核大小
        """
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)  # 初始化SRU模块
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)  # 初始化CRU模块

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (N, C, H, W)
        :return: 经过ScConv模块处理后的输出
        """
        x = self.SRU(x)  # 通过SRU模块
        x = self.CRU(x)  # 通过CRU模块
        return x  # 返回输出


# ---------------------------------------------------------end----------------------------------------------


class CViT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            ScConv(64),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            ScConv(128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            ScConv(256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            ScConv(256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features2 = nn.Sequential(

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
        self.ggca = GGCA(512, 7, 7)  # channel, h, w

    def forward(self, img, mask=None):
        p = self.patch_size
        # img: torch.Size([32, 3, 224, 224])
        x = self.features1(img)  # torch.Size([32, 32, 112, 112])
        x = self.features2(x)
        x1 = self.ggca(x)
        x = x + x1
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        y = self.patch_to_embedding(y)  # 32, 1, 1024
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        # ======加入注意力模块==============
        # n, c, h, w = shape, 2, 32, 32
        # x = x.view(n, c, h, w)
        # x = self.mdfa(x)
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
#
