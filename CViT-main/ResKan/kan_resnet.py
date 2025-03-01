import sys
from kan import KAN

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class BasicBlock(nn.Module):
    expansion = 1  # 残差块的扩展系数，用于计算输出通道数与输入通道数的比例
 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        初始化函数，用于创建BasicBlock对象。

        参数:
            in_channel (int): 输入通道数，即输入特征图的深度。
            out_channel (int): 输出通道数，即输出特征图的深度。
            stride (int, optional): 卷积步长，默认为1。
            downsample (nn.Sequential, optional): 下采样层，用于调整输入特征图与输出特征图之间的维度匹配。默认为None。
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层：3x3卷积，stride用于控制卷积步长，padding为1保持特征图大小不变，bias设为False表示不使用偏置
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 批归一化层，对卷积输出进行批归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        # 第二个卷积层，同样是3x3卷积，但不改变特征图大小
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样层，用于调整输入特征图与输出特征图之间的维度匹配
        self.downsample = downsample
 
    def forward(self, x):
        """
        前向传播函数，定义了BasicBlock的前向计算过程。

        参数:
            x (torch.Tensor): 输入特征图，形状为(batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 输出特征图，形状为(batch_size, out_channel, H, W)。
        """
        # 将输入保存为恒等映射，用于后续的残差连接
        identity = x
        # 如果有下采样层，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        # 第一个卷积层：卷积 + 批归一化 + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二个卷积层：卷积 + 批归一化
        out = self.conv2(out)
        out = self.bn2(out)
        # 将卷积输出与输入特征图进行残差连接
        out += identity
        # 经过ReLU激活函数
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4 # 残差块的扩展系数，用于计算输出通道数与输入通道数的比例
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        """
        初始化函数，用于创建Bottleneck对象。

        参数:
            in_channel (int): 输入通道数，即输入特征图的深度。
            out_channel (int): 输出通道数，即输出特征图的深度。
            stride (int, optional): 卷积步长，默认为1。
            downsample (nn.Sequential, optional): 下采样层，用于调整输入特征图与输出特征图之间的维度匹配。默认为None。
            groups (int, optional): 分组卷积的组数，默认为1。
            width_per_group (int, optional): 分组卷积的每组通道数，默认为64。
        """
        super(Bottleneck, self).__init__()
        # 计算Bottleneck中间卷积层的输出通道数
        width = int(out_channel * (width_per_group / 64.)) * groups
        # 第一个卷积层：1x1卷积，用于减少通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # 第二个卷积层：3x3卷积，使用分组卷积减少计算量
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # 第三个卷积层：1x1卷积，用于恢复通道数，并进行扩展
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True) # 非线性激活函数ReLU
        self.downsample = downsample  # 下采样层，用于调整输入特征图与输出特征图之间的维度匹配
 
    def forward(self, x):
        """
        前向传播函数，定义了Bottleneck的前向计算过程。

        参数:
            x (torch.Tensor): 输入特征图，形状为(batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 输出特征图，形状为(batch_size, out_channel * expansion, H, W)。
        """
        identity = x # 将输入保存为恒等映射，用于后续的残差连接
        if self.downsample is not None: # 如果有下采样层，则对输入进行下采样
            identity = self.downsample(x)
        # 第一个卷积层：1x1卷积 + 批归一化 + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二个卷积层：3x3卷积（分组卷积） + 批归一化 + ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 第三个卷积层：1x1卷积 + 批归一化
        out = self.conv3(out)
        out = self.bn3(out)
        # 将卷积输出与输入特征图进行残差连接
        out += identity
        out = self.relu(out) # 经过ReLU激活函数
 
        return out
 
 
class ResNet(nn.Module):
 
    def __init__(self,
                 block,
                 blocks_num,
                 set_device: None,       # Set_device“是为 Kan 配置训练设备。此参数需要在训练期间设置。
                 num_classes=1000,
                 include_top=False,      # 如果要使用标准 ResNet 进行分类，请将其设置为 True。
                 include_top_kan = True, # 如果要使用 ResNet+KAN 进行分类，请将其设置为 True。
                 groups=1,
                 width_per_group=64):
        """
        初始化函数，用于创建 ResNet 网络。

        参数:
            block (nn.Module): 残差块类型，可以是 BasicBlock 或 Bottleneck。
            blocks_num (list): 包含每个阶段的残差块数量的列表。
            set_device (None): 用于配置训练设备的参数。这个参数需要在训练过程中设置。
            num_classes (int, optional): 分类任务的类别数量，默认为 1000。
            include_top (bool, optional): 如果想要使用标准的 ResNet 进行分类，请设置为 True。默认为 False。
            include_top_kan (bool, optional): 如果想要使用 ResNet+KAN 进行分类，请设置为 True。默认为 True。
            groups (int, optional): 分组卷积的组数，默认为 1。
            width_per_group (int, optional): 分组卷积的每组通道数，默认为 64。
        """
        super().__init__()
        # 设置是否包含顶层分类器和是否包含 KAN
        self.include_top = include_top
        self.include_top_kan = include_top_kan
        self.in_channel = 64 # 输入通道数
        # 分组卷积的参数
        self.groups = groups
        self.width_per_group = width_per_group
         # 第一个卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 创建每个阶段的残差块序列
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # 如果包含顶层分类器，则创建全局平均池化层和线性分类器
        if self.include_top: 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 如果包含 KAN，则创建 KAN 层
        if self.include_top_kan:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # torch.Size([2, 512, 1, 1])
            # self.linear = nn.Linear(512,64* block.expansion)
            self.kan = KAN([512 * block.expansion, 64, num_classes])
        # 初始化所有卷积层的参数        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    def _make_layer(self, block, channel, block_num, stride=1):
        """
        创建一个残差块序列。

        参数:
            block (nn.Module): 残差块类型，可以是 BasicBlock 或 Bottleneck。
            channel (int): 残差块中间卷积层的输出通道数。
            block_num (int): 残差块数量。
            stride (int, optional): 第一个残差块的步长，默认为 1。

        返回:
            nn.Sequential: 包含多个残差块的序列。
        """
        downsample = None
        # 如果步长不为 1 或输入通道数不等于输出通道数乘以残差块扩展系数，则添加下采样层
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        # 创建残差块序列
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 创建多个残差块
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        """
        定义了 ResNet 网络的前向传播过程。

        参数:
            x (torch.Tensor): 输入数据，形状为(batch_size, in_channel, H, W)。

        返回:
            torch.Tensor: 输出特征图或分类结果。
        """
        # 第一层卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 多个残差块序列
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 如果包含顶层分类器，进行全局平均池化和线性分类
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        # 如果包含 KAN，进行全局平均池化和 KAN 操作
        if self.include_top_kan:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            # x = self.linear(x)
            x = self.kan(x)
        return x
 
def resnet34(set_device,num_classes=1000, include_top=False,include_top_kan=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], set_device=set_device,num_classes=num_classes, include_top=include_top,include_top_kan=include_top_kan)
