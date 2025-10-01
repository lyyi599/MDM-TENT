import torch
import torch.nn as nn

__all__ = [
    "Senet",
]

class Senet(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Senet 通道加权模块
        
        参数：
            channels: 输入特征图的通道数
            reduction: 压缩的比例，决定全连接网络的隐藏层大小
        """
        super(Senet, self).__init__()

        # 使用全局平均池化（在seq_len维度上池化）
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 在序列长度方向上做平均池化
        self.fc1 = nn.Linear(channels, channels // reduction)  # 压缩
        self.fc2 = nn.Linear(channels // reduction, channels)  # 激励
        self.sigmoid = nn.Sigmoid()  # 输出通道权重

    def forward(self, x):
        # x的形状是 (batch_size, seq_len, n_channels)
        # 先调整x的形状为 (batch_size, n_channels, seq_len)
        x = x.permute(0, 2, 1)
        batch_size, channels, seq_len = x.size()

        # 先对seq_len维度进行池化（在每个通道内求平均）
        avg_pool = self.global_pool(x)
        avg_pool = avg_pool.view(batch_size, channels)

        # 通过全连接层进行压缩与激励
        x_se = self.fc1(avg_pool)
        x_se = self.fc2(x_se)
        # 经过激活函数
        attention = self.sigmoid(x_se).view(batch_size, channels, 1)  # (batch_size, channels, 1)

        x = x * attention  # 逐通道相乘
        # 还原形状为 (batch_size, seq_len, n_channels)
        x = x.permute(0, 2, 1)

        return x
