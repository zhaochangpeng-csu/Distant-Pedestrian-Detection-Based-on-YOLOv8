"""C2f 的通道注意力扩展；保留 cv1/cv2/m 名称以加载相同预训练骨干。"""
from . import runtime
import torch
from torch import nn
from ultralytics.nn.modules.block import C2f


class ECA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # B,C,H,W -> B,1,C；只沿通道维进行局部交互。
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).unsqueeze(-1).sigmoid()
        return x * y


class C2fECA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.eca = ECA(3)

    def forward(self, x):
        return self.eca(super().forward(x))

    def forward_split(self, x):
        return self.eca(super().forward_split(x))
