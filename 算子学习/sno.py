import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time


# ===============================
# 1. 核心算子定义：Spectral Operator (已修复维度问题)
# ===============================
class SpectralFilter(nn.Module):
    def __init__(self, in_channels, out_channels, modes=12):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        batch, c, h, w = x.shape
        # 使用 fft2 替代 rfft2，保持维度完整性
        x_ft = torch.fft.fft2(x)

        out_ft = torch.zeros_like(x_ft)
        # 动态取最小值，防止越界
        m_h = min(self.modes, x_ft.size(-2))
        m_w = min(self.modes, x_ft.size(-1))

        # 频域相乘
        out_ft[:, :, :m_h, :m_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m_h, :m_w],
            self.weights[:, :, :m_h, :m_w]
        )

        # 回到空间域，取实部
        out = torch.fft.ifft2(out_ft).real
        return out


# ===============================
# 2. Operator Tree Residual Block
# ===============================
class OperatorTreeBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # Local Branch
        self.local = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )

        # Global Branch
        self.proj = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or in_planes != planes else nn.Identity()

        self.global_op = SpectralFilter(planes, planes, modes=12)

        # Gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_planes, planes),
            nn.Sigmoid()
        )

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        f_local = self.local(x)
        f_global = self.global_op(self.proj(x))
        out = g * f_local + (1 - g) * f_global
        return F.relu(out + identity)


# ===============================
# 3. Backbone: Adaptive ResNet
# ===============================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ===============================
# 4. 训练与实验设置 (已修复卡死问题)
# ===============================
def train_and_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # 关键修复：num_workers 改为 0，防止 Windows 卡死
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    # 模型初始化
    model = ResNet(OperatorTreeBlock, [2, 2, 2]).to(device)

    # 统计参数
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model Params: {params:.2f}M")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()

    # 简易训练循环 (增加了进度打印)
    print("Starting Training...")
    for epoch in range(1, 11):
        model.train()
        start = time.time()
        total_loss = 0

        # 增加 batch 进度条打印
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 每 100 个 batch 打印一次
            if batch_idx % 100 == 0:
                print(f"  Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainset)}]  Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_time = time.time() - start
        print(f"\nEpoch {epoch} Complete | Avg Loss: {total_loss / len(trainloader):.4f} | Time: {epoch_time:.1f}s\n")


if __name__ == "__main__":
    train_and_eval()