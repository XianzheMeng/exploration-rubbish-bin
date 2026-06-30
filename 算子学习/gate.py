import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ===============================
# 1. 核心专家：Spectral Expert
# ===============================
class SpectralExpert(nn.Module):
    def __init__(self, channels, modes=8):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(
            torch.randn(channels, channels, modes, modes, dtype=torch.cfloat) * 0.02
        )

    def forward(self, x):
        batch, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)

        m_h = min(self.modes, x_ft.size(-2))
        m_w = min(self.modes, x_ft.size(-1))

        out_ft[:, :, :m_h, :m_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m_h, :m_w],
            self.weights[:, :, :m_h, :m_w]
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


# ===============================
# 2. 核心模块：Operator Tree Block (返回 Gate 值用于 Loss)
# ===============================
class OperatorTreeBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.local_expert = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.proj = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or in_planes != planes else nn.Identity()
        self.global_expert = SpectralExpert(planes)

        # 修改点 1: 增强 Gate 的表达能力
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_planes, in_planes // 2),  # 增加隐藏层
            nn.ReLU(),
            nn.Linear(in_planes // 2, 1),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        g_raw = self.gate(x)
        g = g_raw.unsqueeze(-1).unsqueeze(-1)

        out_l = self.local_expert(x)
        out_g = self.global_expert(self.proj(x))

        out = g * out_l + (1 - g) * out_g
        return F.relu(out + self.shortcut(x)), g_raw.mean()  # 返回 Gate 值


# ===============================
# 3. 架构组装 (记录所有 Layer 的 Gate)
# ===============================
class OpMoENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = OperatorTreeBlock(32, 64, stride=2)
        self.layer2 = OperatorTreeBlock(64, 128, stride=2)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.stem(x)
        x, g1 = self.layer1(x)
        x, g2 = self.layer2(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.classifier(x), (g1 + g2) / 2  # 返回分类结果 + 平均 Gate


# ===============================
# 4. 工具函数
# ===============================
def apply_mask(images):
    mask = torch.ones_like(images)
    mask[:, :, 12:16, :] = 0  # 横向切断
    return images * mask


# ===============================
# 5. 强化训练版主函数
# ===============================
def train_and_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[系统] 使用设备: {device}")

    print("[数据] 正在加载 MNIST...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=1000, shuffle=False, num_workers=0)

    model = OpMoENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    print("\n" + "=" * 40)
    print("开始强化训练 (带对比 Gate Loss)")
    print("=" * 40)

    # 训练循环
    for epoch in range(3):  # 稍微多训一点让它学会
        model.train()
        total_ce = 0
        total_gate_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 构造配对数据: 正常图 + 断裂图
            data_normal = data
            data_broken = apply_mask(data)

            optimizer.zero_grad()

            # 修改点 2: 前向传播两次
            output_normal, gate_normal = model(data_normal)
            output_broken, gate_broken = model(data_broken)

            # Loss 1: 标准分类损失 (两个都要算，保证断裂图也能认对)
            loss_ce = ce_loss(output_normal, target) + ce_loss(output_broken, target)

            # 修改点 3: 核心对比损失 (The Magic)
            # 逻辑：我们希望 gate_normal > gate_broken
            # 如果 gate_broken 大于等于 gate_normal，就会产生 Loss
            loss_gate = F.relu(gate_broken - gate_normal + 0.1).mean()
            # +0.1 是 margin，希望它们至少拉开 0.1 的差距

            # 总 Loss
            loss = loss_ce + 0.5 * loss_gate

            loss.backward()
            optimizer.step()

            total_ce += loss_ce.item()
            total_gate_loss += loss_gate.item()

            if batch_idx % 300 == 0:
                print(
                    f"[Epoch {epoch + 1}] Batch {batch_idx:4d} | CE_Loss: {loss_ce.item():.4f} | Gate_Loss: {loss_gate.item():.4f}")
                print(
                    f"           统计 -> 正常图 Gate: {gate_normal.item():.4f} | 断裂图 Gate: {gate_broken.item():.4f}")

        print(
            f"\n[Epoch {epoch + 1} 结束] 平均 CE: {total_ce / len(train_loader):.4f} | 平均 Gate Loss: {total_gate_loss / len(train_loader):.4f}\n")

    # 最终验证
    print("=" * 40)
    print("开始最终验证")
    print("=" * 40)

    model.eval()
    # 取一个 Batch 来可视化
    test_iter = iter(test_loader)
    samples, labels = next(test_iter)

    # 挑一个样本 (比如第 5 个)
    idx = 5
    img_normal = samples[idx:idx + 1].to(device)
    img_broken = apply_mask(img_normal)

    with torch.no_grad():
        _, g_normal = model(img_normal)
        _, g_broken = model(img_broken)

    # 可视化
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_normal[0, 0].cpu(), cmap='gray')
    ax[0].set_title(f"Normal\nCNN Gate: {g_normal.item():.3f}\nSpectral: {1 - g_normal.item():.3f}")

    ax[1].imshow(img_broken[0, 0].cpu(), cmap='gray')
    ax[1].set_title(f"Broken\nCNN Gate: {g_broken.item():.3f}\nSpectral: {1 - g_broken.item():.3f}")

    print("\n[最终结果]")
    print(f"正常图像 -> CNN 权重: {g_normal.item():.4f} | 算子权重: {1 - g_normal.item():.4f}")
    print(f"断裂图像 -> CNN 权重: {g_broken.item():.4f} | 算子权重: {1 - g_broken.item():.4f}")

    if g_broken.item() < g_normal.item():
        print("\n🎉 成功！论证成立！模型学会了根据图像质量动态分配专家！")
    else:
        print("\n❌ 效果仍不明显，建议增加训练 Epoch 或调大 Gate Loss 的系数。")

    plt.show()


if __name__ == "__main__":
    train_and_verify()