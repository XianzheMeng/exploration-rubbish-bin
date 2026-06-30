import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 修复：必须加上这一行
import torch.fft
import matplotlib.pyplot as plt
import numpy as np


# ===============================
# 1. 强化版算子专家
# ===============================
class SpectralExpert(nn.Module):
    def __init__(self, channels, modes=12):
        super().__init__()
        self.modes = modes
        self.scale = 1 / (channels ** 2)
        self.weights = nn.Parameter(self.scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        m_h, m_w = min(self.modes, x_ft.size(-2)), min(self.modes, x_ft.size(-1))

        out_ft[:, :, :m_h, :m_w] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m_h, :m_w],
                                                self.weights[:, :, :m_h, :m_w])
        return torch.fft.irfft2(out_ft, s=(H, W))


# ===============================
# 2. 深度 HO-MoE 模块
# ===============================
class HOMoE_Block(nn.Module):
    def __init__(self, dim, mode_type='hybrid'):
        super().__init__()
        self.mode_type = mode_type
        self.cnn = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1))
        self.spectral = SpectralExpert(dim)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, 1), nn.Sigmoid())
        # 注意：这里 LayerNorm 硬编码了 64x64，如果输入尺寸变了要改
        self.norm = nn.LayerNorm([dim, 64, 64])

    def forward(self, x):
        if self.mode_type == 'cnn_only':
            return x + self.cnn(x)
        if self.mode_type == 'fno_only':
            return x + self.spectral(x)

        g = self.gate(x).view(-1, 1, 1, 1)
        res = g * self.cnn(x) + (1 - g) * self.spectral(x)
        return x + res


# ===============================
# 3. 实验控制器
# ===============================
class PDESolver(nn.Module):
    def __init__(self, mode='hybrid', depth=4):
        super().__init__()
        self.lift = nn.Conv2d(10, 64, 1)
        self.blocks = nn.ModuleList([HOMoE_Block(64, mode_type=mode) for _ in range(depth)])
        self.project = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


# ===============================
# 4. 自动化对比实验脚本
# ===============================
def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    modes = ['cnn_only', 'fno_only', 'hybrid']
    results = {}

    # 模拟数据
    x_test = torch.randn(1, 10, 64, 64).to(device)
    y_test = torch.randn(1, 1, 64, 64).to(device)

    for m in modes:
        print(f"\n正在测试架构: {m}...")
        model = PDESolver(mode=m).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        for i in range(50):
            optimizer.zero_grad()
            out = model(x_test)
            loss = F.mse_loss(out, y_test)  # 现在这里不会报错了
            rel_l2 = torch.norm(out - y_test) / torch.norm(y_test)
            loss.backward()
            optimizer.step()
            losses.append(rel_l2.item())

            if i % 10 == 0:
                print(f"  Step {i:2d} | Rel L2 Error: {rel_l2.item():.4f}")

        results[m] = {'final_error': losses[-1], 'history': losses, 'pred': out.detach().cpu()}
        print(f"  完成 {m} | Final Error: {losses[-1]:.4f}")

    # === 可视化 ===
    print("\n正在生成对比图...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 图 1: 收敛曲线
    for m in modes:
        ax[0].plot(results[m]['history'], label=f'{m} solver')
    ax[0].set_title("Convergence: Relative L2 Error")
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].grid(True, which="both", ls="--")

    # 图 2: 误差空间分布
    error_map = torch.abs(results['hybrid']['pred'] - y_test.cpu()).squeeze()
    im = ax[1].imshow(error_map, cmap='hot')
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("HO-MoE Spatial Error Distribution")

    plt.tight_layout()
    plt.show()
    return results


if __name__ == "__main__":
    run_comparison()