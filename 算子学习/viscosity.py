import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ===============================
# 1. Burgers' Equation 数据生成（改进数值稳定性）
# ===============================
class BurgersDataset(Dataset):
    def __init__(self, n_samples=1000, nx=64, nt=50, nu=0.01):
        self.n_samples = n_samples
        self.nx = nx
        self.nt = nt
        self.nu = nu
        self.inputs, self.outputs = self._generate_data()  # 归一化
        self.input_mean = self.inputs.mean()
        self.input_std = self.inputs.std() + 1e-8
        self.output_mean = self.outputs.mean()
        self.output_std = self.outputs.std() + 1e-8

        self.inputs = (self.inputs - self.input_mean) / self.input_std
        self.outputs = (self.outputs - self.output_mean) / self.output_std

    def _generate_data(self):
        inputs = []
        outputs = []

        dx = 2.0 / (self.nx - 1)
        dt = 0.0005  # 减小时间步长提高稳定性
        x = np.linspace(0, 2, self.nx)

        for _ in range(self.n_samples):
            # 更温和的初始条件
            u0 = 0.5 * np.sin(np.pi * x) + 0.2 * np.sin(3 * np.pi * x)

            u = u0.copy()
            u_history = [u0]

            for _ in range(self.nt):
                un = u.copy()
                # 使用更稳定的差分格式
                u[1:-1] = un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) + \
                          self.nu * dt / dx ** 2 * (un[2:] - 2 * un[1:-1] + un[:-2])
                u[0] = u[-1] = 0

                # 检查数值稳定性
                if np.isnan(u).any() or np.isinf(u).any():
                    u = u0.copy()
                    break
                u_history.append(u.copy())

            # 输入：前5个时间步
            input_frames = np.stack([u_history[min(i, len(u_history) - 1)] for i in range(0, 50, 10)])
            output_frame = u_history[-1]
            inputs.append(input_frames)
            outputs.append(output_frame)

        inputs = torch.FloatTensor(np.array(inputs)).unsqueeze(1)
        outputs = torch.FloatTensor(np.array(outputs))

        return inputs, outputs

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]



# ===============================
# 2. 简化的网络架构
# ===============================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class OperatorTreeBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.global_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        f_local = self.local(x)
        f_global = self.global_branch(x)
        g = self.gate(x).unsqueeze(-1)

        out = g * f_local + (1 - g) * f_global
        out += identity
        return F.relu(out)


class PDESolver(nn.Module):
    def __init__(self, block, num_blocks, input_channels=5, output_size=64):
        super().__init__()
        self.in_channels = 32
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, 32, (input_channels, 1), 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size),
            nn.Conv1d(64, 1, 1)
        )

    def _make_layer(self, block, channels, num_blocks, stride):
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels
        for _ in range(1, num_blocks):
            layers.append(block(channels, channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_proj(x)
        out = out.squeeze(2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.output_proj(out)
        return out.squeeze(1)


# ===============================
# 3. 训练（添加梯度裁剪和NaN 检测）
# ===============================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_model(model, model_name, epochs=50):
    set_seed()

    train_dataset = BurgersDataset(n_samples=800)
    test_dataset = BurgersDataset(n_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    log = {"train_loss": [], "test_error": [], "epoch_time": []}

    print(f"\n========== 训练 {model_name} ==========")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # NaN 检测
            if torch.isnan(loss):
                print(f"警告: Epoch {epoch + 1} 出现 NaN，跳过此批次")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        test_error = evaluate_model(model, test_loader)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)

        log["train_loss"].append(avg_loss)
        log["test_error"].append(test_error)
        log["epoch_time"].append(epoch_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.6f} | "
                  f"Test Error: {test_error:.6f} | Time: {epoch_time:.2f}s")

    print(f"\n✅ {model_name} 训练完成！最终误差: {log['test_error'][-1]:.6f}")
    return model, log


def evaluate_model(model, test_loader):
    model.eval()
    total_error = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            error = torch.norm(outputs - targets, dim=1) / (torch.norm(targets, dim=1) + 1e-8)
            total_error += error.mean().item()

    return total_error / len(test_loader)


def plot_comparison_results(baseline_log, operator_log, baseline_params, operator_params):
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(baseline_log["test_error"]) + 1)

    axes[0].plot(epochs, baseline_log["test_error"], label=f"Baseline ({baseline_params:.2f}M)",
                 color="blue", linewidth=2)
    axes[0].plot(epochs, operator_log["test_error"], label=f"OperatorTree ({operator_params:.2f}M)",
                 color="red", linewidth=2)
    axes[0].set_title("Test Relative L2 Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Relative Error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, baseline_log["train_loss"], label="Baseline", color="blue", linewidth=2)
    axes[1].plot(epochs, operator_log["train_loss"], label="OperatorTree", color="red", linewidth=2)
    axes[1].set_title("Training Loss (MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, baseline_log["epoch_time"], label="Baseline", color="blue", linewidth=2)
    axes[2].plot(epochs, operator_log["epoch_time"], label="OperatorTree", color="red", linewidth=2)
    axes[2].set_title("Epoch Time")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Time (s)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("pde_solver_comparison.png", dpi=300)
    plt.show()


def run_comparison_experiment(epochs=50):
    baseline_model = PDESolver(BasicBlock1D, [2, 2])
    operator_model = PDESolver(OperatorTreeBlock1D, [2, 2])

    baseline_params = count_params(baseline_model)
    operator_params = count_params(operator_model)

    print("=" * 60)
    print("📊 模型参数统计")
    print(f"Baseline: {baseline_params:.2f}M | OperatorTree: {operator_params:.2f}M")
    print(f"参数增加: {(operator_params - baseline_params) / baseline_params * 100:.2f}%")
    print("=" * 60)

    baseline_model, baseline_log = train_model(baseline_model, "Baseline CNN", epochs)
    operator_model, operator_log = train_model(operator_model, "OperatorTree", epochs)

    plot_comparison_results(baseline_log, operator_log, baseline_params, operator_params)

    print("\n" + "=" * 60)
    print("📈 实验结果汇总")
    print(f"Baseline 最终误差: {baseline_log['test_error'][-1]:.6f}")
    print(f"OperatorTree 最终误差: {operator_log['test_error'][-1]:.6f}")
    improvement = (baseline_log['test_error'][-1] - operator_log['test_error'][-1]) / baseline_log['test_error'][
        -1] * 100
    print(f"误差降低: {improvement:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_comparison_experiment(epochs=50)
