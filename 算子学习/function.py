import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Callable, List, Tuple

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ===============================
# 1. 工具函数
# ===============================
def count_params(model):
    """统计可训练参数数量（单位：M）"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ===============================
# 2. 函数数据集定义
# ===============================
class FunctionDataset(Dataset):
    """通用函数拟合数据集"""

    def __init__(self,
                 func: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float] = (-10, 10),
                 num_samples: int = 10000,
                 noise_std: float = 0.01,
                 is_train: bool = True):
        self.func = func
        self.x_range = x_range
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.is_train = is_train

        rng = np.random.RandomState(42 if is_train else 100)
        self.x = rng.uniform(x_range[0], x_range[1], (num_samples, 1))
        self.y = self.func(self.x)
        if self.is_train and noise_std > 0:
            self.y += rng.normal(0, noise_std, self.y.shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y


# 定义6类典型函数
def linear_func(x: np.ndarray) -> np.ndarray:
    """线性函数: y = 2x + 3"""
    return 2 * x + 3


def nonlinear_poly_func(x: np.ndarray) -> np.ndarray:
    """多项式函数: y = x³ - 6x² + 11x - 6"""
    return x ** 3 - 6 * x ** 2 + 11 * x - 6


def periodic_func(x: np.ndarray) -> np.ndarray:
    """周期函数: y = sin(x) + cos(0.5x)"""
    return np.sin(x) + np.cos(0.5 * x)


def piecewise_func(x: np.ndarray) -> np.ndarray:
    """分段函数"""
    y = np.zeros_like(x)
    y[x < -5] = 0.1 * x[x < -5] + 1
    y[(-5 <= x) & (x < 0)] = np.exp(x[(-5 <= x) & (x < 0)] / 2)
    y[(0 <= x) & (x < 5)] = np.log(x[(0 <= x) & (x < 5)] + 1) + 2
    y[x >= 5] = 0.5 * x[x >= 5] - 1
    return y


def high_dim_func(x: np.ndarray) -> np.ndarray:
    """高维函数（输入5维）: y = x1² + sin(x2) + x3*x4 - |x5|"""
    if x.shape[1] == 1:
        x_high = np.hstack([
            x,
            np.random.uniform(-5, 5, (x.shape[0], 1)),
            np.random.uniform(-5, 5, (x.shape[0], 1)),
            np.random.uniform(-5, 5, (x.shape[0], 1)),
            np.random.uniform(-5, 5, (x.shape[0], 1))
        ])
    else:
        x_high = x
    return x_high[:, 0:1] ** 2 + np.sin(x_high[:, 1:2]) + x_high[:, 2:3] * x_high[:, 3:4] - np.abs(x_high[:, 4:5])


def multi_output_func(x: np.ndarray) -> np.ndarray:
    """多输出函数: [sin(x), cos(x), x²]"""
    return np.hstack([np.sin(x), np.cos(x), x ** 2])


# 函数集合（全量测试）
FUNCTIONS = {
    "linear": (linear_func, (-10, 10), 1),
    "polynomial": (nonlinear_poly_func, (-10, 10), 1),
    "periodic": (periodic_func, (-10, 10), 1),
    "piecewise": (piecewise_func, (-10, 10), 1),
    "high_dim": (high_dim_func, (-5, 5), 1),
    "multi_output": (multi_output_func, (-10, 10), 3)
}


# ===============================
# 3. 模型定义（和你一致的结构）
# ===============================
class BasicBlock(nn.Module):
    """Baseline 基础残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class OriginalOperatorTreeBlock(nn.Module):
    """原始Operator Tree 残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            nn.Conv1d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.global_branch = nn.Sequential(
            nn.Conv1d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            nn.Conv1d(planes, planes, 1, 1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_planes, planes),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        f_local = self.local(x)
        f_global = self.global_branch(x)
        g = self.gate(x).unsqueeze(-1)
        out = g * f_local + (1 - g) * f_global
        out += identity
        return F.relu(out)


class LightweightOperatorTreeBlock(nn.Module):
    """轻量化Operator Tree残差块"""

    def __init__(self, in_planes, planes, stride=1, reduction=4):
        super().__init__()
        mid_planes = planes // reduction
        self.shared_conv = nn.Sequential(
            nn.Conv1d(in_planes, mid_planes, 3, stride, 1, bias=False),
            nn.BatchNorm1d(mid_planes),
            nn.ReLU()
        )
        self.local = nn.Sequential(
            nn.Conv1d(mid_planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.global_branch = nn.Sequential(
            nn.Conv1d(mid_planes, planes, 1, 1, bias=False),
            nn.BatchNorm1d(planes)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_planes, planes, 1, bias=False),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x_shared = self.shared_conv(x)
        f_local = self.local(x_shared)
        f_global = self.global_branch(x_shared)
        g = self.gate(x)
        out = g * f_local + (1 - g) * f_global
        out += identity
        return F.relu(out)


class ResNetForFitting(nn.Module):
    """适配函数拟合的ResNet"""

    def __init__(self, block, num_blocks, input_dim=1, output_dim=1):
        super().__init__()
        self.in_planes = 64
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 1))
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if block == LightweightOperatorTreeBlock:
                layers.append(block(self.in_planes, planes, s, reduction=4))
            else:
                layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_proj(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_head(out)
        return out


# ===============================
# 4. 数据加载 + 训练 + 评估
# ===============================
def get_function_dataloaders(func_name: str, batch_size: int = 128):
    """获取函数拟合的数据加载器"""
    func, x_range, output_dim = FUNCTIONS[func_name]
    train_dataset = FunctionDataset(func, x_range, num_samples=10000, noise_std=0.01, is_train=True)
    test_dataset = FunctionDataset(func, x_range, num_samples=2000, noise_std=0.0, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, output_dim


def train_model_for_fitting(model, model_name, func_name, epochs=50):
    """训练函数拟合模型（兼容PyTorch 1.x）"""
    set_seed()
    train_loader, test_loader, _ = get_function_dataloaders(func_name)
    model = model.to(device)

    # 分模型调整学习率
    if "Original" in model_name:
        lr = 5e-5
    elif "Lightweight" in model_name:
        lr = 8e-5
    else:
        lr = 1e-4

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.MSELoss()

    log = {"train_loss": [], "test_mse": [], "epoch_time": [], "lr": []}
    best_mse = float('inf')
    patience = 10
    patience_counter = 0

    print(f"\n========== 开始训练 {model_name} 拟合 {func_name} 函数 ==========")
    print(f"总训练轮数: {epochs} | 初始学习率: {lr:.6f} | 权重衰减: 5e-6")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        # 评估
        model.eval()
        test_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_mse += F.mse_loss(output, target, reduction='sum').item()
                total_samples += target.size(0)
        test_mse /= total_samples

        # 学习率调度
        scheduler.step()

        # 早停
        if test_mse < best_mse * 0.99:
            best_mse = test_mse
            patience_counter = 0
            # 关键修改：去掉weights_only参数（兼容PyTorch 1.x）
            torch.save(model.state_dict(), f"best_{model_name}_{func_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发！Epoch {epoch + 1}，最优MSE: {best_mse:.6f}")
                break

        # 记录
        avg_loss = total_loss / len(train_loader)
        log["train_loss"].append(avg_loss)
        log["test_mse"].append(test_mse)
        log["epoch_time"].append(time.time() - epoch_start)
        log["lr"].append(optimizer.param_groups[0]['lr'])

        # 打印
        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience - 1:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.6f} | Test MSE: {test_mse:.6f} | Best MSE: {best_mse:.6f}")

    # 加载最优模型（同样去掉weights_only）
    model.load_state_dict(torch.load(f"best_{model_name}_{func_name}.pth"))
    final_mse = test_mse

    print(f"\n✅ {model_name} 拟合 {func_name} 函数完成！")
    print(f"最优MSE: {best_mse:.6f} | 最终MSE: {final_mse:.6f}")
    print(f"平均耗时: {np.mean(log['epoch_time']):.2f}s/epoch")

    return model, log, best_mse, np.mean(log['epoch_time'])


# ===============================
# 5. 主函数：全函数测试
# ===============================
def run_full_function_test(epochs=50):
    """全函数测试主函数"""
    # 1. 统计参数量
    baseline_model = ResNetForFitting(BasicBlock, [2, 2, 2])
    original_model = ResNetForFitting(OriginalOperatorTreeBlock, [2, 2, 2])
    light_model = ResNetForFitting(LightweightOperatorTreeBlock, [2, 2, 2])

    baseline_params = count_params(baseline_model)
    original_params = count_params(original_model)
    light_params = count_params(light_model)

    print("=" * 60)
    print("📊 模型参数统计")
    print(f"Baseline ResNet: {baseline_params:.2f}M")
    print(
        f"Original OT ResNet: {original_params:.2f}M (↑{(original_params - baseline_params) / baseline_params * 100:.2f}%)")
    print(
        f"Lightweight OT ResNet: {light_params:.2f}M (↑{(light_params - baseline_params) / baseline_params * 100:.2f}%)")
    print("=" * 60)

    # 2. 全函数测试
    all_results = {}
    for func_name in FUNCTIONS.keys():
        print(f"\n{'=' * 60}")
        print(f"📝 测试函数: {func_name}")
        print(f"{'=' * 60}")

        # 获取输出维度
        _, _, output_dim = FUNCTIONS[func_name]
        input_dim = 5 if func_name == "high_dim" else 1

        # 初始化模型
        baseline = ResNetForFitting(BasicBlock, [2, 2, 2], input_dim, output_dim)
        original = ResNetForFitting(OriginalOperatorTreeBlock, [2, 2, 2], input_dim, output_dim)
        light = ResNetForFitting(LightweightOperatorTreeBlock, [2, 2, 2], input_dim, output_dim)

        # 训练模型
        _, _, baseline_best_mse, baseline_time = train_model_for_fitting(baseline, "Baseline ResNet", func_name, epochs)
        _, _, original_best_mse, original_time = train_model_for_fitting(original, "Original OperatorTree ResNet",
                                                                         func_name, epochs)
        _, _, light_best_mse, light_time = train_model_for_fitting(light, "Lightweight OperatorTree ResNet", func_name,
                                                                   epochs)

        # 计算改善率
        original_improve = (baseline_best_mse - original_best_mse) / baseline_best_mse * 100
        light_improve = (baseline_best_mse - light_best_mse) / baseline_best_mse * 100

        # 保存结果
        all_results[func_name] = {
            "baseline": {"mse": baseline_best_mse, "time": baseline_time, "params": baseline_params},
            "original": {"mse": original_best_mse, "time": original_time, "params": original_params,
                         "improve": original_improve},
            "light": {"mse": light_best_mse, "time": light_time, "params": light_params, "improve": light_improve}
        }

    # 3. 打印汇总表
    print("\n" + "=" * 100)
    print("📈 全函数测试结果汇总表")
    print("=" * 100)
    print(
        f"{'函数类型':<12} | {'模型':<20} | {'参数量(M)':<10} | {'最优MSE':<10} | {'相对改善(%)':<12} | {'单轮耗时(s)':<10}")
    print("-" * 100)

    for func_name in all_results.keys():
        res = all_results[func_name]
        # Baseline
        print(
            f"{func_name:<12} | {'Baseline':<20} | {res['baseline']['params']:<10.2f} | {res['baseline']['mse']:<10.6f} | {'-':<12} | {res['baseline']['time']:<10.2f}")
        # Original OT
        print(
            f"{' ':<12} | {'Original OT':<20} | {res['original']['params']:<10.2f} | {res['original']['mse']:<10.6f} | {res['original']['improve']:<12.2f} | {res['original']['time']:<10.2f}")
        # Light OT
        print(
            f"{' ':<12} | {'Lightweight OT':<20} | {res['light']['params']:<10.2f} | {res['light']['mse']:<10.6f} | {res['light']['improve']:<12.2f} | {res['light']['time']:<10.2f}")
        print("-" * 100)

    return all_results


# ===============================
# 执行测试
# ===============================
if __name__ == "__main__":
    # 训练50轮
    all_results = run_full_function_test(epochs=50)
    print("\n🎉 全函数测试完成！")