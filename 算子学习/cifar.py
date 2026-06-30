import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ===============================
# 1. 工具函数：参数统计 + 结果记录
# ===============================
def count_params(model):
    """统计可训练参数数量（单位：M）"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6  # 转换为百万


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ===============================
# 2. 网络结构定义（保留你的核心代码）
# ===============================
class BasicBlock(nn.Module):
    """Baseline 基础残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class OperatorTreeBlock(nn.Module):
    """Operator Tree 残差块"""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # local branch
        self.local = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )

        # global branch (1x1 conv 模拟全局 mixing)
        self.global_branch = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )

        # gate (channel-wise)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_planes, planes),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        f_local = self.local(x)
        f_global = self.global_branch(x)
        g = self.gate(x).unsqueeze(-1).unsqueeze(-1)

        out = g * f_local + (1 - g) * f_global
        out += identity
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR专用ResNet"""

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
# 3. 数据加载（优化增强 + 标准化）
# ===============================
def get_cifar10_dataloaders():
    """获取CIFAR10数据加载器（带标准化）"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, test_loader


# ===============================
# 4. 训练函数（带完整日志记录）
# ===============================
def train_model(model, model_name, epochs=100):
    """
    训练模型并记录详细日志
    返回：训练日志（loss、acc、耗时）
    """
    # 初始化
    set_seed()
    train_loader, test_loader = get_cifar10_dataloaders()
    model = model.to(device)

    # 优化器和调度器（保持你的配置）
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 80], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    # 日志记录
    log = {
        "train_loss": [],
        "test_acc": [],
        "epoch_time": [],
        "lr": []
    }

    print(f"\n========== 开始训练 {model_name} ==========")
    print(f"总训练轮数: {epochs} | 初始学习率: 0.1 | 权重衰减: 5e-4")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        # 训练一轮
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证精度
        test_acc = evaluate_model(model, test_loader)

        # 更新调度器
        scheduler.step()

        # 记录日志
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        log["train_loss"].append(avg_loss)
        log["test_acc"].append(test_acc)
        log["epoch_time"].append(epoch_time)
        log["lr"].append(current_lr)

        # 打印进度（每10轮详细打印，其余简略）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
        else:
            print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.4f}")

    # 训练完成
    print(f"\n✅ {model_name} 训练完成！")
    print(f"最终测试精度: {log['test_acc'][-1]:.4f}")
    print(f"平均每轮耗时: {np.mean(log['epoch_time']):.2f}s")

    return model, log


def evaluate_model(model, test_loader):
    """独立的验证函数"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


# ===============================
# 5. 对比实验主函数
# ===============================
def run_comparison_experiment(epochs=100):
    """运行Baseline vs OperatorTree的完整对比实验"""
    # 1. 初始化模型
    baseline_model = ResNet(BasicBlock, [2, 2, 2])
    operator_model = ResNet(OperatorTreeBlock, [2, 2, 2])

    # 2. 统计参数
    baseline_params = count_params(baseline_model)
    operator_params = count_params(operator_model)
    params_increase = (operator_params - baseline_params) / baseline_params * 100

    print("=" * 60)
    print("📊 模型参数统计（单位：百万）")
    print(f"Baseline ResNet: {baseline_params:.2f} M")
    print(f"OperatorTree ResNet: {operator_params:.2f} M")
    print(f"参数增加比例: {params_increase:.2f}%")
    print("=" * 60)

    # 3. 训练两个模型
    baseline_model, baseline_log = train_model(baseline_model, "Baseline ResNet", epochs)
    operator_model, operator_log = train_model(operator_model, "OperatorTree ResNet", epochs)

    # 4. 结果可视化
    plot_comparison_results(baseline_log, operator_log, baseline_params, operator_params)

    # 5. 结果汇总
    summarize_results(baseline_log, operator_log, baseline_params, operator_params)

    # 6. 保存模型（可选）
    torch.save(baseline_model.state_dict(), "baseline_resnet_cifar10.pth")
    torch.save(operator_model.state_dict(), "operatortree_resnet_cifar10.pth")
    print("\n💾 模型权重已保存！")


# ===============================
# 6. 可视化函数
# ===============================
def plot_comparison_results(baseline_log, operator_log, baseline_params, operator_params):
    """绘制对比曲线图"""
    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.figsize"] = (15, 10)

    fig, axes = plt.subplots(2, 2)

    # 子图1：测试精度对比
    epochs = range(1, len(baseline_log["test_acc"]) + 1)
    axes[0, 0].plot(epochs, baseline_log["test_acc"], label=f"Baseline (Params: {baseline_params:.2f}M)",
                    color="blue", linewidth=2, marker="o", markersize=3)
    axes[0, 0].plot(epochs, operator_log["test_acc"], label=f"OperatorTree (Params: {operator_params:.2f}M)",
                    color="red", linewidth=2, marker="s", markersize=3)
    axes[0, 0].set_title("Test Accuracy Comparison")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim(0.7, 1.0)

    # 子图2：训练Loss对比
    axes[0, 1].plot(epochs, baseline_log["train_loss"], label="Baseline",
                    color="blue", linewidth=2)
    axes[0, 1].plot(epochs, operator_log["train_loss"], label="OperatorTree",
                    color="red", linewidth=2)
    axes[0, 1].set_title("Training Loss Comparison")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 子图3：每轮耗时对比
    axes[1, 0].plot(epochs, baseline_log["epoch_time"], label="Baseline",
                    color="blue", linewidth=2)
    axes[1, 0].plot(epochs, operator_log["epoch_time"], label="OperatorTree",
                    color="red", linewidth=2)
    axes[1, 0].set_title("Epoch Time Comparison")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 子图4：学习率变化
    axes[1, 1].plot(epochs, baseline_log["lr"], label="Baseline LR",
                    color="blue", linewidth=2)
    axes[1, 1].plot(epochs, operator_log["lr"], label="OperatorTree LR",
                    color="red", linewidth=2, linestyle="--")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("resnet_comparison_cifar10.png", dpi=300, bbox_inches="tight")
    plt.show()


# ===============================
# 7. 结果汇总函数
# ===============================
def summarize_results(baseline_log, operator_log, baseline_params, operator_params):
    """汇总对比结果"""
    print("\n" + "=" * 60)
    print("📈 实验结果汇总")
    print("=" * 60)

    # 精度对比
    baseline_final_acc = baseline_log["test_acc"][-1]
    operator_final_acc = operator_log["test_acc"][-1]
    acc_improvement = (operator_final_acc - baseline_final_acc) * 100

    # 耗时对比
    baseline_avg_time = np.mean(baseline_log["epoch_time"])
    operator_avg_time = np.mean(operator_log["epoch_time"])
    time_increase = (operator_avg_time - baseline_avg_time) / baseline_avg_time * 100

    # Loss对比
    baseline_final_loss = baseline_log["train_loss"][-1]
    operator_final_loss = operator_log["train_loss"][-1]

    print(f"1. 精度对比:")
    print(f"   - Baseline最终精度: {baseline_final_acc:.4f} ({baseline_final_acc * 100:.2f}%)")
    print(f"   - OperatorTree最终精度: {operator_final_acc:.4f} ({operator_final_acc * 100:.2f}%)")
    print(f"   - 精度提升: {acc_improvement:.2f} 个百分点")

    print(f"\n2. 耗时对比:")
    print(f"   - Baseline平均每轮耗时: {baseline_avg_time:.2f}s")
    print(f"   - OperatorTree平均每轮耗时: {operator_avg_time:.2f}s")
    print(f"   - 耗时增加: {time_increase:.2f}%")

    print(f"\n3. Loss对比:")
    print(f"   - Baseline最终Loss: {baseline_final_loss:.4f}")
    print(f"   - OperatorTree最终Loss: {operator_final_loss:.4f}")

    print(f"\n4. 参数量对比:")
    print(f"   - Baseline参数量: {baseline_params:.2f}M")
    print(f"   - OperatorTree参数量: {operator_params:.2f}M")
    print(f"   - 参数量增加: {((operator_params - baseline_params) / baseline_params * 100):.2f}%")


# ===============================
# 执行入口
# ===============================
if __name__ == "__main__":
    # 设置训练轮数（可以根据需要调整，默认100轮）
    TRAIN_EPOCHS = 100
    run_comparison_experiment(epochs=TRAIN_EPOCHS)