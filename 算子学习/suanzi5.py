import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import json
import torch.optim as optim
from torchvision.models.resnet import ResNet
import numpy as np
import time

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# ======================== 工具函数 ========================
def move_module_to_device(module, device):
    module.to(device)
    for child in module.children():
        move_module_to_device(child, device)
    return module


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


# ======================== 算子库 ========================
class OperatorLibrary(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, stride=1, seed=42, device=DEVICE):
        super().__init__()
        self.device = device
        self.stride = stride
        set_seed(seed)

        self.in_ch = in_ch
        self.out_ch = out_ch

        # 1. 局部卷积
        self.local_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.local_conv.weight, mode='fan_out', nonlinearity='relu')

        # 2. 扩张卷积
        self.dilated_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=stride, padding=1, dilation=1, bias=False)
        nn.init.kaiming_normal_(self.dilated_conv.weight, mode='fan_out', nonlinearity='relu')

        # 3. 1x1卷积堆叠
        class Conv1x1Stack(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.layers = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 1, stride=1, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
                for m in self.layers:
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                return self.layers(x).contiguous()

        self.conv1x1_stack = Conv1x1Stack(self.in_ch, self.out_ch, stride=stride)

        # 4. 频域算子
        class FreqOperator(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                return self.bn(self.proj(x)).contiguous()

        self.freq_op = FreqOperator(self.in_ch, self.out_ch, stride=stride)

        # 5. 形态学算子
        class MorphologyOp(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
                self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                out = self.pool(x)
                return self.bn(self.proj(out)).contiguous()

        self.morph_op = MorphologyOp(self.in_ch, self.out_ch, stride=stride)

        # 6. 通道重排
        class ChannelPermute(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, device=DEVICE):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.stride = stride
                self.device = device
                self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                self.base_perm = torch.randperm(in_ch).to(device)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                if self.in_ch <= len(self.base_perm):
                    x_perm = x[:, self.base_perm[:self.in_ch], :, :].contiguous()
                else:
                    x_perm = x
                return self.bn(self.proj(x_perm)).contiguous()

        self.channel_perm = ChannelPermute(self.in_ch, self.out_ch, stride=stride, device=self.device)

        self.operators = {
            'local_conv': self.local_conv,
            'dilated_conv': self.dilated_conv,
            '1x1_conv': self.conv1x1_stack,
            'freq_op': self.freq_op,
            'morph_op': self.morph_op,
            'channel_perm': self.channel_perm
        }
        move_module_to_device(self, self.device)

    def forward(self, x, op_name):
        return F.relu(self.operators[op_name](x))

    def get_operator_list(self):
        return list(self.operators.keys())


# ======================== 数据 ========================
def get_cifar10_dataloaders(batch_size=128, train=True):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True)


# ======================== 单算子 Block（核心修复） ========================
class SingleOpBlock(nn.Module):
    expansion = 1

    # 修复：兼容ResNet的所有默认参数
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 op_name='local_conv', device=DEVICE):
        super().__init__()
        # 接收ResNet传入的所有参数（即使不用），避免参数不匹配
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation
        self.norm_layer = norm_layer or nn.BatchNorm2d

        self.downsample = downsample
        self.stride = stride
        self.device = device
        self.inplanes = inplanes
        self.planes = planes
        self.op_name = op_name

        self.op_lib = OperatorLibrary(
            in_ch=inplanes,
            out_ch=planes * self.expansion,
            stride=stride,
            device=device
        )
        self.bn = self.norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.op_lib(x, self.op_name)
        out = self.bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# ======================== 单算子 ResNet18（修复传参） ========================
def resnet18_single_op(op_name, num_classes=10, device=DEVICE):
    # 第一步：创建基础ResNet框架
    model = ResNet(
        SingleOpBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        # 显式传入ResNet需要的默认参数，避免自动传参出错
        groups=1,
        width_per_group=64,
        norm_layer=None
    )

    # 第二步：初始化权重
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    # 第三步：为所有Block设置指定的算子名称
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for block in layer:
                if isinstance(block, SingleOpBlock):
                    block.op_name = op_name
                    # 重新初始化算子库，确保参数正确
                    block.op_lib = OperatorLibrary(
                        in_ch=block.inplanes,
                        out_ch=block.planes * block.expansion,
                        stride=block.stride,
                        device=device
                    )
                    move_module_to_device(block, device)

    return model.to(device)


# ======================== 训练配置 ========================
# 测试模式：快速验证代码，完整训练改回100
TEST_MODE = True
if TEST_MODE:
    TRAIN_EPOCHS = 2
    TRAIN_SEEDS = [42]
    BATCH_SIZE = 64
else:
    TRAIN_EPOCHS = 100
    TRAIN_SEEDS = [42, 43, 44]
    BATCH_SIZE = 128


# ======================== 单算子训练 ========================
def train_single_op(op_name, seed=42, epochs=TRAIN_EPOCHS, device=DEVICE):
    set_seed(seed)
    train_loader = get_cifar10_dataloaders(BATCH_SIZE, train=True)
    test_loader = get_cifar10_dataloaders(1000, train=False)

    # 创建模型
    model = resnet18_single_op(op_name, device=device)

    # 优化器配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    best_acc = 0.0
    acc_history = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"{op_name} Epoch {epoch + 1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(x)
            loss = criterion(outputs, y)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (len(pbar))})

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total
        acc_history.append(acc)

        if acc > best_acc:
            best_acc = acc

        scheduler.step()
        print(f"[{op_name}] Seed {seed} | Epoch {epoch + 1} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    return {
        "best_acc": float(best_acc),
        "acc_std": float(np.std(acc_history)),
        "seed": seed
    }


# ======================== 主函数：只跑单算子 ========================
if __name__ == "__main__":
    # 6个算子列表
    op_list = ['local_conv', 'dilated_conv', '1x1_conv', 'freq_op', 'morph_op', 'channel_perm']
    all_results = []

    # 逐个训练算子
    for op in op_list:
        print("\n" + "=" * 50)
        print(f"          训练单算子：{op}")
        print("=" * 50)

        # 多种子训练
        seed_results = [train_single_op(op, s) for s in TRAIN_SEEDS]

        # 计算平均结果
        avg_acc = float(np.mean([r['best_acc'] for r in seed_results]))
        avg_std = float(np.mean([r['acc_std'] for r in seed_results]))

        all_results.append({
            "op_name": op,
            "avg_best_acc": avg_acc,
            "avg_acc_std": avg_std,
            "seed_results": seed_results
        })

    # 保存结果到JSON
    all_results_py = convert_numpy_to_python(all_results)
    with open("single_op_results.json", "w", encoding='utf-8') as f:
        json.dump(all_results_py, f, indent=4, ensure_ascii=False)

    # 打印最终排名
    print("\n" + "=" * 60)
    print("        单算子准确率最终排名")
    print("=" * 60)
    # 按平均准确率降序排序
    sorted_results = sorted(all_results, key=lambda x: x['avg_best_acc'], reverse=True)
    for idx, res in enumerate(sorted_results):
        print(
            f"排名{idx + 1:2d}. {res['op_name']:15s} | 平均准确率: {res['avg_best_acc']:6.2f}% | 标准差: {res['avg_acc_std']:.4f}")

    print(f"\n结果已保存到：single_op_results.json")