import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import json
import torch.optim as optim
from torchvision.models.resnet import ResNet
import numpy as np
import time

# 全局配置
SAVE_PATH = "./cifar10_anchor_dataset.pkl"
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


# 新增：JSON序列化辅助函数（转换numpy类型为Python原生类型）
def convert_numpy_to_python(obj):
    """递归将numpy类型转换为Python原生类型"""
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


# ======================== 修复后的轻量化算子库 ========================
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

        # 3. 简化1×1卷积堆叠
        class Conv1x1Stack(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch  # 显式定义属性
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

        # 4. 简化频域算子
        class FreqOperator(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch  # 显式定义属性
                self.out_ch = out_ch
                self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                return self.bn(self.proj(x)).contiguous()

        self.freq_op = FreqOperator(self.in_ch, self.out_ch, stride=stride)

        # 5. 简化形态学算子
        class MorphologyOp(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.in_ch = in_ch  # 显式定义属性
                self.out_ch = out_ch
                self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
                self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                out = self.pool(x)
                return self.bn(self.proj(out)).contiguous()

        self.morph_op = MorphologyOp(self.in_ch, self.out_ch, stride=stride)

        # 6. 简化通道重排算子（核心修复：显式定义in_ch/out_ch属性）
        class ChannelPermute(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, device=DEVICE):
                super().__init__()
                self.in_ch = in_ch  # 关键：显式定义in_ch属性
                self.out_ch = out_ch
                self.stride = stride
                self.device = device
                self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                # 修复：确保perm长度和in_ch一致
                self.base_perm = torch.randperm(in_ch).to(device)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                # 现在能正确引用self.in_ch
                if self.in_ch <= len(self.base_perm):
                    x_perm = x[:, self.base_perm[:self.in_ch], :, :].contiguous()
                else:
                    x_perm = x
                return self.bn(self.proj(x_perm)).contiguous()

        # 修复：传入device参数
        self.channel_perm = ChannelPermute(self.in_ch, self.out_ch, stride=stride, device=self.device)

        # 算子字典
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
        if op_name not in self.operators:
            raise ValueError(f"无效算子：{op_name}，可选：{list(self.operators.keys())}")
        x = x.to(self.device)
        out = self.operators[op_name](x)
        return F.relu(out)

    def get_operator_list(self):
        return list(self.operators.keys())


# ======================== 数据加载 ========================
def get_cifar10_dataloaders(batch_size=128, train=True):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform
        )
        dataloader = DataLoader(
            dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True
        )
    return dataloader


def load_cifar10_anchor_set(batch_size=64, num_samples=1000):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform
    )

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return dataloader


# ======================== Gram矩阵计算 ========================
def compute_gram_matrix(op_lib, dataloader, device=DEVICE):
    op_lib.eval()
    op_names = op_lib.get_operator_list()
    num_ops = len(op_names)

    gram_matrix = torch.zeros(num_ops, num_ops, device=device)
    total_samples = 0

    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    feature_extractor = move_module_to_device(feature_extractor, device)
    feature_extractor.eval()

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="计算Gram矩阵"):
            images = images.to(device, non_blocking=True)
            B = images.shape[0]
            total_samples += B

            x = feature_extractor(images).contiguous()

            op_outputs = []
            for op_name in op_names:
                out = op_lib(x, op_name)
                out_flat = out.reshape(B, -1)
                out_flat = F.normalize(out_flat, p=2, dim=1)
                op_outputs.append(out_flat)

            for i in range(num_ops):
                for j in range(num_ops):
                    inner_product = (op_outputs[i] * op_outputs[j]).sum(dim=1).mean()
                    gram_matrix[i, j] += inner_product * B

        gram_matrix /= total_samples

    return gram_matrix.cpu().numpy(), op_names


def save_gram_matrix(gram_matrix, op_names, save_path=SAVE_PATH):
    data = {
        'gram_matrix': gram_matrix,
        'op_names': op_names,
        'dataset': 'CIFAR-10',
        'num_samples': 1000,
        'timestamp': str(np.datetime64('now'))  # 修复：转换为字符串，避免JSON序列化问题
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nGram矩阵已保存到：{save_path}")
    return data


# ======================== 算子组合生成 ========================
def generate_op_combinations(gram_matrix, op_names):
    combs = list(itertools.combinations(op_names, 2))
    comb_scores = {}

    for (op1, op2) in combs:
        i = op_names.index(op1)
        j = op_names.index(op2)
        sub_matrix = gram_matrix[[i, j], :][:, [i, j]]
        det = np.linalg.det(sub_matrix)
        comb_scores[(op1, op2)] = det

    sorted_combs = sorted(comb_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n=== 算子组合Volume Score排名 ===")
    for idx, (comb, score) in enumerate(sorted_combs):
        print(f"{idx + 1}. {comb[0]}+{comb[1]}: {score:.6f}")

    return sorted_combs, comb_scores


# ======================== 修复后的ResNet替换模块 ========================
class OpCombinationBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 op1_name='local_conv', op2_name='1x1_conv', device=DEVICE):
        super().__init__()
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation
        self.norm_layer = norm_layer if norm_layer is not None else nn.BatchNorm2d

        self.downsample = downsample
        self.stride = stride
        self.device = device
        self.inplanes = inplanes
        self.planes = planes

        self.op_lib = OperatorLibrary(
            in_ch=inplanes,
            out_ch=planes * self.expansion,
            stride=stride,
            device=device
        )
        self.op1_name = op1_name
        self.op2_name = op2_name

        self.gate1 = nn.Parameter(torch.randn(1).to(device) * 0.01)
        self.gate2 = nn.Parameter(torch.randn(1).to(device) * 0.01)
        self.bn = self.norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out1 = self.op_lib(x, self.op1_name)
        out2 = self.op_lib(x, self.op2_name)

        out1 = F.adaptive_avg_pool2d(out1, output_size=out2.shape[-2:])

        gate1 = torch.sigmoid(self.gate1)
        gate2 = torch.sigmoid(self.gate2)
        out = gate1 * out1 + gate2 * out2

        out = self.bn(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


def resnet18_op_comb(op1_name, op2_name, num_classes=10, device=DEVICE):
    model = ResNet(
        OpCombinationBlock,
        [2, 2, 2, 2],
        num_classes=num_classes
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for block in layer:
                if isinstance(block, OpCombinationBlock):
                    block.op1_name = op1_name
                    block.op2_name = op2_name
                    block.op_lib = OperatorLibrary(
                        in_ch=block.inplanes,
                        out_ch=block.planes * block.expansion,
                        stride=block.stride,
                        device=device
                    )
                    move_module_to_device(block, device)

    return model.to(device)


# ======================== 训练配置 + 修复后的训练函数 ========================
# 核心修改1：关闭测试模式，启用100轮训练
TEST_MODE = False
if TEST_MODE:
    TRAIN_EPOCHS = 5
    TRAIN_SEEDS = [42]
    TRAIN_TOP_COMBS = 1  # 测试模式下训练1个组合
    BATCH_SIZE = 64
else:
    TRAIN_EPOCHS = 100  # 完整训练：100轮
    TRAIN_SEEDS = [42, 43, 44]  # 3个随机种子，结果更稳定
    TRAIN_TOP_COMBS = 15  # 核心修改2：训练前15个算子组合
    BATCH_SIZE = 128  # 完整训练用更大的batch_size


def train_op_combination(op1_name, op2_name, seed=42, epochs=TRAIN_EPOCHS, device=DEVICE):
    set_seed(seed)

    train_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE, train=True)
    test_loader = get_cifar10_dataloaders(batch_size=1000, train=False)

    model = resnet18_op_comb(op1_name, op2_name, num_classes=10, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    grad_clip = 1.0

    best_acc = 0.0
    acc_history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})

        model.eval()
        test_acc = 0.0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                test_acc += (predicted == targets).sum().item()

        test_acc = 100. * test_acc / total
        acc_history.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
        scheduler.step()

        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_time = avg_time_per_epoch * (epochs - epoch - 1)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"[{op1_name}+{op2_name}] Seed {seed} | Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss / (batch_idx + 1):.4f} | "
            f"Acc: {test_acc:.2f}% | Best: {best_acc:.2f}% | LR: {current_lr:.6f} | Remaining: {remaining_time / 60:.1f} mins")

    return {
        "best_acc": float(best_acc),  # 强制转换为Python float
        "acc_std": float(np.std(acc_history)),  # 强制转换为Python float
        "op_comb": (op1_name, op2_name),
        "seed": int(seed)  # 强制转换为Python int
    }


# ======================== 主流程 ========================
if __name__ == "__main__":
    print("=" * 80)
    print(f"开始运行 | 设备：{DEVICE} | 模式：{'测试' if TEST_MODE else '完整训练'}")
    print(f"训练配置：{TRAIN_EPOCHS}轮 | 前{TRAIN_TOP_COMBS}个算子组合 | 3个随机种子")
    print("=" * 80)

    # 步骤1：计算Gram矩阵
    print("\n步骤1：计算Gram矩阵")
    op_lib = OperatorLibrary(in_ch=64, out_ch=64, stride=1, device=DEVICE)
    dataloader = load_cifar10_anchor_set(batch_size=64, num_samples=1000)
    gram_matrix, op_names = compute_gram_matrix(op_lib, dataloader, device=DEVICE)
    gram_data = save_gram_matrix(gram_matrix, op_names, save_path=SAVE_PATH)

    # 可视化
    plt.figure(figsize=(10, 8))
    sns.heatmap(gram_matrix, annot=True, fmt='.3f', xticklabels=op_names, yticklabels=op_names, cmap='viridis')
    plt.title('Gram Matrix of Operators on CIFAR-10 Anchor Set')
    plt.tight_layout()
    plt.savefig('./cifar10_gram_matrix.png', dpi=300)
    plt.show()

    # 步骤2：生成组合
    print("\n步骤2：生成算子组合")
    sorted_combs, comb_scores = generate_op_combinations(gram_matrix, op_names)

    # 步骤3：训练
    print("\n步骤3：开始训练算子组合")
    all_results = []
    # 取前TRAIN_TOP_COMBS个组合（15个）
    combs = [c[0] for c in sorted_combs[:TRAIN_TOP_COMBS]]

    for idx, comb in enumerate(combs):
        op1, op2 = comb
        seed_results = []
        print(f"\n===== 训练组合 {idx+1}/{TRAIN_TOP_COMBS}：{op1}+{op2} =====")

        for seed in TRAIN_SEEDS:
            res = train_op_combination(op1, op2, seed=seed, epochs=TRAIN_EPOCHS, device=DEVICE)
            seed_results.append(res)

        # 强制转换为Python原生类型
        avg_acc = float(np.mean([r["best_acc"] for r in seed_results]))
        avg_std = float(np.mean([r["acc_std"] for r in seed_results]))

        all_results.append({
            "op_comb": f"{op1}+{op2}",
            "volume_score": float(comb_scores[comb]),  # 强制转换为Python float
            "avg_best_acc": avg_acc,
            "avg_acc_std": avg_std,
            "seed_results": seed_results
        })

    # 修复：先转换所有numpy类型为Python原生类型，再保存JSON
    all_results_python = convert_numpy_to_python(all_results)
    with open("./op_comb_results_15combs_100epochs.json", "w", encoding='utf-8') as f:
        json.dump(all_results_python, f, indent=4, ensure_ascii=False)

    # 汇总打印
    print("\n" + "=" * 80)
    print("训练完成！结果汇总（前15个组合 | 100轮训练）：")
    print("=" * 80)
    # 按平均准确率排序打印
    sorted_all_results = sorted(all_results, key=lambda x: x["avg_best_acc"], reverse=True)
    for idx, res in enumerate(sorted_all_results):
        print(
            f"排名{idx+1}. {res['op_comb']} | Volume Score: {res['volume_score']:.6f} | "
            f"平均准确率: {res['avg_best_acc']:.2f}% | 方差: {res['avg_acc_std']:.4f}")

    print(f"\n文件保存位置：")
    print(f"- Gram矩阵：{SAVE_PATH}")
    print(f"- 训练结果：./op_comb_results_15combs_100epochs.json")
    print(f"- Gram可视化：./cifar10_gram_matrix.png")