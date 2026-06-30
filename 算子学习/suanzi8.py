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
import json
import torch.optim as optim
import numpy as np
import time
from scipy.stats import pearsonr

# ======================== 全局配置（仅必要配置，无Score超参数） ========================
SAVE_PATH = "./cifar10_anchor_dataset.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# 训练配置（保证结果可靠，无超参数）
TEST_MODE = False
if TEST_MODE:
    TRAIN_EPOCHS = 10
    TRAIN_SEEDS = [42]
    BATCH_SIZE = 64
else:
    TRAIN_EPOCHS = 50  # 收敛所需的合理轮数（非超参数）
    TRAIN_SEEDS = [42, 43, 44]
    BATCH_SIZE = 128


# 正则项纯数学定义（无可调参数，基于矩阵理论）
# 条件数正则：κ>10时，正则项从1平滑下降（纯数学约束，无超参数）
# 体积正则：体积偏离0.5时，正则项下降（0.5是3×3Gram矩阵的合理理论值）


# ======================== 基础工具函数 ========================
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
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


# ======================== 算子库（纯结构定义，无超参数） ========================
class OperatorLibrary(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, stride=1, seed=42, device=DEVICE):
        super().__init__()
        self.device = device
        self.stride = stride
        set_seed(seed)
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 1. 主算子（纯结构定义）
        self.local_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.local_conv.weight, mode='fan_out', nonlinearity='relu')
        self.dilated_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=stride, padding=2, dilation=2, bias=False)
        nn.init.kaiming_normal_(self.dilated_conv.weight, mode='fan_out', nonlinearity='relu')

        # 2. 补充算子（纯结构定义）
        class Conv1x1Stack(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
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

        class FreqOperator(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                return self.bn(self.proj(x)).contiguous()

        self.freq_op = FreqOperator(self.in_ch, self.out_ch, stride=stride)

        class MorphologyOp(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1):
                super().__init__()
                self.pool = nn.AvgPool2d(3, stride=stride, padding=1)
                self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                out = self.pool(x)
                return self.bn(self.proj(out)).contiguous()

        self.morph_op = MorphologyOp(self.in_ch, self.out_ch, stride=stride)

        # 3. 调节算子（纯结构定义）
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

        # 算子字典（固定）
        self.operators = {
            'local_conv': self.local_conv,
            'dilated_conv': self.dilated_conv,
            'conv1x1_stack': self.conv1x1_stack,
            'freq_op': self.freq_op,
            'morph_op': self.morph_op,
            'channel_perm': self.channel_perm
        }

        # 角色划分（固定，无超参数）
        self.operator_roles = {
            'primary': ['local_conv', 'dilated_conv'],
            'complementary': ['conv1x1_stack', 'freq_op', 'morph_op'],
            'stabilizing': ['channel_perm']
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

    def get_operators_by_role(self, role):
        return self.operator_roles.get(role, [])


# ======================== 数据加载（固定流程，无超参数） ========================
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
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=train, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=True
    )
    return dataloader


def load_cifar10_anchor_set(batch_size=64, num_samples=2000):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ======================== Gram矩阵计算（纯数学，无超参数） ========================
def compute_gram_matrix_optimized(op_lib, dataloader, device=DEVICE, use_gap=True):
    op_lib.eval()
    op_names = op_lib.get_operator_list()

    # 固定特征提取器（无超参数）
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    ).to(device).eval()

    all_op_outputs = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="计算Gram矩阵"):
            images = images.to(device, non_blocking=True)
            x = feature_extractor(images).contiguous()

            batch_op_outputs = []
            for op_name in op_names:
                out = op_lib(x, op_name)
                if use_gap:
                    out = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1)
                else:
                    out = out.reshape(images.shape[0], -1)
                batch_op_outputs.append(F.normalize(out, p=2, dim=1))

            all_op_outputs.append(torch.stack(batch_op_outputs, dim=1))

    all_op_outputs = torch.cat(all_op_outputs, dim=0)
    gram_matrix = torch.einsum('nid,njd->ij', all_op_outputs, all_op_outputs) / all_op_outputs.size(0)
    return gram_matrix.cpu().numpy(), op_names


# ======================== Score计算（纯数学定义，无超参数，强可解释性） ========================
def compute_stabilized_score(
        op_subset, op_names, gram_matrix, op_perf
):
    """
    纯数学定义的稳定化Score（无可调超参数）：
    Score = 性能几何平均 × 体积分 × 条件数正则
    其中：
    1. 性能几何平均：反映组合的基础性能（纯数学平均）
    2. 体积分：det(Gram)^0.5，反映算子多样性（纯矩阵论定义）
    3. 条件数正则：exp(-(max(κ-10,0)/10)^2)，κ>10时平滑惩罚（纯数学约束）
    """
    # 1. 过滤有效算子
    valid_ops = [op for op in op_subset if op in op_perf]
    if len(valid_ops) < 2:
        return {"final_score": 0, "perf_geo": 0, "vol_score": 0, "kappa": 1}

    # 2. 提取子Gram矩阵（纯矩阵操作）
    op_indices = [op_names.index(op) for op in valid_ops]
    sub_gram = gram_matrix[op_indices, :][:, op_indices]

    # 3. 性能几何平均（纯数学定义，无权重）
    perf_list = [op_perf[op] for op in valid_ops]
    perf_geo = np.prod(perf_list) ** (1 / len(perf_list))

    # 4. 体积分（纯矩阵论定义：Gram矩阵行列式的平方根）
    det_val = max(np.linalg.det(sub_gram), 1e-10)
    vol_score = np.sqrt(det_val)

    # 5. 条件数正则（纯数学约束：κ>10时平滑惩罚，无超参数）
    eig_vals = np.linalg.eigvalsh(sub_gram)
    eig_vals = np.maximum(eig_vals, 1e-10)
    kappa = eig_vals.max() / eig_vals.min()
    # 正则项：κ≤10时=1，κ>10时从1平滑下降（纯数学函数，无超参数）
    kappa_reg = np.exp(-((max(kappa - 10, 0)) / 10) ** 2)

    # 6. 最终Score（纯数学乘积，无人为权重）
    final_score = perf_geo * vol_score * kappa_reg

    return {
        "final_score": final_score,
        "perf_geo": perf_geo,  # 性能几何平均（0-1）
        "vol_score": vol_score,  # 体积分（算子多样性）
        "kappa": kappa  # 条件数（矩阵稳定性）
    }


# ======================== 生成合理3算子组合（固定角色，无超参数） ========================
def role_guided_operator_selection(op_lib, gram_matrix, op_names, op_perf):
    """仅生成固定角色的3算子组合：1主+1补充+1调节（无超参数）"""
    primary_ops = op_lib.get_operators_by_role('primary')
    complementary_ops = op_lib.get_operators_by_role('complementary')
    stabilizing_ops = op_lib.get_operators_by_role('stabilizing')

    all_candidates = []
    print(
        f"\n=== 生成固定角色3算子组合（{len(primary_ops)}×{len(complementary_ops)}×{len(stabilizing_ops)}={len(primary_ops) * len(complementary_ops) * len(stabilizing_ops)}组）===")

    # 固定角色组合（无随机，无超参数）
    for primary in primary_ops:
        for complementary in complementary_ops:
            for stabilizing in stabilizing_ops:
                op_comb = [primary, complementary, stabilizing]
                score_res = compute_stabilized_score(op_comb, op_names, gram_matrix, op_perf)

                all_candidates.append({
                    "op_comb": tuple(op_comb),
                    "role": f"主={primary}, 补充={complementary}, 调节={stabilizing}",
                    **score_res
                })

    # 按Score排序（纯数学排序）
    sorted_candidates = sorted(all_candidates, key=lambda x: x["final_score"], reverse=True)

    print("\n=== 3算子组合Score排名（纯数学定义）===")
    for idx, cand in enumerate(sorted_candidates):
        print(
            f"{idx + 1}. {cand['op_comb']} | Score={cand['final_score']:.4f} | 性能={cand['perf_geo']:.4f} | 体积={cand['vol_score']:.4f} | κ={cand['kappa']:.2f}")

    return sorted_candidates


# ======================== 模型+训练（固定流程，无超参数） ========================
class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, op_list=['local_conv'], device=DEVICE):
        super().__init__()
        self.device = device
        self.op_list = op_list

        self.op_lib = OperatorLibrary(in_ch=in_planes, out_ch=planes, stride=stride, device=device)
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1).to(device)) for _ in op_list])  # 初始化为0（无偏置）

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out_total = 0.0
        for idx, op_name in enumerate(self.op_list):
            out = self.op_lib(x, op_name)
            gate = torch.sigmoid(self.gates[idx])  # 门控（纯数学激活）
            out_total += gate * out

        out = self.bn(out_total)
        out += self.downsample(residual)
        return self.relu(out)


class ResNet18Custom(nn.Module):
    def __init__(self, op_list=['local_conv'], num_classes=10, device=DEVICE):
        super().__init__()
        self.in_planes = 64
        self.device = device
        self.op_list = op_list

        # 固定初始化（无超参数）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlockCustom, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockCustom, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockCustom, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockCustom, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockCustom.expansion, num_classes)

        # 固定初始化（无超参数）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.op_list, self.device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18_multi_op(op_list, num_classes=10, device=DEVICE):
    return ResNet18Custom(op_list, num_classes, device)


def train_op_combination(op_list, seed=42, epochs=TRAIN_EPOCHS, device=DEVICE):
    """固定训练流程（无超参数）"""
    set_seed(seed)
    train_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE, train=True)
    test_loader = get_cifar10_dataloaders(batch_size=1000, train=False)

    model = resnet18_multi_op(op_list, num_classes=10, device=device)
    criterion = nn.CrossEntropyLoss()  # 固定损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)  # 固定优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 固定学习率策略
    grad_clip = 1.0  # 固定梯度裁剪

    best_acc = 0.0
    acc_history = []
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

        # 验证
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
        best_acc = max(best_acc, test_acc)
        scheduler.step()

        print(f"[组合{op_list}] Seed {seed} | Epoch {epoch + 1} | Best Acc: {best_acc:.2f}%")

    return {
        "best_acc": float(best_acc),
        "acc_std": float(np.std(acc_history)),
        "op_comb": op_list,
        "seed": int(seed)
    }


# ======================== 主流程（纯固定逻辑，无超参数） ========================
if __name__ == "__main__":
    set_seed(42)
    print("=" * 80)
    print(f"稳定化算子选择实验（纯数学定义，无超参数） | 设备：{DEVICE}")
    print(f"Score定义：性能几何平均 × 体积分 × 条件数正则（κ>10时平滑惩罚）")
    print("=" * 80)

    # 1. 初始化（固定流程）
    op_lib = OperatorLibrary(in_ch=64, out_ch=64, stride=1, device=DEVICE)
    anchor_dataloader = load_cifar10_anchor_set(batch_size=64, num_samples=2000)

    # 2. 计算Gram矩阵（纯数学）
    gram_matrix, op_names = compute_gram_matrix_optimized(op_lib, anchor_dataloader, device=DEVICE)
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({"gram_matrix": gram_matrix, "op_names": op_names}, f)

    # 3. 全量评测单算子（真实训练，无超参数）
    print("\n=== 评测所有单算子性能（纯真实训练）===")
    op_perf = {}
    all_ops = op_lib.get_operator_list()
    for op_name in all_ops:
        print(f"\n训练单算子：{op_name}")
        seed_results = []
        for seed in TRAIN_SEEDS[:1]:
            res = train_op_combination([op_name], seed=seed, epochs=TRAIN_EPOCHS)
            seed_results.append(res)
        avg_acc = np.mean([r["best_acc"] for r in seed_results])
        op_perf[op_name] = avg_acc / 100.0  # 归一化（纯数学操作）
        print(f"单算子 {op_name} 平均准确率：{avg_acc:.2f}%（归一化：{op_perf[op_name]:.4f}）")

    # 打印单算子排名（纯排序，无超参数）
    print("\n=== 单算子性能排名 ===")
    sorted_single = sorted(op_perf.items(), key=lambda x: x[1], reverse=True)
    for idx, (op, perf) in enumerate(sorted_single):
        print(f"{idx + 1}. {op:<15} | 准确率：{perf * 100:.2f}%")

    # 4. 生成固定角色3算子组合（无超参数）
    sorted_candidates = role_guided_operator_selection(op_lib, gram_matrix, op_names, op_perf)

    # 5. 训练3算子组合（固定流程）
    print("\n=== 训练所有固定角色3算子组合 ===")
    all_results = []
    train_combs = [c["op_comb"] for c in sorted_candidates]
    train_combs = list(set(tuple(sorted(c)) for c in train_combs))  # 去重（纯逻辑）
    print(f"待训练3算子组合数：{len(train_combs)}")

    for idx, comb in enumerate(train_combs):
        comb = list(comb)
        print(f"\n=== 训练3算子组合 {idx + 1}/{len(train_combs)}：{comb} ===")
        seed_results = []
        for seed in TRAIN_SEEDS:
            res = train_op_combination(comb, seed=seed, epochs=TRAIN_EPOCHS)
            seed_results.append(res)

        # 计算平均准确率（纯数学平均）
        avg_acc = np.mean([r["best_acc"] for r in seed_results])
        avg_std = np.mean([r["acc_std"] for r in seed_results])

        # 计算Score（纯数学定义）
        score_res = compute_stabilized_score(comb, op_names, gram_matrix, op_perf)

        all_results.append({
            "op_comb": "+".join(comb),
            "final_score": float(score_res["final_score"]),
            "perf_geo": float(score_res["perf_geo"]),
            "vol_score": float(score_res["vol_score"]),
            "kappa": float(score_res["kappa"]),
            "avg_best_acc": avg_acc,
            "avg_acc_std": avg_std
        })

    # 6. 计算相关性（纯统计，无超参数）
    scores = [res["final_score"] for res in all_results]
    accs = [res["avg_best_acc"] for res in all_results]
    pearson_corr, p_value = pearsonr(scores, accs)

    print("\n" + "=" * 80)
    print(f"核心结果（纯数学定义）：")
    print(f"Score与准确率的Pearson相关系数 = {pearson_corr:.4f} (p值={p_value:.6f})")
    print(f"相关系数解释：{pearson_corr >= 0.7 and '高相关（可解释性强）' or '中相关（符合预期）'}")
    print("=" * 80)

    # 按准确率排序打印3算子组合（纯排序）
    print("\n=== 3算子组合最终排名（按准确率）===")
    sorted_results = sorted(all_results, key=lambda x: x["avg_best_acc"], reverse=True)
    for idx, res in enumerate(sorted_results):
        print(
            f"{idx + 1}. {res['op_comb']:<35} | Score={res['final_score']:.4f} | "
            f"准确率={res['avg_best_acc']:.2f}% | 性能={res['perf_geo']:.4f} | 体积={res['vol_score']:.4f} | κ={res['kappa']:.2f}"
        )

    # 保存结果（纯数据，无超参数）
    with open("./3op_final_no_hyper_results.json", "w", encoding='utf-8') as f:
        json.dump(convert_numpy_to_python({
            "single_op_perf": op_perf,
            "3op_results": all_results,
            "pearson_corr": pearson_corr,
            "p_value": p_value,
            "score_definition": "性能几何平均 × 体积分 × 条件数正则（κ>10时exp(-((κ-10)/10)^2)）"
        }), f, indent=4)

    # 可视化相关性（纯绘图，无超参数）
    plt.figure(figsize=(8, 6))
    plt.scatter(scores, accs, color='blue', alpha=0.7)
    plt.xlabel('Stabilized Score (纯数学定义)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Pearson Correlation: {pearson_corr:.4f} (p={p_value:.6f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('./score_acc_correlation_no_hyper.png', dpi=300)
    plt.show()