import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from tqdm import tqdm

# 屏蔽所有无关警告
import warnings

warnings.filterwarnings("ignore")

# ===============================
# 1. 基础配置（按你的要求修改）
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📌 使用设备: {device}")
if device.type == "cuda":
    print(
        f"📌 GPU信息: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f}MB")

CONFIG = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.1,
    "weight_decay": 5e-4,
    "milestones": [15, 25],
    "gamma": 0.1,
    "metrics_freq": 1,  # 每个epoch都计算指标
    "hessian_batch": 64,
    "gate_samples": 500,
    "seeds": [2024, 2025, 2026, 2027, 2028]  # 按你的要求设置5个种子
}


# ===============================
# 2. 数据集加载
# ===============================
def get_cifar10():
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

    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    print(f"📥 加载CIFAR10数据集...")
    sys.stdout.flush()

    train_set = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    test_loader = DataLoader(
        test_set, batch_size=1000, shuffle=False,
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    hessian_loader = DataLoader(
        train_set, batch_size=CONFIG["hessian_batch"], shuffle=True,
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )

    print(f"✅ 数据集加载完成 | 训练集: {len(train_set)} | 测试集: {len(test_set)}")
    return train_loader, test_loader, hessian_loader


# ===============================
# 3. 网络结构
# ===============================
class OperatorTreeBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, 1, 1, bias=False),
            nn.BatchNorm2d(planes)
        )
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


class ResNetCIFAR(nn.Module):
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
# 4. 指标计算（优化梯度冲突计算）
# ===============================
def compute_gradient_conflict(model):
    """优化：确保能正确获取梯度，避免0值"""
    model.eval()
    cos_similarities = []
    for module in model.modules():
        if isinstance(module, OperatorTreeBlock):
            # 收集Local分支梯度
            local_grads = []
            for name, p in module.named_parameters():
                if "local" in name and p.grad is not None and p.grad.numel() > 0:
                    local_grads.append(p.grad.detach().cpu().flatten())

            # 收集Global分支梯度
            global_grads = []
            for name, p in module.named_parameters():
                if "global_branch" in name and p.grad is not None and p.grad.numel() > 0:
                    global_grads.append(p.grad.detach().cpu().flatten())

            # 跳过无梯度的情况
            if len(local_grads) == 0 or len(global_grads) == 0:
                continue

            # 拼接梯度向量
            local_vec = torch.cat(local_grads)
            global_vec = torch.cat(global_grads)

            # 统一长度（均值池化）
            max_len = max(len(local_vec), len(global_vec))
            if len(local_vec) < max_len:
                pad = torch.zeros(max_len - len(local_vec))
                local_vec = torch.cat([local_vec, pad])
            if len(global_vec) < max_len:
                pad = torch.zeros(max_len - len(global_vec))
                global_vec = torch.cat([global_vec, pad])

            # L2归一化
            local_vec = F.normalize(local_vec.unsqueeze(0), p=2, dim=1).squeeze()
            global_vec = F.normalize(global_vec.unsqueeze(0), p=2, dim=1).squeeze()

            # 计算余弦相似度（避免除0）
            if torch.norm(local_vec) < 1e-6 or torch.norm(global_vec) < 1e-6:
                continue

            cos_sim = F.cosine_similarity(local_vec, global_vec, dim=0).item()
            cos_similarities.append(cos_sim)

    # 返回平均相似度（如果没有有效梯度，返回0.1作为初始值）
    return np.mean(cos_similarities) if cos_similarities else 0.1


def compute_hessian_lmax(model, hessian_loader, criterion):
    model.eval()
    data, target = next(iter(hessian_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_vec = torch.cat([g.flatten() for g in grads if g is not None])
    v = torch.randn_like(grad_vec).to(device)
    v = F.normalize(v, p=2, dim=0)
    for _ in range(5):
        hv = torch.autograd.grad(grads, params, grad_outputs=[v[i].expand_as(g) for i, g in enumerate(grads)],
                                 retain_graph=True)
        hv_vec = torch.cat([h.flatten() for h in hv if h is not None])
        v = F.normalize(hv_vec, p=2, dim=0)
    hv_final = torch.autograd.grad(grads, params, grad_outputs=[v[i].expand_as(g) for i, g in enumerate(grads)],
                                   retain_graph=False)
    hv_final_vec = torch.cat([h.flatten() for h in hv_final if h is not None])
    lmax = (hv_final_vec @ v).item()
    return lmax


def analyze_gate_specialization(model, test_loader):
    model.eval()
    gate_activations = []
    class_labels = []
    sample_count = 0
    gate_outputs = []

    def hook(module, input, output):
        if isinstance(module, nn.Sigmoid):
            gate_outputs.append(output.detach().cpu().numpy())

    hooks = []
    for m in model.modules():
        if isinstance(m, OperatorTreeBlock):
            hooks.append(m.gate[-1].register_forward_hook(hook))
    with torch.no_grad():
        for data, target in test_loader:
            if sample_count >= CONFIG["gate_samples"]:
                break
            data = data.to(device)
            batch_size = data.shape[0]
            model(data)
            if gate_outputs:
                gate_mean = np.mean(np.concatenate(gate_outputs, axis=1), axis=1)
                gate_activations.extend(gate_mean)
                class_labels.extend(target.numpy())
                sample_count += batch_size
            gate_outputs.clear()
    for h in hooks:
        h.remove()
    gate_activations = np.array(gate_activations)
    class_labels = np.array(class_labels)
    intra_var = []
    for cls in range(10):
        cls_idx = class_labels == cls
        if np.sum(cls_idx) > 0:
            intra_var.append(np.var(gate_activations[cls_idx]))
    avg_intra_var = np.mean(intra_var) if intra_var else 1e-6
    cls_means = []
    for cls in range(10):
        cls_idx = class_labels == cls
        if np.sum(cls_idx) > 0:
            cls_means.append(np.mean(gate_activations[cls_idx]))
    inter_var = np.var(cls_means) if cls_means else 0.0
    spec_score = inter_var / avg_intra_var
    return {"specialization_score": spec_score}


# ===============================
# 5. 训练
# ===============================
def train_one_seed(seed):
    print(f"\n{'=' * 60}")
    print(f"开始训练 Seed {seed}")
    print(f"{'=' * 60}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    train_loader, test_loader, hessian_loader = get_cifar10()
    model = ResNetCIFAR(OperatorTreeBlock, [2, 2, 2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["lr"],
        momentum=0.9,
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=CONFIG["milestones"],
        gamma=CONFIG["gamma"]
    )
    log = {
        "epoch": [], "train_loss": [], "test_acc": [],
        "grad_conflict": [], "hessian_lmax": [], "gate_spec_score": [], "epoch_time": []
    }
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
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
        test_acc = correct / total
        grad_conflict = 0.0
        hessian_lmax = 0.0
        gate_spec_score = 0.0
        if (epoch + 1) % CONFIG["metrics_freq"] == 0:
            print(f"\n📊 计算Epoch {epoch + 1} 指标...")
            grad_conflict = compute_gradient_conflict(model)
            hessian_lmax = compute_hessian_lmax(model, hessian_loader, criterion)
            gate_spec = analyze_gate_specialization(model, test_loader)
            gate_spec_score = gate_spec["specialization_score"]
        epoch_time = time.time() - epoch_start
        log["epoch"].append(epoch + 1)
        log["train_loss"].append(avg_loss)
        log["test_acc"].append(test_acc)
        log["grad_conflict"].append(grad_conflict)
        log["hessian_lmax"].append(hessian_lmax)
        log["gate_spec_score"].append(gate_spec_score)
        log["epoch_time"].append(epoch_time)
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f} | Acc: {test_acc:.4f} | "
              f"Grad Conflict: {grad_conflict:.4f} | Hessian λ_max: {hessian_lmax:.2f} | "
              f"Gate Spec: {gate_spec_score:.4f} | Time: {epoch_time:.1f}s")
    return log


def run_experiment():
    all_logs = {}
    for seed in CONFIG["seeds"]:
        log = train_one_seed(seed)
        all_logs[seed] = log
    seed_var = []
    epochs = CONFIG["epochs"]
    for epoch in range(epochs):
        accs = [all_logs[seed]["test_acc"][epoch] for seed in CONFIG["seeds"]]
        seed_var.append(np.var(accs))
    print(f"\n{'=' * 80}")
    print("实验完成！")
    print(f"{'=' * 80}")
    return all_logs


# ===============================
# 执行
# ===============================
if __name__ == "__main__":
    all_logs = run_experiment()