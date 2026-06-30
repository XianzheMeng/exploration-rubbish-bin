import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# 1. 工具
# ===============================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# 2. OperatorTree Block
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


# ===============================
# 3. ResNet
# ===============================

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)

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
# 4. 数据
# ===============================

def get_dataloaders():
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

    train_dataset = datasets.CIFAR10(root="./data", train=True,
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader


# ===============================
# 5. 梯度冲突计算
# ===============================

def compute_gradient_conflict(model):
    local_grads = []
    global_grads = []

    for module in model.modules():
        if isinstance(module, OperatorTreeBlock):

            for p in module.local.parameters():
                if p.grad is not None:
                    local_grads.append(p.grad.view(-1))

            for p in module.global_branch.parameters():
                if p.grad is not None:
                    global_grads.append(p.grad.view(-1))

    if len(local_grads) == 0 or len(global_grads) == 0:
        return None

    local_vec = torch.cat(local_grads)
    global_vec = torch.cat(global_grads)

    cos_sim = F.cosine_similarity(local_vec, global_vec, dim=0).item()
    return cos_sim


# ===============================
# 6. 训练
# ===============================

def train(model, epochs=50):
    train_loader, test_loader = get_dataloaders()
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 40], gamma=0.1
    )

    criterion = nn.CrossEntropyLoss()

    gradient_cos_history = []
    conflict_ratio_history = []

    for epoch in range(epochs):
        model.train()
        epoch_cos = []

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            cos_sim = compute_gradient_conflict(model)
            if cos_sim is not None:
                epoch_cos.append(cos_sim)

            optimizer.step()

        scheduler.step()

        avg_cos = np.mean(epoch_cos)
        conflict_ratio = np.mean(np.array(epoch_cos) < 0)

        gradient_cos_history.append(avg_cos)
        conflict_ratio_history.append(conflict_ratio)

        acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1} | Acc: {acc:.4f} | "
              f"Avg Cos: {avg_cos:.4f} | Conflict Ratio: {conflict_ratio:.4f}")

    return gradient_cos_history, conflict_ratio_history


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


# ===============================
# 7. 主函数
# ===============================

if __name__ == "__main__":
    set_seed()

    model = ResNet(OperatorTreeBlock, [2, 2, 2])

    print("Total Params:", count_params(model))

    cos_history, conflict_history = train(model, epochs=50)

    # 画图
    plt.figure()
    plt.plot(cos_history)
    plt.title("Gradient Cosine Similarity")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.show()

    plt.figure()
    plt.plot(conflict_history)
    plt.title("Gradient Conflict Ratio (cos < 0)")
    plt.xlabel("Epoch")
    plt.ylabel("Conflict Ratio")
    plt.show()