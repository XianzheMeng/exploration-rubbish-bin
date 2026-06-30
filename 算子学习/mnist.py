import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 参数统计函数
# ===============================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===============================
# 公平 MLP
# ===============================
class MLP(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# ===============================
# 公平 Operator Tree
# ===============================
class FairOperatorTree(nn.Module):
    def __init__(self, hidden=256, branch_ratio=0.5):
        super().__init__()

        branch_dim = int(hidden * branch_ratio)

        # 共享底层
        self.shared = nn.Linear(28*28, hidden)

        # 两个分支（降维以控制参数量）
        self.branch1 = nn.Linear(hidden, branch_dim)
        self.branch2 = nn.Linear(hidden, branch_dim)

        # 门控（同维度）
        self.gate = nn.Linear(hidden, branch_dim)

        # 输出层
        self.out = nn.Linear(branch_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        h = torch.relu(self.shared(x))

        f1 = torch.relu(self.branch1(h))
        f2 = torch.relu(self.branch2(h))
        g = torch.sigmoid(self.gate(h))

        y = g * f1 + (1 - g) * f2
        return self.out(y)

# ===============================
# 训练函数
# ===============================
def train(model, train_loader, test_loader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Test Acc: {acc:.4f}")

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
# 数据加载
# ===============================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ===============================
# 运行对比
# ===============================
mlp = MLP(hidden=256)
op_tree = FairOperatorTree(hidden=256, branch_ratio=0.5)

print("MLP params:", count_params(mlp))
print("OperatorTree params:", count_params(op_tree))

print("\nTraining MLP...")
train(mlp, train_loader, test_loader)

print("\nTraining OperatorTree...")
train(op_tree, train_loader, test_loader)