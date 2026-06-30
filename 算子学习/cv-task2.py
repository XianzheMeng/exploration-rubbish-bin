import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# =========================
# 1. Toy Dataset
# =========================
def create_dataset(n=1000):
    x = torch.randn(n, 1, 28, 28)
    y = (x.mean(dim=[1,2,3]) > 0).long()
    return TensorDataset(x, y)

train_loader = DataLoader(create_dataset(2000), batch_size=64, shuffle=True)
anchor_loader = DataLoader(create_dataset(64), batch_size=32, shuffle=False)

# =========================
# 2. Backbone Network
# =========================
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8,8))

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.pool(x)

# =========================
# 3. Heterogeneous Operators
# =========================
class LocalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
    def forward(self, x):
        return F.relu(self.conv(x))

class DilatedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=2, dilation=2)
    def forward(self, x):
        return F.relu(self.conv(x))

class SpectralConv(nn.Module):
    def forward(self, x):
        fft = torch.fft.fft2(x)
        return torch.real(torch.fft.ifft2(fft))

# =========================
# 4. Full Model
# =========================
class Model(nn.Module):
    def __init__(self, operators):
        super().__init__()
        self.backbone = Backbone()
        self.operators = nn.ModuleList(operators)
        self.classifier = nn.Linear(16*8*8, 2)

    def forward(self, x):
        z = self.backbone(x)
        outs = [op(z) for op in self.operators]
        z = sum(outs) / len(outs)
        z = z.flatten(1)
        return self.classifier(z)

# =========================
# 5. Dynamic Geometry Tracker
# =========================
class GeometryTracker:
    def __init__(self, model, anchor_loader):
        self.model = model
        self.anchor_loader = anchor_loader
        self.gram_history = []
        self.volume_history = []
        self.trajectory_length = 0.0

    @torch.no_grad()
    def compute_gram(self):
        self.model.eval()
        K = len(self.model.operators)
        features = []

        for op in self.model.operators:
            op_feats = []
            for x, _ in self.anchor_loader:
                x = x.to(device)
                z = self.model.backbone(x)
                h = op(z)
                h = h.flatten(1)
                h = F.normalize(h, dim=1)
                op_feats.append(h)
            op_feats = torch.cat(op_feats, dim=0)
            features.append(op_feats)

        G = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                G[i,j] = torch.mean(torch.sum(features[i]*features[j], dim=1))

        return G

    def update(self):
        G = self.compute_gram()
        self.gram_history.append(G)

        det = torch.det(G).clamp(min=0)
        volume = torch.sqrt(det)
        self.volume_history.append(volume.item())

        if len(self.gram_history) > 1:
            diff = torch.norm(G - self.gram_history[-2], p="fro")
            self.trajectory_length += diff.item()

    def analyze_last(self):
        G = self.gram_history[-1]
        eigvals = torch.linalg.eigvals(G).real
        condition_number = (eigvals.max() / eigvals.min()).item()
        p = eigvals / eigvals.sum()
        entropy = -(p * torch.log(p + 1e-8)).sum().item()
        return condition_number, entropy

# =========================
# 6. Training
# =========================
operators = [LocalConv(), DilatedConv(), SpectralConv()]
model = Model(operators).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

tracker = GeometryTracker(model, anchor_loader)

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    if epoch % 3 == 0:
        tracker.update()
        print(f"Epoch {epoch}: Volume={tracker.volume_history[-1]:.4f}")

# =========================
# 7. Results
# =========================
print("\nTrajectory Length:", tracker.trajectory_length)

cond, entropy = tracker.analyze_last()
print("Final Condition Number:", cond)
print("Final Spectral Entropy:", entropy)

plt.plot(tracker.volume_history)
plt.title("Dynamic Operator Volume")
plt.xlabel("Time Step")
plt.ylabel("Volume")
plt.show()