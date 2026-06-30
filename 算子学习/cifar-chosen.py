import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import pickle

# ===================== 1. 基础配置（保证可复现） =====================
# 设置随机种子，确保每次运行选取的图片都相同
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if torch.cuda.is_available() else None

# 配置参数
DATA_ROOT = "./data"  # CIFAR-10数据集下载/读取路径
ANCHOR_SIZE = 5000    # 锚点数据集大小
SAVE_PATH = "./cifar10_anchor_dataset.pkl"  # 锚点数据集保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 2. 定义预处理流程（符合CIFAR-10标准） =====================
# 预处理：先转为Tensor，再标准化（使用CIFAR-10官方均值和标准差）
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图片转为Tensor，像素值从[0,255]归一化到[0,1]
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10训练集RGB通道均值
        std=[0.2023, 0.1994, 0.2010]    # CIFAR-10训练集RGB通道标准差
    )
])

# ===================== 3. 加载CIFAR-10训练集 =====================
# 下载/加载CIFAR-10训练集（仅训练集）
trainset = torchvision.datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=True,  # 首次运行会自动下载，后续改为False
    transform=transform
)

# ===================== 4. 随机选取5000张图片（可复现） =====================
# 获取训练集总长度，生成随机索引（无重复）
total_train_samples = len(trainset)
anchor_indices = np.random.choice(total_train_samples, size=ANCHOR_SIZE, replace=False)

# 根据索引选取子集
anchor_dataset = Subset(trainset, anchor_indices)

# ===================== 5. 整理数据并保存为固定文件 =====================
# 整理数据：包含图片张量、标签、原始索引（方便溯源）
anchor_data = {
    "images": [],
    "labels": [],
    "indices": anchor_indices.tolist()  # 保存选取的原始索引，便于复现
}

# 遍历锚点数据集，收集数据（可批量加载，避免内存溢出）
for idx in range(len(anchor_dataset)):
    img, label = anchor_dataset[idx]
    anchor_data["images"].append(img.to(DEVICE).cpu().numpy())  # 转为numpy便于保存
    anchor_data["labels"].append(label)

# 转换为numpy数组（更高效的存储和读取）
anchor_data["images"] = np.array(anchor_data["images"])
anchor_data["labels"] = np.array(anchor_data["labels"])

# 创建保存目录（如果不存在）
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 保存为pickle文件（二进制格式，保留数据结构）
with open(SAVE_PATH, "wb") as f:
    pickle.dump(anchor_data, f)

print(f"锚点数据集已保存至: {SAVE_PATH}")
print(f"数据集规模：{len(anchor_data['images'])} 张图片")
print(f"图片形状：{anchor_data['images'].shape} (样本数, 通道数, 高度, 宽度)")
print(f"标签范围：{np.min(anchor_data['labels'])} ~ {np.max(anchor_data['labels'])}")

# ===================== 6. 验证加载（可选） =====================
def load_anchor_dataset(save_path):
    """加载保存的锚点数据集"""
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    # 还原为tensor（如需使用）
    data["images"] = torch.from_numpy(data["images"]).to(DEVICE)
    data["labels"] = torch.from_numpy(data["labels"]).to(DEVICE)
    return data

# 验证加载
loaded_data = load_anchor_dataset(SAVE_PATH)
print("\n加载验证：")
print(f"加载后图片数量：{len(loaded_data['images'])}")
print(f"第一张图片形状：{loaded_data['images'][0].shape}")
print(f"第一张图片标签：{loaded_data['labels'][0].item()}")