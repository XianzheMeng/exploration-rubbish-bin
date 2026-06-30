import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略所有警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')


# ======================== 工具函数 ========================
def move_module_to_device(module, device):
    """递归将模块及其所有子模块移到指定设备"""
    module.to(device)
    for child in module.children():
        move_module_to_device(child, device)
    return module


# ======================== 1. 算子库定义（修复张量连续性） ========================
class OperatorLibrary(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, seed=42, device='cpu'):
        super().__init__()
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.in_ch = in_ch
        self.out_ch = out_ch

        # 基础算子
        self.local_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1, bias=False)
        self.dilated_conv = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=2, dilation=2, bias=False)

        # 1×1卷积堆叠（确保输出连续）
        class Conv1x1Stack(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.layers = nn.Sequential(
                    *[nn.Conv2d(in_ch, in_ch, 1, bias=False) for _ in range(8)],
                    nn.Conv2d(in_ch, out_ch, 1, bias=False)
                )
                for m in self.layers:
                    nn.init.eye_(m.weight.squeeze())

            def forward(self, x):
                out = self.layers(x)
                return out.contiguous()  # 确保张量连续

        self.conv1x1_stack = Conv1x1Stack(self.in_ch, self.out_ch)

        # 频域算子（修复张量连续性）
        class FreqOperator(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.freq_mul = nn.Parameter(torch.ones(in_ch, 23, 25))
                self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                nn.init.eye_(self.proj.weight.squeeze())

            def forward(self, x):
                x_fft = fft.fft2(x, dim=(-2, -1), norm='ortho')
                # 修复插值后的张量连续性
                freq_mul = F.interpolate(
                    self.freq_mul.unsqueeze(0),
                    size=x.shape[-2:],
                    mode='nearest'
                ).squeeze(0).contiguous()
                x_fft = x_fft * freq_mul
                x_out = fft.ifft2(x_fft, dim=(-2, -1), norm='ortho').real
                out = self.proj(x_out)
                return out.contiguous()  # 确保连续

        self.freq_op = FreqOperator(self.in_ch, self.out_ch)

        # 形态学算子（修复张量连续性）
        class MorphologyOp(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.weights = nn.Parameter(torch.ones(out_ch, in_ch, 9))
                nn.init.constant_(self.weights, 1.0 / 9)

            def forward(self, x):
                x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
                x_unfold = F.unfold(x_pad, 3)
                B, C9, HW = x_unfold.shape
                H, W = x.shape[2], x.shape[3]

                # 修复unfold后的张量形状和连续性
                x_unfold = x_unfold.reshape(B, self.in_ch, 9, H, W).contiguous()
                x_median = x_unfold.median(dim=2)[0].contiguous()

                x_median = x_median.permute(0, 2, 3, 1).contiguous()
                # 修复einsum后的张量连续性
                x_out = torch.einsum(
                    'bhwc, oci -> bhwo',
                    x_median,
                    self.weights.mean(dim=2, keepdim=True).contiguous()
                ).contiguous()
                out = x_out.permute(0, 3, 1, 2).contiguous()
                return out

        self.morph_op = MorphologyOp(self.in_ch, self.out_ch)

        # 通道重排算子（修复张量连续性）
        class ChannelPermute(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.perm_mats = nn.ParameterList([
                    nn.Parameter(torch.eye(in_ch)) for _ in range(7)
                ])
                self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.base_perm = torch.randperm(in_ch)
                nn.init.eye_(self.proj.weight.squeeze())

            def forward(self, x):
                # 修复索引后的张量连续性
                x_perm = x[:, self.base_perm, :, :].contiguous()
                for mat in self.perm_mats:
                    x_perm = torch.einsum('bchw, dc -> bdhw', x_perm, mat).contiguous()
                out = self.proj(x_perm)
                return out.contiguous()

        self.channel_perm = ChannelPermute(self.in_ch, self.out_ch)

        # 算子字典
        self.operators = {
            'local_conv': self.local_conv,
            'dilated_conv': self.dilated_conv,
            '1x1_conv': self.conv1x1_stack,
            'freq_op': self.freq_op,
            'morph_op': self.morph_op,
            'channel_perm': self.channel_perm
        }

        # 迁移所有参数到指定设备
        move_module_to_device(self, self.device)

    def forward(self, x, op_name):
        if op_name not in self.operators:
            raise ValueError(f"无效算子：{op_name}，可选：{list(self.operators.keys())}")
        x = x.to(self.device)
        return self.operators[op_name](x)

    def get_operator_list(self):
        return list(self.operators.keys())


# ======================== 2. CIFAR-10数据加载 ========================
def load_cifar10_anchor_set(batch_size=64, num_samples=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.Resize((32, 32)),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return dataloader


# ======================== 3. Gram矩阵计算（核心修复：用reshape替代view） ========================
def compute_gram_matrix(op_lib, dataloader, device='cpu'):
    op_lib.eval()
    op_names = op_lib.get_operator_list()
    num_ops = len(op_names)

    gram_matrix = torch.zeros(num_ops, num_ops, device=device)
    total_samples = 0

    # 特征提取器
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True)
    )
    feature_extractor = move_module_to_device(feature_extractor, device)
    feature_extractor.eval()

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="计算Gram矩阵"):
            images = images.to(device)
            B = images.shape[0]
            total_samples += B

            # 提取特征
            x = feature_extractor(images).contiguous()

            # 计算每个算子输出
            op_outputs = []
            for op_name in op_names:
                out = op_lib(x, op_name)
                # 核心修复：用reshape替代view，兼容不连续张量
                out_flat = out.reshape(B, -1)  # 替换view为reshape
                out_flat = F.normalize(out_flat, p=2, dim=1)
                op_outputs.append(out_flat)

            # 计算内积
            for i in range(num_ops):
                for j in range(num_ops):
                    inner_product = (op_outputs[i] * op_outputs[j]).sum(dim=1).mean()
                    gram_matrix[i, j] += inner_product * B

        gram_matrix /= total_samples

    return gram_matrix.cpu().numpy(), op_names


# ======================== 4. 结果保存与可视化 ========================
def save_gram_matrix(gram_matrix, op_names, save_path):
    data = {
        'gram_matrix': gram_matrix,
        'op_names': op_names,
        'dataset': 'CIFAR-10',
        'num_samples': 1000,
        'timestamp': np.datetime64('now')
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Gram矩阵已保存到：{save_path}")


def plot_gram_matrix(gram_matrix, op_names, save_fig=True):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        gram_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=op_names,
        yticklabels=op_names,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    plt.title('Gram Matrix of Operators on CIFAR-10 Anchor Set', fontsize=14)
    plt.xlabel('Operators', fontsize=12)
    plt.ylabel('Operators', fontsize=12)
    plt.tight_layout()
    if save_fig:
        plt.savefig('./cifar10_gram_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# ======================== 5. 主执行流程 ========================
if __name__ == "__main__":
    # 配置参数
    SAVE_PATH = "./cifar10_anchor_dataset.pkl"
    BATCH_SIZE = 64
    NUM_SAMPLES = 1000
    DEVICE = 'cpu'  # 稳定优先，推荐使用CPU
    print(f"使用设备：{DEVICE}")

    # 初始化算子库
    op_lib = OperatorLibrary(in_ch=64, out_ch=64, device=DEVICE)

    # 加载数据
    print("\n加载CIFAR-10锚点集...")
    dataloader = load_cifar10_anchor_set(batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES)

    # 计算Gram矩阵
    print("\n开始计算Gram矩阵...")
    gram_matrix, op_names = compute_gram_matrix(op_lib, dataloader, device=DEVICE)

    # 保存结果
    save_gram_matrix(gram_matrix, op_names, SAVE_PATH)

    # 可视化
    print("\n可视化Gram矩阵...")
    plot_gram_matrix(gram_matrix, op_names)

    # 打印关键信息
    print("\n=== Gram矩阵计算完成 ===")
    print(f"算子列表：{op_names}")
    print(f"Gram矩阵形状：{gram_matrix.shape}")
    try:
        det = np.linalg.det(gram_matrix)
        print(f"Gram矩阵行列式：{det:.6f}")
    except Exception as e:
        print(f"Gram矩阵行列式计算失败：{e}")
    print(f"Gram矩阵迹：{np.trace(gram_matrix):.6f}")

    print("\n✅ 所有流程完成，无报错！")