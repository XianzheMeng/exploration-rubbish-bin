import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import json
import torch.optim as optim
import numpy as np
import time
from PIL import Image
import os
import cv2
import torchvision.transforms as transforms

# ======================== 全局配置 ========================
SAVE_PATH = "./sr_anchor_dataset.pkl"
RESULT_PATH = "./sr_op_combination_results.json"

# 【GPU修改1】自动检测GPU，优先使用CUDA
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔧 使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"📌 GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"📌 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

SEED = 42
SCALE = 2
# 【显存适配】RTX 4050 6GB显存，批次降到4，避免溢出
BATCH_SIZE = 4
EPOCHS = 50  # 缩短训练轮数，加快运行
ANCHOR_NUM_SAMPLES = 100  # 锚点集样本数
OP_NAMES = [
    'pixel_shuffle', 'edge_enhance', 'non_local',
    'attention_conv', 'residual_dense', 'freq_attention'
]

# 数据集配置（本地生成，无需下载）
DATASET_NAME = "LocalGenerated"
DATASET_PATH = "./SR_LocalDataset"
HR_PATH = os.path.join(DATASET_PATH, "HR")
LR_PATH = os.path.join(DATASET_PATH, f"LR_x{SCALE}")
GENERATE_IMAGE_NUM = 50  # 生成50张不同的图像
# 【显存适配】图像尺寸从256→128，降低显存占用
IMAGE_SIZE = (128, 128)

# 【GPU修改3】开启cuDNN加速（GPU专属优化）
warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 自动选择最优卷积算法
    torch.backends.cudnn.deterministic = True  # 保证结果可复现
else:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


# ======================== 工具函数 ========================
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 【GPU修改4】设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def move_module_to_device(module, device):
    module.to(device)
    return module


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
    else:
        return obj


def generate_local_sr_dataset():
    """本地生成超分数据集（50张不同纹理的RGB图像）"""
    if os.path.exists(HR_PATH) and len(os.listdir(HR_PATH)) >= GENERATE_IMAGE_NUM:
        print(f"✅ 本地数据集已存在，共{len(os.listdir(HR_PATH))}张图像")
        return

    # 创建目录
    os.makedirs(HR_PATH, exist_ok=True)
    os.makedirs(LR_PATH, exist_ok=True)

    print(f"🎨 正在生成{GENERATE_IMAGE_NUM}张本地超分图像...")
    for idx in tqdm(range(GENERATE_IMAGE_NUM), desc="生成HR/LR图像"):
        # 生成不同纹理的随机图像（模拟自然场景）
        # 混合噪声、条纹、块纹理，增加多样性
        np.random.seed(SEED + idx)  # 每张图种子不同，保证多样性

        # 生成基础纹理
        base_texture = np.random.rand(*IMAGE_SIZE, 3) * 255

        # 添加条纹纹理
        stripe_freq = np.random.randint(5, 20)
        for c in range(3):
            base_texture[:, :, c] += 30 * np.sin(np.linspace(0, stripe_freq * np.pi, IMAGE_SIZE[1]))

        # 添加块纹理
        block_size = np.random.randint(10, 30)
        for i in range(0, IMAGE_SIZE[0], block_size):
            for j in range(0, IMAGE_SIZE[1], block_size):
                block_val = np.random.rand() * 50
                base_texture[i:i + block_size, j:j + block_size, :] += block_val

        # 归一化到0-255
        base_texture = np.clip(base_texture, 0, 255).astype(np.uint8)

        # 保存HR图像
        hr_img = Image.fromarray(base_texture)
        hr_filename = f"img_{idx:03d}.png"
        hr_img.save(os.path.join(HR_PATH, hr_filename))

        # 生成LR图像（下采样）
        lr_size = (IMAGE_SIZE[0] // SCALE, IMAGE_SIZE[1] // SCALE)
        lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        lr_img.save(os.path.join(LR_PATH, hr_filename))

    print(f"✅ 本地数据集生成完成：")
    print(f"   - HR目录：{HR_PATH} | 数量：{len(os.listdir(HR_PATH))}")
    print(f"   - LR目录：{LR_PATH} | 数量：{len(os.listdir(LR_PATH))}")
    print(f"   - 图像尺寸：HR={IMAGE_SIZE}, LR={IMAGE_SIZE[0] // SCALE}x{IMAGE_SIZE[1] // SCALE}")


# ======================== 6个SR专属算子库（维度统一版） ========================
class SROperatorLibrary(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, scale=SCALE, device=DEVICE):
        super().__init__()
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale = scale
        set_seed(SEED)

        # 1. PixelShuffle算子
        class PixelShuffleOp(nn.Module):
            def __init__(self, in_ch, out_ch, scale):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.scale = scale
                self.up_conv = nn.Conv2d(in_ch, out_ch * scale * scale, 3, padding=1, bias=False)
                self.ps = nn.PixelShuffle(scale)
                self.down_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=scale, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                nn.init.kaiming_normal_(self.up_conv.weight, mode='fan_out', nonlinearity='relu')

            def forward(self, x):
                input_size = x.shape[-2:]
                x = self.up_conv(x)
                x = self.ps(x)
                x = self.down_conv(x)
                if x.shape[-2:] != input_size:
                    x = F.interpolate(x, size=input_size, mode='bilinear')
                return self.bn(x)

        self.pixel_shuffle = PixelShuffleOp(in_ch, out_ch, scale)

        # 2. 边缘增强算子
        class EdgeEnhanceOp(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.edge_kernel = nn.Parameter(
                    torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
                    .unsqueeze(0).unsqueeze(0).repeat(in_ch, 1, 1, 1)
                )
                self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                edge = F.conv2d(x, self.edge_kernel, padding=1, groups=self.in_ch)
                x = x + 0.1 * edge
                x = self.conv(x)
                return self.relu(self.bn(x))

        self.edge_enhance = EdgeEnhanceOp(in_ch, out_ch)

        # 3. 非局部相似性算子
        class NonLocalOp(nn.Module):
            def __init__(self, in_ch, out_ch, reduction=2):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.ch = in_ch // reduction
                self.query = nn.Conv2d(in_ch, self.ch, 1, bias=False)
                self.key = nn.Conv2d(in_ch, self.ch, 1, bias=False)
                self.value = nn.Conv2d(in_ch, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)

            def forward(self, x):
                B, C, H, W = x.shape
                q = self.query(x).reshape(B, self.ch, -1).permute(0, 2, 1)
                k = self.key(x).reshape(B, self.ch, -1)
                v = self.value(x).reshape(B, self.out_ch, -1).permute(0, 2, 1)

                attn = F.softmax(torch.bmm(q, k) / np.sqrt(self.ch), dim=-1)
                out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, self.out_ch, H, W)
                return self.bn(out + x)

        self.non_local = NonLocalOp(in_ch, out_ch)

        # 4. 通道注意力卷积
        class AttentionConvOp(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(in_ch, in_ch // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_ch // 4, in_ch),
                    nn.Sigmoid()
                )
                self.bn = nn.BatchNorm2d(out_ch)

            def forward(self, x):
                B, C = x.shape[:2]
                attn = self.avg_pool(x).reshape(B, C)
                attn = self.fc(attn).reshape(B, C, 1, 1)
                x = x * attn
                x = self.conv(x)
                return self.bn(x)

        self.attention_conv = AttentionConvOp(in_ch, out_ch)

        # 5. 残差密集连接算子
        class ResidualDenseOp(nn.Module):
            def __init__(self, in_ch, out_ch, growth_rate=16):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.conv1 = nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False)
                self.conv2 = nn.Conv2d(in_ch + growth_rate, growth_rate, 3, padding=1, bias=False)
                self.conv3 = nn.Conv2d(in_ch + 2 * growth_rate, growth_rate, 3, padding=1, bias=False)
                self.conv4 = nn.Conv2d(in_ch + 3 * growth_rate, out_ch, 1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                x1 = self.relu(self.conv1(x))
                x2 = self.relu(self.conv2(torch.cat([x, x1], dim=1)))
                x3 = self.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
                x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
                return self.bn(x4 + x)

        self.residual_dense = ResidualDenseOp(in_ch, out_ch)

        # 6. 频域注意力算子
        class FreqAttentionOp(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
                self.bn = nn.BatchNorm2d(out_ch)
                self.fc = nn.Sequential(
                    nn.Linear(out_ch, out_ch // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_ch // 4, out_ch),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # 【GPU修改5】GPU下FFT需要指定float32（避免精度问题）
                x = x.to(torch.float32)
                x_fft = torch.fft.fft2(x, dim=(-2, -1))
                x_amp = torch.abs(x_fft)
                amp_avg = torch.mean(x_amp, dim=(-2, -1))
                attn = self.fc(amp_avg).unsqueeze(-1).unsqueeze(-1)
                x_ifft = torch.fft.ifft2(x_fft, dim=(-2, -1)).real
                x = self.conv(x_ifft * attn)
                return self.bn(x)

        self.freq_attention = FreqAttentionOp(in_ch, out_ch)

        # 算子字典
        self.operators = {
            'pixel_shuffle': self.pixel_shuffle,
            'edge_enhance': self.edge_enhance,
            'non_local': self.non_local,
            'attention_conv': self.attention_conv,
            'residual_dense': self.residual_dense,
            'freq_attention': self.freq_attention
        }

        move_module_to_device(self, self.device)

    def forward(self, x, op_name):
        x = x.to(self.device)
        out = self.operators[op_name](x)
        assert out.shape == x.shape, f"{op_name}维度错误：{out.shape} != {x.shape}"
        return F.relu(out)

    def get_operator_list(self):
        return list(self.operators.keys())


# ======================== 增强版SR数据集（本地生成） ========================
class SRDataset(Dataset):
    def __init__(self, hr_path=HR_PATH, lr_path=LR_PATH, scale=SCALE, is_train=True):
        self.scale = scale
        self.is_train = is_train
        self.hr_files = [f for f in os.listdir(hr_path) if f.endswith(('.png', '.jpg'))]
        self.lr_files = [f for f in os.listdir(lr_path) if f.endswith(('.png', '.jpg'))]

        # 确保HR/LR文件匹配
        self.common_files = list(set(self.hr_files) & set(self.lr_files))
        if len(self.common_files) == 0:
            raise ValueError("❌ 未找到匹配的HR/LR图像对")

        print(f"📊 数据集统计：共{len(self.common_files)}对HR/LR图像")

        # 数据变换
        crop_size = 64 if is_train else None  # 适配128尺寸，裁剪到64
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),  # 随机裁剪
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.hr_path = hr_path
        self.lr_path = lr_path

    def __len__(self):
        # 训练时重复采样，增加样本量
        return ANCHOR_NUM_SAMPLES if self.is_train else len(self.common_files)

    def __getitem__(self, idx):
        # 循环取图，避免索引越界
        file_name = self.common_files[idx % len(self.common_files)]

        # 读取HR和LR图像
        hr_img = Image.open(os.path.join(self.hr_path, file_name)).convert('RGB')
        lr_img = Image.open(os.path.join(self.lr_path, file_name)).convert('RGB')

        # 应用变换
        if self.is_train:
            # 同步裁剪/翻转（保证HR/LR变换一致）
            seed = torch.Generator().manual_seed(torch.initial_seed())
            # HR变换
            hr_img = self.transform(hr_img)
            # 重置种子，保证LR和HR变换一致
            torch.manual_seed(torch.initial_seed())
            lr_img = self.transform(lr_img)
        else:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img


def load_sr_anchor_set(batch_size=BATCH_SIZE):
    """加载锚点集（训练集）"""
    dataset = SRDataset(is_train=True)
    # 【GPU修改6】DataLoader开启pin_memory（GPU数据传输优化），调整num_workers
    num_workers = 0  # Windows下num_workers=4易报错，改为0更稳定
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader


def get_sr_dataloaders(batch_size=BATCH_SIZE):
    """获取训练/测试加载器"""
    train_dataset = SRDataset(is_train=True)
    test_dataset = SRDataset(is_train=False)

    # 【GPU修改7】GPU下DataLoader优化（Windows下num_workers=0）
    num_workers = 0
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader


# ======================== Gram矩阵计算 ========================
def compute_gram_matrix(op_lib, dataloader):
    op_lib.eval()
    op_names = op_lib.get_operator_list()
    num_ops = len(op_names)
    gram_matrix = torch.zeros(num_ops, num_ops, device=DEVICE)  # 【GPU修改8】直接在GPU创建矩阵
    total_samples = 0

    # 适配RGB图像（3通道→64通道）
    feature_extractor = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((64, 64))
    ).to(DEVICE)

    with torch.no_grad():
        for lr_imgs, _ in tqdm(dataloader, desc="计算Gram矩阵"):
            lr_imgs = lr_imgs.to(DEVICE, non_blocking=True)  # 【GPU修改9】non_blocking加速数据传输
            B = lr_imgs.shape[0]
            total_samples += B

            x = feature_extractor(lr_imgs)
            op_outputs = []
            for op_name in op_names:
                out = op_lib(x, op_name)
                out_flat = out.reshape(B, -1)
                out_flat = F.normalize(out_flat, p=2, dim=1)
                op_outputs.append(out_flat)

            # 计算Gram矩阵
            for i in range(num_ops):
                for j in range(num_ops):
                    inner_product = (op_outputs[i] * op_outputs[j]).sum(dim=1).mean()
                    gram_matrix[i, j] += inner_product * B

        # 修正：归一化放在循环外（原代码缩进错误）
        gram_matrix /= total_samples if total_samples > 0 else 1

    # 【GPU修改10】仅在最后将结果移到CPU转numpy（减少数据传输）
    return gram_matrix.cpu().numpy(), op_names


# ======================== SR评估指标 ========================
def calculate_psnr(img1, img2, max_val=1.0):
    # 【GPU修改11】保证在同一设备计算
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_val / torch.sqrt(mse)) if mse > 0 else float('inf')


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    # 【GPU修改12】SSIM计算适配GPU，窗口在GPU创建
    img1 = img1.to(DEVICE)
    img2 = img2.to(DEVICE)
    B, C, H, W = img1.shape
    window = torch.ones(window_size, window_size, device=img1.device) / (window_size ** 2)
    window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1 * mu2
    sigma1 = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1 ** 2
    sigma2 = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2 ** 2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return torch.mean(ssim)


# ======================== SR网络（适配RGB） ========================
class SRNet(nn.Module):
    def __init__(self, op_names_list, scale=SCALE):
        super().__init__()
        self.op_names = op_names_list
        self.scale = scale

        # 特征提取（适配RGB 3通道）
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 算子库
        self.op_lib = SROperatorLibrary(in_ch=64, out_ch=64, scale=scale)
        self.op_weights = nn.Parameter(torch.ones(len(op_names_list)) / len(op_names_list))

        # 上采样头（输出RGB 3通道）
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

        move_module_to_device(self, DEVICE)

    def forward(self, x):
        x = self.feat_extract(x)
        # 算子组合融合
        op_outputs = [self.op_lib(x, name) for name in self.op_names]
        weights = F.softmax(self.op_weights, dim=0)
        x = sum([w * out for w, out in zip(weights, op_outputs)])
        # 上采样
        x = self.upsample(x)
        x = (x + 1) / 2  # 归一化到[0,1]
        return x


# ======================== 算子组合评估 ========================
def evaluate_op_combination(op_names_list):
    """评估单个算子/算子组合的SR性能"""
    set_seed(SEED)
    train_loader, test_loader = get_sr_dataloaders(batch_size=BATCH_SIZE)

    # 构建模型
    model = SRNet(op_names_list, scale=SCALE).to(DEVICE)
    criterion = nn.MSELoss().to(DEVICE)  # 【GPU修改13】损失函数移到GPU
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 调整学习率

    best_psnr = 0.0
    best_ssim = 0.0

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} | 组合：{'+'.join(op_names_list)}")

        for lr_imgs, hr_imgs in pbar:
            # 【GPU修改14】non_blocking加速GPU数据传输
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE, non_blocking=True), hr_imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            sr_imgs = model(lr_imgs)
            # 裁剪HR到SR尺寸
            hr_imgs = F.interpolate(hr_imgs, size=sr_imgs.shape[-2:], mode='bicubic')
            loss = criterion(sr_imgs, hr_imgs)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        # 测试
        model.eval()
        test_psnr = 0.0
        test_ssim = 0.0
        total = 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in test_loader:
                lr_imgs, hr_imgs = lr_imgs.to(DEVICE, non_blocking=True), hr_imgs.to(DEVICE, non_blocking=True)
                sr_imgs = model(lr_imgs)
                hr_imgs = F.interpolate(hr_imgs, size=sr_imgs.shape[-2:], mode='bicubic')

                test_psnr += calculate_psnr(sr_imgs, hr_imgs, max_val=1.0).item()
                test_ssim += calculate_ssim(sr_imgs, hr_imgs, max_val=1.0).item()
                total += 1

        avg_psnr = test_psnr / total if total > 0 else 0
        avg_ssim = test_ssim / total if total > 0 else 0

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ssim = avg_ssim

        scheduler.step()

        # 【GPU修改15】训练中清理GPU缓存（避免显存溢出）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "op_comb": op_names_list,
        "best_psnr": round(best_psnr, 2),
        "best_ssim": round(best_ssim, 4),
        "seed": SEED
    }


# ======================== 主流程 ========================
if __name__ == "__main__":
    print("=" * 80)
    print(f"开始SR算子全组合评估（{DATASET_NAME}数据集 + 6算子）")
    print(f"🔧 运行设备: {DEVICE}")
    print("=" * 80)

    # 步骤0：生成本地数据集
    print("\n【步骤0/5】生成本地超分数据集...")
    generate_local_sr_dataset()

    # 步骤1：初始化算子库
    print("\n【步骤1/5】初始化SR算子库...")
    op_lib = SROperatorLibrary(in_ch=64, out_ch=64, scale=SCALE)
    print(f"✅ 算子库初始化完成，可用算子：{op_lib.get_operator_list()}")

    # 步骤2：计算Gram矩阵
    print("\n【步骤2/5】计算6个算子的Gram矩阵")
    anchor_loader = load_sr_anchor_set(batch_size=BATCH_SIZE)
    gram_matrix, op_names = compute_gram_matrix(op_lib, anchor_loader)

    # 保存+可视化Gram矩阵
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump({'gram_matrix': gram_matrix, 'op_names': op_names}, f)
    plt.figure(figsize=(10, 8))
    sns.heatmap(gram_matrix, annot=True, fmt='.3f', xticklabels=op_names, yticklabels=op_names,
                cmap='viridis')
    plt.title(f'{DATASET_NAME} SR算子Gram矩阵（{SCALE}×）')
    plt.tight_layout()
    plt.savefig('./sr_gram_matrix.png', dpi=300)
    plt.show()
    print(f"✅ Gram矩阵已保存到 {SAVE_PATH}")

    # 步骤3：评估6个单算子
    print("\n【步骤3/5】评估6个单算子")
    single_results = []
    for op_name in OP_NAMES:
        print(f"\n--- 评估单算子：{op_name} ---")
        res = evaluate_op_combination([op_name])
        single_results.append(res)
        print(f"单算子 {op_name} | PSNR: {res['best_psnr']}dB | SSIM: {res['best_ssim']}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 步骤4：评估所有两算子组合
    print("\n【步骤4/5】评估所有两算子组合（共15个）")
    two_combs = list(itertools.combinations(OP_NAMES, 2))
    two_results = []
    for idx, comb in enumerate(two_combs):
        comb_list = list(comb)
        print(f"\n--- 评估两算子组合 {idx + 1}/15：{'+'.join(comb_list)} ---")
        res = evaluate_op_combination(comb_list)
        two_results.append(res)
        print(f"组合 {comb_list[0]}+{comb_list[1]} | PSNR: {res['best_psnr']}dB | SSIM: {res['best_ssim']}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 步骤5：评估所有三算子组合
    print("\n【步骤5/5】评估所有三算子组合（共20个）")
    three_combs = list(itertools.combinations(OP_NAMES, 3))
    three_results = []
    for idx, comb in enumerate(three_combs):
        comb_list = list(comb)
        print(f"\n--- 评估三算子组合 {idx + 1}/20：{'+'.join(comb_list)} ---")
        res = evaluate_op_combination(comb_list)
        three_results.append(res)
        print(f"组合 {'+'.join(comb_list)} | PSNR: {res['best_psnr']}dB | SSIM: {res['best_ssim']}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总结果
    all_results = {
        "dataset": DATASET_NAME,
        "single_op": single_results,
        "two_op": two_results,
        "three_op": three_results,
        "config": {
            "scale": SCALE,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "operators": OP_NAMES,
            "image_count": len(os.listdir(HR_PATH)),
            "image_size": IMAGE_SIZE,
            "device": str(DEVICE)  # 【GPU修改16】记录运行设备
        }
    }

    # 保存结果
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_python(all_results), f, indent=4, ensure_ascii=False)

    # 输出排名
    print("\n" + "=" * 80)
    print(f"{DATASET_NAME}数据集评估结果汇总")
    print("=" * 80)

    # 单算子排名
    single_sorted = sorted(single_results, key=lambda x: x['best_psnr'], reverse=True)
    print("\n【单算子PSNR排名】")
    for i, res in enumerate(single_sorted):
        print(f"{i + 1}. {res['op_comb'][0]}: {res['best_psnr']}dB (SSIM: {res['best_ssim']})")

    # 两算子组合排名（前5）
    two_sorted = sorted(two_results, key=lambda x: x['best_psnr'], reverse=True)
    print("\n【两算子组合PSNR排名（前5）】")
    for i, res in enumerate(two_sorted[:15]):
        print(f"{i + 1}. {'+'.join(res['op_comb'])}: {res['best_psnr']}dB (SSIM: {res['best_ssim']})")

    # 三算子组合排名（前5）
    three_sorted = sorted(three_results, key=lambda x: x['best_psnr'], reverse=True)
    print("\n【三算子组合PSNR排名（前5）】")
    for i, res in enumerate(three_sorted[:20]):
        print(f"{i + 1}. {'+'.join(res['op_comb'])}: {res['best_psnr']}dB (SSIM: {res['best_ssim']})")

    print(f"\n✅ 所有结果已保存到：{RESULT_PATH}")
    print(f"✅ Gram矩阵可视化：./sr_gram_matrix.png")
    print(f"✅ 本地数据集路径：{DATASET_PATH}")

    # 最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🗑️ GPU缓存已清理，显存使用：{torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")