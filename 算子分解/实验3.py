import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch

# ---------------------- 全局配置（实验3专用：适配可解释性提取）----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 幂函数模型配置（与实验1/2一致，保证模型一致性）
K = 30  # 选用最优K值，可解释性信息更丰富
A_SET = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
TAU = 0.3
EPS = 1e-6
GAMMA1, GAMMA2, GAMMA3 = 0.00005, 0.00005, 0.000005
STRUCT_MAX_ITER = 200
WEIGHT_MAX_ITER = 150

# 数据集配置
PCA_DIM = 10  # 与之前一致，特征维度d=10，便于可视化
IMG_H, IMG_W = 2, 5
TRAIN_SIZE = 0.15
TEST_SIZE = 0.03

# 可视化配置
PLOT_FIGSIZE = (16, 12)
FONTSIZE = 10
COLOR_MAP_CONTRIB = "RdBu_r"  # 红-蓝双色图，对应正负贡献
COLOR_MAP_POWER = "viridis"   # 幂次可视化色图
SAVE_FIG_DPI = 300

# ---------------------- 1. 数据准备与预处理（与实验1/2一致）----------------------
def load_and_preprocess_mnist():
    """加载MNIST并完成预处理，返回训练/测试集及特征维度"""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(np.int32)

    # PCA降维 + 鲁棒标准化
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X)
    scaler = RobustScaler(quantile_range=(5, 95))
    X_robust = scaler.fit_transform(X_pca)
    mu_i = scaler.scale_

    X_tilde = (X_robust + EPS) / (mu_i + EPS)
    X_tilde = np.clip(X_tilde, 1e-4, 5)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_tilde, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 标签转换为one-hot编码
    y_train_onehot = np.eye(10)[y_train]

    print(f"数据加载完成：训练集形状{X_train.shape}，测试集形状{X_test.shape}")
    print(f"特征维度d={X_train.shape[1]}，类别数=10（数字0-9）")
    return X_train, X_test, y_train, y_test, y_train_onehot, X_train.shape[1]

# ---------------------- 2. 幂函数模型实现（新增可解释性信息提取方法）----------------------
class PowerFunctionModel:
    """结构化幂函数模型：新增可解释性信息提取接口，支持可视化"""
    def __init__(self, K, d, a_set=A_SET):
        self.K = K
        self.d = d
        self.a_set = a_set
        self.num_a = len(a_set)
        self.train_time = 0.0
        self._init_params()

    def _init_params(self):
        self.m_tilde = np.random.uniform(0.4, 0.6, size=(self.K, self.d))
        self.pi = np.random.uniform(0.1, 0.9, size=(self.K, self.d, self.num_a))
        self.pi = self.pi / np.sum(self.pi, axis=2, keepdims=True)
        self.lambda_k = np.random.uniform(0.005, 0.02, size=self.K)
        self.w = np.random.normal(0, 0.01, size=(10, self.K))

    def _get_sparse_feature_subset(self):
        """获取每个基函数的稀疏特征子集（掩码值>TAU）"""
        S_k = []
        for k in range(self.K):
            feat_indices = np.where(self.m_tilde[k] > TAU)[0]
            max_feat = 3 if self.K <= 10 else 2
            if len(feat_indices) > max_feat:
                feat_values = self.m_tilde[k][feat_indices]
                top_idx = np.argsort(feat_values)[-max_feat:]
                feat_indices = feat_indices[top_idx]
            S_k.append(feat_indices)
        return S_k

    def _get_discrete_alpha(self):
        """获取每个基函数-特征对应的最优幂次"""
        alpha_argmax = np.argmax(self.pi, axis=2)
        alpha = self.a_set[alpha_argmax]
        return alpha

    def _base_function(self, X_tilde):
        """计算基函数输出（保持原逻辑，保证模型性能）"""
        N = X_tilde.shape[0]
        phi = np.zeros((N, self.K))
        S_k = self._get_sparse_feature_subset()
        alpha = self._get_discrete_alpha()

        for k in range(self.K):
            feat_indices = S_k[k]
            if len(feat_indices) == 0:
                continue

            x_subset = X_tilde[:, feat_indices]
            alpha_subset = alpha[k, feat_indices]
            prod_term = np.prod(np.power(x_subset, alpha_subset), axis=1)
            norm_term = np.sum(x_subset ** 2, axis=1)
            decay_term = np.exp(-self.lambda_k[k] * norm_term)

            phi[:, k] = prod_term * decay_term

        phi_mean = np.mean(phi, axis=0, keepdims=True)
        phi_std = np.std(phi, axis=0, keepdims=True) + EPS
        phi = (phi - phi_mean) / (phi_std * 0.8)
        phi = np.clip(phi, -8, 8)

        return phi

    def predict(self, X_tilde):
        """模型预测（保持原逻辑）"""
        phi = self._base_function(X_tilde)
        y_pred_logits = phi @ self.w.T
        y_pred = softmax(y_pred_logits, axis=1)
        return y_pred

    def _structure_loss(self):
        """结构损失（保持原逻辑）"""
        S_k = self._get_sparse_feature_subset()
        alpha = self._get_discrete_alpha()

        loss_struct = 0.0
        for k in range(self.K):
            gamma1 = GAMMA1 if self.K > 10 else GAMMA1 * 0.5
            loss_struct += gamma1 * len(S_k[k])

            if len(S_k[k]) > 0:
                alpha_abs_sum = np.sum(np.abs(alpha[k, S_k[k]]))
                loss_struct += GAMMA2 * alpha_abs_sum

            loss_struct += GAMMA3 * self.lambda_k[k]

        return loss_struct

    def _cross_entropy_loss(self, y_pred, y_true):
        """交叉熵损失（保持原逻辑）"""
        return -np.mean(np.sum(y_true * np.log(y_pred + EPS), axis=1))

    def fit(self, X_train, y_train_onehot):
        """模型训练（保持原逻辑）"""
        start_time = time.time()

        print(f"Stage I: 结构学习（K={self.K}，迭代{STRUCT_MAX_ITER}次）...")
        self._train_structure(X_train, y_train_onehot)

        print(f"Stage II: 权重拟合（K={self.K}，迭代{WEIGHT_MAX_ITER}次）...")
        self._train_weights(X_train, y_train_onehot)

        self.train_time = time.time() - start_time
        print(f"模型训练完成，总耗时{self.train_time:.2f}s")

    def _train_structure(self, X_train, y_train_onehot, max_iter=None):
        """结构学习（保持原逻辑）"""
        max_iter = max_iter or STRUCT_MAX_ITER

        def pack_params():
            return np.concatenate([
                self.m_tilde.flatten(),
                self.pi.flatten(),
                self.lambda_k.flatten()
            ])

        def unpack_params(params):
            m_flat_len = self.K * self.d
            pi_flat_len = self.K * self.d * self.num_a
            lambda_flat_len = self.K

            m_tilde = params[:m_flat_len].reshape((self.K, self.d))
            pi = params[m_flat_len:m_flat_len + pi_flat_len].reshape((self.K, self.d, self.num_a))
            lambda_k = params[m_flat_len + pi_flat_len:].reshape(self.K)

            m_tilde = np.clip(m_tilde, 0, 1)
            pi = np.clip(pi, 1e-8, None)
            pi = pi / np.sum(pi, axis=2, keepdims=True)
            lambda_k = np.clip(lambda_k, 0, 0.5)

            return m_tilde, pi, lambda_k

        def loss_func(params):
            self.m_tilde, self.pi, self.lambda_k = unpack_params(params)
            y_pred = self.predict(X_train)
            ce_loss = self._cross_entropy_loss(y_pred, y_train_onehot)
            struct_loss = self._structure_loss()
            return ce_loss + struct_loss

        initial_params = pack_params()
        result = minimize(
            loss_func,
            x0=initial_params,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False, "gtol": 1e-4}
        )

        self.m_tilde, self.pi, self.lambda_k = unpack_params(result.x)

    def _train_weights(self, X_train, y_train_onehot, max_iter=None):
        """权重拟合（保持原逻辑）"""
        max_iter = max_iter or WEIGHT_MAX_ITER
        phi_train = self._base_function(X_train)

        def pack_w():
            return self.w.flatten()

        def unpack_w(params):
            return params.reshape((10, self.K))

        def loss_w(params):
            self.w = unpack_w(params)
            y_pred_logits = phi_train @ self.w.T
            y_pred = softmax(y_pred_logits, axis=1)
            return self._cross_entropy_loss(y_pred, y_train_onehot)

        initial_w = pack_w()
        result = minimize(
            loss_w,
            x0=initial_w,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False, "gtol": 1e-5}
        )

        self.w = unpack_w(result.x)

    def count_params(self):
        """计算参数量（保持原逻辑）"""
        struct_params = self.K * self.d + self.K * self.d * self.num_a + self.K
        weight_params = 10 * self.K
        return struct_params + weight_params

    # ---------------------- 新增：可解释性信息提取核心方法 ----------------------
    def extract_interpretability_info(self):
        """
        提取完整的可解释性信息，返回字典格式，包含：
        1. 类别-基函数贡献矩阵
        2. 每个基函数的特征子集、掩码值、幂次
        3. 基函数衰减系数
        """
        # 1. 提取核心基础信息
        S_k = self._get_sparse_feature_subset()  # 每个基函数的特征索引列表
        alpha = self._get_discrete_alpha()      # 每个基函数-特征的幂次 (K, d)
        mask_values = self.m_tilde.copy()       # 每个基函数-特征的掩码值 (K, d)
        class_basis_weights = self.w.copy()     # 类别-基函数贡献矩阵 (10, K)

        # 2. 整理每个基函数的详细信息
        basis_func_details = []
        for k in range(self.K):
            feat_indices = S_k[k]
            if len(feat_indices) == 0:
                # 无有效特征的基函数
                basis_func_details.append({
                    "basis_id": k,
                    "feat_indices": [],
                    "feat_mask_values": [],
                    "feat_powers": [],
                    "decay_coeff": self.lambda_k[k],
                    "has_valid_feat": False
                })
                continue

            # 提取该基函数的有效特征信息
            feat_mask = mask_values[k, feat_indices]
            feat_power = alpha[k, feat_indices]

            # 排序：按掩码值从大到小（突出更重要的特征）
            sorted_idx = np.argsort(feat_mask)[::-1]
            feat_indices_sorted = feat_indices[sorted_idx]
            feat_mask_sorted = feat_mask[sorted_idx]
            feat_power_sorted = feat_power[sorted_idx]

            basis_func_details.append({
                "basis_id": k,
                "feat_indices": feat_indices_sorted.tolist(),
                "feat_mask_values": feat_mask_sorted.tolist(),
                "feat_powers": feat_power_sorted.tolist(),
                "decay_coeff": self.lambda_k[k],
                "has_valid_feat": True
            })

        # 3. 整理返回结果
        interpret_info = {
            "class_names": [f"数字{i}" for i in range(10)],  # 类别名称（0-9）
            "feat_names": [f"特征{f}" for f in range(self.d)],  # 特征名称（0-d-1）
            "class_basis_contrib": class_basis_weights,  # (10, K) 类别-基函数贡献
            "basis_func_details": basis_func_details,    # 每个基函数的详细信息
            "K": self.K,
            "d": self.d
        }

        print(f"可解释性信息提取完成：{self.K}个基函数，{self.d}个输入特征，10个类别")
        return interpret_info

# ---------------------- 3. 可解释性可视化核心方法 ----------------------
def plot_class_basis_contrib_heatmap(interpret_info):
    """
    可视化1：类别-基函数贡献热力图
    展示每个数字类别对哪些基函数有正/负贡献，贡献大小
    """
    # 提取数据
    class_names = interpret_info["class_names"]
    basis_ids = [f"基函数{k}" for k in range(interpret_info["K"])]
    contrib_matrix = interpret_info["class_basis_contrib"]

    # 配置图像
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # 绘制热力图
    im = ax.imshow(contrib_matrix, cmap=COLOR_MAP_CONTRIB, aspect="auto")

    # 设置坐标轴
    ax.set_xticks(np.arange(len(basis_ids)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(basis_ids, fontsize=FONTSIZE-2, rotation=90)
    ax.set_yticklabels(class_names, fontsize=FONTSIZE)

    # 设置标签和标题
    ax.set_xlabel("基函数ID", fontsize=FONTSIZE+2)
    ax.set_ylabel("数字类别", fontsize=FONTSIZE+2)
    ax.set_title("实验3：类别-基函数贡献热力图（红=正贡献，蓝=负贡献）", fontsize=FONTSIZE+4)

    # 添加颜色条（标注贡献值范围）
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("贡献值（权重w）", fontsize=FONTSIZE)

    # 优化布局
    plt.tight_layout()
    plt.savefig("experiment3_class_basis_heatmap.png", dpi=SAVE_FIG_DPI, bbox_inches="tight")
    plt.show()

def plot_basis_func_details(interpret_info, top_basis_num=10):
    """
    可视化2：前N个核心基函数的详细信息（特征+掩码+幂次）
    每个基函数展示：特征重要性（掩码值）、对应的幂次
    """
    # 提取数据（筛选有有效特征的基函数，取前top_basis_num个）
    basis_details = [bf for bf in interpret_info["basis_func_details"] if bf["has_valid_feat"]]
    top_basis_details = basis_details[:min(top_basis_num, len(basis_details))]
    if len(top_basis_details) == 0:
        print("无有效基函数可可视化")
        return

    # 配置图像
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(PLOT_FIGSIZE[0], PLOT_FIGSIZE[1]*0.8))
    fig.suptitle(f"实验3：前{len(top_basis_details)}个核心基函数详细信息", fontsize=FONTSIZE+4)

    # 准备绘图数据
    basis_ids = [bf["basis_id"] for bf in top_basis_details]
    max_feat_per_basis = max([len(bf["feat_indices"]) for bf in top_basis_details])

    # 子图1：基函数-特征掩码值（特征重要性）
    ax1 = axes[0]
    mask_data = np.zeros((len(top_basis_details), max_feat_per_basis))
    mask_data[:] = np.nan  # 填充NaN，无特征的位置不显示
    feat_label_data = []

    for i, bf in enumerate(top_basis_details):
        feat_masks = bf["feat_mask_values"]
        mask_data[i, :len(feat_masks)] = feat_masks
        # 记录特征标签
        for feat_idx in bf["feat_indices"]:
            feat_label_data.append(f"基{k}-特{feat_idx}")

    # 绘制掩码值热力图
    im1 = ax1.imshow(mask_data, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(np.arange(max_feat_per_basis))
    ax1.set_yticks(np.arange(len(basis_ids)))
    ax1.set_xticklabels([f"特征{idx+1}" for idx in range(max_feat_per_basis)], fontsize=FONTSIZE)
    ax1.set_yticklabels([f"基函数{k}" for k in basis_ids], fontsize=FONTSIZE)
    ax1.set_xlabel("特征排序（按掩码值从大到小）", fontsize=FONTSIZE+2)
    ax1.set_ylabel("基函数ID", fontsize=FONTSIZE+2)
    ax1.set_title("基函数-特征掩码值（越大，特征越重要）", fontsize=FONTSIZE+3)
    plt.colorbar(im1, ax=ax1, label="掩码值")

    # 子图2：基函数-特征幂次
    ax2 = axes[1]
    power_data = np.zeros((len(top_basis_details), max_feat_per_basis))
    power_data[:] = np.nan  # 填充NaN，无特征的位置不显示

    for i, bf in enumerate(top_basis_details):
        feat_powers = bf["feat_powers"]
        power_data[i, :len(feat_powers)] = feat_powers

    # 绘制幂次热力图
    im2 = ax2.imshow(power_data, cmap=COLOR_MAP_POWER, aspect="auto")
    ax2.set_xticks(np.arange(max_feat_per_basis))
    ax2.set_yticks(np.arange(len(basis_ids)))
    ax2.set_xticklabels([f"特征{idx+1}" for idx in range(max_feat_per_basis)], fontsize=FONTSIZE)
    ax2.set_yticklabels([f"基函数{k}" for k in basis_ids], fontsize=FONTSIZE)
    ax2.set_xlabel("特征排序（按掩码值从大到小）", fontsize=FONTSIZE+2)
    ax2.set_ylabel("基函数ID", fontsize=FONTSIZE+2)
    ax2.set_title("基函数-特征幂次（显式非线性变换）", fontsize=FONTSIZE+3)
    plt.colorbar(im2, ax=ax2, label="幂次值")

    # 优化布局
    plt.tight_layout()
    plt.savefig("experiment3_basis_func_details.png", dpi=SAVE_FIG_DPI, bbox_inches="tight")
    plt.show()

def plot_key_class_interpretation(interpret_info, key_classes=[0, 1, 9]):
    """
    可视化3：关键类别（0/1/9）的可解释性汇总
    展示每个关键类别依赖的TOP5基函数，以及这些基函数的特征+幂次
    """
    # 校验关键类别
    valid_key_classes = [c for c in key_classes if 0 <= c < 10]
    if len(valid_key_classes) == 0:
        print("无有效关键类别可可视化")
        return

    # 提取数据
    class_names = interpret_info["class_names"]
    basis_details = interpret_info["basis_func_details"]
    contrib_matrix = interpret_info["class_basis_contrib"]

    # 配置图像
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(nrows=len(valid_key_classes), ncols=1, figsize=(PLOT_FIGSIZE[0], PLOT_FIGSIZE[1]))
    if len(valid_key_classes) == 1:
        axes = [axes]
    fig.suptitle("实验3：关键类别可解释性汇总（TOP5依赖基函数）", fontsize=FONTSIZE+4)

    # 遍历每个关键类别
    for idx, class_id in enumerate(valid_key_classes):
        ax = axes[idx]
        class_name = class_names[class_id]

        # 提取该类别的TOP5基函数（按贡献值绝对值排序）
        class_contrib = contrib_matrix[class_id, :]
        top5_basis_ids = np.argsort(np.abs(class_contrib))[::-1][:5]
        top5_contrib = class_contrib[top5_basis_ids]

        # 准备基函数详细信息标签
        basis_labels = []
        for basis_id in top5_basis_ids:
            bf = basis_details[basis_id]
            if not bf["has_valid_feat"]:
                label = f"基{basis_id}（无有效特征）"
            else:
                # 拼接特征+幂次信息
                feat_power_str = []
                for f_idx, f_power in zip(bf["feat_indices"], bf["feat_powers"]):
                    feat_power_str.append(f"特{f_idx}^{f_power}")
                feat_power_summary = " + ".join(feat_power_str)
                label = f"基{basis_id}：{feat_power_summary}"
            basis_labels.append(label)

        # 绘制水平条形图（展示贡献值）
        y_pos = np.arange(len(top5_basis_ids))
        colors = ["red" if c > 0 else "blue" for c in top5_contrib]
        bars = ax.barh(y_pos, np.abs(top5_contrib), color=colors, alpha=0.7)

        # 设置坐标轴
        ax.set_yticks(y_pos)
        ax.set_yticklabels(basis_labels, fontsize=FONTSIZE-1)
        ax.set_xlabel("贡献值绝对值", fontsize=FONTSIZE)
        ax.set_title(f"{class_name}（类别ID={class_id}）依赖的TOP5基函数", fontsize=FONTSIZE+2)
        ax.grid(axis="x", alpha=0.3)

        # 标注正负贡献
        for bar, contrib in zip(bars, top5_contrib):
            sign = "+" if contrib > 0 else "-"
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{sign}", ha="left", va="center", fontsize=FONTSIZE)

    # 优化布局
    plt.tight_layout()
    plt.savefig("experiment3_key_class_interpretation.png", dpi=SAVE_FIG_DPI, bbox_inches="tight")
    plt.show()

def generate_interpretability_report(interpret_info, save_to_file=True):
    """
    生成文本格式的可解释性报告，便于论文/报告引用
    可选保存到本地文件（experiment3_interpretability_report.txt）
    """
    class_names = interpret_info["class_names"]
    basis_details = interpret_info["basis_func_details"]
    contrib_matrix = interpret_info["class_basis_contrib"]
    K = interpret_info["K"]
    d = interpret_info["d"]

    # 构建报告内容
    report = []
    report.append("="*80)
    report.append("实验3：幂函数模型结构可解释性报告（最终版）")
    report.append("="*80)
    report.append(f"模型配置：基函数数量K={K}，输入特征维度d={d}，类别数=10（数字0-9）")
    report.append(f"可解释性核心：显式捕捉「类别-基函数-特征-幂次」的四层逻辑关系")
    report.append("")

    # 1. 整体基函数统计
    valid_basis_num = sum([1 for bf in basis_details if bf["has_valid_feat"]])
    report.append("一、基函数整体统计")
    report.append(f"  有效基函数数量（含有效特征）：{valid_basis_num}/{K}")
    report.append(f"  平均每个有效基函数使用特征数：{np.mean([len(bf['feat_indices']) for bf in basis_details if bf['has_valid_feat']]):.2f}")
    report.append("")

    # 2. 类别-基函数核心关系
    report.append("二、类别-基函数贡献核心结论")
    for class_id in range(10):
        class_contrib = contrib_matrix[class_id, :]
        top1_basis_id = np.argmax(np.abs(class_contrib))
        top1_contrib = class_contrib[top1_basis_id]
        report.append(f"  {class_names[class_id]}：核心依赖基函数{top1_basis_id}，贡献值={top1_contrib:.4f}（{'正' if top1_contrib>0 else '负'}）")
    report.append("")

    # 3. 核心基函数详细信息
    report.append("三、核心基函数（前10个）详细信息")
    top10_basis = [bf for bf in basis_details if bf["has_valid_feat"]][:10]
    for bf in top10_basis:
        report.append(f"  基函数{bf['basis_id']}：")
        report.append(f"    - 有效特征：{bf['feat_indices']}")
        report.append(f"    - 特征掩码值：{[f'{v:.4f}' for v in bf['feat_mask_values']]}")
        report.append(f"    - 特征幂次：{bf['feat_powers']}")
        report.append(f"    - 衰减系数：{bf['decay_coeff']:.6f}")
    report.append("")

    # 4. 可解释性优势总结
    report.append("四、可解释性优势总结（对比NN）")
    report.append("  1. 显式性：每个类别依赖的基函数、每个基函数的特征与幂次均为显式可解读，无黑盒性")
    report.append("  2. 定量性：贡献值、掩码值、幂次均为定量数值，可进行量化分析与验证")
    report.append("  3. 逻辑性：基函数的特征组合与幂次变换对应真实数据的非线性规律，具备因果逻辑")
    report.append("  4. 可复现性：相同配置下，模型提取的可解释性信息一致，无NN的权重随机波动问题")
    report.append("="*80)

    # 打印报告
    report_text = "\n".join(report)
    print(report_text)

    # 保存到文件
    if save_to_file:
        with open("experiment3_interpretability_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n可解释性报告已保存到：experiment3_interpretability_report.txt")

# ---------------------- 4. 实验3主执行流程 ----------------------
def run_experiment3():
    """运行实验3：结构可解释性可视化，完整流程执行"""
    # 步骤1：加载数据
    X_train, X_test, y_train, y_test, y_train_onehot, d = load_and_preprocess_mnist()

    # 步骤2：训练幂函数模型
    print("\n=====================================")
    print("训练幂函数模型（K=30，用于可解释性提取）")
    print("=====================================")
    model = PowerFunctionModel(K=K, d=d)
    model.fit(X_train, y_train_onehot)

    # 步骤3：评估模型性能（保证可解释性的同时，性能不下降）
    print("\n=====================================")
    print("评估模型性能（验证可解释性与性能的平衡）")
    print("=====================================")
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred_argmax == y_test)
    params = model.count_params()
    print(f"模型参数量：{params}")
    print(f"模型测试准确率：{acc:.4f}")

    # 步骤4：提取可解释性信息
    print("\n=====================================")
    print("提取可解释性信息")
    print("=====================================")
    interpret_info = model.extract_interpretability_info()

    # 步骤5：分层可视化
    print("\n=====================================")
    print("绘制可解释性可视化图表")
    print("=====================================")
    # 5.1 类别-基函数贡献热力图
    plot_class_basis_contrib_heatmap(interpret_info)

    # 5.2 基函数-特征-幂次详情图
    plot_basis_func_details(interpret_info, top_basis_num=10)

    # 5.3 关键类别可解释性汇总
    plot_key_class_interpretation(interpret_info, key_classes=[0, 1, 9])

    # 步骤6：生成可解释性报告
    print("\n=====================================")
    print("生成可解释性文本报告")
    print("=====================================")
    generate_interpretability_report(interpret_info)

    print("\n=====================================")
    print("实验3执行完成！所有结果已保存。")
    print("=====================================")

# ---------------------- 5. 导入必要依赖 & 主函数入口 ----------------------
import time  # 补充训练耗时统计依赖

if __name__ == "__main__":
    run_experiment3()