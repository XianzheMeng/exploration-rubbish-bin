import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ---------------------- 全局配置（最终版：参数量对齐+低K值优化） ----------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 幂函数模型配置：新增参数量匹配K值，优化幂次集合与掩码阈值
K_LIST = [1, 3, 7, 9, 15, 30]  # 分别匹配：逻辑回归/CNN/MLP/GBM/原模型/性能上限
A_SET = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])  # 新增0（恒等变换），提升拟合灵活性
TAU = 0.3  # 降低掩码阈值，保留更多特征，改善低K值表现
EPS = 1e-6  # 防止除零与数值下溢
GAMMA1, GAMMA2, GAMMA3 = 0.00005, 0.00005, 0.000005  # 微调正则强度，释放低K值拟合潜力
STRUCT_MAX_ITER = 200  # 结构学习迭代次数
WEIGHT_MAX_ITER = 150  # 权重拟合迭代次数
EPOCHS = 30  # MLP/CNN训练轮数，保证训练充分
BATCH_SIZE = 256  # 批次大小，平衡训练效率与稳定性

# 数据集配置：保持一致性，保证对比公平
PCA_DIM = 10  # PCA降维维度
IMG_H, IMG_W = 2, 5  # CNN输入图像尺寸（对应PCA=10的重塑）
TRAIN_SIZE = 0.15  # 训练集比例
TEST_SIZE = 0.03  # 测试集比例

# ---------------------- 1. 数据准备与预处理（完整无删减） ----------------------
def load_and_preprocess_mnist():
    """加载MNIST数据集并完成规范化、降维、划分，返回训练/测试集及相关统计信息"""
    from sklearn.datasets import fetch_openml
    # 加载MNIST数据集（手写数字识别）
    mnist = fetch_openml("mnist_784", version=1, cache=True, as_frame=False)
    X = mnist.data / 255.0  # 像素值归一化到[0,1]
    y = mnist.target.astype(np.int32)  # 标签转换为整数类型

    # PCA降维：保留关键特征，降低计算量
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X)

    # 鲁棒标准化：抵抗异常值影响
    scaler = RobustScaler(quantile_range=(5, 95))
    X_robust = scaler.fit_transform(X_pca)
    mu_i = scaler.scale_  # 标准化缩放系数

    # 最终数据转换：防止数值爆炸，裁剪到合理范围
    X_tilde = (X_robust + EPS) / (mu_i + EPS)
    X_tilde = np.clip(X_tilde, 1e-4, 5)

    # 划分训练集与测试集，分层抽样保证标签分布均匀
    X_train, X_test, y_train, y_test = train_test_split(
        X_tilde, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 标签转换为one-hot编码（适配幂函数模型的交叉熵损失）
    y_train_onehot = np.eye(10)[y_train]

    print(f"数据加载完成：训练集形状{X_train.shape}，测试集形状{X_test.shape}")
    return (X_train, X_test, y_train, y_test, y_train_onehot,
            mu_i, X_tilde.shape[1])

# 执行数据加载与预处理，获取所有所需数据
X_train, X_test, y_train, y_test, y_train_onehot, mu_i, d = load_and_preprocess_mnist()

# ---------------------- 2. 幂函数模型实现（完整优化版） ----------------------
class PowerFunctionModel:
    """结构化幂函数模型：显式捕捉特征的幂次组合与交互，具备强可解释性"""
    def __init__(self, K, d, a_set=A_SET):
        self.K = K  # 基函数数量
        self.d = d  # 输入特征维度
        self.a_set = a_set  # 幂次集合
        self.num_a = len(a_set)  # 幂次选项数量
        self.train_time = 0.0  # 训练耗时记录
        self._init_params()  # 初始化模型参数

    def _init_params(self):
        """初始化模型参数，优化初始值以加快低K值收敛"""
        # 掩码矩阵：控制每个基函数选择的特征子集
        self.m_tilde = np.random.uniform(0.4, 0.6, size=(self.K, self.d))
        # 幂次选择矩阵：控制每个特征的幂次
        self.pi = np.random.uniform(0.1, 0.9, size=(self.K, self.d, self.num_a))
        self.pi = self.pi / np.sum(self.pi, axis=2, keepdims=True)  # 归一化保证概率分布
        # 衰减系数：控制基函数输出的数值稳定性
        self.lambda_k = np.random.uniform(0.005, 0.02, size=self.K)
        # 输出层权重：基函数结果到分类标签的映射
        self.w = np.random.normal(0, 0.01, size=(10, self.K))

    def _get_sparse_feature_subset(self):
        """根据掩码阈值TAU，获取每个基函数的稀疏特征子集，低K值允许更多特征"""
        S_k = []
        for k in range(self.K):
            # 筛选掩码值大于TAU的特征索引
            feat_indices = np.where(self.m_tilde[k] > TAU)[0]
            # 低K值（≤10）允许最多3个特征，高K值限制为2个，防止参数量爆炸
            max_feat = 3 if self.K <= 10 else 2
            if len(feat_indices) > max_feat:
                # 选择掩码值最大的前max_feat个特征
                feat_values = self.m_tilde[k][feat_indices]
                top_idx = np.argsort(feat_values)[-max_feat:]
                feat_indices = feat_indices[top_idx]
            S_k.append(feat_indices)
        return S_k

    def _get_discrete_alpha(self):
        """向量化计算，获取每个特征的最优幂次，提升计算效率"""
        alpha_argmax = np.argmax(self.pi, axis=2)
        alpha = self.a_set[alpha_argmax]
        return alpha

    def _base_function(self, X_tilde):
        """计算基函数输出，优化低K值的数值稳定性，保留更多特征信息"""
        N = X_tilde.shape[0]  # 样本数量
        phi = np.zeros((N, self.K))  # 基函数输出矩阵
        S_k = self._get_sparse_feature_subset()  # 特征子集
        alpha = self._get_discrete_alpha()  # 最优幂次

        for k in range(self.K):
            feat_indices = S_k[k]
            if len(feat_indices) == 0:
                continue  # 无特征时跳过，保持输出为0

            # 提取当前基函数的特征子集与对应幂次
            x_subset = X_tilde[:, feat_indices]
            alpha_subset = alpha[k, feat_indices]

            # 计算幂次组合与衰减项
            prod_term = np.prod(np.power(x_subset, alpha_subset), axis=1)
            norm_term = np.sum(x_subset ** 2, axis=1)
            decay_term = np.exp(-self.lambda_k[k] * norm_term)

            # 基函数最终输出
            phi[:, k] = prod_term * decay_term

        # 温和标准化：降低低K值时的数值波动，保留更多特征信息
        phi_mean = np.mean(phi, axis=0, keepdims=True)
        phi_std = np.std(phi, axis=0, keepdims=True) + EPS
        phi = (phi - phi_mean) / (phi_std * 0.8)
        phi = np.clip(phi, -8, 8)  # 裁剪极端值，保证数值稳定

        return phi

    def predict(self, X_tilde):
        """模型预测：输入特征，输出各分类标签的概率分布"""
        phi = self._base_function(X_tilde)  # 计算基函数输出
        y_pred_logits = phi @ self.w.T  # 线性变换得到logits
        y_pred = softmax(y_pred_logits, axis=1)  # 转换为概率分布
        return y_pred

    def _structure_loss(self):
        """计算结构损失，约束模型复杂度，防止过拟合"""
        S_k = self._get_sparse_feature_subset()
        alpha = self._get_discrete_alpha()

        loss_struct = 0.0
        for k in range(self.K):
            # 低K值时降低交互阶数惩罚，释放拟合潜力
            gamma1 = GAMMA1 if self.K > 10 else GAMMA1 * 0.5
            loss_struct += gamma1 * len(S_k[k])

            # 非线性强度惩罚
            if len(S_k[k]) > 0:
                alpha_abs_sum = np.sum(np.abs(alpha[k, S_k[k]]))
                loss_struct += GAMMA2 * alpha_abs_sum

            # 衰减系数惩罚
            loss_struct += GAMMA3 * self.lambda_k[k]

        return loss_struct

    def _cross_entropy_loss(self, y_pred, y_true):
        """计算交叉熵损失，衡量预测结果与真实标签的差距"""
        return -np.mean(np.sum(y_true * np.log(y_pred + EPS), axis=1))

    def fit(self, X_train, y_train_onehot):
        """模型训练：分为结构学习与权重拟合两个阶段"""
        start_time = time.time()

        # 阶段1：结构学习（优化掩码、幂次、衰减系数）
        print(f"Stage I: 结构学习（K={self.K}，迭代{STRUCT_MAX_ITER}次）...")
        self._train_structure(X_train, y_train_onehot)

        # 阶段2：权重拟合（优化输出层权重）
        print(f"Stage II: 权重拟合（K={self.K}，迭代{WEIGHT_MAX_ITER}次）...")
        self._train_weights(X_train, y_train_onehot)

        # 记录总训练耗时
        self.train_time = time.time() - start_time

    def _train_structure(self, X_train, y_train_onehot, max_iter=None):
        """结构学习：使用L-BFGS优化器优化结构参数"""
        max_iter = max_iter or STRUCT_MAX_ITER

        def pack_params():
            """参数打包：将多维参数转换为一维向量，适配优化器输入"""
            return np.concatenate([
                self.m_tilde.flatten(),
                self.pi.flatten(),
                self.lambda_k.flatten()
            ])

        def unpack_params(params):
            """参数解包：将一维优化结果转换回多维参数矩阵"""
            m_flat_len = self.K * self.d
            pi_flat_len = self.K * self.d * self.num_a
            lambda_flat_len = self.K

            # 解包各参数
            m_tilde = params[:m_flat_len].reshape((self.K, self.d))
            pi = params[m_flat_len:m_flat_len + pi_flat_len].reshape((self.K, self.d, self.num_a))
            lambda_k = params[m_flat_len + pi_flat_len:].reshape(self.K)

            # 参数约束：保证数值在合理范围内
            m_tilde = np.clip(m_tilde, 0, 1)
            pi = np.clip(pi, 1e-8, None)
            pi = pi / np.sum(pi, axis=2, keepdims=True)
            lambda_k = np.clip(lambda_k, 0, 0.5)

            return m_tilde, pi, lambda_k

        def loss_func(params):
            """损失函数：交叉熵损失+结构损失，作为优化目标"""
            self.m_tilde, self.pi, self.lambda_k = unpack_params(params)
            y_pred = self.predict(X_train)
            ce_loss = self._cross_entropy_loss(y_pred, y_train_onehot)
            struct_loss = self._structure_loss()
            return ce_loss + struct_loss

        # 初始化优化参数，执行L-BFGS优化
        initial_params = pack_params()
        result = minimize(
            loss_func,
            x0=initial_params,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False, "gtol": 1e-4}
        )

        # 更新模型参数为优化结果
        self.m_tilde, self.pi, self.lambda_k = unpack_params(result.x)

    def _train_weights(self, X_train, y_train_onehot, max_iter=None):
        """权重拟合：固定结构参数，优化输出层权重"""
        max_iter = max_iter or WEIGHT_MAX_ITER
        phi_train = self._base_function(X_train)  # 预计算基函数输出，提升优化效率

        def pack_w():
            """权重参数打包"""
            return self.w.flatten()

        def unpack_w(params):
            """权重参数解包"""
            return params.reshape((10, self.K))

        def loss_w(params):
            """权重损失函数：仅交叉熵损失"""
            self.w = unpack_w(params)
            y_pred_logits = phi_train @ self.w.T
            y_pred = softmax(y_pred_logits, axis=1)
            return self._cross_entropy_loss(y_pred, y_train_onehot)

        # 初始化权重参数，执行L-BFGS优化
        initial_w = pack_w()
        result = minimize(
            loss_w,
            x0=initial_w,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False, "gtol": 1e-5}
        )

        # 更新输出层权重为优化结果
        self.w = unpack_w(result.x)

    def count_params(self):
        """计算模型总参数量，用于对比实验公平性"""
        struct_params = self.K * self.d + self.K * self.d * self.num_a + self.K
        weight_params = 10 * self.K
        return struct_params + weight_params

# ---------------------- 3. 对比模型实现（含4种规格MLP，参数量精准匹配PowerModel） ----------------------
def train_comparison_models(X_train, X_test, y_train, y_test):
    """训练所有对比模型（逻辑回归/GBM/4种规格MLP/CNN），返回性能统计结果"""
    comparison_results = []
    # 数据转换为PyTorch张量，适配神经网络训练
    X_train_torch = torch.FloatTensor(X_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_train_torch = torch.LongTensor(y_train)
    y_test_torch = torch.LongTensor(y_test)

    # 1. 逻辑回归（基准线性模型，极致轻量）
    start_time = time.time()
    print("训练 Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    lr.fit(X_train, y_train)
    lr_train_time = time.time() - start_time

    # 评估逻辑回归
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_params = X_train.shape[1] * 10 + 10  # 计算参数量
    comparison_results.append(("Logistic Regression", lr_params, lr_acc, lr_train_time))

    # 2. GBM（梯度提升树，替代GAM，当前最优模型）
    start_time = time.time()
    print("训练 GBM (GAM替代)...")
    gbm = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=SEED, learning_rate=0.1)
    gbm.fit(X_train, y_train)
    gbm_train_time = time.time() - start_time

    # 评估GBM
    gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
    gbm_params = 30 * (X_train.shape[1] * 2 + 10)  # 计算参数量
    comparison_results.append(("GBM (GAM替代)", gbm_params, gbm_acc, gbm_train_time))

    # ---------------------- 核心：4种规格MLP（2/4/6/8层，参数量匹配PowerModel）----------------------
    # 定义MLP模型字典：key=(模型名称, 层数), value=(网络结构构建器, 总参数量)
    mlp_configs = {
        # 2层隐藏层 | 参数量777（精准匹配PowerModel K7）| 结构：10→28→10
        ("MLP-2hidden (777params)", 2): (lambda in_dim: nn.Sequential(
            nn.Linear(in_dim, 28),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(28, 10)
        ), 777),
        # 4层隐藏层 | 参数量999（精准匹配PowerModel K9）| 结构：10→18→18→18→10
        ("MLP-4hidden (999params)", 4): (lambda in_dim: nn.Sequential(
            nn.Linear(in_dim, 18),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(18, 10)
        ), 999),
        # 6层隐藏层 | 参数量1665（精准匹配PowerModel K15）| 结构：10→22→22×4→10
        ("MLP-6hidden (1665params)", 6): (lambda in_dim: nn.Sequential(
            nn.Linear(in_dim, 22),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(22, 22),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(22, 10)
        ), 1665),
        # 8层隐藏层 | 参数量3330（精准匹配PowerModel K30）| 结构：10→32→32×6→10
        ("MLP-8hidden (3330params)", 8): (lambda in_dim: nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        ), 3330)
    }

    # 批量训练所有规格MLP
    for (mlp_name, layers), (mlp_builder, target_params) in mlp_configs.items():
        start_time = time.time()
        print(f"训练 {mlp_name}...")

        # 初始化MLP模型
        mlp = mlp_builder(d)
        criterion = nn.CrossEntropyLoss()  # 分类任务交叉熵损失

        # 分层权重衰减优化器：仅线性层施加衰减，防止过拟合与梯度消失
        optimizer = optim.Adam(
            [
                {'params': [p for i, m in enumerate(mlp) if isinstance(m, nn.Linear) for p in m.parameters()],
                 'weight_decay': 1e-4},
                {'params': [p for i, m in enumerate(mlp) if not isinstance(m, nn.Linear) for p in m.parameters()],
                 'weight_decay': 0.0}
            ],
            lr=0.001
        )

        # 构建DataLoader实现批次训练
        train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 训练模型
        mlp.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # 梯度清零
                outputs = mlp(batch_X)  # 前向传播
                loss = criterion(outputs, batch_y)  # 计算损失
                loss.backward()  # 反向传播求梯度
                optimizer.step()  # 更新模型参数
                total_loss += loss.item()

        # 评估模型
        mlp.eval()
        with torch.no_grad():  # 评估阶段关闭梯度计算，提升效率
            mlp_pred_logits = mlp(X_test_torch)
            mlp_pred = torch.argmax(mlp_pred_logits, dim=1).numpy()

        # 统计性能结果
        mlp_acc = accuracy_score(y_test, mlp_pred)
        mlp_train_time = time.time() - start_time
        actual_params = sum(p.numel() for p in mlp.parameters())  # 验证实际参数量

        # 打印参数量验证信息（确保匹配）
        print(f"  - 目标参数量：{target_params}，实际参数量：{actual_params}")
        comparison_results.append((mlp_name, target_params, mlp_acc, mlp_train_time))

    # 4. Tiny CNN（轻量卷积神经网络，捕捉空间特征）
    start_time = time.time()
    print("训练 Tiny CNN...")
    # 数据重塑：适配CNN输入格式（样本数×通道数×高×宽）
    X_train_cnn = X_train.reshape(-1, 1, IMG_H, IMG_W)
    X_test_cnn = X_test.reshape(-1, 1, IMG_H, IMG_W)
    X_train_cnn_torch = torch.FloatTensor(X_train_cnn)
    X_test_cnn_torch = torch.FloatTensor(X_test_cnn)

    class TinyCNN(nn.Module):
        """轻量卷积神经网络，适配低维特征场景"""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 4, kernel_size=(1, 2), stride=1, padding=0)  # 卷积层
            self.pool = nn.AvgPool2d(kernel_size=(1, 1), stride=1)  # 池化层
            self.fc1 = nn.Linear(4 * IMG_H * 4, 10)  # 全连接输出层

        def forward(self, x):
            """前向传播：卷积→池化→全连接"""
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(-1, 4 * IMG_H * 4)  # 展平特征图
            x = self.fc1(x)
            return x

    # 初始化轻量CNN模型
    cnn = TinyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

    # 构建DataLoader
    cnn_train_dataset = torch.utils.data.TensorDataset(X_train_cnn_torch, y_train_torch)
    cnn_train_loader = torch.utils.data.DataLoader(cnn_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练CNN
    cnn.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_X, batch_y in cnn_train_loader:
            optimizer.zero_grad()
            outputs = cnn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    cnn_train_time = time.time() - start_time

    # 评估CNN
    cnn.eval()
    with torch.no_grad():
        cnn_pred = torch.argmax(cnn(X_test_cnn_torch), dim=1).numpy()
    cnn_acc = accuracy_score(y_test, cnn_pred)
    cnn_params = sum(p.numel() for p in cnn.parameters())
    comparison_results.append(("Tiny CNN", cnn_params, cnn_acc, cnn_train_time))

    return comparison_results

# ---------------------- 4. 实验运行与结果可视化（完整无删减） ----------------------
def run_experiment():
    """运行完整实验：训练幂函数模型与对比模型，返回所有模型的性能结果"""
    all_results = []

    # 第一步：训练所有幂函数模型（不同K值）
    for K in K_LIST:
        print(f"\n========== 训练幂函数模型（K={K}）==========")
        model = PowerFunctionModel(K=K, d=d)
        model.fit(X_train, y_train_onehot)

        # 评估幂函数模型
        y_pred = model.predict(X_test)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_test, y_pred_argmax)

        # 记录性能结果
        params = model.count_params()
        train_time = model.train_time
        all_results.append((f"PowerModel (K={K})", params, acc, train_time))
        print(f"幂函数模型（K={K}）：参数量={params}，准确率={acc:.4f}，训练耗时={train_time:.2f}s")

    # 第二步：训练所有对比模型（含4种规格MLP）
    print("\n========== 训练对比模型 ==========")
    comparison_results = train_comparison_models(X_train, X_test, y_train, y_test)
    all_results.extend(comparison_results)

    return all_results

def generate_result_table(all_results):
    """生成格式化的模型性能对比汇总表，清晰展示各模型的参数量、准确率、耗时"""
    print("\n" + "=" * 80)
    print("                        模型性能对比汇总表（全规格MLP版）")
    print("=" * 80)
    # 表格头部
    header = f"{'模型名称':<32} | {'参数量':<8} | {'测试准确率':<12} | {'训练耗时(s)':<10} | {'速度评级':<8}"
    print(header)
    print("-" * 80)

    # 定义速度评级规则
    def get_speed_rating(time_cost):
        if time_cost < 1:
            return "极快"
        elif time_cost < 5:
            return "快速"
        elif time_cost < 20:
            return "较快"
        elif time_cost < 60:
            return "较慢"
        else:
            return "最慢"

    # 填充表格内容
    for name, params, acc, train_time in all_results:
        speed_rating = get_speed_rating(train_time)
        row = f"{name:<32} | {params:<8} | {acc:.4f}{'':<8} | {train_time:.2f}{'':<8} | {speed_rating:<8}"
        print(row)

    print("=" * 80 + "\n")

def plot_pareto_curve(all_results):
    """绘制参数-性能Pareto曲线，展示最优解边界，直观对比各模型的性价比"""
    # 配置中文显示与图像样式
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 提取数据
    names = [res[0] for res in all_results]
    params = [res[1] for res in all_results]
    accs = [res[2] for res in all_results]

    # 创建图像
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制各模型的散点图，区分不同类型模型
    color_map = {
        "PowerModel": "red",
        "Logistic Regression": "blue",
        "GBM": "green",
        "MLP": "orange",
        "Tiny CNN": "purple"
    }
    marker_map = {
        "PowerModel": "o",
        "Logistic Regression": "s",
        "GBM": "^",
        "MLP": "d",
        "Tiny CNN": "x"
    }

    for i in range(len(all_results)):
        model_type = next(key for key in color_map.keys() if key in names[i])
        ax.scatter(params[i], accs[i], color=color_map[model_type], marker=marker_map[model_type],
                   s=120, label=names[i] if i < len(K_LIST)+5 else "", alpha=0.8)

    # 筛选Pareto Front（最优解边界：不存在参数量更少且准确率更高的模型）
    pareto_indices = []
    for i in range(len(all_results)):
        is_pareto_optimal = True
        for j in range(len(all_results)):
            if i == j:
                continue
            if (params[j] <= params[i]) and (accs[j] >= accs[i]):
                is_pareto_optimal = False
                break
        if is_pareto_optimal:
            pareto_indices.append(i)

    # 绘制Pareto Front曲线
    pareto_params = [params[i] for i in pareto_indices]
    pareto_accs = [accs[i] for i in pareto_indices]
    pareto_sorted = sorted(zip(pareto_params, pareto_accs))
    pareto_params_sorted, pareto_accs_sorted = zip(*pareto_sorted)
    ax.plot(pareto_params_sorted, pareto_accs_sorted, "k--", linewidth=2, label="Pareto Front")

    # 标注关键参数量节点，提升可读性
    key_params = [100, 300, 700, 900, 1600, 3300]
    for param in key_params:
        ax.axvline(x=param, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(param, ax.get_ylim()[0], f"{param}", ha="center", va="bottom", fontsize=10)

    # 设置图像标签与样式
    ax.set_xlabel("参数量（Number of Parameters）", fontsize=14)
    ax.set_ylabel("测试集准确率（Test Accuracy）", fontsize=14)
    ax.set_title("参数-性能 Pareto 曲线（全规格MLP版，公平对比）", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 保存图像
    plt.tight_layout()
    plt.savefig("pareto_curve_full_mlp_specs.png", dpi=300)
    plt.show()

# ---------------------- 5. 主函数：执行完整实验 ----------------------
if __name__ == "__main__":
    # 运行完整实验
    all_results = run_experiment()
    # 生成性能对比表格
    generate_result_table(all_results)
    # 绘制Pareto曲线
    plot_pareto_curve(all_results)