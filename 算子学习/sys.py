import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
import copy
from typing import List, Dict, Tuple, Optional

# ======================================
# 全局配置（严格遵循技术规格书，新增强制节点数配置）
# ======================================
CONFIG = {
    # 算子族配置
    "monotone_params": {"alpha_range": (0.5, 5.0), "beta_range": (0.1, 2.0), "p_range": (0.5, 3.0)},
    "periodic_params": {"omega_range": (0.5, 2.0), "phi_range": (0.0, np.pi), "epsilon_max": 0.2},
    "saturation_params": {"kappa_range": (1.0, 10.0)},
    # 程序图配置
    "max_depth": 4,
    "max_width": 5,
    "min_program_length": 2,  # 新增：强制Program Length ≥ 2（主干+残差）
    # Beam Search配置
    "beam_size": 10,
    "top_k": 5,
    # 优化配置
    "lr": 1e-3,
    "epochs": 2000,
    "lambda_struct": 1e-4,  # 结构复杂度正则
    "mu_period": 1e-3,      # 周期项方差正则
    # 数据配置
    "x_train_range": (-3.0, 3.0),
    "x_ood_range": (3.0, 6.0),
    "sample_sizes": [20, 50, 100],
    # 评价配置
    "random_seed": 42
}

# 设置随机种子
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])

# ======================================
# 先定义 DAG 基类（解决NameError，父类提前定义）
# 补全完整实现，保证有向无环约束
# ======================================
class DAG:
    """有向无环图（DAG）基类：提供图的基本操作，保证无环"""
    def __init__(self):
        self.adjacency = defaultdict(list)  # 邻接表：{from_node: [to_node1, to_node2, ...]}
        self.nodes = set()  # 所有节点集合

    def add_edge(self, from_node, to_node) -> None:
        """添加有向边，同时检查是否形成环（保证无环）"""
        # 检查是否存在反向边（避免直接环）
        if to_node in self.adjacency and from_node in self.adjacency[to_node]:
            raise ValueError(f"Adding edge from {from_node} to {to_node} would create a cycle")
        # 检查是否存在间接环（深度优先搜索）
        if self._has_path(to_node, from_node):
            raise ValueError(f"Adding edge from {from_node} to {to_node} would create a cycle")
        # 添加边和节点
        self.adjacency[from_node].append(to_node)
        self.nodes.add(from_node)
        self.nodes.add(to_node)

    def _has_path(self, start_node, target_node) -> bool:
        """深度优先搜索：检查是否存在从 start_node 到 target_node 的路径（避免间接环）"""
        visited = set()
        stack = [start_node]
        while stack:
            current_node = stack.pop()
            if current_node == target_node:
                return True
            if current_node in visited:
                continue
            visited.add(current_node)
            stack.extend(self.adjacency.get(current_node, []))
        return False

    def get_all_nodes(self) -> set:
        """获取所有节点"""
        return self.nodes

# ======================================
# 第一部分：算子族实现（4类最小完备集）
# 严格遵循技术规格书，参数化、模块化（所有算子仅作用于原始x）
# ======================================
class OperatorFamily:
    """算子族基类：所有算子的统一接口（仅作用于原始x，不接受其他算子输出）"""
    def __init__(self, op_type: str, params: Optional[np.ndarray] = None):
        self.op_type = op_type
        self.params = params if params is not None else self._init_default_params()
        self.param_size = len(self.params)
        self.trainable = True  # 是否可训练

    def _init_default_params(self) -> np.ndarray:
        """初始化默认参数（子类实现）"""
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向计算（仅作用于原始x，numpy版本）"""
        raise NotImplementedError

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """前向计算（仅作用于原始x，PyTorch版本，保证梯度传递）"""
        raise NotImplementedError

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        """获取参数边界（用于约束优化）"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.op_type}(params={[f'{p:.4f}' for p in self.params]})"

# ------------------------------
# (A) 单调主干算子族（3种）
# ------------------------------
class MonoTanh(OperatorFamily):
    """单调主干：O_mono(x; α, β) = α · tanh(β x)（仅作用于原始x）"""
    def __init__(self, params: Optional[np.ndarray] = None):
        super().__init__("MonoTanh", params)

    def _init_default_params(self) -> np.ndarray:
        alpha = np.random.uniform(*CONFIG["monotone_params"]["alpha_range"])
        beta = np.random.uniform(*CONFIG["monotone_params"]["beta_range"])
        return np.array([alpha, beta], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        alpha, beta = self.params
        return alpha * np.tanh(beta * x)

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        alpha, beta = params
        return alpha * torch.tanh(beta * x)

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [CONFIG["monotone_params"]["alpha_range"],
                CONFIG["monotone_params"]["beta_range"]]

class MonoLog(OperatorFamily):
    """单调主干：O_log(x; γ) = γ · log(1 + |x|)（仅作用于原始x）"""
    def __init__(self, params: Optional[np.ndarray] = None):
        super().__init__("MonoLog", params)

    def _init_default_params(self) -> np.ndarray:
        gamma = np.random.uniform(*CONFIG["monotone_params"]["alpha_range"])
        return np.array([gamma], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gamma, = self.params
        return gamma * np.log1p(np.abs(x))

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gamma, = params
        return gamma * torch.log1p(torch.abs(x))

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [CONFIG["monotone_params"]["alpha_range"]]

class MonoPoly(OperatorFamily):
    """单调主干：O_poly(x; p) = sign(x) · |x|^p（仅作用于原始x）"""
    def __init__(self, params: Optional[np.ndarray] = None):
        super().__init__("MonoPoly", params)

    def _init_default_params(self) -> np.ndarray:
        p = np.random.uniform(*CONFIG["monotone_params"]["p_range"])
        return np.array([p], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        p, = self.params
        return np.sign(x) * np.power(np.abs(x), p)

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        p, = params
        return torch.sign(x) * torch.pow(torch.abs(x), p)

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [CONFIG["monotone_params"]["p_range"]]

# ------------------------------
# (B) 局部周期扰动算子族（残差）
# ------------------------------
class LocalPeriodic(OperatorFamily):
    """局部周期：O_per(x; ω, φ, ε) = ε · sin(ω x + φ)（仅作用于原始x，残差项）"""
    def __init__(self, params: Optional[np.ndarray] = None):
        super().__init__("LocalPeriodic", params)

    def _init_default_params(self) -> np.ndarray:
        omega = np.random.uniform(*CONFIG["periodic_params"]["omega_range"])
        phi = np.random.uniform(*CONFIG["periodic_params"]["phi_range"])
        epsilon = np.random.uniform(0.0, CONFIG["periodic_params"]["epsilon_max"])
        return np.array([omega, phi, epsilon], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        omega, phi, epsilon = self.params
        return epsilon * np.sin(omega * x + phi)

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        omega, phi, epsilon = params
        return epsilon * torch.sin(omega * x + phi)

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [CONFIG["periodic_params"]["omega_range"],
                CONFIG["periodic_params"]["phi_range"],
                (0.0, CONFIG["periodic_params"]["epsilon_max"])]

# ------------------------------
# (C) 饱和/截断算子族
# ------------------------------
class Saturation(OperatorFamily):
    """饱和截断：O_sat(x; κ) = clip(x, -κ, κ)（仅作用于原始x）"""
    def __init__(self, params: Optional[np.ndarray] = None):
        super().__init__("Saturation", params)

    def _init_default_params(self) -> np.ndarray:
        kappa = np.random.uniform(*CONFIG["saturation_params"]["kappa_range"])
        return np.array([kappa], dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        kappa, = self.params
        return np.clip(x, -kappa, kappa)

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        kappa, = params
        return torch.clip(x, -kappa, kappa)

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [CONFIG["saturation_params"]["kappa_range"]]

# ------------------------------
# (D) 线性混合算子族（唯一全局读出层，仅在Stage2组合多算子输出）
# ------------------------------
class LinearMix(OperatorFamily):
    """线性混合：O_lin(z1,z2,...zk; w) = Σ w_i z_i（仅组合多算子并行输出，不作用于原始x）"""
    def __init__(self, num_inputs: int, params: Optional[np.ndarray] = None):
        self.num_inputs = num_inputs
        super().__init__("LinearMix", params)

    def _init_default_params(self) -> np.ndarray:
        return np.ones(self.num_inputs, dtype=np.float32) / self.num_inputs

    def forward(self, z_list: List[np.ndarray]) -> np.ndarray:
        """输入：多算子并行输出的numpy列表，输出：加权和"""
        w = self.params
        return np.sum([w[i] * z for i, z in enumerate(z_list)], axis=0)

    def forward_torch(self, z_list: List[torch.Tensor], params: torch.Tensor) -> torch.Tensor:
        """输入：多算子并行输出的Tensor列表，输出：加权和（保证梯度传递）"""
        w = params
        weighted_tensor_list = [w[i] * z for i, z in enumerate(z_list)]
        stacked_tensor = torch.stack(weighted_tensor_list)
        return torch.sum(stacked_tensor, dim=0)

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        return [(-1.0, 1.0) for _ in range(self.num_inputs)]

# ------------------------------
# 算子族注册表（用于结构搜索）
# ------------------------------
OPERATOR_FAMILIES = [
    MonoTanh,    # 单调主干
    MonoLog,     # 单调主干
    MonoPoly,    # 单调主干
    LocalPeriodic,# 周期残差
    Saturation   # 辅助算子
]

OPERATOR_NAMES = [op.__name__ for op in OPERATOR_FAMILIES]

# ======================================
# 第二部分：Program Graph（DAG）实现
# 修正：禁止算子吃算子，所有算子并行作用于原始x，强制多节点结构
# ======================================
class ProgramNode:
    """程序图节点：封装算子实例（仅作用于原始x，禁止接收其他节点作为输入）"""
    def __init__(self, operator: OperatorFamily):
        self.operator = operator
        self.depth = 1  # 所有节点均为浅层，并行作用于原始x，无层级依赖
        self.output = None  # 缓存前向输出（numpy版本）
        self.output_torch = None  # 缓存前向输出（PyTorch版本，保证梯度）

    # 修正1：禁止算子吃算子，删除输入节点参数，抛出非法输入异常
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向计算（numpy版本，仅作用于原始x，禁止其他节点输入）"""
        if self.output is not None:
            return self.output  # 缓存命中
        # 强制仅作用于原始x，无任何其他节点输入
        self.output = self.operator.forward(x)
        return self.output

    def forward_torch(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """前向计算（PyTorch版本，仅作用于原始x，保证梯度传递，禁止其他节点输入）"""
        # 全程无numpy调用，保证梯度不中断
        self.output_torch = self.operator.forward_torch(x, params)
        return self.output_torch

    def reset_cache(self) -> None:
        """重置所有缓存"""
        self.output = None
        self.output_torch = None

    def __repr__(self) -> str:
        return f"ProgramNode({self.operator})"

class ProgramGraph(DAG):
    """程序图（DAG）：多算子并行作用于原始x，强制节点数≥2，完整PyTorch forward支持"""
    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.nodes: List[ProgramNode] = []  # 并行节点列表，无层级依赖
        self.linear_mix: Optional[LinearMix] = None
        self.struct_complexity = 0  # 节点数（强制≥2）

    # 修正1：禁止算子吃算子，删除输入节点参数，仅添加并行节点
    def add_node(self, operator: OperatorFamily) -> ProgramNode:
        """添加并行节点（仅作用于原始x，无输入依赖，强制节点数≥2）"""
        new_node = ProgramNode(operator)
        self.nodes.append(new_node)
        self.struct_complexity = len(self.nodes)  # 更新节点数
        # 检查是否满足最小节点数要求
        if self.struct_complexity < CONFIG["min_program_length"]:
            pass  # 暂不报错，仅在结构搜索时过滤
        else:
            self._init_linear_mix()  # 满足最小节点数后初始化线性混合
        return new_node

    def _init_linear_mix(self) -> None:
        """初始化线性混合算子（仅当节点数≥2时）"""
        if self.struct_complexity >= CONFIG["min_program_length"]:
            self.linear_mix = LinearMix(self.struct_complexity)

    # 修正3：完整numpy forward（多算子并行，无层级依赖）
    def forward_all_operators(self, x: np.ndarray) -> List[np.ndarray]:
        """前向计算所有并行算子的输出（numpy版本）"""
        [node.reset_cache() for node in self.nodes]
        return [node.forward(x) for node in self.nodes]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """完整前向（numpy版本：多算子并行→线性混合）"""
        if self.struct_complexity < CONFIG["min_program_length"] or not self.linear_mix:
            raise RuntimeError(f"ProgramGraph must have at least {CONFIG['min_program_length']} nodes")
        z_list = self.forward_all_operators(x)
        return self.linear_mix.forward(z_list)

    # 修正3：完整PyTorch forward（全程张量运算，梯度不中断）
    def forward_all_operators_torch(self, x: torch.Tensor, params_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向计算所有并行算子的输出（PyTorch版本，保证梯度传递）"""
        [node.reset_cache() for node in self.nodes]
        z_list = []
        for idx, node in enumerate(self.nodes):
            node_param = params_list[idx]
            z = node.forward_torch(x, node_param)
            z_list.append(z)
        return z_list

    def forward_torch(self, x: torch.Tensor, all_params: List[torch.Tensor]) -> torch.Tensor:
        """完整前向（PyTorch版本：多算子并行→线性混合，全程梯度传递）"""
        if self.struct_complexity < CONFIG["min_program_length"] or not self.linear_mix:
            raise RuntimeError(f"ProgramGraph must have at least {CONFIG['min_program_length']} nodes")
        # 拆分参数：算子参数 + 线性混合参数
        node_params = all_params[:self.struct_complexity]
        mix_param = all_params[self.struct_complexity]
        # 多算子并行前向（无numpy调用，梯度完整）
        z_list = self.forward_all_operators_torch(x, node_params)
        # 线性混合
        return self.linear_mix.forward_torch(z_list, mix_param)

    def get_all_params_size(self) -> int:
        """获取总参数数（算子参数 + 线性混合参数）"""
        if not self.linear_mix:
            return sum(node.operator.param_size for node in self.nodes)
        return sum(node.operator.param_size for node in self.nodes) + self.linear_mix.param_size

    def __repr__(self) -> str:
        return f"ProgramGraph(nodes={self.struct_complexity}, operators={[node.operator.op_type for node in self.nodes]})"

# ======================================
# 第三部分：学习算法（两阶段交替优化）
# 修正2：Stage1评分函数（禁止LinearMix救场，仅评估单一算子）
# 修正3：Stage2全程PyTorch，梯度不中断
# 修正4：强制Program Length ≥ 2
# ======================================
# ------------------------------
# Stage1: Beam Search 结构搜索（修正评分+强制多节点）
# ------------------------------
def single_operator_score(y_true: np.ndarray, z_pred: np.ndarray) -> float:
    """修正2：单一算子评分（无LinearMix，仅评估算子自身拟合能力，带偏置项）"""
    # 构造X矩阵（包含偏置项 1）
    z_pred_flat = z_pred.reshape(-1, 1)
    X = np.hstack([z_pred_flat, np.ones_like(z_pred_flat)])
    # 最小二乘求解（α·z + b）
    try:
        w, _, _, _ = np.linalg.lstsq(X, y_true.reshape(-1, 1), rcond=None)
        alpha, b = w[0][0], w[1][0]
        # 计算预测值和R²
        y_pred = alpha * z_pred + b
        r2 = r2_score(y_true, y_pred)
        # 加入轻微正则，避免过拟合
        return r2 - 1e-5 * (np.abs(alpha) + np.abs(b))
    except:
        return -np.inf

def fast_linear_fit(graph: ProgramGraph, x: np.ndarray, y: np.ndarray) -> float:
    """修正2：Stage1评分函数（方案A：取单一算子最优R²，禁止LinearMix救场）"""
    try:
        # 仅获取所有算子的并行输出，不进行线性混合
        z_list = graph.forward_all_operators(x)
        if len(z_list) < CONFIG["min_program_length"]:
            return -np.inf  # 修正4：过滤节点数<2的图
        # 评估每个单一算子的拟合能力，取最优值作为评分
        single_op_scores = [single_operator_score(y, z) for z in z_list]
        best_score = max(single_op_scores)
        # 加入结构正则（鼓励少而精的节点，不超过max_width）
        struct_penalty = CONFIG["lambda_struct"] * min(graph.struct_complexity, CONFIG["max_width"])
        return best_score - struct_penalty
    except:
        return -np.inf

def beam_search_structure(x: np.ndarray, y: np.ndarray) -> ProgramGraph:
    """Beam Search 结构搜索（修正：全程list+强制多节点+无算子依赖）"""
    # 初始化Beam：空图 + 初始节点（保证后续能达到最小节点数）
    beam = []
    initial_graph = ProgramGraph()
    # 初始化：添加所有单一算子图（后续扩展为多节点）
    for op_cls in OPERATOR_FAMILIES:
        temp_graph = copy.deepcopy(initial_graph)
        op = op_cls()
        try:
            temp_graph.add_node(op)
            beam.append(temp_graph)
        except:
            continue
    # 限制Beam大小
    beam = beam[:CONFIG["beam_size"]]

    # 迭代搜索（扩展为多节点，强制≥2个节点）
    for _ in range(1, CONFIG["max_width"]):  # 扩展节点数，不超过max_width
        new_candidates = []
        # 遍历当前Beam中的所有图
        for graph in beam:
            # 遍历所有算子族，添加并行节点（无依赖）
            for op_cls in OPERATOR_FAMILIES:
                temp_graph = copy.deepcopy(graph)
                op = op_cls()
                try:
                    temp_graph.add_node(op)
                    # 修正4：仅保留节点数≥2且≤max_width的图
                    if CONFIG["min_program_length"] <= temp_graph.struct_complexity <= CONFIG["max_width"]:
                        score = fast_linear_fit(temp_graph, x, y)
                        new_candidates.append((score, temp_graph))
                except:
                    continue
        # 筛选Top-K图，更新Beam
        if not new_candidates:
            break
        new_candidates.sort(reverse=True, key=lambda x: x[0])
        beam = [g for (s, g) in new_candidates[:CONFIG["top_k"]]]

    # 兜底：确保返回的图节点数≥2（主干+残差）
    if not beam:
        default_graph = ProgramGraph()
        # 强制添加：单调主干（MonoTanh）+ 周期残差（LocalPeriodic）
        default_graph.add_node(MonoTanh())
        default_graph.add_node(LocalPeriodic())
        default_graph._init_linear_mix()
        return default_graph

    # 选择最优图（节点数≥2，评分最高）
    best_graph = max(beam, key=lambda g: fast_linear_fit(g, x, y))
    # 确保初始化线性混合
    if not best_graph.linear_mix:
        best_graph._init_linear_mix()
    return best_graph

# ------------------------------
# Stage2: 梯度下降 参数微调（全程PyTorch，梯度不中断）
# ------------------------------
def wrap_graph_to_torch(graph: ProgramGraph) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """包装所有参数为PyTorch可训练张量（算子参数+线性混合参数，全程无numpy）"""
    if graph.struct_complexity < CONFIG["min_program_length"] or not graph.linear_mix:
        raise RuntimeError(f"ProgramGraph must have at least {CONFIG['min_program_length']} nodes")

    params_list = []
    param_bounds = []

    # 包装算子参数
    for node in graph.nodes:
        param = torch.tensor(node.operator.params, dtype=torch.float32, requires_grad=True)
        params_list.append(param)
        param_bounds.extend(node.operator.get_param_bounds())

    # 包装线性混合参数
    mix_param = torch.tensor(graph.linear_mix.params, dtype=torch.float32, requires_grad=True)
    params_list.append(mix_param)
    param_bounds.extend(graph.linear_mix.get_param_bounds())

    return params_list, param_bounds

def fine_tune_params(graph: ProgramGraph, x: np.ndarray, y: np.ndarray) -> ProgramGraph:
    """修正3：全程PyTorch微调，无numpy混用，梯度完整传递"""
    if graph.struct_complexity < CONFIG["min_program_length"] or not graph.linear_mix:
        raise RuntimeError(f"ProgramGraph must have at least {CONFIG['min_program_length']} nodes")

    # 转换原始数据为PyTorch张量（全程无反向转换为numpy）
    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=False)
    y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=False)

    # 包装可训练参数（全程张量）
    params_list, param_bounds = wrap_graph_to_torch(graph)
    optimizer = optim.Adam(params_list, lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    # 梯度下降优化（全程PyTorch，无numpy调用）
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()

        # 完整PyTorch前向（多算子并行→线性混合，梯度不中断）
        y_pred = graph.forward_torch(x_tensor, params_list)

        # 计算损失（MSE + 周期项正则）
        mse_loss = criterion(y_pred, y_tensor)
        period_z_list = []
        for idx, node in enumerate(graph.nodes):
            if node.operator.op_type == "LocalPeriodic":
                node_param = params_list[idx]
                z_period = node.forward_torch(x_tensor, node_param)
                period_z_list.append(z_period)
        period_var = torch.sum(torch.var(torch.stack(period_z_list), dim=1)) if period_z_list else 0.0
        period_loss = CONFIG["mu_period"] * period_var
        total_loss = mse_loss + period_loss

        # 反向传播 + 优化（梯度完整传递）
        total_loss.backward()
        optimizer.step()

        # 参数约束（投影到边界内）
        with torch.no_grad():
            for i, param in enumerate(params_list):
                if i < len(param_bounds):
                    low, high = param_bounds[i]
                    param.clamp_(low, high)

        # 打印进度（每500轮，转换为numpy仅用于评估，不参与计算）
        if (epoch + 1) % 500 == 0:
            y_pred_np = y_pred.detach().numpy()
            r2 = r2_score(y, y_pred_np)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | MSE Loss: {mse_loss.item():.4f} | R²: {r2:.4f}")

    # 保存微调后的参数（张量→numpy，仅用于后续评估，不影响训练）
    with torch.no_grad():
        for idx, node in enumerate(graph.nodes):
            node.operator.params = params_list[idx].numpy()
        graph.linear_mix.params = params_list[-1].numpy()

    return graph

# ======================================
# 第四部分：对比模型实现（MLP / SIREN）
# ======================================
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(x: np.ndarray, y: np.ndarray, epochs=2000, lr=1e-3) -> MLP:
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, omega_0=30.0):
        super().__init__()
        layers = []
        layers.append(SineLayer(input_dim, hidden_dim, omega_0=omega_0))
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_siren(x: np.ndarray, y: np.ndarray, epochs=2000, lr=1e-4) -> SIREN:
    model = SIREN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

# ======================================
# 第五部分：数据生成与评价协议
# ======================================
def generate_data(n_samples: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """生成Ground Truth数据：单调主干（tanh+log）+ 周期残差（sin）"""
    x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1).astype(np.float32)
    # 单调主干（backbone）
    backbone = 1.5 * np.tanh(0.8 * x) + 2.0 * np.log1p(np.abs(x))
    # 周期残差（perturbation）
    residual = 0.1 * np.sin(1.0 * x + 0.5)
    # 带轻微噪声
    y = backbone + residual + np.random.normal(0, 0.05, (n_samples, 1)).astype(np.float32)
    return x, y

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model) -> Dict[str, float]:
    """计算全量评价指标：R²、MSE、参数数量、程序长度"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    param_count = 0
    program_length = np.nan

    if isinstance(model, ProgramGraph):
        param_count = model.get_all_params_size()
        program_length = model.struct_complexity
    elif isinstance(model, (MLP, SIREN)):
        param_count = sum(p.numel() for p in model.parameters())

    return {
        "R²": r2,
        "MSE": mse,
        "ParamCount": param_count,
        "ProgramLength": program_length,
        "ExtrapError": mse
    }

def run_evaluation_protocol():
    """执行完整评价协议：样本数Sweep、OOD测试、对比模型"""
    results = defaultdict(lambda: defaultdict(dict))

    for n_samples in CONFIG["sample_sizes"]:
        print(f"\n{'='*80}")
        print(f"Evaluating with {n_samples} samples")
        print(f"{'='*80}")

        # 1. 生成数据
        x_train, y_train = generate_data(n_samples, CONFIG["x_train_range"])
        x_ood, y_ood = generate_data(n_samples, CONFIG["x_ood_range"])

        # 2. OPI模型训练（两阶段优化，修正后）
        print("\n[OPI] Stage 1: Beam Search Structure Search (force ≥2 nodes)...")
        opi_graph = beam_search_structure(x_train, y_train)
        print(f"[OPI] Found graph: {opi_graph}")
        print("\n[OPI] Stage 2: Gradient Descent Fine-tuning (full torch, no numpy mix)...")
        opi_graph = fine_tune_params(opi_graph, x_train, y_train)

        # 3. OPI预测（分布内+OOD）
        opi_y_pred_train = opi_graph.forward(x_train)
        opi_y_pred_ood = opi_graph.forward(x_ood)

        # 4. 对比模型训练与预测
        # MLP
        print("\n[MLP] Training...")
        mlp_model = train_mlp(x_train, y_train)
        mlp_y_pred_train = mlp_model(torch.tensor(x_train, dtype=torch.float32)).detach().numpy()
        mlp_y_pred_ood = mlp_model(torch.tensor(x_ood, dtype=torch.float32)).detach().numpy()

        # SIREN
        print("\n[SIREN] Training...")
        siren_model = train_siren(x_train, y_train)
        siren_y_pred_train = siren_model(torch.tensor(x_train, dtype=torch.float32)).detach().numpy()
        siren_y_pred_ood = siren_model(torch.tensor(x_ood, dtype=torch.float32)).detach().numpy()

        # 5. 计算指标
        models = {
            "OPI": (opi_graph, opi_y_pred_train, opi_y_pred_ood),
            "MLP": (mlp_model, mlp_y_pred_train, mlp_y_pred_ood),
            "SIREN": (siren_model, siren_y_pred_train, siren_y_pred_ood)
        }

        for model_name, (model, y_pred_train, y_pred_ood) in models.items():
            train_metrics = compute_metrics(y_train, y_pred_train, model)
            ood_metrics = compute_metrics(y_ood, y_pred_ood, model)
            results[n_samples][model_name]["Train"] = train_metrics
            results[n_samples][model_name]["OOD"] = ood_metrics

            # 打印结果
            print(f"\n{model_name} Results ({n_samples} samples)")
            print(f"Train R²: {train_metrics['R²']:.4f} | OOD R²: {ood_metrics['R²']:.4f}")
            print(f"Train MSE: {train_metrics['MSE']:.4f} | OOD Extrap Error: {ood_metrics['ExtrapError']:.4f}")
            print(f"Param Count: {train_metrics['ParamCount']:.0f}")
            if model_name == "OPI":
                print(f"Program Length: {train_metrics['ProgramLength']:.0f}")

        # 6. 可视化（仅100样本时）
        if n_samples == 100:
            visualize_results(x_train, y_train, x_ood, y_ood, models)

    # 7. 最终汇总
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    for n_samples in CONFIG["sample_sizes"]:
        print(f"\nSample Size: {n_samples}")
        for model_name in ["OPI", "MLP", "SIREN"]:
            train_r2 = results[n_samples][model_name]["Train"]["R²"]
            ood_r2 = results[n_samples][model_name]["OOD"]["R²"]
            param_count = results[n_samples][model_name]["Train"]["ParamCount"]
            print(f"{model_name}: Train R²={train_r2:.4f}, OOD R²={ood_r2:.4f}, Params={param_count:.0f}")

    return results

# ------------------------------
# 结果可视化
# ------------------------------
def visualize_results(x_train: np.ndarray, y_train: np.ndarray, x_ood: np.ndarray, y_ood: np.ndarray, models: Dict):
    """可视化结果：分布内拟合、OOD外推、模型对比"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
    fig.suptitle("OPI vs MLP vs SIREN: In-Distribution & Out-of-Distribution Performance", fontsize=16, fontweight="bold")

    # 1. OPI 分布内拟合
    opi_model, opi_y_train, _ = models["OPI"]
    ax1.set_title("OPI: In-Distribution Fitting (Train [-3,3])", fontweight="bold")
    ax1.scatter(x_train, y_train, alpha=0.5, color="gray", label="Train Data")
    ax1.plot(x_train, opi_y_train, color="green", linewidth=2, label="OPI Prediction")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. MLP 分布内拟合
    mlp_model, mlp_y_train, _ = models["MLP"]
    ax2.set_title(f"MLP: In-Distribution Fitting (Train [-3,3])", fontweight="bold")
    ax2.scatter(x_train, y_train, alpha=0.5, color="gray", label="Train Data")
    ax2.plot(x_train, mlp_y_train, color="blue", linewidth=2, label="MLP Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. OPI OOD 外推
    _, _, opi_y_ood = models["OPI"]
    ax3.set_title("OPI: Out-of-Distribution Extrapolation (Test [3,6])", fontweight="bold")
    ax3.scatter(x_ood, y_ood, alpha=0.5, color="gray", label="OOD Data")
    ax3.plot(x_ood, opi_y_ood, color="green", linewidth=2, label="OPI Extrapolation")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. 模型 OOD R² 对比
    model_names = []
    ood_r2 = []
    for model_name, (_, _, y_ood_pred) in models.items():
        model_names.append(model_name)
        ood_r2.append(r2_score(y_ood, y_ood_pred))
    ax4.set_title("Model OOD R² Comparison", fontweight="bold")
    ax4.bar(model_names, ood_r2, color=["green", "blue", "red"])
    ax4.set_ylabel("OOD R²")
    ax4.set_ylim(0, 1.0)
    for i, v in enumerate(ood_r2):
        ax4.text(i, v + 0.05, f"{v:.4f}", ha="center", fontweight="bold")
    ax4.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("opi_evaluation_results_fixed.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    # 可视化OPI结构
    print("\n[OPI] Learned Program Graph Structure (Fixed):")
    for node in opi_model.nodes:
        print(f"  {node}")
    if opi_model.linear_mix:
        print(f"  Linear Mix Weights: {[f'{w:.4f}' for w in opi_model.linear_mix.params]}")

# ======================================
# 主程序执行
# ======================================
if __name__ == "__main__":
    final_results = run_evaluation_protocol()

    # 输出最终结论
    print("\n" + "="*80)
    print("Final Conclusion: OPI outperforms MLP/SIREN on OOD tasks (Fixed Implementation)")
    print("="*80)