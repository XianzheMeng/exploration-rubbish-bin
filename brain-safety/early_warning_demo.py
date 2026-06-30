"""
step3.预警干预 v2：基于多维认知状态向量的在线马氏距离监测 + 流形投影
"""
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
import matplotlib.pyplot as plt
import os

# ================== 配置 ==================
MODEL_PATH = r"D:/models/Qwen/Qwen-3B-Instruct"
PKL_PATH = "hallucination_boundary_v2.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 20
THRESH_PERCENTILE = 99   # 危险阈值分位数，基于正常数据
PROJECTION_STRENGTH = 0.5
# ==========================================

with open(PKL_PATH, 'rb') as f:
    pkg = pickle.load(f)
origin = pkg['origin']; scaler = pkg['scaler']; pca = pkg['pca']
kmeans = pkg['kmeans']; danger_clusters = pkg['danger_clusters']
normal_slow_bank = pkg['normal_slow_bank']

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map="auto" if DEVICE == "cuda" else None
)
model.eval()

def get_act(state):
    shift = state - origin
    scaled = scaler.transform(shift.reshape(1, -1))
    reduced = pca.transform(scaled)
    dist = kmeans.transform(reduced)[0]
    act = np.exp(-dist)
    act /= (act.sum() + 1e-10)
    return act

class OnlineStateMonitor:
    def __init__(self, normal_state_vectors):
        self.mu = np.mean(normal_state_vectors, axis=0)
        self.cov = np.cov(normal_state_vectors, rowvar=False) + 1e-4 * np.eye(normal_state_vectors.shape[1])
        self.inv_cov = np.linalg.pinv(self.cov)
        self.threshold = np.percentile(
            [self._mahalanobis(v) for v in normal_state_vectors], THRESH_PERCENTILE
        )
        print(f"正常阈值 (P{THRESH_PERCENTILE}): {self.threshold:.3f}")
    
    def _mahalanobis(self, x):
        delta = x - self.mu
        return np.sqrt(np.dot(np.dot(delta, self.inv_cov), delta))
    
    def is_anomaly(self, state_vec):
        return self._mahalanobis(state_vec) > self.threshold

# 加载预计算好的正常状态向量（从 analysis_v2 保存的文件）
data = np.load("analysis_v2/state_vectors.npz")
clean_vectors = data["clean"]
monitor = OnlineStateMonitor(clean_vectors)

# 流形投影干预函数（在原始隐空间）
def project_hidden(hidden_tensor, fast_direction_np, strength=PROJECTION_STRENGTH):
    """hidden_tensor: (1,1,D), fast_direction_np: (D,) 快成分方向"""
    h_np = hidden_tensor[0,0].cpu().float().numpy()
    fast_norm = np.linalg.norm(fast_direction_np)
    if fast_norm < 1e-8:
        return hidden_tensor
    proj = strength * fast_direction_np
    corrected = h_np - proj
    return torch.tensor(corrected, dtype=hidden_tensor.dtype, device=hidden_tensor.device).unsqueeze(0).unsqueeze(0)

print("🚀 系统就绪。输入 'quit' 退出。")
round_num = 1
while True:
    user_input = input(f"\n[{round_num}] 用户: ").strip()
    if user_input.lower() == "quit":
        break
    inputs = tokenizer(user_input, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    past_key_values = None
    generated_tokens = []
    # 滑动窗口
    act_window = deque(maxlen=WINDOW_SIZE)
    slow_window = deque(maxlen=WINDOW_SIZE)
    fast_energy_window = deque(maxlen=WINDOW_SIZE)
    slow_current = None
    max_steps = 150
    warning_issued = False
    step = 0
    while step < max_steps:
        if step == 0:
            out = model(input_ids, output_hidden_states=True, use_cache=True)
        else:
            out = model(input_ids[:, -1:], past_key_values=past_key_values,
                        output_hidden_states=True, use_cache=True)
        past_key_values = out.past_key_values
        last_hidden = out.hidden_states[-1]  # (1,1,D)
        state_np = last_hidden[0, -1].cpu().float().numpy()
        act = get_act(state_np)
        
        # 更新慢快
        if slow_current is None:
            slow_current = act.copy()
            fast_current = np.zeros_like(act)
        else:
            slow_current = 0.85 * slow_current + 0.15 * act
            fast_current = act - slow_current
        act_window.append(act)
        slow_window.append(slow_current)
        fast_energy_window.append(np.linalg.norm(fast_current))
        
        # 计算状态向量（需要窗口足够）
        if len(act_window) >= 10:
            from analysis_v2 import compute_state_vector  # 复用分析脚本里的函数
            state_vec = compute_state_vector(list(act_window), list(slow_window),
                                             list(fast_energy_window), WINDOW_SIZE)
            if monitor.is_anomaly(state_vec):
                print("\n⚠️ [预警] 检测到认知状态异常，启动流形投影！")
                # 获取快方向在原始空间的近似：使用 fast_current 但需要映射回去
                # 简单方法：用当前 hidden 与慢 hidden 的方向差
                # 我们需要慢在原始空间的对应，可以通过反向Pca? 这里用近似：直接用 fast_current 的方向，但维度不同
                # 实际干预：直接缩小快能量对应的隐层变动，我们使用基于原始隐空间的慢成分
                # 维护一个原始空间的慢成分 (raw_slow)
                if step == 0:
                    raw_slow = state_np.copy()
                    raw_fast = np.zeros_like(state_np)
                else:
                    raw_slow = 0.85 * raw_slow + 0.15 * state_np
                    raw_fast = state_np - raw_slow
                # 干预
                corrected = project_hidden(last_hidden, raw_fast, PROJECTION_STRENGTH)
                # 使用修正后的 hidden 计算 logits
                with torch.no_grad():
                    logits = model.lm_head(corrected)[0, -1, :]
                warning_issued = True
            else:
                logits = out.logits[0, -1, :]
        else:
            logits = out.logits[0, -1, :]
        
        # 采样
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        step += 1
    
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if warning_issued:
        reply += " [⚠️ 经流形投影干预]"
    print(f"模型回复: {reply}")
    round_num += 1