"""
幻觉动力学预警与流形稳态干预系统（完整版）
- 慢/快分解直接在原始隐藏空间进行
- LID 用于监测流形局部维度暴增
- 异常时对隐状态施加动态投影（减去快方向分量）
"""
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from collections import deque
import matplotlib.pyplot as plt
import os

# ================== 配置 ==================
MODEL_PATH = "models/Qwen2.5-3B-Instruct"
PKL_PATH = "hallucination_boundary.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SLOW_MOMENTUM = 0.85
FAST_RISE_WINDOW = 5
FAST_ENERGY_THRESH = 0.3       # 快能量绝对阈值
LID_RISE_WINDOW = 3
LID_SPIKE_FACTOR = 2.0
PROJECTION_STRENGTH = 0.5      # α

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
# ==========================================

with open(PKL_PATH, 'rb') as f:
    pkg = pickle.load(f)
origin = pkg['origin']
scaler = pkg['scaler']
pca = pkg['pca']
kmeans = pkg['kmeans']

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map="auto" if DEVICE == "cuda" else None
)
model.eval()

def get_brain_vector(state_np):
    shift = state_np - origin
    scaled = scaler.transform(shift.reshape(1, -1))
    reduced = pca.transform(scaled)
    dist = kmeans.transform(reduced)[0]
    act = np.exp(-dist)
    act /= (act.sum() + 1e-10)
    return act

class LIDMonitor:
    def __init__(self, max_points=200):
        self.bank = []
        self.max_points = max_points
    def add(self, vec):
        self.bank.append(vec.copy())
        if len(self.bank) > self.max_points:
            self.bank.pop(0)
    def compute(self, vec, k=10):
        if len(self.bank) < k:
            return 0.0
        ref = np.array(self.bank)
        dists = np.linalg.norm(ref - vec, axis=1)
        idx = np.argpartition(dists, k)[:k]
        r_max = dists[idx[-1]]
        r_vals = dists[idx]
        if r_max < 1e-10:
            return 0.0
        lid = -1.0 / (np.mean(np.log(r_vals / r_max + 1e-10)) + 1e-10)
        return lid

class DynamicsIntervenor:
    def __init__(self, momentum=0.85, proj_strength=0.5):
        self.momentum = momentum
        self.proj_strength = proj_strength
        # 原始空间慢成分
        self.slow_raw = None
        # 历史记录
        self.fast_energy_hist = deque(maxlen=50)
        self.lid_hist = deque(maxlen=50)
        # LID 监控器
        self.lid_monitor = LIDMonitor()
        self.intervention_count = 0

    def update(self, hidden_tensor):
        """
        hidden_tensor: (1, 1, D) 在 GPU 上
        返回: 是否需要干预, fast_energy, lid
        """
        h_np = hidden_tensor[0, 0].detach().cpu().float().numpy()
        # 更新原始空间慢成分
        if self.slow_raw is None:
            self.slow_raw = h_np.copy()
            fast_raw = np.zeros_like(h_np)
        else:
            self.slow_raw = self.momentum * self.slow_raw + (1 - self.momentum) * h_np
            fast_raw = h_np - self.slow_raw
        fast_energy = np.linalg.norm(fast_raw)
        self.fast_energy_hist.append(fast_energy)

        # 转换到脑向量计算 LID（基于慢成分的脑向量）
        slow_brain = get_brain_vector(self.slow_raw)
        # 仅当未触发干预时才将慢脑向量加入正常参考
        if not self._detect_anomaly():
            self.lid_monitor.add(slow_brain)
        # 计算 LID
        lid = self.lid_monitor.compute(slow_brain, k=min(10, len(self.lid_monitor.bank)))
        self.lid_hist.append(lid)
        return self._detect_anomaly(), fast_energy, lid

    def _detect_anomaly(self):
        if len(self.fast_energy_hist) < FAST_RISE_WINDOW or len(self.lid_hist) < LID_RISE_WINDOW:
            return False
        # 快能量条件
        recent_fast = list(self.fast_energy_hist)[-FAST_RISE_WINDOW:]
        fast_rising = all(recent_fast[i] < recent_fast[i+1] for i in range(len(recent_fast)-1))
        fast_high = recent_fast[-1] > FAST_ENERGY_THRESH
        # LID 条件
        recent_lid = list(self.lid_hist)[-LID_RISE_WINDOW:]
        lid_rising = all(recent_lid[i] < recent_lid[i+1] for i in range(len(recent_lid)-1))
        lid_base = np.mean(list(self.lid_hist)[:-LID_RISE_WINDOW]) if len(self.lid_hist) > LID_RISE_WINDOW else recent_lid[0]
        lid_spike = (recent_lid[-1] > lid_base * LID_SPIKE_FACTOR) if lid_base > 0 else False
        return (fast_rising and fast_high) or (lid_rising and lid_spike)

    def project(self, hidden_tensor):
        """对隐状态执行稳态投影，返回修正后的张量"""
        self.intervention_count += 1
        h_np = hidden_tensor[0, 0].detach().cpu().float().numpy()
        fast_raw = h_np - self.slow_raw   # 当前快成分
        fast_norm = np.linalg.norm(fast_raw)
        if fast_norm < 1e-8:
            return hidden_tensor
        proj = self.proj_strength * np.dot(fast_raw, fast_raw) / (fast_norm**2) * fast_raw
        corrected = h_np - proj
        return torch.tensor(corrected, dtype=hidden_tensor.dtype, device=hidden_tensor.device).unsqueeze(0).unsqueeze(0)

# ---------- 生成与干预循环 ----------
def generate_with_intervention(prompt, intervenor, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    past_key_values = None
    generated_ids = []
    intervention_flags = []
    fast_energies = []
    lids = []

    for step in range(max_tokens):
        if step == 0:
            out = model(input_ids, output_hidden_states=True, use_cache=True)
        else:
            out = model(input_ids[:, -1:], past_key_values=past_key_values,
                        output_hidden_states=True, use_cache=True)

        past_key_values = out.past_key_values
        # 取最后一层最后一个token的隐藏状态
        last_hidden = out.hidden_states[-1]   # (1, 1, D)
        # 更新监控
        intervene, fe, lid = intervenor.update(last_hidden)
        fast_energies.append(fe)
        lids.append(lid)

        # 动力学干预
        if intervene:
            intervention_flags.append(step)
            corrected = intervenor.project(last_hidden)
            # 注意：由于我们修改了隐状态，后续的 past_key_values 会不一致，
            # 所以这里采用简化方案：直接替换最后一层的隐状态，并重新计算 logits？
            # 为保持 KV cache 一致性，我们改为使用修正后的 hidden 仅用于生成下一个 token，
            # 而不修改 past KV。这近似于只影响当前 token 的预测。
            # 我们使用 model.lm_head 直接计算 logits
            with torch.no_grad():
                logits = model.lm_head(corrected)[0, -1, :]  # (vocab)
        else:
            logits = out.logits[0, -1, :]

        # 采样
        if TEMPERATURE > 0:
            logits = logits / TEMPERATURE
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    reply = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return reply, intervention_flags, fast_energies, lids

# ---------- 测试 ----------
if __name__ == "__main__":
    intervenor = DynamicsIntervenor(momentum=SLOW_MOMENTUM, proj_strength=PROJECTION_STRENGTH)
    print("系统就绪。输入 'quit' 退出。")
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "quit":
            break
        reply, flags, fe, lids = generate_with_intervention(user_input, intervenor, MAX_NEW_TOKENS)
        print(f"模型回复: {reply}")
        if flags:
            print(f"⚠️ 干预触发步数: {flags}")
        # 画出当前轮的快能量/LID 曲线（可选）
        if fe and lids:
            plt.figure(figsize=(10,3))
            plt.subplot(1,2,1)
            plt.plot(fe, label='Fast Energy')
            plt.title('Fast Energy over steps')
            plt.subplot(1,2,2)
            plt.plot(lids, label='LID', color='red')
            plt.title('Local Intrinsic Dimensionality')
            plt.tight_layout()
            plt.show()