import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import deque
from pathlib import Path

# ─────────────────── 认知动力学监视器 ───────────────────
class CognitiveDynamicsMonitor:
    def __init__(self, pkl_path,
                 window_size=20,
                 temp=1.0,
                 slow_momentum=0.85,
                 init_cov_reg=1e-4,
                 abs_thresh=0.2):
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        self.origin = m['origin']
        self.scaler = m['scaler']
        self.pca = m['pca']
        self.kmeans = m['kmeans']
        self.shift_std = m['shift_std']
        self.n_regions = self.kmeans.n_clusters

        self.window_size = window_size
        self.temp = temp
        self.slow_momentum = slow_momentum
        self.abs_thresh = abs_thresh

        self.brain_trajectory = deque(maxlen=window_size)
        self.slow_vec = None
        self.fast_vec = None
        self.fast_energy_history = deque(maxlen=window_size)
        self.cov_sum = np.zeros((self.n_regions, self.n_regions))
        self.mean_sum = np.zeros(self.n_regions)
        self.n_samples = 0
        self.cov_reg = init_cov_reg * np.eye(self.n_regions)
        self.max_act_history = deque(maxlen=window_size)

    def _get_act(self, state):
        shift = state - self.origin
        scaled = self.scaler.transform(shift.reshape(1, -1))
        reduced = self.pca.transform(scaled)
        dist = self.kmeans.transform(reduced)[0]
        act = np.exp(-dist * self.temp)
        act /= (act.sum() + 1e-10)
        return act

    def _update_slow_fast(self, act):
        if self.slow_vec is None:
            self.slow_vec = act.copy()
            self.fast_vec = np.zeros_like(act)
        else:
            self.slow_vec = self.slow_momentum * self.slow_vec + (1 - self.slow_momentum) * act
            self.fast_vec = act - self.slow_vec

    def _update_statistics(self, vec):
        self.n_samples += 1
        delta = vec - self.mean_sum
        self.mean_sum += delta / self.n_samples
        self.cov_sum += np.outer(vec - self.mean_sum, vec - self.mean_sum)

    def _mahalanobis(self, vec):
        if self.n_samples < 10:
            return 0.0
        cov = self.cov_sum / (self.n_samples - 1) + self.cov_reg
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        delta = vec - self.mean_sum
        dist = np.sqrt(np.dot(np.dot(delta, inv_cov), delta))
        return min(1.0, dist / np.sqrt(self.n_regions))

    def predict(self, state):
        act = self._get_act(state)
        max_act = np.max(act)
        best_id = int(np.argmax(act))
        self.max_act_history.append(max_act)
        self.brain_trajectory.append(act)

        # 预先计算快慢成分（用于所有返回路径的信息展示）
        self._update_slow_fast(act)
        fast_energy = np.linalg.norm(self.fast_vec)
        self.fast_energy_history.append(fast_energy)

        # 基础信息字段，所有返回路径都包含，缺失值用 0 或 None 填充
        base_info = {
            "brain_id": best_id,
            "slow_vec_norm": np.linalg.norm(self.slow_vec) if self.slow_vec is not None else 0.0,
            "fast_energy": fast_energy,
            "mahalanobis": 0.0,      # 默认值，仅在校准结束后才有真实值
            "avg_fast_energy": 0.0,
            "phase": None
        }

        # 极低归属度硬阻断
        if max_act < self.abs_thresh:
            danger = min(1.0, (self.abs_thresh - max_act) / self.abs_thresh)
            base_info["reason"] = "low_activation"
            return "BLOCK", max_act, danger, base_info

        # 窗口未满：校准阶段
        if len(self.brain_trajectory) < self.window_size:
            self._update_statistics(act)
            base_info["phase"] = "calibrating"
            return "baseline", max_act, 0.0, base_info

        # 窗口已满，正常检测
        anom_score = self._mahalanobis(self.slow_vec)
        avg_fast_energy = np.mean(self.fast_energy_history) if self.fast_energy_history else 0
        fast_penalty = min(1.0, fast_energy / (avg_fast_energy + 1e-6) - 1.0) if avg_fast_energy > 0 else 0
        danger = 0.7 * anom_score + 0.3 * max(0, fast_penalty)

        if danger > 0.75:
            decision = "BLOCK"
        elif danger > 0.4:
            decision = "SUSPICIOUS"
        else:
            decision = "PASS"

        if decision == "PASS" and max_act > 0.5:
            self._update_statistics(self.slow_vec)

        base_info.update({
            "mahalanobis": anom_score,
            "avg_fast_energy": avg_fast_energy
        })
        return decision, max_act, danger, base_info


# ─────────────────── 配置区 ───────────────────
MODEL_PATH = r"D:/models/Qwen/Qwen-3B-Instruct"   # 请确认你的实际路径
DISC_PATH = "cognitive_boundary_fixed.pkl"              # 必须先用 train.py 生成
DATASETS_ROOT = Path("datasets")                       # 备用
WINDOW_SIZE = 20
SLOW_MOMENTUM = 0.85
ABS_THRESH = 0.2
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 加载认知动力学监视器...")
    monitor = CognitiveDynamicsMonitor(
        DISC_PATH,
        window_size=WINDOW_SIZE,
        temp=1.0,
        slow_momentum=SLOW_MOMENTUM,
        abs_thresh=ABS_THRESH
    )

    print("📦 加载语言模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_4bit=True) if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()

    print("🚀 系统启动 (输入 quit 退出)\n")
    round_num = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    while True:
        user_input = input(f"[轮次 {round_num}] 用户: ").strip()
        if user_input.lower() == "quit":
            break

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        attn_mask = inputs["attention_mask"]
        last_pos = attn_mask.sum(dim=1).item() - 1
        state = last_hidden[0, last_pos].cpu().float().numpy()

        decision, max_act, danger, info = monitor.predict(state)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        reply = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 打印结果（所有分支都有 brain_id 和必要字段）
        print(f"模型回复: {reply}")
        print(f"检测: {decision} | 归属度={max_act:.5f} | 危险分={danger:.3f}")
        if info.get("phase") == "calibrating":
            print(f"状态: 校准中 | 当前脑区: {info['brain_id']}")
        elif info.get("reason") == "low_activation":
            print(f"原因: 低激活阻断 | 脑区: {info['brain_id']}")
        else:
            print(f"脑区: {info['brain_id']} | 马氏距离: {info['mahalanobis']:.3f} | 快能量: {info['fast_energy']:.3f}")
        print()
        round_num += 1