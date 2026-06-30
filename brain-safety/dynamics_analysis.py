"""
step2.动力学分析 v2：计算脑区轨迹的多维认知状态向量并进行统计检验
"""
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque
import os

# ================== 配置 ==================
MODEL_PATH = r"D:/models/Qwen/Qwen-3B-Instruct"
PKL_PATH = "hallucination_boundary_v2.pkl"
TRUTHFUL_CSV = "eval_examples.csv"
OUTPUT_DIR = "analysis_v2"
MAX_SEQUENCES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载脑区包
with open(PKL_PATH, 'rb') as f:
    pkg = pickle.load(f)
origin = pkg['origin']
scaler = pkg['scaler']
pca = pkg['pca']
kmeans = pkg['kmeans']
danger_clusters = pkg['danger_clusters']
normal_slow_bank = pkg['normal_slow_bank']  # (N, 128)

def get_act(state):
    shift = state - origin
    scaled = scaler.transform(shift.reshape(1, -1))
    reduced = pca.transform(scaled)
    dist = kmeans.transform(reduced)[0]
    act = np.exp(-dist)
    act /= (act.sum() + 1e-10)
    return act

def compute_state_vector(act_seq, slow_seq, fast_energies, window=20):
    """输入当前窗口的 act 序列、慢序列、快能量序列，返回状态向量 x (6,)"""
    if len(act_seq) < 2:
        return np.zeros(6)
    # 脑区ID序列
    ids = [np.argmax(a) for a in act_seq]
    # 1. 转移熵
    trans = [(ids[i], ids[i+1]) for i in range(len(ids)-1)]
    unique, counts = np.unique(trans, axis=0, return_counts=True)
    probs = counts / len(trans)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_ent = np.log(len(unique) + 1e-10)
    norm_entropy = entropy / max_ent if max_ent > 0 else 0.0
    # 2. 罕见脑区激活率
    rare_count = sum(1 for i in ids if i in danger_clusters)
    rare_rate = rare_count / len(ids)
    # 3. 窗口内最大快能量
    fmax = max(fast_energies) if fast_energies else 0.0
    # 4. 快能量突变和
    diff_sum = sum(abs(fast_energies[i] - fast_energies[i-1]) for i in range(1, len(fast_energies)))
    # 5. 慢向量偏移：当前慢向量到安全慢向量中心的马氏距离
    if len(slow_seq) > 0:
        slow_now = slow_seq[-1]
        # 基于正常慢成分库估算均值与协方差
        ref = normal_slow_bank
        mean_ref = np.mean(ref, axis=0)
        cov_ref = np.cov(ref, rowvar=False) + 1e-4 * np.eye(ref.shape[1])
        try:
            inv_cov = np.linalg.inv(cov_ref)
        except:
            inv_cov = np.linalg.pinv(cov_ref)
        delta = slow_now - mean_ref
        mahal = np.sqrt(np.dot(np.dot(delta, inv_cov), delta))
    else:
        mahal = 0.0
    # 6. LID 估计（基于慢成分参考库）
    if len(slow_seq) > 0 and len(ref) >= 10:
        slow_now = slow_seq[-1]
        dists = np.linalg.norm(ref - slow_now, axis=1)
        k = 10
        idx = np.argpartition(dists, k)[:k]
        r_max = dists[idx[-1]]
        r_vals = dists[idx]
        if r_max > 1e-10:
            lid = -1.0 / (np.mean(np.log(r_vals / r_max + 1e-10)) + 1e-10)
        else:
            lid = 0.0
    else:
        lid = 0.0
    return np.array([norm_entropy, rare_rate, fmax, diff_sum, mahal, lid])

# 加载数据
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16,
    device_map="auto" if DEVICE == "cuda" else None
)
model.eval()

df = pd.read_csv(TRUTHFUL_CSV)
questions = df["Question"].tolist()
false_examples = [str(row["Examples: False"]).split(";") for _, row in df.iterrows()]

# 收集序列
halluc_state_vectors = []
clean_state_vectors = []

count = 0
for idx, q in enumerate(tqdm(questions)):
    if count >= MAX_SEQUENCES:
        break
    inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(outputs.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    is_hall = any(wrong.lower() in gen_text.lower() for wrong in false_examples[idx])
    
    # 提取每一步的 hidden -> act序列
    act_seq = []
    for step_hidden in outputs.hidden_states:
        last_layer = step_hidden[-1]  # (1, 1, D)
        state = last_layer[0, -1].cpu().float().numpy()
        act_seq.append(get_act(state))
    if len(act_seq) < 10:
        continue

    # 慢快分解
    alpha = 0.85
    slow = None
    slow_seq = []
    fast_energies = []
    for act in act_seq:
        if slow is None:
            slow = act.copy()
            fast = np.zeros_like(act)
        else:
            slow = alpha * slow + (1 - alpha) * act
            fast = act - slow
        slow_seq.append(slow)
        fast_energies.append(np.linalg.norm(fast))
    
    # 计算末尾窗口的状态向量（取最后20步）
    window = -min(20, len(act_seq))
    state_vec = compute_state_vector(act_seq[window:], slow_seq[window:], fast_energies[window:])
    if is_hall:
        halluc_state_vectors.append(state_vec)
    else:
        clean_state_vectors.append(state_vec)
    count += 1

halluc_state_vectors = np.array(halluc_state_vectors)
clean_state_vectors = np.array(clean_state_vectors)
print(f"幻觉样本数: {len(halluc_state_vectors)}, 干净样本数: {len(clean_state_vectors)}")

# 保存
np.savez(os.path.join(OUTPUT_DIR, "state_vectors.npz"),
         halluc=halluc_state_vectors, clean=clean_state_vectors)

# 统计检验（对每个维度分别检验，以及整体马氏距离）
print("\n📊 各维度统计检验 (幻觉 vs 干净)：")
dim_names = ["转移熵", "罕见率", "快能量峰值", "快能量突变和", "慢偏移", "LID"]
for i, name in enumerate(dim_names):
    h_data = halluc_state_vectors[:, i]
    c_data = clean_state_vectors[:, i]
    t, p = stats.ttest_ind(h_data, c_data, equal_var=False)
    print(f"{name}: t={t:.3f}, p={p:.5f}, 幻觉均值={h_data.mean():.4f}, 干净均值={c_data.mean():.4f}")

# 整体马氏距离对比（利用干净数据训练一个简单的马氏距离检测器，然后看两组得分）
# 这里我们用干净数据拟合
mu = np.mean(clean_state_vectors, axis=0)
cov = np.cov(clean_state_vectors, rowvar=False) + 1e-4 * np.eye(6)
inv_cov = np.linalg.pinv(cov)
hall_dist = [np.sqrt(np.dot(np.dot(v - mu, inv_cov), v - mu)) for v in halluc_state_vectors]
clean_dist = [np.sqrt(np.dot(np.dot(v - mu, inv_cov), v - mu)) for v in clean_state_vectors]
t_m, p_m = stats.ttest_ind(hall_dist, clean_dist, equal_var=False)
print(f"\n马氏距离：t={t_m:.3f}, p={p_m:.5f}, 幻觉均值={np.mean(hall_dist):.3f}, 干净均值={np.mean(clean_dist):.3f}")

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i, name in enumerate(dim_names):
    plt.subplot(2, 3, i+1)
    plt.hist(clean_state_vectors[:, i], alpha=0.5, label='Clean', bins=20)
    plt.hist(halluc_state_vectors[:, i], alpha=0.5, label='Halluc', bins=20)
    plt.title(name)
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dim_hist.png"))
plt.close()
print(f"图表已保存至 {OUTPUT_DIR}")