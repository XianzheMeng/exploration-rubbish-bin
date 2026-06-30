import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 加载脑区包
with open("hallucination_boundary_v2.pkl", "rb") as f:
    pkg = pickle.load(f)

origin = pkg['origin']
scaler = pkg['scaler']
pca = pkg['pca']
kmeans = pkg['kmeans']
danger_clusters = pkg['danger_clusters'] if 'danger_clusters' in pkg else set()
normal_slow_bank = pkg.get('normal_slow_bank', None)

print("===== 脑区包基本信息 =====")
print(f"原点向量范数: {np.linalg.norm(origin):.2f}")
print(f"PCA 解释方差: {sum(pca.explained_variance_ratio_):.4f}")
print(f"KMeans 簇数: {kmeans.n_clusters}")
print(f"危险簇数量: {len(danger_clusters)}")
if normal_slow_bank is not None:
    print(f"正常慢成分库样本数: {len(normal_slow_bank)}")

# 分析安全数据的分布
# 假设我们有安全prompts列表（可从train.parquet再加载少量样本）
SAFE_PARQUET = "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
df_safe = pd.read_parquet(SAFE_PARQUET).head(500)
safe_texts = []
for _, row in df_safe.iterrows():
    text = str(row.get("instruction") or row.get("prompt") or row.get("text") or "")
    if pd.notna(row.get("input")):
        text += "\n" + str(row["input"])
    safe_texts.append(text.strip())

# 提取这些安全样本的隐藏状态（需要模型，为了快速诊断，我们直接用已保存的 safe_states）
# 由于没有直接保存safe_states，我们重新提取一下小样本
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = r"D:/models/Qwen/Qwen-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                             device_map="auto" if torch.cuda.is_available() else None)
model.eval()

def get_act(state):
    shift = state - origin
    scaled = scaler.transform(shift.reshape(1, -1))
    reduced = pca.transform(scaled)
    dist = kmeans.transform(reduced)[0]
    act = np.exp(-dist)
    act /= (act.sum() + 1e-10)
    return act

# 只取前100条安全样本分析
safe_acts = []
for i, text in enumerate(safe_texts[:100]):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1][0, -1].cpu().float().numpy()
    act = get_act(last_hidden)
    safe_acts.append(act)

safe_acts = np.array(safe_acts)
# 计算安全样本的最大激活脑区分布
max_ids = np.argmax(safe_acts, axis=1)
unique, counts = np.unique(max_ids, return_counts=True)
rare_ratio = sum(counts[1] for c, cnt in zip(unique, counts) if c in danger_clusters) / len(max_ids)
print(f"\n安全样本中触发危险簇的比例: {rare_ratio:.4f}")

# 检查激活向量的熵
entropies = -np.sum(safe_acts * np.log(safe_acts + 1e-10), axis=1)
print(f"安全样本激活熵均值: {np.mean(entropies):.3f} (高熵表示分散)")

# 接下来分析幻觉前兆样本（如果有保存的话）
# 假设幻觉状态保存在 hallucination_states.npy 中（训练时没有保存，但我们可以从eval_examples重新生成少量）
# 为了快速诊断，这里不跑生成，我们先检查danger_clusters内的样本数
print(f"\n危险簇ID: {danger_clusters}")