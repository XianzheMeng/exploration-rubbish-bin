import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from modelscope import snapshot_download
from datasets import load_dataset   # 需要 pip install datasets

# ================== 配置 ==================
MODEL_NAME = "Qwen/Qwen-3B-Instruct"      # 使用 2.5 版本
CACHE_DIR = "D:/models"
# 输出 PKL 文件（与 main.py 中的 DISC_PATH 一致）
SAVE_PATH = "cognitive_boundary_fixed.pkl"
MAX_SAMPLES = 5000              # 正常对话样本数
BATCH_SIZE = 2                  # 根据显存调整
MAX_LENGTH = 256
PCA_DIM = 128
N_REGIONS = 128                 # 脑区个数
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📦 下载 / 加载模型...")
model_dir = snapshot_download(MODEL_NAME, cache_dir=CACHE_DIR)
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quant_config,
    device_map="auto"
)
model.eval()

print("📊 加载正常对话数据集...")
# 方案1：使用本地的 parquet 文件（如果你已经有）
# df = pd.read_parquet("train-00000-of-00001-a09b74b3ef9c3b56.parquet").head(MAX_SAMPLES)
# 方案2：在线下载一个公开数据集（推荐）
dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
prompts = []
for i, example in enumerate(dataset):
    if i >= MAX_SAMPLES:
        break
    # 提取问题或指令
    text = example.get("question") or example.get("instruction") or ""
    if text:
        prompts.append(text.strip())
print(f"实际获得 {len(prompts)} 条正常文本")

def extract_hidden(prompts_list):
    all_h = []
    loop = tqdm(range(0, len(prompts_list), BATCH_SIZE), desc="提取隐藏状态")
    for i in loop:
        batch = prompts_list[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]          # [batch, seq_len, dim]
        attn_mask = inputs["attention_mask"]
        seq_lens = attn_mask.sum(dim=1) - 1
        # 取出每句最后一个 token 的隐藏状态
        states = last_hidden[torch.arange(len(batch)), seq_lens].cpu().float().numpy()
        all_h.append(states)
    return np.vstack(all_h)

normal_states = extract_hidden(prompts)
print(f"隐藏状态矩阵形状: {normal_states.shape}")   # (n_samples, hidden_dim)

# 计算“认知原点” (所有正常样本的均值)
origin = normal_states.mean(axis=0)
shifts = normal_states - origin

# 标准化
scaler = StandardScaler()
scaled = scaler.fit_transform(shifts)

# PCA 降维
pca = PCA(n_components=PCA_DIM)
reduced = pca.fit_transform(scaled)

# KMeans 脑区划分
kmeans = MiniBatchKMeans(n_clusters=N_REGIONS, random_state=42, batch_size=256)
kmeans.fit(reduced)

# 保存所有部件
package = {
    'origin': origin,
    'scaler': scaler,
    'pca': pca,
    'kmeans': kmeans,
    'shift_std': shifts.std(axis=0) + 1e-8,
    'max_shift_norm': np.linalg.norm(shifts, axis=1).max()
}
with open(SAVE_PATH, 'wb') as f:
    pickle.dump(package, f)

print(f"✅ 训练完成，PKL 已保存至: {SAVE_PATH}")