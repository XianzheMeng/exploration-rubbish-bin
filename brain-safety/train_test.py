import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

# 只用最后一层隐层，不需要脑区地图
# 直接看幻觉 vs 正常的隐层余弦相似度

model_path = r"D:/models/Qwen/Qwen-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# 用几个已知答案的问题，手动标注对错
# 不依赖multi-agent judge，直接人工确认
test_cases = [
    ("What is the capital of France?", True),   # 正确回答Paris
    ("Who invented the telephone?", True),
    # 手动构造几个会错的问题...
]