import pandas as pd
import json

print("===== eval_examples.csv =====")
df_eval = pd.read_csv("eval_examples.csv")
print("形状:", df_eval.shape)
print("列名:", list(df_eval.columns))
print("前2行:\n", df_eval.head(2))
print()

print("===== finetune_truth.jsonl =====")
truth_data = []
with open("finetune_truth.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        truth_data.append(json.loads(line))
print("总行数:", len(truth_data))
print("第一条的键:", list(truth_data[0].keys()))
print("第一条示例:\n", json.dumps(truth_data[0], indent=2, ensure_ascii=False)[:500])
print()

print("===== finetune_info.jsonl =====")
info_data = []
with open("finetune_info.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        info_data.append(json.loads(line))
print("总行数:", len(info_data))
print("第一条的键:", list(info_data[0].keys()))
print("第一条示例:\n", json.dumps(info_data[0], indent=2, ensure_ascii=False)[:500])