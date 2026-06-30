import numpy as np
from scipy import stats

# ==========================================
# 1. 先整理你的实验数据
# ==========================================
# 单算子准确率（从你给的结果里提取）
single_op_acc = {
    "dilated_conv": 0.5070,
    "local_conv": 0.4727,
    "morph_op": 0.4183,
    "channel_perm": 0.2078,
    "freq_op": 0.2070,
    "1x1_conv": 0.1950
}

# 所有15个组合的数据（从你给的结果里提取）
comb_data = [
    # (组合名, 算子1, 算子2, Volume Score, 平均准确率)
    ("local_conv+dilated_conv", "local_conv", "dilated_conv", 0.923685, 0.8044),
    ("dilated_conv+channel_perm", "dilated_conv", "channel_perm", 0.890702, 0.8000),
    ("local_conv+channel_perm", "local_conv", "channel_perm", 0.913059, 0.7977),
    ("dilated_conv+freq_op", "dilated_conv", "freq_op", 0.891584, 0.7961),
    ("local_conv+freq_op", "local_conv", "freq_op", 0.909278, 0.7940),
    ("local_conv+1x1_conv", "local_conv", "1x1_conv", 0.931636, 0.7936),
    ("local_conv+morph_op", "local_conv", "morph_op", 0.902677, 0.7927),
    ("dilated_conv+1x1_conv", "dilated_conv", "1x1_conv", 0.880608, 0.7909),
    ("dilated_conv+morph_op", "dilated_conv", "morph_op", 0.884780, 0.7865),
    ("morph_op+channel_perm", "morph_op", "channel_perm", 0.776836, 0.7036),
    ("1x1_conv+morph_op", "1x1_conv", "morph_op", 0.906328, 0.7012),
    ("freq_op+morph_op", "freq_op", "morph_op", 0.856251, 0.6974),
    ("freq_op+channel_perm", "freq_op", "channel_perm", 0.855342, 0.2738),
    ("1x1_conv+channel_perm", "1x1_conv", "channel_perm", 0.886391, 0.2700),
    ("1x1_conv+freq_op", "1x1_conv", "freq_op", 0.884661, 0.2677)
]

# ==========================================
# 2. 计算每个组合的 sqrt(acc1*acc2) * Volume
# ==========================================
metric_list = []  # 存储指标：sqrt(acc1*acc2) * Volume
acc_list = []  # 存储平均准确率

for comb_name, op1, op2, volume, avg_acc in comb_data:
    # 获取两个单算子的准确率
    acc1 = single_op_acc[op1]
    acc2 = single_op_acc[op2]

    # 计算指标：sqrt(acc1*acc2) * Volume
    metric = np.sqrt(acc1 * acc2) * volume

    # 存入列表
    metric_list.append(metric)
    acc_list.append(avg_acc)

    # 打印每个组合的计算结果（可选）
    print(f"{comb_name}:")
    print(f"  sqrt(acc1*acc2) = {np.sqrt(acc1 * acc2):.4f}, Volume = {volume:.4f}")
    print(f"  指标 = {metric:.4f}, 平均准确率 = {avg_acc:.4f}\n")

# ==========================================
# 3. 计算皮尔逊相关系数
# ==========================================
corr, p_value = stats.pearsonr(metric_list, acc_list)

# ==========================================
# 4. 打印最终结果
# ==========================================
print("=" * 80)
print(f"【最终结果】")
print(f"指标（sqrt(acc1*acc2)*Volume）与平均准确率的皮尔逊相关系数：{corr:.4f}")
print(f"p值：{p_value:.8f}")
print("=" * 80)

# 解释结果
if corr > 0.7 and p_value < 0.05:
    print("结论：指标与准确率呈【显著强正相关】，可以作为核心预筛选指标！")
elif corr > 0.5 and p_value < 0.05:
    print("结论：指标与准确率呈【显著中等正相关】，可以作为辅助筛选指标。")
else:
    print("结论：指标与准确率相关性不显著，需要调整指标设计。")