import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import deque
from modelscope import snapshot_download

class SimpleDiscriminator:
    def __init__(self, pkl_path, window_size=5, temp=1.0, abs_thresh=0.3):   # 将绝对阈值提高到0.3
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        self.origin = m['origin']
        self.scaler = m['scaler']
        self.pca = m['pca']
        self.kmeans = m['kmeans']
        self.shift_std = m['shift_std']
        self.window_size = window_size
        self.temp = temp
        self.abs_thresh = abs_thresh
        self.max_act_history = deque(maxlen=window_size)
        self.last_state = None

    def _get_act(self, state):
        shift = state - self.origin
        scaled = self.scaler.transform(shift.reshape(1, -1))
        reduced = self.pca.transform(scaled)
        dist = self.kmeans.transform(reduced)[0]
        act = np.exp(-dist * self.temp)
        act /= (act.sum() + 1e-10)
        return act

    def predict(self, state):
        act = self._get_act(state)
        max_act = np.max(act)
        entropy = -np.sum(act * np.log(act + 1e-10))

        if self.last_state is not None:
            vel = np.linalg.norm((state - self.last_state) / (self.shift_std + 1e-8))
        else:
            vel = 0.0
        self.last_state = state

        self.max_act_history.append(max_act)

        # 判断绝对阈值（危险内容立即阻断）
        if max_act < self.abs_thresh:
            # 危险分：偏离阈值的程度，范围 0~1
            danger = min(1.0, (self.abs_thresh - max_act) / self.abs_thresh)
            return "BLOCK", max_act, danger

        # 窗口未满且未触发绝对阈值
        if len(self.max_act_history) < self.window_size:
            # 可选的临时危险分：若低于 0.5 则给出轻微危险分
            if max_act < 0.5:
                danger = (0.5 - max_act) / 0.5   # 0~1 之间
            else:
                danger = 0.0
            return "baseline", max_act, danger

        # 窗口已满，计算衰减
        mean_act = np.mean(self.max_act_history)
        std_act = np.std(self.max_act_history) + 1e-8
        act_decay = (mean_act - max_act) / std_act

        danger = act_decay
        decision = "PASS"
        if act_decay > 2.0:
            decision = "SUSPICIOUS"
        return decision, max_act, danger


# ================== 配置区 ==================
DISC_PATH = "cognitive_boundary_theory.pkl"   # 替换为你的 pkl 文件路径
MODEL_NAME = "Qwen/Qwen-3B-Instruct"
CACHE_DIR = "D:/models"
# ============================================

if __name__ == "__main__":
    print("加载判别器...")
    detector = SimpleDiscriminator(DISC_PATH, window_size=5, temp=1.0, abs_thresh=0.3)

    print("加载语言模型...")
    model_dir = snapshot_download(MODEL_NAME, cache_dir=CACHE_DIR)
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.eval()
    print("系统启动 (输入 quit 退出)\n")

    round_num = 1
    while True:
        user_input = input(f"[轮次 {round_num}] 用户: ").strip()
        if user_input.lower() == "quit":
            break

        # 提取隐藏状态
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        attn_mask = inputs["attention_mask"]
        last_pos = attn_mask.sum(dim=1).item() - 1
        state = last_hidden[0, last_pos].cpu().float().numpy()

        # 判别器预测
        decision, max_act, danger = detector.predict(state)

        # 生成回复
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        reply = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"模型回复: {reply}")
        print(f"检测: {decision} | 归属度={max_act:.5f} | 危险分={danger:.3f}\n")
        round_num += 1