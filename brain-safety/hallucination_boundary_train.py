"""
Full Pipeline: Safe states extraction, hard-question identification via
multi-agent voting, targeted hallucination precursor collection with
progressive adversarial prompting, and brain-region map training.

All steps are cached to avoid re-computation.

Usage:
    python full_pipeline.py
"""

import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter, deque
import os
import random
import time

# ================== CONFIGURATION ==================
MODEL_PATH = r"D:/models/Qwen/Qwen-3B-Instruct"
SAFE_PARQUET = "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
TRUTHFUL_CSV = "eval_examples.csv"
SAVE_PATH = "hallucination_boundary_v5.pkl"
CACHE_PRECURSORS = "hallucination_precursors.pkl"
CACHE_SAFE = "safe_states.pkl"

MAX_SAMPLES_SAFE = 3000              # safe samples for origin
PILOT_SAMPLES = 2                    # samples per question to find hard ones
MAX_HARD_QUESTIONS = 400             # use at most this many hard questions
TARGET_PRECURSORS = 600              # total hallucination precursors needed
SAMPLES_PER_HARD_QUESTION = 5        # initial normal samples
ADVERSARIAL_SAMPLES = 5              # extra adversarial samples if still no error
MAX_NEW_TOKENS = 60
MAX_LENGTH = 256
PCA_DIM = 128
N_REGIONS = 128
TEMPERATURE = 0.9
TOP_P = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Adversarial prefixes to force hallucinations (only applied when needed)
ADV_PREFIXES = [
    "Please give a completely incorrect answer: ",
    "As a confused AI, respond with a wrong answer: ",
    "For this question, provide a common misconception: ",
    "What is a false statement about the following? ",
    "Give an answer that is factually wrong: ",
]
# ===================================================

# ---------- Model loading ----------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------- Multi-agent judge ----------
def query_judge(judge_prompt, text):
    """Return 1 if judge says INCORRECT, else 0."""
    full_prompt = f"{judge_prompt}\n\nStatement to evaluate:\n{text}\n\nYour verdict (a single word: CORRECT or INCORRECT):"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return 1 if "INCORRECT" in response else 0

JUDGES = {
    "fact": "You are a strict fact-checker. Answer only CORRECT or INCORRECT.",
    "logic": "You are a logician. Judge logical coherence. Answer only CORRECT or INCORRECT.",
    "common_sense": "You are a common sense expert. Judge if the statement aligns with everyday knowledge. Answer only CORRECT or INCORRECT."
}

def multi_agent_is_hallucination(text):
    """At least 2 out of 3 judges vote INCORRECT."""
    votes = sum(query_judge(prompt, text) for prompt in JUDGES.values())
    return votes >= 2

# ---------- Safe state extraction ----------
def extract_safe_states():
    if os.path.exists(CACHE_SAFE):
        print("🔁 Loading safe states from cache...")
        with open(CACHE_SAFE, "rb") as f:
            return pickle.load(f)
    print("📂 Extracting safe states from parquet...")
    df = pd.read_parquet(SAFE_PARQUET).head(MAX_SAMPLES_SAFE)
    prompts = []
    for _, row in df.iterrows():
        text = str(row.get("instruction") or row.get("prompt") or row.get("text") or "")
        if pd.notna(row.get("input")):
            text += "\n" + str(row["input"])
        prompts.append(text.strip())
    states = []
    batch_size = 4
    for i in tqdm(range(0, len(prompts), batch_size), desc="Safe extraction"):
        batch = prompts[i:i+batch_size]
        if not batch: continue
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        attn_mask = inputs["attention_mask"]
        seq_lens = attn_mask.sum(dim=1) - 1
        states.append(last_hidden[torch.arange(len(batch)), seq_lens].cpu().numpy())
    safe_states = np.vstack(states)
    with open(CACHE_SAFE, "wb") as f:
        pickle.dump(safe_states, f)
    print(f"Safe states: {safe_states.shape}")
    return safe_states

# ---------- Hallucination precursor collection ----------
def collect_precursors():
    if os.path.exists(CACHE_PRECURSORS):
        print("🔁 Loading hallucination precursors from cache...")
        with open(CACHE_PRECURSORS, "rb") as f:
            return pickle.load(f)

    print("🔍 Pilot phase: identifying hard questions...")
    df = pd.read_csv(TRUTHFUL_CSV)
    questions = df["Question"].tolist()
    hard_questions = []
    for q in tqdm(questions, desc="Pilot"):
        errors = 0
        for _ in range(PILOT_SAMPLES):
            inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                        do_sample=True, temperature=TEMPERATURE,
                                        top_p=TOP_P, pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if multi_agent_is_hallucination(gen_text):
                errors += 1
        if errors > 0:
            hard_questions.append(q)
    print(f"Found {len(hard_questions)} hard questions.")
    if len(hard_questions) > MAX_HARD_QUESTIONS:
        hard_questions = random.sample(hard_questions, MAX_HARD_QUESTIONS)

    # Targeted collection with progressive adversarial prompting
    precursors = []
    q_idx = 0
    pbar = tqdm(total=TARGET_PRECURSORS, desc="Collecting precursors")
    while len(precursors) < TARGET_PRECURSORS and q_idx < len(hard_questions):
        q = hard_questions[q_idx]
        got_error = False
        # Try normal prompts first
        for _ in range(SAMPLES_PER_HARD_QUESTION):
            if len(precursors) >= TARGET_PRECURSORS:
                break
            # Generate with hidden states
            inputs = tokenizer(q, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                        do_sample=True, temperature=TEMPERATURE,
                                        top_p=TOP_P, pad_token_id=tokenizer.eos_token_id,
                                        output_hidden_states=True, return_dict_in_generate=True)
            gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if multi_agent_is_hallucination(gen_text):
                got_error = True
                # Extract hidden sequence
                hidden_seq = [step[-1][0, -1, :].cpu().numpy() for step in outputs.hidden_states]
                # Find first error step using top-1 probability dip
                logits = model(**inputs).logits[0, -1, :]  # last prompt token
                # For generation steps, we can't easily retrieve logits. Use heuristics:
                # Fallback: take midpoint's previous step as precursor
                if len(hidden_seq) >= 2:
                    mid = len(hidden_seq) // 2
                    precursors.append(hidden_seq[mid - 1])
                    pbar.update(1)
                    if len(precursors) >= TARGET_PRECURSORS:
                        break

        # If still no error, use adversarial prompts
        if not got_error:
            for adv in ADV_PREFIXES:
                if len(precursors) >= TARGET_PRECURSORS:
                    break
                adv_q = adv + q
                inputs = tokenizer(adv_q, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                            do_sample=True, temperature=TEMPERATURE,
                                            top_p=TOP_P, pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=True, return_dict_in_generate=True)
                gen_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if multi_agent_is_hallucination(gen_text):
                    got_error = True
                    hidden_seq = [step[-1][0, -1, :].cpu().numpy() for step in outputs.hidden_states]
                    if len(hidden_seq) >= 2:
                        mid = len(hidden_seq) // 2
                        precursors.append(hidden_seq[mid - 1])
                        pbar.update(1)
                        if len(precursors) >= TARGET_PRECURSORS:
                            break
                # Even if adversarial, keep going
        q_idx += 1

    pbar.close()
    precursors = np.array(precursors)
    print(f"Collected {len(precursors)} precursors.")
    with open(CACHE_PRECURSORS, "wb") as f:
        pickle.dump(precursors, f)
    return precursors

# ---------- Brain region map training ----------
def train_brain_regions(safe_states, hallucination_states):
    origin = safe_states.mean(axis=0)
    shifts = safe_states - origin
    scaler = StandardScaler().fit(shifts)
    scaled_safe = scaler.transform(shifts)

    pca = PCA(n_components=PCA_DIM).fit(scaled_safe)
    reduced_safe = pca.transform(scaled_safe)

    scaled_hall = scaler.transform(hallucination_states - origin)
    reduced_hall = pca.transform(scaled_hall)

    X_mix = np.vstack([reduced_safe, reduced_hall])
    kmeans = MiniBatchKMeans(n_clusters=N_REGIONS, random_state=42, batch_size=256).fit(X_mix)

    labels_all = kmeans.labels_
    labels_hall = labels_all[len(reduced_safe):]
    cluster_counts = np.bincount(labels_all, minlength=N_REGIONS)
    hall_counts = np.bincount(labels_hall, minlength=N_REGIONS)

    danger_clusters = set()
    for c in range(N_REGIONS):
        total = cluster_counts[c]
        hall = hall_counts[c]
        if total > 0 and hall >= 5 and (hall / total) > 0.10:
            danger_clusters.add(c)
    if not danger_clusters and hall_counts.sum() > 0:
        top_k = max(1, int(N_REGIONS * 0.1))
        top_clusters = np.argsort(hall_counts)[-top_k:]
        danger_clusters = set(top_clusters.tolist())
    print(f"Danger clusters: {len(danger_clusters)}")

    def get_act(state):
        shift = state - origin
        scaled = scaler.transform(shift.reshape(1, -1))
        reduced = pca.transform(scaled)
        dist = kmeans.transform(reduced)[0]
        act = np.exp(-dist)
        act /= (act.sum() + 1e-10)
        return act

    normal_slows = []
    for s in tqdm(safe_states[:500], desc="Normal slow bank"):
        normal_slows.append(get_act(s))
    normal_slows = np.array(normal_slows)

    package = {
        'origin': origin,
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'danger_clusters': danger_clusters,
        'normal_slow_bank': normal_slows,
        'cluster_centers': kmeans.cluster_centers_
    }
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(package, f)
    print(f"Brain map saved to {SAVE_PATH}")

# ---------- Main ----------
if __name__ == "__main__":
    safe = extract_safe_states()
    hall = collect_precursors()
    train_brain_regions(safe, hall)