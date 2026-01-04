import os
import re
import torch
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 配置与映射 ---
LABELS = [
    "fresh_food", "processed_food", "sugary_drink", "single_use_plastic",
    "household_chemical", "eco_friendly", "non_essential", "other"
]
id2label = {i: l for i, l in enumerate(LABELS)}

# --- 2. 文本归一化 (保留队友的清洗逻辑) ---
def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"\b(rm|myr|usd)\b", " ", s)
    s = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b", " ", s)  # prices
    s = re.sub(r"\b\d+\b", " ", s)  # numbers
    s = re.sub(r"[^a-z\s\-\&\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- 3. 核心行过滤逻辑 (去掉收据中的杂质) ---
def looks_like_non_item(line: str) -> bool:
    if not isinstance(line, str): return True
    low = line.strip().lower()
    if len(low) < 2: return True
    if sum(c.isalpha() for c in low) <= 1: return True

    blacklist = [
        "total", "subtotal", "tax", "gst", "sst", "vat", "cash", "change", 
        "invoice", "receipt", "thank", "date", "time", "table", "cashier", 
        "server", "rounding", "service", "summary", "amount", "balance",
        "tel", "phone", "address",
    ]
    if any(k in low for k in blacklist): return True
    if " rm " in f" {low} ": return True
    return False

# --- 4. 提取候选商品行 ---
def extract_candidate_item_lines(ocr_text: str, max_lines: int = 25) -> List[str]:
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        return []

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    items = []

    for ln in lines:
        # 去掉行尾的价格数字
        ln2 = re.sub(r"\s+\d{1,3}(?:,\d{3})*(?:\.\d{2})\s*$", "", ln.strip())
        if looks_like_non_item(ln2):
            continue
        clean = normalize_text(ln2)
        if clean and len(clean) >= 2:
            items.append(clean)

    seen, uniq = set(), []
    for it in items:
        if it not in seen:
            uniq.append(it)
            seen.add(it)
    return uniq[:max_lines]

# --- 5. 本地模型预测函数 ---
@torch.inference_mode()
def predict_items(item_lines: List[str], threshold: float = 0.45) -> List[Dict]:
    # 路径确保指向你本地的 models 文件夹
    MODEL_DIR = "models/item_classifier_model"
    
    if not os.path.exists(MODEL_DIR):
        return [{"line": "Error", "category": "Model path not found", "confidence": 0}]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.eval()

    cleaned = [normalize_text(x) for x in item_lines]
    if not cleaned: return []
    
    enc = tok(cleaned, padding=True, truncation=True, return_tensors="pt")
    outputs = mdl(**enc)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    results = []
    for raw, cln, p in zip(item_lines, cleaned, probs):
        best_id = int(np.argmax(p))
        conf = float(np.max(p))
        label = id2label[best_id]
        if conf < threshold:
            label = "other"
        results.append({
            "line": raw,
            "clean": cln,
            "category": label,
            "confidence": round(conf, 4)
        })
    return results