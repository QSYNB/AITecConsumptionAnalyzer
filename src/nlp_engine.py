import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 对应 M4 定义的 8 个标签
LABELS = ["fresh_food", "processed_food", "sugary_drink", "single_use_plastic", 
          "household_chemical", "eco_friendly", "non_essential", "other"]
id2label = {i: l for i, l in enumerate(LABELS)}

# 2. 文本归一化函数（必须与训练时一致）
def normalize_text(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"\b(rm|myr|usd)\b", " ", s) # 去除货币符号
    s = re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b", " ", s) # 去除价格数字
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 3. 推理核心逻辑
@torch.inference_mode()
def predict_items(item_lines, model_path="models/item_classifier_model", threshold=0.45):
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    cleaned = [normalize_text(x) for x in item_lines] # 预处理 OCR 结果
    
    # 编码输入
    inputs = tokenizer(cleaned, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    
    # 计算概率
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    results = []
    for raw, p in zip(item_lines, probs):
        best_id = int(np.argmax(p))
        conf = float(np.max(p))
        
        # 逻辑：低于阈值则标记为 "other"
        label = id2label[best_id] if conf >= threshold else "other"
        
        results.append({
            "item": raw,           # 原始文本
            "category": label,     # 分类标签
            "confidence": conf     # 置信度
        })
    return results