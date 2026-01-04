import openai
import json
import re
import streamlit as st


HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize the OpenAI client to connect to models on Hugging Face
client = openai.OpenAI(
    api_key=HF_TOKEN, 
    base_url="https://router.huggingface.co/v1" 
)



def get_eco_report_from_deepseek(raw_ocr_text):
    """
    Calling DeepSeek on Hugging Face via OpenAI SDK
    """
    prompt = f"""
    ### ROLE ###
    You are a Senior Sustainability Audit Expert. Your goal is to analyze the provided OCR text from a shopping receipt and generate a human-centric, professional sustainability report.

    ### LOGIC & TASKS ###
    1. **Data Extraction**: Reconstruct the shopping list with product names and prices. Correct OCR artifacts (e.g., '0' to 'O', '1' to 'I').
    2. **Classification**: Categorize the transaction into: [Dining & Eating Out / Grocery & Fresh Food / Household & Living / Others].
    3. **Differential Audit**:
    - For [Dining]: Evaluate nutritional balance and energy contribution.
    - For [Grocery]: Assess the Veg-to-Meat ratio and identify local/low-carbon items.
    - For [Household]: Focus on packaging minimalism and eco-friendly materials.
    4. **SDG Mapping**: Link the consumption to a specific UN Sustainable Development Goal.

    ### OUTPUT FORMAT (STRICT JSON) ###
    You must return a valid JSON object ONLY. Use the following structure:
    {{
    "header": "A catchy, emotional title (e.g., 'The Green Gourmet’s Lunch')",
    "receipt_summary": {{
        "items": [
        {{"name": "Cleaned Product Name", "price": "Price with Currency"}}
        ],
        "total_amount": "Total sum found"
    }},
    "consumption_category": "The chosen category",
    "audit_details": {{
        "positives": ["Point 1: Sustainability/Health benefit", "Point 2..."],
        "concerns": ["A gentle observation of a potential risk"],
        "suggestion": "A warm, friend-like tip for next time (avoid preachy tone)"
    }},
    "sdg_impact": {{
        "target": "SDG Number and Name (e.g., SDG 12: Responsible Consumption)",
        "explanation": "Why this purchase relates to this SDG"
    }},
    "score": 85,
    "soul_quote": "A warm, inspiring 'Daily Wisdom' quote"
    }}

    ### RAW OCR TEXT TO ANALYZE ###
    {raw_ocr_text}
    """

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", 
            messages=[
                {"role": "system", "content": "You are a professional sustainability auditor that outputs ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            # Limit the number of tokens output to save overhead.
            max_tokens=1000,
            temperature=0.7
        )

        result_text = completion.choices[0].message.content
        
        # JSON extraction logic
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        return {"error": "Invalid JSON format in model response."}

    except Exception as e:
        return {"error": f"LLM Call Failed: {str(e)}"}
    
def parse_llm_json(raw_response):
    """
    Clean the text returned by the LLM and parse it into a Python dictionary
    """
    try:
        # 1. 使用正则提取 JSON 块（防止模型返回多余的 Markdown 标记或解释词）
        # 1. Use regex to extract the JSON block (to avoid extra markdown or explanatory words from the model)
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, raw_response, re.DOTALL)
        
        if match:
            json_str = match.group()
            return json.loads(json_str)
        else:
            # 如果没找到 JSON 括号，尝试直接解析整个字符串
            # If no JSON braces found, try parsing the whole string directly
            return json.loads(raw_response)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return None 