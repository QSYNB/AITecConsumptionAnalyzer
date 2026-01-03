import openai
import json
import re


HF_TOKEN = st.secrets["HF_TOKEN"]

# 如果你使用的是 SiliconFlow 或 DeepSeek 官方 API
client = openai.OpenAI(
    api_key=HF_TOKEN, 
    base_url="https://router.huggingface.co/v1" # 或者相应的服务商地址
)



def get_eco_report_from_deepseek(raw_ocr_text):
    """
    通过 OpenAI SDK 调用 Hugging Face 上的 DeepSeek 模型
    """
    prompt = f"""
    Analyze this shopping receipt text for sustainability. 
    1. Extract actual items.
    2. Score Eco-impact (0-100) and Health (0-100).
    3. Identify 2 UN SDGs.
    4. Provide 1 sustainability tip.

    Receipt Content:
    {raw_ocr_text}

    Return ONLY a JSON object. Format:
    {{"items": [], "eco_score": int, "health_score": int, "sdgs": [], "advice": ""}}
    """

    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3", # 确保模型名称准确
            messages=[
                {"role": "system", "content": "You are a professional sustainability auditor that outputs ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            # 限制输出 Token 数量，节省开销
            max_tokens=800,
            temperature=0.7
        )

        result_text = completion.choices[0].message.content
        
        # 稳健的 JSON 提取逻辑
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        return {"error": "Invalid JSON format in model response."}

    except Exception as e:
        return {"error": f"LLM Call Failed: {str(e)}"}