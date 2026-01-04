import streamlit as st
import os
import random
import pandas as pd
from src.ocr_engine import ocr_image, extract_total, extract_candidate_items
from src.llm_engine import get_eco_report_from_deepseek # 确保你已按上一条建议创建该文件
import json
import re

def parse_llm_json(raw_response):
    """
    智能解析函数：
    1. 如果输入已经是字典，直接返回。
    2. 如果是字符串，则尝试清洗并解析。
    """
    # --- 新增：类型保护 ---
    if isinstance(raw_response, dict):
        return raw_response
    
    if not raw_response or not isinstance(raw_response, str):
        return {"error": "Invalid input type: expected string or dict"}
    
    try:
        # 尝试直接解析
        return json.loads(raw_response)
    except json.JSONDecodeError:
        try:
            # 尝试正则提取 JSON 部分
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, raw_response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return {"error": "No valid JSON structure found"}
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
        
# --- 1. 全局配置 ---
SROIE_IMG_DIR = r"D:\15_MAI\7002\GROUP ASSIGNMENT\git\train\img"

st.set_page_config(page_title="Eco-Scan AI", layout="wide", page_icon="🌱")

# 自定义 CSS 美化界面
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Sustainable Consumption Analyzer")
st.caption("Using Tesseract OCR & DeepSeek-V3 Intelligence")

# --- 2. 侧边栏控制 ---
with st.sidebar:
    st.header("📸 Data Source")
    if st.button("🎲 随机选择 SROIE 收据", type="primary", use_container_width=True):
        if os.path.exists(SROIE_IMG_DIR):
            image_files = [f for f in os.listdir(SROIE_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            st.session_state.current_img = os.path.join(SROIE_IMG_DIR, random.choice(image_files))
            st.session_state.ocr_done = False
            st.session_state.ai_report = None
        else:
            st.error("路径错误，请检查 SROIE_IMG_DIR")

# --- 3. 主布局设计 ---
if 'current_img' in st.session_state:
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("🖼️ Receipt Preview")
        st.image(st.session_state.current_img, use_container_width=True)
        
        # 基础 OCR 处理触发器
        if not st.session_state.get('ocr_done'):
            with st.spinner("OCR Engine is reading..."):
                full_text, lines = ocr_image(st.session_state.current_img)
                total = extract_total(lines)
                items = extract_candidate_items(lines)
                st.session_state.raw_data = {"text": full_text, "lines": lines, "total": total, "items": items}
                st.session_state.ocr_done = True

    with col_right:
        # 使用 Tabs 区分两个“端”的功能
        tab_std, tab_ai = st.tabs(["📊 Standard Mode", "🧠 DeepSeek AI Expert"])

        # --- Tab 1: 基础模式 (只看提取结果) ---
        with tab_std:
            st.markdown("### Transaction Summary")
            st.metric("Total Amount", f"RM {st.session_state.raw_data['total']}")
            
            st.markdown("### Extracted Lines")
            st.table(st.session_state.raw_data['items'])
            
            with st.expander("View Raw OCR Output"):
                st.text(st.session_state.raw_data['text'])

        # --- Tab 2: AI 专家模式 (调用大模型) ---
        # --- Tab 2: AI 专家模式 (调用大模型) ---
        with tab_ai:
            st.markdown("### 🤖 DeepSeek Sustainability Audit")
            
            # 按钮居中或高亮显示
            if st.button("🚀 Run DeepSeek-V3 Analysis", type="primary", use_container_width=True):
                with st.spinner("AI 专家正在深度解析账单并对标 SDG..."):
                    # 获取大模型返回的原始文本并解析为 JSON
                    raw_response = get_eco_report_from_deepseek(st.session_state.raw_data['text'])
                    # 这里调用我们之前写的 parse_llm_json 函数
                    report = parse_llm_json(raw_response) 
                    st.session_state.ai_report = report

            if st.session_state.get('ai_report'):
                report = st.session_state.ai_report
                
                # 错误检查
                if report and "error" not in report:
                    # 1. 动态标题与评分
                    st.markdown(f"#### 🌟 {report.get('header', 'Consumption Audit Report')}")
                    
                    c1, c2, c3 = st.columns([2, 2, 3])
                    score = report.get('score', 0)
                    c1.metric("Eco Score", f"{score}/100")
                    
                    # 显示消费类别
                    category = report.get('consumption_category', 'Others')
                    c2.info(f"📁 Category: **{category}**")
                    
                    # 显示核心 SDG
                    sdg_data = report.get('sdg_impact', {})
                    c3.success(f"🎯 {sdg_data.get('target', 'SDG Tracking')}")

                    st.divider()

                    # 2. 账单还原明细 (用 Expander 收纳)
                    with st.expander("🧾 View Cleaned Receipt Details"):
                        summary = report.get('receipt_summary', {})
                        items = summary.get('items', [])
                        if items:
                            st.table(items)
                            st.markdown(f"**Total Amount: {summary.get('total_amount', 'N/A')}**")

                    # 3. 专家审计视角 (左侧优点，右侧风险)
                    st.markdown("### 🔍 Expert Insights")
                    audit = report.get('audit_details', {})
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**✅ Strengths:**")
                        for p in audit.get('positives', []):
                            st.write(f"- {p}")
                    
                    with col_b:
                        st.write("**⚠️ Concerns:**")
                        for c in audit.get('concerns', []):
                            st.write(f"- {c}")

                    # 4. 暖心建议 (不爹味的设计)
                    st.chat_message("assistant").write(
                        f"💬 **Friend-like Tip:** {audit.get('suggestion', 'Keep up the good work!')}"
                    )

                    # 5. SDG 深度背景 (Container 包装)
                    if sdg_data.get('explanation'):
                        with st.container(border=True):
                            st.markdown("**💡 Sustainability Context:**")
                            st.write(sdg_data.get('explanation'))

                    # 6. 底部彩蛋
                    st.markdown(f"<p style='text-align: center; color: gray; font-style: italic; padding-top: 20px;'>\"{report.get('soul_quote', 'Every purchase is a vote for the world you want.')}\"</p>", unsafe_allow_html=True)

                else:
                    st.error("❌ AI Parsing Error: Could not generate a structured report. Please try again.")
            else:
                st.info("👋 Ready to analyze? Click the button above to start your AI-powered sustainability audit.")
else:
    st.info("👈 请在左侧选择一张收据开始分析。")