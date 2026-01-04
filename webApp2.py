import streamlit as st
import os
import random
import pandas as pd
from src.ocr_engine import ocr_image, extract_total, extract_candidate_items
from src.llm_engine import get_eco_report_from_deepseek 
import json
import re
from src.nlp_engine import extract_candidate_item_lines, predict_items

def parse_llm_json(raw_response):
    """
    智能解析函数：
    1. 如果输入已经是字典，直接返回。
    2. 如果是字符串，则尝试清洗并解析。
    INTELIGENT PARSING FUNCTION
    1. If input is already a dict, return as is.
    2. If it's a string, attempt to clean and parse it.
    """
    # --- 类型保护 ---
    # 1. Type Guarding ---
    if isinstance(raw_response, dict):
        return raw_response
    
    if not raw_response or not isinstance(raw_response, str):
        return {"error": "Invalid input type: expected string or dict"}
    
    try:
        # 尝试直接解析
        # Try direct parsing
        return json.loads(raw_response)
    except json.JSONDecodeError:
        try:
            # 尝试正则提取 JSON 部分
            # Try regex extraction of JSON part
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, raw_response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                return {"error": "No valid JSON structure found"}
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
        
# --- 1. 全局配置 ---
# --- 1. Global Config ---
SROIE_IMG_DIR = r"D:\15_MAI\7002\GROUP ASSIGNMENT\git\train\img"

st.set_page_config(page_title="Eco-Scan AI", layout="wide", page_icon="🌱")

# 自定义 CSS 美化界面
# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Sustainable Consumption Analyzer")
st.caption("Using Tesseract OCR & DeepSeek-V3 Intelligence")

# --- 2. 侧边栏控制 ---
# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("📸 Data Source")
    if st.button("🎲 Randomly SROIE receipt ", type="primary", use_container_width=True):
        if os.path.exists(SROIE_IMG_DIR):
            image_files = [f for f in os.listdir(SROIE_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            st.session_state.current_img = os.path.join(SROIE_IMG_DIR, random.choice(image_files))
            st.session_state.ocr_done = False
            st.session_state.ai_report = None
        else:
            st.error("The path is incorrect, please check SROIE_IMG_DIR")
    
    # 选项 B: 从本地文件夹选择 (你的新需求)
    # Option B: Choose from local upload (your new requirement)
    uploaded_file = st.file_uploader("📂 Choose receipts from documents", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        temp_upload_path = "temp_upload.jpg"
        with open(temp_upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        if st.session_state.get('last_uploaded') != uploaded_file.name:
            st.session_state.current_img = temp_upload_path
            st.session_state.ocr_done = False
            st.session_state.ai_report = None
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.is_uploaded = True

    

# --- 3. 主布局设计 ---
# --- 3. Main Layout Design ---
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
        # Use Tabs to separate two "modes" of functionality
        tab_std, tab_ai = st.tabs(["📊 Standard Mode", "🧠 DeepSeek AI Expert"])

    # --- Tab 1: 基础模式 (本地模型驱动) ---
    # --- Tab 1. Standard Mode (Local Model Driven) ---
    with tab_std:
        st.markdown("### 📊 Local NLP Audit (DistilBERT)")
        
        raw_text = st.session_state.raw_data.get('text', "")
        
        if raw_text:
            # 1. 提取候选行 (调用队友的过滤逻辑)
            # 1. Extract candidate lines (using teammate's filtering logic)
            with st.spinner("Filtering receipt lines..."):
                candidate_items = extract_candidate_item_lines(raw_text)
            
            if candidate_items:
                with st.spinner("Classifying items using local model..."):
                    results = predict_items(candidate_items)
                
                
                filtered_results = [res for res in results if res['category'] != 'other']
                
                if filtered_results:
                    # 转换成 DataFrame 方便统计
                    # Convert to DataFrame for easier stats
                    df_res = pd.DataFrame(filtered_results)
                    
                    # 布局：左边显示指标和表格，右边显示饼图
                    # Layout: Left for metrics and table, right for pie chart
                    c_metrics, c_chart = st.columns([1, 1])
                    
                    with c_metrics:
                        st.metric("Identified Specific Items", len(filtered_results))
                        st.write("Below are the categorized items (excluding 'other').")
                    
                    with c_chart:
                        # 使用 plotly 画饼图
                        # Use plotly to draw pie chart
                        import plotly.express as px
                        cat_counts = df_res['category'].value_counts().reset_index()
                        cat_counts.columns = ['Category', 'Count']
                        
                        fig = px.pie(
                            cat_counts, 
                            values='Count', 
                            names='Category', 
                            title='Consumption Distribution',
                            hole=0.4, 
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)

                   
                    st.markdown("#### 🛒 Detailed Classification")
                    table_data = []
                    for res in results:
                        if  res['category'] != 'other':
                            table_data.append({
                                "Item": res['line'],
                                "Category": res['category'],
                                "Confidence": f"{res['confidence']:.2%}"
                            })
                    st.table(table_data)
                else:
                    st.warning("All detected items were classified as 'other'. No chart to display.")
                    st.table(results) 
            else:
                st.warning("No valid items detected by the local filters.")
        else:
            st.info("Please upload a receipt first.")

        # --- Tab 2: AI 专家模式 (调用大模型) ---
        # --- Tab 2: AI Expert Mode (Calling Large Model) ---
        with tab_ai:
            st.markdown("### 🤖 DeepSeek Sustainability Audit")
            
            
            if st.button("🚀 Run DeepSeek-V3 Analysis", type="primary", use_container_width=True):
                with st.spinner("AI experts are conducting in-depth analysis of the bills..."):
                    # 获取大模型返回的原始文本并解析为 JSON
                    # Get raw response from large model and parse to JSON
                    raw_response = get_eco_report_from_deepseek(st.session_state.raw_data['text'])
                    # 这里调用我们之前写的 parse_llm_json 函数
                    # Here we use our previously defined parse_llm_json function
                    report = parse_llm_json(raw_response) 
                    st.session_state.ai_report = report

            if st.session_state.get('ai_report'):
                report = st.session_state.ai_report
                
                
                if report and "error" not in report:
                    # 1. 动态标题与评分
                    # 1. Dynamic Title and Score
                    st.markdown(f"#### 🌟 {report.get('header', 'Consumption Audit Report')}")
                    
                    c1, c2, c3 = st.columns([2, 2, 3])
                    score = report.get('score', 0)
                    c1.metric("Eco Score", f"{score}/100")
                    
                    # 显示消费类别
                    # Display Consumption Category
                    category = report.get('consumption_category', 'Others')
                    c2.info(f"📁 Category: **{category}**")
                    
                    # 显示核心 SDG
                    # Display Core SDG
                    sdg_data = report.get('sdg_impact', {})
                    c3.success(f"🎯 {sdg_data.get('target', 'SDG Tracking')}")

                    st.divider()

                    # 2. 账单还原明细 (用 Expander 收纳)
                    # 2. Receipt Details (Using Expander)
                    with st.expander("🧾 View Cleaned Receipt Details",expanded=True):
                        summary = report.get('receipt_summary', {})
                        items = summary.get('items', [])
                        if items:
                            st.table(items)
                            st.markdown(f"**Total Amount: {summary.get('total_amount', 'N/A')}**")

                    # 3. 专家审计视角 (左侧优点，右侧风险)
                    # 3. Expert Audit Perspective (Left: Positives, Right: Risks)
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
                    # 4. Friendly Suggestions (Non-preachy Design)
                    st.chat_message("assistant").write(
                        f"💬 **Friend-like Tip:** {audit.get('suggestion', 'Keep up the good work!')}"
                    )

                    # 5. SDG 深度背景 (Container 包装)
                    # 5. SDG In-depth Context (Using Container)
                    if sdg_data.get('explanation'):
                        with st.container(border=True):
                            st.markdown("**💡 Sustainability Context:**")
                            st.write(sdg_data.get('explanation'))

                    # 6. 底部彩蛋
                    # 6. Footer Easter Egg
                    st.markdown(f"<p style='text-align: center; color: gray; font-style: italic; padding-top: 20px;'>\"{report.get('soul_quote', 'Every purchase is a vote for the world you want.')}\"</p>", unsafe_allow_html=True)

                else:
                    st.error("❌ AI Parsing Error: Could not generate a structured report. Please try again.")
            else:
                st.info("👋 Ready to analyze? Click the button above to start your AI-powered sustainability audit.")
else:
    st.info("👈 select first")