import streamlit as st
import os
import random
import pandas as pd
from src.ocr_engine import ocr_image, extract_total, extract_candidate_items
from src.llm_engine import get_eco_report_from_deepseek # 确保你已按上一条建议创建该文件

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
        with tab_ai:
            st.markdown("### DeepSeek Sustainability Audit")
            
            if st.button("🚀 Run DeepSeek-V3 Analysis", type="primary"):
                with st.spinner("Analyzing with DeepSeek-V3..."):
                    # 直接把 OCR 的全文本发给大模型，让它自己去重、清洗和评分
                    report = get_eco_report_from_deepseek(st.session_state.raw_data['text'])
                    st.session_state.ai_report = report

            if st.session_state.get('ai_report'):
                report = st.session_state.ai_report
                if "error" not in report:
                    # 1. 分数展示
                    c1, c2 = st.columns(2)
                    c1.metric("Eco-Score", f"{report.get('eco_score', 0)}/100")
                    c2.metric("Health Score", f"{report.get('health_score', 'N/A')}")
                    
                    # 2. 商品清单（大模型清洗后的版本）
                    st.write("**Cleaned Items:**")
                    st.write(", ".join(report.get('items', [])))

                    # 3. SDG 关联
                    st.write("**Targeted SDGs:**")
                    sdg_cols = st.columns(len(report.get('sdgs', [])) if report.get('sdgs') else 1)
                    for i, sdg in enumerate(report.get('sdgs', [])):
                        sdg_cols[i].info(sdg)

                    # 4. 专家建议
                    st.warning(f"💡 **AI Advice:** {report.get('advice', 'No advice available.')}")
                else:
                    st.error(f"AI Engine Error: {report['error']}")
            else:
                st.info("Click the button above to start DeepSeek-V3 analysis.")

else:
    st.info("👈 请在左侧选择一张收据开始分析。")