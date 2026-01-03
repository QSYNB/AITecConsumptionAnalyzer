import streamlit as st
import os
import random
import pandas as pd
# 导入你之前封装好的两个核心引擎
from src.ocr_engine import ocr_image, extract_total, extract_candidate_items
from src.nlp_engine import predict_items

# --- 1. 基础配置 ---
# 确保路径指向你存放 SROIE 训练集图片的文件夹
SROIE_IMG_DIR = r"D:\15_MAI\7002\GROUP ASSIGNMENT\git\train\img" 
# 指向 M4 提供给你的模型文件夹
MODEL_PATH = r"D:\15_MAI\7002\GROUP ASSIGNMENT\git\models\item_classifier_model"

st.set_page_config(page_title="Eco-Consumer Analyzer", layout="wide")

st.title("🌱 Smart Sustainable Consumption Analyzer")
st.markdown("---")

# --- 2. 侧边栏与输入控制 ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if st.button("🎲 随机选择 SROIE 收据", type="primary"):
        if os.path.exists(SROIE_IMG_DIR):
            image_files = [f for f in os.listdir(SROIE_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random_img = random.choice(image_files)
            st.session_state.current_img = os.path.join(SROIE_IMG_DIR, random_img)
            st.session_state.run_analysis = True
        else:
            st.error("找不到图片目录，请检查路径配置。")

# --- 3. 主界面布局 ---
col_img, col_data = st.columns([1, 1.2])

if 'current_img' in st.session_state:
    with col_img:
        st.subheader("🖼️ Selected Receipt")
        st.image(st.session_state.current_img, use_container_width=True)
        
    with col_data:
        st.subheader("🔍 Analysis Results")
        
        if st.session_state.get('run_analysis'):
            with st.spinner('正在执行 OCR 与 NLP 深度分析...'):
                # 第一步：OCR 提取文字和总额
                full_text, lines = ocr_image(st.session_state.current_img)
                total_price = extract_total(lines)
                
                # 第二步：筛选商品候选行
                candidate_lines = extract_candidate_items(lines)
                
                # 第三步：调用 NLP 模型进行分类
                # 此处调用 M4 训练的 DistilBERT 模型
                classification_results = predict_items(candidate_lines, model_path=MODEL_PATH)
                
                # --- 展示核心指标 ---
                st.metric(label="Total Amount", value=f"RM {total_price}")
                
                # --- 展示商品分类详情 ---
                



                st.write("### 🛒 Item Classification")
                if classification_results:
                    # 转化为 DataFrame 并在 UI 展示
                    df_results = pd.DataFrame(classification_results)
                    
                    
                    # 定义类别颜色映射
                    color_map = {
                        "fresh_food": "green",
                        "sugary_drink": "red",
                        "processed_food": "orange",
                        "single_use_plastic": "gray"
                    }
                    
                    # 美化表格显示
                    st.dataframe(
                        df_results[['item', 'category', 'confidence']],
                        column_config={
                            "item": "Product Name",
                            "category": st.column_config.SelectboxColumn("Category", options=color_map.keys()),
                            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.warning("未能识别出具体的商品行。")

                # --- 原始数据折叠栏 ---
                with st.expander("📄 查看 OCR 原始文本"):
                    st.text(full_text)

# --- 4. 页脚提示 ---
st.markdown("---")
st.caption("Powered by Tesseract OCR & DistilBERT | Team Lead Integrated Version")