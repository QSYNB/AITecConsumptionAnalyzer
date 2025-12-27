import streamlit as st
from PIL import Image

# 页面配置
st.set_page_config(page_title="Smart Sustainable Analyzer", layout="wide")

# 自定义一些样式，让按钮和容器更好看
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .result-container { padding: 20px; border: 1px solid #ccc; border-radius: 10px; background-color: #fafafa; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Smart Sustainable Consumption Analyzer")
st.markdown("---")

# 创建主要布局：三栏 — 左：上传（40%），中：设置（20%），右：结果（40%）
col_upload, col_settings, col_result = st.columns([2, 1, 3])

# --- 第一部分：上传图片区域 ---
with col_upload:
    st.subheader("1️⃣ Upload Receipt")
    uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 显示缩略图
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Preview", use_container_width=True)

        # 如果是新上传的文件，重置触发状态，等待用户点击 Generate
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != getattr(uploaded_file, 'name', None):
            st.session_state.last_uploaded = getattr(uploaded_file, 'name', None)
            st.session_state.trigger_analysis = False

    else:
        st.info("Please upload an image to unlock analysis settings.")

# --- 第二部分：按钮与配置（放在中间列） ---
with col_settings:
    st.subheader("2️⃣ Analysis Settings")

    settings_disabled = uploaded_file is None

    do_eco = st.checkbox("Environmental Impact Analysis (SDG 12)", value=True, disabled=settings_disabled)
    do_health = st.checkbox("Health & Nutrition Analysis", value=False, disabled=settings_disabled)
    do_spending = st.checkbox("Spending Insights", value=True, disabled=settings_disabled)

    # 大按钮：生成（上传前禁用）
    if st.button("🚀 Generate Insights", disabled=settings_disabled):
        st.session_state.trigger_analysis = True

# --- 第三部分：结果生成区域 ---
with col_result:
    st.subheader("3️⃣ Analysis Report")
    
    # 初始状态：等待上传
    if 'trigger_analysis' not in st.session_state or not st.session_state.trigger_analysis:
        st.info("Results will appear here once you click 'Generate'.")
        # 这里可以放一个精美的占位图
        st.image("https://via.placeholder.com/600x400.png?text=Waiting+for+Data+Processing...", use_container_width=True)
    
    else:
        # 当点击生成后的展示逻辑
        with st.container(border=True):
            st.success("Analysis Complete!")
            

            # 模拟三个区域的结果
            if do_eco:
                with st.expander("🍀 Environmental Impact", expanded=True):
                    st.metric(label="Eco Score", value="82/100", delta="Excellent")
                    st.write("- Found 3 organic items.\n- Plastic packaging detected in 2 items.")
            
            if do_health:
                with st.expander("🍎 Health Analysis", expanded=True):
                    st.write("- High sugar content detected in: 'Coca Cola'.\n- Good protein source: 'Chicken Breast'.")
            
            if do_spending:
                with st.expander("💰 Spending Insights", expanded=False):
                    st.bar_chart({"Category": ["Food", "Household", "Other"], "Spend": [45, 12, 5]})

# 重置按钮逻辑 (可选)
if st.sidebar.button("Reset All"):
    st.session_state.trigger_analysis = False
    st.rerun()