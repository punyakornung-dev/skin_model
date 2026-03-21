import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 1. ตั้งค่าหน้าเว็บสไตล์คลินิก ---
st.set_page_config(page_title="Derm-AI Specialist", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 ระบบสนับสนุนการวินิจฉัยโรคผิวหนัง (Derm-AI)")
st.caption("The Next Decade Hackathon 2026 - AI for Healthcare Edition")

# --- 2. โหลดโมเดล (รองรับชื่อไฟล์ V2) ---
@st.cache_resource
def load_skin_model():
    # ลองโหลด V2 ก่อน ถ้าไม่มีให้โหลด V1
    try:
        return load_model('super_skin_model_v2.keras')
    except:
        return load_model('super_skin_model.keras')

model = load_skin_model()

# แผนที่ชื่อโรค
disease_map_th = {
    0: 'AKIEC (แสงแดดแผดเผา)', 1: 'BCC (มะเร็งเบซัล)', 2: 'BEN_OTH (ทั่วไป)',
    3: 'BKL (ปาน/กระ)', 4: 'DF (เนื้องอกเส้นใย)', 5: 'INF (ติดเชื้อ)',
    6: 'MAL_OTH (มะเร็งอื่นๆ)', 7: 'MEL (มะเร็งไฝ)', 8: 'NV (ไฝปกติ)',
    9: 'SCCKA (มะเร็งสความัส)', 10: 'VASC (หลอดเลือด)'
}

# --- 3. ส่วนรับข้อมูลผู้ป่วย (Side Bar) ---
with st.sidebar:
    st.header("📋 ข้อมูลผู้ป่วย")
    patient_name = st.text_input("ชื่อ-นามสกุล", "คนไข้ทั่วไป")
    age = st.number_input("อายุ", min_value=0, max_value=120, value=25)
    gender = st.selectbox("เพศ", ["ชาย", "หญิง", "อื่นๆ"])
    location = st.text_input("ตำแหน่งที่พบรอยโรค", "เช่น แขนซ้าย, หน้าอก")
    st.divider()
    st.info("คำแนะนำ: รูปถ่ายควรมีแสงสว่างเพียงพอและเห็นรอยโรคชัดเจน")

# --- 4. ส่วนอัปโหลดและวิเคราะห์ (Main Area) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="รูปภาพต้นฉบับ", use_container_width=True)

with col2:
    st.subheader("🔬 ผลการวิเคราะห์จาก AI")
    if uploaded_file and st.button("🚀 เริ่มการวิเคราะห์เชิงลึก"):
        with st.spinner("AI กำลังประมวลผลข้อมูล..."):
            # เตรียมรูป
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # ทายผล
            preds = model.predict(img_array)[0]
            
            # ดึง Top 3
            top_3_idx = np.argsort(preds)[-3:][::-1]
            
            # โชว์อันดับ 1 แบบเด่นๆ
            best_idx = top_3_idx[0]
            confidence = preds[best_idx] * 100
            disease_name = disease_map_th[best_idx]
            
            st.metric(label="โรคที่มีความเป็นไปได้สูงสุด", value=disease_name)
            
            # โชว์กราฟความน่าจะเป็น
            st.write("📊 **ระดับความมั่นใจ 3 อันดับแรก:**")
            chart_data = pd.DataFrame({
                'โรค': [disease_map_th[i] for i in top_3_idx],
                'ความมั่นใจ (%)': [preds[i]*100 for i in top_3_idx]
            })
            st.bar_chart(chart_data.set_index('โรค'))
            
            # สรุปทางการแพทย์
            st.markdown("---")
            st.markdown(f"**📝 รายงานสรุปสำหรับคุณ {patient_name}:**")
            if best_idx in [1, 6, 7, 9]:
                st.error(f"⚠️ ตรวจพบรอยโรคกลุ่มเสี่ยงสูง ({disease_name}) ควรส่งพบแพทย์เฉพาะทางทันที")
            else:
                st.success(f"✅ ตรวจพบรอยโรคกลุ่มเสี่ยงต่ำ ({disease_name}) แนะนำให้ติดตามอาการ")

            st.caption(f"ID ผู้ป่วย: {np.random.randint(1000,9999)} | วิเคราะห์เมื่อ: 2026-04-28")