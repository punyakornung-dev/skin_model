import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 1. ตั้งค่าหน้าเว็บสไตล์คลินิกยุคใหม่ (Modern UI) ---
st.set_page_config(page_title="Derm-AI Specialist", page_icon="🩺", layout="wide")

# Custom CSS เพื่อความสวยงาม
st.markdown("""
    <style>
    /* สีพื้นหลังรวม */
    .stApp { background-color: #f8fafc; }
    
    /* ตกแต่งปุ่มหลัก */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.2em; 
        background: linear-gradient(135deg, #2563eb, #1d4ed8); 
        color: white; 
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        background: linear-gradient(135deg, #1d4ed8, #1e40af); 
    }
    
    /* ตกแต่งกล่องข้อความ */
    .css-1n76uvr { border-radius: 10px; }
    
    /* ปรับขนาดฟอนต์ของ Metric (ผลลัพธ์อันดับ 1) */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #059669;
        font-weight: 800;
    }
    </style>
    """, unsafe_allow_html=True)

# ส่วนหัวของเว็บไซต์
col_header1, col_header2 = st.columns([1, 11])
with col_header1:
    st.image("https://cdn-icons-png.flaticon.com/512/2867/2867295.png", width=60) # ไอคอนแพทย์
with col_header2:
    st.title("ระบบสนับสนุนการวินิจฉัยโรคผิวหนัง (Derm-AI)")
    st.caption("The Next Decade Hackathon 2026 - AI for Healthcare Edition ✨")

st.markdown("---")

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
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063116.png", width=100)
    st.header("📋 ข้อมูลผู้ป่วย")
    patient_name = st.text_input("ชื่อ-นามสกุล", "คนไข้ทั่วไป")
    age = st.number_input("อายุ", min_value=0, max_value=120, value=25)
    gender = st.selectbox("เพศ", ["ชาย", "หญิง", "อื่นๆ"])
    location = st.text_input("ตำแหน่งที่พบรอยโรค", "เช่น แขนซ้าย, หน้าอก")
    
    st.divider()
    
    st.warning("📌 **ข้อควรระวังทางการแพทย์**\n\nระบบนี้เป็นเพียง AI ช่วยสนับสนุนการตัดสินใจเบื้องต้น ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ผู้เชี่ยวชาญได้")
    st.info("💡 **คำแนะนำการถ่ายภาพ:**\n- ถ่ายในที่มีแสงสว่างเพียงพอ\n- โฟกัสรอยโรคให้ชัดเจนที่สุด")

# --- 4. ส่วนอัปโหลดและวิเคราะห์ (Main Area) ---
col1, col_spacer, col2 = st.columns([1.2, 0.1, 1.2])

with col1:
    st.subheader("📸 1. อัปโหลดรูปภาพรอยโรค")
    uploaded_file = st.file_uploader("ลากไฟล์มาวาง หรือคลิกเพื่อเลือกไฟล์ (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        # ตกแต่งกรอบรูปภาพนิดหน่อย
        st.markdown("<div style='border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        st.image(img, caption="รูปภาพที่ต้องการวิเคราะห์", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("🔬 2. ผลการวิเคราะห์จาก AI")
    
    if not uploaded_file:
        st.info("👈 กรุณาอัปโหลดรูปภาพทางด้านซ้ายเพื่อเริ่มต้นการวิเคราะห์")
    
    if uploaded_file and st.button("🚀 เริ่มการประมวลผลเชิงลึก"):
        with st.spinner("🧠 AI กำลังประมวลผลและเทียบเคียงฐานข้อมูล..."):
            # เตรียมรูป
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # ทายผล
            preds = model.predict(img_array)[0]
            
            # ดึง Top 3
            top_3_idx = np.argsort(preds)[-3:][::-1]
            best_idx = top_3_idx[0]
            confidence = preds[best_idx] * 100
            disease_name = disease_map_th[best_idx]
            
            # --- ส่วนแสดงผลแบบใหม่ ---
            st.success("ประมวลผลเสร็จสิ้น!")
            
            # โชว์อันดับ 1 แบบเด่นๆ
            st.metric(label="🩺 โรคที่มีความเป็นไปได้สูงสุด", value=disease_name, delta=f"ความมั่นใจ {confidence:.1f}%", delta_color="normal")
            
            st.markdown("---")
            st.markdown("📊 **ระดับความมั่นใจ 3 อันดับแรก**")
            
            # ใช้ Progress Bar แทน Bar Chart เพื่อความสวยงามและอ่านง่าย
            for idx in top_3_idx:
                d_name = disease_map_th[idx]
                conf_score = preds[idx] * 100
                st.write(f"**{d_name}** ({conf_score:.1f}%)")
                st.progress(int(conf_score))
            
            # สรุปทางการแพทย์
            st.markdown("---")
            st.subheader(f"📝 รายงานสรุปสำหรับคุณ {patient_name}")
            
            if best_idx in [1, 6, 7, 9]:
                st.error(f"⚠️ **ความเสี่ยงระดับสูง:** รอยโรคจัดอยู่ในกลุ่มเสี่ยง ({disease_name})\n\n**คำแนะนำ:** ควรรีบนำผลการประเมินนี้ไปปรึกษาแพทย์ผิวหนังเฉพาะทางทันทีเพื่อทำการตรวจชิ้นเนื้อ (Biopsy) หรือวินิจฉัยเพิ่มเติม")
            else:
                st.success(f"✅ **ความเสี่ยงระดับต่ำ:** รอยโรคจัดอยู่ในกลุ่มเสี่ยงต่ำ ({disease_name})\n\n**คำแนะนำ:** แนะนำให้สังเกตอาการและขนาดของรอยโรคอย่างต่อเนื่อง หากมีการเปลี่ยนแปลงรูปร่าง สี หรือขนาด ควรพบแพทย์")

            st.caption(f"🆔 รหัสอ้างอิง: REQ-{np.random.randint(1000,9999)} | 📅 วันที่วิเคราะห์: 2026-04-28")
