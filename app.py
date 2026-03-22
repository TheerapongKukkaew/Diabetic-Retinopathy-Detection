import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- 1. ตั้งค่าหน้าเว็บให้ดูง่าย ---
st.set_page_config(page_title="DR Classification Pipeline", layout="wide")

# --- 2. ฟังก์ชัน Ben Graham's Processing (1024px) ---
def apply_ben_graham(img, size=1024):
    # แปลง PIL เป็น OpenCV (BGR)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize เป็น 1024 ตามที่เทรนมา
    img_res = cv2.resize(img_bgr, (size, size))
    
    # สูตร Ben Graham: เพิ่มความชัดของเส้นเลือดและรอยโรค
    # Image * 4 + Blur * -4 + 128
    img_ben = cv2.addWeighted(img_res, 4, cv2.GaussianBlur(img_res, (0, 0), size / 30), -4, 128)
    return img_ben

# --- 3. โหลดโมเดล Classification ---
@st.cache_resource
def load_models():
    # ตรวจสอบชื่อไฟล์ .pt ใน GitHub ให้ตรงกันนะครับ
    m_bin = YOLO("best_Binary_26.pt")   # สำหรับแยก Normal / DR
    m_grad = YOLO("best_5Class_26.pt")  # สำหรับแยก Grade 0-4
    return m_bin, m_grad

# --- เริ่มส่วนการแสดงผล ---
st.title("👁️ DR Classification: Dual-Model Pipeline (1024px)")
st.write("กระบวนการ: Original ➡️ Ben Graham Processing ➡️ Classification Result")

try:
    bin_model, grad_model = load_models()
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    st.stop()

uploaded_file = st.file_uploader("อัปโหลดรูปภาพจอประสาทตา (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    raw_img = Image.open(uploaded_file)
    
    # แสดงรูปภาพเปรียบเทียบ 2 ขั้นตอน (Original vs Processed)
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.subheader("1. Original Image")
        st.image(raw_img, use_container_width=True)

    with st.spinner('กำลังประมวลผล Ben Graham และทำนายผล...'):
        # ทำ Preprocessing
        processed_bgr = apply_ben_graham(raw_img)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        
        with col_img2:
            st.subheader("2. Ben Graham (1024px)")
            st.image(processed_rgb, use_container_width=True)

        # สั่งทำนายผล (Inference) ด้วยขนาด 1024
        # .predict() สำหรับ Classification จะคืนค่าเป็น Probs
        res_bin = bin_model.predict(processed_bgr, imgsz=1024)[0]
        res_grad = grad_model.predict(processed_bgr, imgsz=1024)[0]

    # --- ส่วนการแสดงผลลัพธ์ความแม่นยำ (Confidence) ---
    st.divider()
    st.subheader("📊 รายงานผลการวิเคราะห์ (Classification Report)")
    
    res_col1, res_col2 = st.columns(2)
    
    # โมเดล 1: Binary (Normal vs DR)
    with res_col1:
        st.markdown("### **Step 1: Screening Model**")
        if res_bin.probs is not None:
            top1_idx = res_bin.probs.top1
            label_bin = res_bin.names[top1_idx]
            conf_bin = res_bin.probs.top1conf.item()
            
            # ตกแต่งสีตามผลลัพธ์
            if "Normal" in label_bin:
                st.success(f"**สถานะ:** {label_bin}")
            else:
                st.error(f"**สถานะ:** {label_bin}")
                
            st.metric("ความแม่นยำ (Confidence)", f"{conf_bin:.2%}")
        else:
            st.warning("โมเดลนี้ไม่ใช่ไฟล์สำหรับ Classification")

    # โมเดล 2: Grading (0, 1, 2, 3, 4)
    with res_col2:
        st.markdown("### **Step 2: Severity Grading**")
        if res_grad.probs is not None:
            top1_idx_grad = res_grad.probs.top1
            label_grad = res_grad.names[top1_idx_grad]
            conf_grad = res_grad.probs.top1conf.item()
            
            st.warning(f"**ระดับที่ตรวจพบ:** {label_grad}")
            st.metric("ความแม่นยำ (Confidence)", f"{conf_grad:.2%}")
            
            # แสดงค่า Prob ของทุก Class เป็นตารางเล็กๆ (เผื่ออาจารย์ถาม)
            with st.expander("ดูความมั่นใจในทุกระดับ (All Classes)"):
                for i, name in res_grad.names.items():
                    val = res_grad.probs.data[i].item()
                    st.write(f"{name}: {val:.4f}")
        else:
            st.info("ไม่พบข้อมูลผลลัพธ์จาก Grading Model")

    # สรุปผลสุดท้าย
    st.divider()
    if "Normal" in label_bin:
        st.balloons()
        st.success("🏁 สรุปผล: ไม่พบความผิดปกติของโรคเบาหวานขึ้นจอตา")
    else:
        st.error(f"🏁 สรุปผล: พบความผิดปกติในระดับ {label_grad}")
