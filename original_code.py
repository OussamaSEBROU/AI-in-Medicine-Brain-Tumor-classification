import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import time

# استيراد tensorflow بطريقة تجنبنا التحميل الكامل للمكتبة الضخمة
import tensorflow as tf

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# 2. القاموس اللغوي
# ---------------------------------------------------------
translations = {
    "ar": {
        "dir": "rtl",
        "title": "نظام NeuroScan للتشخيص",
        "subtitle": "تحليل صور الرنين المغناطيسي للكشف عن الأورام",
        "mode_select": "طريقة الإدخال",
        "mode_camera": "📸 الكاميرا المباشرة",
        "mode_upload": "📂 رفع ملف",
        "result_header": "تقرير الفحص الآلي",
        "pos_result": "⚠️ اشتباه بوجود كتلة",
        "neg_result": "✅ النتيجة سليمة",
        "advice_title": "الخطوات الموصى بها:",
        "footer": "Developed by Oussama SEBROU"
    },
    "en": {
        "dir": "ltr",
        "title": "NeuroScan AI System",
        "subtitle": "AI-Powered Brain Tumor Detection",
        "mode_select": "Input Method",
        "mode_camera": "📸 Live Camera",
        "mode_upload": "📂 Upload File",
        "result_header": "Analysis Report",
        "pos_result": "⚠️ Potential Abnormality",
        "neg_result": "✅ Scan is Normal",
        "advice_title": "Recommended Steps:",
        "footer": "Developed by Oussama SEBROU"
    }
}

# ---------------------------------------------------------
# 3. معالجة الصور والنموذج
# ---------------------------------------------------------
@st.cache_resource
def load_tm_model():
    # تحميل الموديل مع تعطيل التجميع لحل مشكلة 'groups'
    return tf.keras.models.load_model('keras_model.h5', compile=False)

def predict(img, model):
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    normalized_image = (img_array / 127.5) - 1.0
    data = np.expand_dims(normalized_image, axis=0)
    return model.predict(data)

# ---------------------------------------------------------
# 4. واجهة المستخدم
# ---------------------------------------------------------
with st.sidebar:
    lang = st.selectbox("Language", ["العربية", "English"])
    t = translations["ar" if lang == "العربية" else "en"]
    mode = st.radio(t['mode_select'], [t['mode_camera'], t['mode_upload']])

st.markdown(f"<h1 style='text-align:center;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'>{t['subtitle']}</p>", unsafe_allow_html=True)

img_file = st.camera_input("Scan") if mode == t['mode_camera'] else st.file_uploader("Upload")

if img_file:
    img = Image.open(img_file).convert('RGB')
    st.image(img, width=300)
    
    if st.button("Analyze / تحليل"):
        with st.spinner("Processing..."):
            model = load_tm_model()
            res = predict(img, model)
            idx = np.argmax(res)
            
            st.markdown("---")
            if idx == 1:
                st.error(t['pos_result'])
            else:
                st.success(t['neg_result'])
            st.write(f"Confidence: {res[0][idx]*100:.1f}%")

st.markdown(f"<div style='text-align:center; margin-top:50px;'>{t['footer']}</div>", unsafe_allow_html=True)

