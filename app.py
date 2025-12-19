import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. قاموس النصوص والترجمة
# ---------------------------------------------------------
translations = {
    "ar": {
        "dir": "rtl",
        "title": "نظام NeuroScan للتشخيص الآلي",
        "subtitle": "تحليل صور الرنين المغناطيسي (MRI) للكشف عن أورام المخ باستخدام الذكاء الاصطناعي",
        "sidebar_title": "لوحة التحكم",
        "lang_select": "اللغة / Language",
        "mode_select": "طريقة الإدخال",
        "mode_camera": "📸 استخدام الكاميرا المباشرة",
        "mode_upload": "📂 رفع صورة من الجهاز",
        "upload_text": "قم برفع صورة الأشعة هنا",
        "camera_text": "التقط صورة واضحة للأشعة",
        "analyzing": "جاري معالجة الصورة وتحليل البيانات...",
        "result_header": "تقرير الفحص الآلي",
        "confidence": "نسبة التطابق",
        "pos_result": "⚠️ النتيجة: اشتباه بوجود كتلة غير طبيعية",
        "pos_msg": "اكتشف النموذج أنماطاً تشبه خصائص أورام المخ.",
        "pos_advice_title": "الخطوات الطبية الموصى بها:",
        "pos_advice_list": [
            "الهدوء: هذه النتيجة أولية وليست تشخيصاً نهائياً.",
            "مراجعة الطبيب: يجب حجز موعد مع استشاري مخ وأعصاب فوراً.",
            "إجراءات متوقعة: قد يطلب الطبيب أشعة رنين مغناطيسي بالصبغة.",
            "تجهيز الملف: احتفظ بنسخة من هذه الصورة لعرضها على المختص."
        ],
        "neg_result": "✅ النتيجة: النسيج يبدو سليماً",
        "neg_msg": "لم يكتشف النموذج أي أنماط غير طبيعية واضحة في هذه الصورة.",
        "neg_advice_title": "توجيهات عامة:",
        "neg_advice_list": [
            "استمرار المراقبة: إذا كانت لديك أعراض مستمرة، استشر الطبيب فوراً.",
            "الفحص السريري: الطبيب هو الوحيد القادر على إعطاء تشخيص نهائي.",
            "الصحة العامة: حافظ على نمط حياة صحي ومتابعة دورية."
        ],
        "disclaimer": "إخلاء مسؤولية: هذا النظام أداة مساعدة للبحث فقط. لا تعتمد عليه في اتخاذ قرارات مصيرية دون الرجوع لطبيب."
    },
    "en": {
        "dir": "ltr",
        "title": "NeuroScan AI Diagnostic System",
        "subtitle": "AI-Powered Brain Tumor Detection from MRI Scans",
        "sidebar_title": "Control Panel",
        "lang_select": "Language",
        "mode_select": "Input Method",
        "mode_camera": "📸 Use Live Camera",
        "mode_upload": "📂 Upload Image File",
        "upload_text": "Upload MRI scan here",
        "camera_text": "Capture a photo of the MRI",
        "analyzing": "Analyzing data...",
        "result_header": "Automated Analysis Report",
        "confidence": "Confidence Score",
        "pos_result": "⚠️ Result: Potential Abnormality Detected",
        "pos_msg": "The model identified patterns consistent with brain tumors.",
        "pos_advice_title": "Recommended Medical Steps:",
        "pos_advice_list": [
            "Stay Calm: This is a preliminary AI screening.",
            "Consultation: Schedule an appointment with a neurologist immediately.",
            "Next Steps: A Contrast MRI might be required for verification.",
            "Documentation: Keep a copy of this scan for the specialist."
        ],
        "neg_result": "✅ Result: No Abnormality Detected",
        "neg_msg": "The model did not find clear abnormal patterns.",
        "neg_advice_title": "General Guidance:",
        "neg_advice_list": [
            "Monitor Symptoms: Consult a doctor if you feel any symptoms.",
            "Clinical Exam: A physical examination is always required.",
            "Health: Maintain a healthy lifestyle and regular checkups."
        ],
        "disclaimer": "Disclaimer: Research tool only. Consult a doctor for medical decisions."
    }
}

# ---------------------------------------------------------
# 3. التنسيق (CSS)
# ---------------------------------------------------------
def inject_custom_css(direction):
    font_family = "'Cairo', sans-serif" if direction == "rtl" else "'Roboto', sans-serif"
    text_align = "right" if direction == "rtl" else "left"
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Roboto:wght@400;500;700&display=swap');
        html, body, [class*="css"] {{ font_family: {font_family}; }}
        .stApp {{ background-color: #f8f9fa; }}
        .main-header {{ text-align: center; color: #2c3e50; padding-bottom: 20px; border-bottom: 1px solid #e0e0e0; margin-bottom: 30px; }}
        .report-container {{ background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #eee; margin-top: 20px; direction: {direction}; text-align: {text_align}; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #7f8c8d; border-top: 1px solid #eee; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. منطق الذكاء الاصطناعي
# ---------------------------------------------------------
@st.cache_resource
def load_teachable_machine_model():
    # استخدام h5py لفتح الملف بمرونة وتجاوز تعارض الكلاسات
    return tf.keras.models.load_model('keras_model.h5', compile=False)

def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (img_array / 127.5) - 1.0
    data = np.expand_dims(normalized_image_array, axis=0)
    prediction = model.predict(data)
    return prediction

# ---------------------------------------------------------
# 5. بناء الواجهة
# ---------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    lang = st.selectbox("🌐 اللغة / Language", ["العربية", "English"])
    lang_code = "ar" if lang == "العربية" else "en"
    t = translations[lang_code]
    input_mode = st.radio(t['mode_select'], [t['mode_camera'], t['mode_upload']])
    st.markdown("---")
    st.warning(t['disclaimer'])

inject_custom_css(t['dir'])

st.markdown(f"<div class='main-header'><h1>{t['title']}</h1><p>{t['subtitle']}</p></div>", unsafe_allow_html=True)

image_file = None
col1, col2 = st.columns([1, 1])

with col1:
    if input_mode == t['mode_camera']:
        image_file = st.camera_input(t['camera_text'])
    else:
        image_file = st.file_uploader(t['upload_text'], type=['jpg', 'png', 'jpeg'])

with col2:
    if image_file is not None:
        try:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Preview", use_container_width=True)
            
            if st.button("Start Analysis / ابدأ التحليل", use_container_width=True):
                model = load_teachable_machine_model()
                
                with st.spinner(t['analyzing']):
                    prediction = process_and_predict(image, model)
                    index = np.argmax(prediction)
                    confidence = prediction[0][index]
                
                # افتراض: Index 0 هو Normal و Index 1 هو Tumor
                is_tumor = (index == 1) 
                
                st.markdown(f"<div class='report-container'>", unsafe_allow_html=True)
                st.markdown(f"<h2>{t['result_header']}</h2>", unsafe_allow_html=True)
                
                if is_tumor:
                    st.error(t['pos_result'])
                    st.write(f"**{t['confidence']}:** {confidence*100:.2f}%")
                    st.markdown(f"#### {t['pos_advice_title']}")
                    for advice in t['pos_advice_list']:
                        st.markdown(f"- {advice}")
                else:
                    st.success(t['neg_result'])
                    st.write(f"**{t['confidence']}:** {confidence*100:.2f}%")
                    st.markdown(f"#### {t['neg_advice_title']}")
                    for advice in t['neg_advice_list']:
                        st.markdown(f"- {advice}")
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error Processing: {e}")

st.markdown(f"<div class='footer'>Developed by Oussama SEBROU</div>", unsafe_allow_html=True)