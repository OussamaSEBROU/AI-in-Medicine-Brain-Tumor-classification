import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI - Diagnostic Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. قاموس النصوص والترجمة (موسع بدقة طبية)
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
        "upload_text": "قم بسحب وإفلات صورة الأشعة هنا",
        "camera_text": "التقط صورة واضحة للأشعة",
        "analyzing": "جاري معالجة الصورة وتحليل البيانات...",
        "result_header": "تقرير الفحص الآلي",
        "confidence": "نسبة التطابق مع الأنماط المتعلمة",
        # نتائج الورم
        "pos_result": "⚠️ النتيجة: اشتباه بوجود كتلة غير طبيعية",
        "pos_msg": "اكتشف النموذج أنماطاً تشبه خصائص أورام المخ.",
        "pos_advice_title": "الخطوات الطبية الموصى بها:",
        "pos_advice_list": [
            "الهدوء: هذه النتيجة أولية من ذكاء اصطناعي وليست تشخيصاً نهائياً.",
            "مراجعة الطبيب: يجب حجز موعد مع استشاري مخ وأعصاب في أقرب وقت.",
            "إجراءات متوقعة: قد يطلب الطبيب أشعة بالصبغة (Contrast MRI) للتأكد.",
            "تجهيز الملف: احتفظ بنسخة من هذه الصورة والتقرير لعرضه على المختص."
        ],
        # نتائج السليم
        "neg_result": "✅ النتيجة: النسيج يبدو سليماً",
        "neg_msg": "لم يكتشف النموذج أي أنماط غير طبيعية واضحة في هذه الصورة.",
        "neg_advice_title": "توجيهات عامة:",
        "neg_advice_list": [
            "استمرار المراقبة: إذا كانت لديك أعراض (صداع، زغللة، تشنجات) فلا تعتمد على هذه النتيجة فقط.",
            "الاستشارة: الفحص السريري عند الطبيب هو الفيصل دائماً.",
            "نمط الحياة: حافظ على نوم منتظم وتجنب الإجهاد الذهني المفرط."
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
        "upload_text": "Drag and drop MRI scan here",
        "camera_text": "Capture a clear photo of the MRI",
        "analyzing": "Processing image and analyzing data...",
        "result_header": "Automated Analysis Report",
        "confidence": "Pattern Match Confidence",
        # Tumor Results
        "pos_result": "⚠️ Result: Potential Abnormality Detected",
        "pos_msg": "The model identified patterns consistent with brain tumors.",
        "pos_advice_title": "Recommended Medical Steps:",
        "pos_advice_list": [
            "Stay Calm: This is a preliminary AI screening, not a final diagnosis.",
            "Consultation: Schedule an appointment with a neurologist immediately.",
            "Next Steps: The doctor may request a Contrast MRI for verification.",
            "Documentation: Keep a copy of this scan and report for the specialist."
        ],
        # Normal Results
        "neg_result": "✅ Result: No Abnormality Detected",
        "neg_msg": "The model did not find clear abnormal patterns in this image.",
        "neg_advice_title": "General Guidance:",
        "neg_advice_list": [
            "Monitor Symptoms: If you have symptoms (headache, vision blur, seizures), do not rely solely on this result.",
            "Clinical Exam: A physical examination by a doctor is always required.",
            "Lifestyle: Maintain regular sleep and avoid excessive mental stress."
        ],
        "disclaimer": "Disclaimer: This system is a research tool. Do NOT make medical decisions without consulting a doctor."
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
        
        html, body, [class*="css"] {{
            font_family: {font_family};
        }}
        
        .stApp {{
            background-color: #f8f9fa;
        }}
        
        .main-header {{
            text-align: center;
            color: #2c3e50;
            padding-bottom: 20px;
            border-bottom: 1px solid #e0e0e0;
            margin-bottom: 30px;
        }}
        
        .report-container {{
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid #eee;
            margin-top: 20px;
            direction: {direction};
            text-align: {text_align};
        }}
        
        .advice-box {{
            background-color: #f0f7ff;
            border-right: 5px solid #007bff;
            padding: 15px;
            margin-top: 15px;
            border-radius: 4px;
        }}
        
        /* ضبط اتجاه القوائم */
        ul {{
            direction: {direction};
            text-align: {text_align};
        }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. منطق الذكاء الاصطناعي
# ---------------------------------------------------------
@st.cache_resource
def load_teachable_machine_model():
    # تأكد من وجود ملف keras_model.h5 في نفس المجلد
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    return model

def process_and_predict(image_data, model):
    # تحضير الصورة كما يتطلب Teachable Machine
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction

# ---------------------------------------------------------
# 5. بناء الواجهة الرئيسية
# ---------------------------------------------------------

# الشريط الجانبي للإعدادات
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=70)
    lang = st.selectbox("🌐 اللغة / Language", ["العربية", "English"])
    lang_code = "ar" if lang == "العربية" else "en"
    t = translations[lang_code]
    
    st.header(t['sidebar_title'])
    
    # اختيار وضع الإدخال (كاميرا أو رفع)
    input_mode = st.radio(t['mode_select'], [t['mode_camera'], t['mode_upload']])
    
    st.markdown("---")
    st.info(t['disclaimer'])

# حقن التصميم
inject_custom_css(t['dir'])

# العنوان الرئيسي
st.markdown(f"<div class='main-header'><h1>{t['title']}</h1><p style='color:#7f8c8d;'>{t['subtitle']}</p></div>", unsafe_allow_html=True)

# منطق إدخال الصورة
image_file = None

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### " + t['mode_select'])
    if input_mode == t['mode_camera']:
        image_file = st.camera_input(t['camera_text'])
    else:
        image_file = st.file_uploader(t['upload_text'], type=['jpg', 'png', 'jpeg'])

with col2:
    if image_file is not None:
        try:
            image = Image.open(image_file)
            st.image(image, caption="Scan Preview", width=300)
            
            # زر التحليل (يظهر فقط عند وجود صورة)
            if st.button("Start Analysis / ابدأ التحليل", use_container_width=True):
                model = load_teachable_machine_model()
                
                # شريط التقدم
                progress_text = t['analyzing']
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.015)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                my_bar.empty()
                
                # التوقع
                prediction = process_and_predict(image, model)
                index = np.argmax(prediction)
                confidence = prediction[0][index]
                
                # افتراض: Index 0 = سليم، Index 1 = ورم
                # يرجى تعديل الشرط أدناه إذا كان ترتيب نموذجك مختلفاً
                is_tumor = (index == 1) 
                
                # عرض التقرير الطبي
                st.markdown(f"<div class='report-container'>", unsafe_allow_html=True)
                st.markdown(f"<h2>{t['result_header']}</h2>", unsafe_allow_html=True)
                st.markdown("---")
                
                if is_tumor:
                    st.error(t['pos_result'])
                    st.write(f"**{t['confidence']}:** {confidence*100:.2f}%")
                    st.write(t['pos_msg'])
                    
                    st.markdown(f"#### {t['pos_advice_title']}")
                    for advice in t['pos_advice_list']:
                        st.markdown(f"- {advice}")
                else:
                    st.success(t['neg_result'])
                    st.write(f"**{t['confidence']}:** {confidence*100:.2f}%")
                    st.write(t['neg_msg'])
                    
                    st.markdown(f"#### {t['neg_advice_title']}")
                    for advice in t['neg_advice_list']:
                        st.markdown(f"- {advice}")
                        
                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error("Error Processing Image.")
            st.error(f"Details: {e}")
            
    else:
        # رسالة انتظار صورة
        st.info(t['camera_text'] if input_mode == t['mode_camera'] else t['upload_text'])


