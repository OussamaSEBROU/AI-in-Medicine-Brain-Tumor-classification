import streamlit as st
import tensorflow. keras
from PIL import Image, ImageOps
import numpy as np
import io
import os

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)

# --- Internationalization (i18n) Messages ---
MESSAGES = {
    "en": {
        "app_title": "AI NeuroScan",
        "intro_text": "Welcome to AI NeuroScan Our advanced neural network provides rapid, automated screening of brain MRI scans Designed to support clinical workflows, this tool identifies struc[...]",
        "subtitle": "Leveraging advanced AI for early, non-invasive screening of brain MRI scans.  Precision in every pixel.",
        "sidebar_title": "Settings & Information",
        "language_label": "Select Language",
        "input_mode_label": "Select Input Method",
        "mode_upload":  "Upload Image",
        "mode_camera":  "Live Camera",
        "upload_help": "Upload a brain MRI image (JPG, PNG, JPEG)",
        "camera_button": "Capture Image",
        "processing":  "Processing image...",
        "result_header": "Analysis Result",
        "result_yes_title": "Anomaly Detected (Tumor Found)",
        "result_yes_text": "I want to speak with you with complete transparency and empathy. The AI analysis has detected an abnormal growth in your brain scan that requires immediate medical atte[...]",
        "result_no_title": "Scan Clear (No Tumor Found)",
        "result_no_text": "I'm pleased to share positive news with you. The AI analysis of your brain MRI scan shows no evidence of tumor or significant abnormal growth.",
        "invalid_image_msg": "⚠️ Invalid Image - Not a Brain MRI Scan",
        "invalid_image_details": "The uploaded/captured image is not clear or is not a brain MRI scan. Please ensure you upload a high-quality MRI brain scan image for accurate analysis",
        "developer_credit": "Developed by Erinov AI Club",
        "about_title": "About technology",
        "about_text":  "This app uses a Deep Learning model (Convolutional Neural Network) trained on thousands of MRI images",
        "how_to_use_title": "How to Use",
        "how_to_use_text": "1. Select 'Upload' or 'Camera'.\n2. Provide a clear MRI image.\n3. Wait for the AI analysis",
        "references_title": "Disclaimer",
        "references_text":  "This is a prototype for educational and screening purposes. It is NOT a definitive medical diagnosis",
        "developers_title": "Developers Team",
        "supervisor":  "Under the supervision of *M. Oussama SEBROU*"
    },
    "ar":  {
        "app_title":  "AI NeuroScan",
        "intro_text": "مرحباً بكم، توفر منصتنا فحصاً ذكياً وفورياً لصور الرنين المغناطيسي للدماغ باستخدام تقنيات الت[...]",
        "subtitle":  "تسخير الذكاء الاصطناعي المتقدم للكشف المبكر وغير الجراحي عن أورام الدماغ",
        "sidebar_title": "الإعدادات والمعلومات",
        "language_label": "اختر اللغة",
        "input_mode_label": "اختر طريقة الإدخال",
        "mode_upload": "تحميل صورة",
        "mode_camera": "الكاميرا المباشرة",
        "upload_help": "قم بتحميل صورة رنين مغناطيسي (MRI) للدماغ",
        "camera_button": "التقاط الصورة",
        "processing": "جاري معالجة الصورة.. .",
        "result_header":  "نتيجة التحليل",
        "result_yes_title": "تم الكشف عن شذوذ (وجود ورم)",
        "result_yes_text": "أود أن أتحدث معك بشفافية كاملة وتعاطف. أظهر التحليل بواسطة الذكاء الاصطناعي وجود نمو غير طبي[...]",
        "result_no_title": "المسح سليم (لا يوجد ورم)",
        "result_no_text":  "يسعدني مشاركة أخبار إيجابية معك. أظهر تحليل الذكاء الاصطناعي لفحص الرنين المغناطيسي للدماغ [...]",
        "invalid_image_msg": "⚠️ صورة غير صالحة - ليست صورة رنين مغناطيسي",
        "invalid_image_details": "الصورة التي تم رفعها أو التقاطها ليست واضحة أو ليست صورة رنين مغناطيسي للدماغ (MRI Brain Scan). ي[...]",
        "developer_credit": "Developed by Erinov AI Club",
        "about_title":  "حول التقنية المستخدمة",
        "about_text": "يعتمد التطبيق على خوارزميات التعلم العميق (Deep Learning) وبالتحديد الشبكات العصبية التلافيفية (CNN)",
        "how_to_use_title": "كيفية الاستخدام",
        "how_to_use_text": "1. اختر 'تحميل صورة' أو 'الكاميرا'.\n2. ارفع صورة MRI واضحة.\n3. انتظر معالجة الذكاء الاصطناعي",
        "references_title": "تنبيه هام",
        "references_text":  "هذا التطبيق هو نموذج أولي للأغراض التعليمية فقط، ولا يعتبر تشخيصاً طبياً نهائياً",
        "developers_title": "فريق التطوير",
        "supervisor": "تحت إشراف *M. Oussama SEBROU*"
    }
}

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp {
    background-color: #ffffff;
    color: #212529;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Primary H1 title color kept blue for contrast */
h1 {
    color: #007bff;
    text-align: center;
    font-weight: 800;
    font-size: 3em;
    margin-bottom: 10px;
}
. intro-box {
    background:  linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 30px;
    border-radius:  15px;
    border-right: 8px solid #007bff;
    border-left: 8px solid #007bff;
    margin-bottom:  30px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    line-height: 1.6;
    color: #212529;
}
. footer {
    position: fixed;
    left: 0;
    bottom:  0;
    width: 100%;
    background-color:  #f8f9fa;
    color: #6c757d;
    text-align: center;
    padding: 8px;
    font-size:  0.85em;
    border-top: 1px solid #e9ecef;
    z-index: 1000;
}
. rtl-text {
    direction: rtl;
    text-align: right;
    color: #212529;
}

/* File uploader - Browse files button in red */
[data-testid="stFileUploader"] label {
    color: #ff0000 !important;
}

/* result-specific colors (kept, but global color will apply to most text) */
.result-yes { color: #dc3545; }
.result-no { color: #28a745; }
.result-invalid { color: #ff9800; }
.result-header-yes { color: #dc3545; }
.result-header-no { color: #28a745; }
.invalid-details { color: #ff0000; font-weight: 600; }
</style>
"""

@st.cache_resource
def load_model_and_labels():
    try:
        model = tensorflow.keras.models.load_model(MODEL_PATH, compile=False)
        class_names = []
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) > 1:
                        class_names.append(parts[1]. strip())
        if not class_names:
            class_names = ["Yes Have a Tumor in Brain Scan", "No Tumor in Brain Scan", "Not MRI Scan"]
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model:  {e}")
        st.stop()

def preprocess_image(image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image. Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array. astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def main():
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'
    if 'input_mode_key' not in st.session_state:
        st.session_state.input_mode_key = 'upload'
    
    lang = st.session_state.lang
    msg = MESSAGES[lang]
    st. markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title(msg["sidebar_title"])
        lang_choice = st.radio(msg["language_label"], ("English", "العربية"), index=0 if lang == 'en' else 1)
        new_lang = 'en' if lang_choice == "English" else 'ar'
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st. rerun()

        input_mode = st.radio(msg["input_mode_label"], [msg["mode_upload"], msg["mode_camera"]])
        st.session_state.input_mode_key = 'upload' if input_mode == msg["mode_upload"] else 'camera'
        
        st.markdown("---")
        st.subheader(msg["how_to_use_title"])
        # Render each point on a single physical line by converting newlines to <br>
        st.markdown(msg["how_to_use_text"].replace("\n", "<br>"), unsafe_allow_html=True)
        st.subheader(msg["about_title"])
        st.caption(msg["about_text"])
        st.subheader(msg["references_title"])
        st.warning(msg["references_text"])

        # --- Developers Section ---
        st.markdown("---")
        st.subheader(msg["developers_title"])
        st.write(msg["supervisor"])
        st.markdown("""
- Walid Tahkoubit
- Dalia
- B.  Mohamed 
- Omar Slimen
- B. Lokman
- Omar
""")

    st.title(msg["app_title"])
    
    # Developed Introduction Box (RTL class added for Arabic)
    st.markdown(f'<div class="intro-box {"rtl-text" if lang == "ar" else ""}"> {msg["intro_text"]} </div>', unsafe_allow_html=True)

    if lang == 'ar':
        st. markdown('<div class="rtl-text">', unsafe_allow_html=True)
    
    uploaded_file = None
    if st.session_state.input_mode_key == 'upload':
        uploaded_file = st.file_uploader(msg["upload_help"], type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input(msg["camera_button"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        with st.spinner(msg["processing"]):
            try:
                model, class_names = load_model_and_labels()
                data = preprocess_image(image)
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = prediction[0][index]

                if "Yes Have a Tumor in Brain Scan" in class_name: 
                    st.markdown(f'<h2 class="result-header-yes">{msg["result_header"]}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3 class="result-yes">{msg["result_yes_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_yes_text"])
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                elif "No Tumor in Brain Scan" in class_name:
                    st.markdown(f'<h2 class="result-header-no">{msg["result_header"]}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<h3 class="result-no">{msg["result_no_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_no_text"])
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                elif "Not MRI Scan" in class_name:
                    # Display the invalid image details in red and respect Arabic RTL when needed
                    st.markdown(f'<h3 class="result-invalid">{msg["invalid_image_msg"]}</h3>', unsafe_allow_html=True)
                    if lang == 'ar':
                        st.markdown(
                            f'<p class="invalid-details" style="direction: rtl; text-align:  right;">{msg["invalid_image_details"]}</p>',
                            unsafe_allow_html=True
                        )
                    else: 
                        st.markdown(
                            f'<p class="invalid-details">{msg["invalid_image_details"]}</p>',
                            unsafe_allow_html=True
                        )
                else: 
                    st.error(msg["invalid_image_msg"])
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if lang == 'ar':
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="footer">{msg["developer_credit"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__": 
    main()
