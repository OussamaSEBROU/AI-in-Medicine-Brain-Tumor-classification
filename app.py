import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import io
import os

# --- Configuration ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7

# --- Internationalization (i18n) Messages ---
MESSAGES = {
    "en": {
        "app_title": "AI NeuroScan",
        "title": "Brain Tumor Detection (AI-Powered)",
        "subtitle": "Leveraging advanced AI for early, non-invasive screening of brain MRI scans. Precision in every pixel.",
        "sidebar_title": "Settings & Information",
        "language_label": "Select Language",
        "input_mode_label": "Select Input Method",
        "mode_upload": "Upload Image",
        "mode_camera": "Live Camera",
        "upload_help": "Upload a brain MRI image (JPG, PNG, JPEG)",
        "camera_button": "Capture Image",
        "processing": "Processing image...",
        "no_file": "Please select an input method and provide an image.",
        "error_unclear": "The uploaded image is unclear or not recognized as a brain MRI. Please upload a clear brain MRI scan for accurate analysis.",
        "result_header": "Analysis Result",
        "result_yes_title": "Anomaly Detected",
        "result_yes_text": "We have detected an anomaly. Please consult a specialist immediately. Early detection is key to successful treatment. We recommend contacting a neurosurgeon or oncologist.",
        "result_no_title": "Scan Clear",
        "result_no_text": "Great news! The scan looks clear. No tumor detected. The symptoms you are experiencing may be due to other, less serious causes. Please follow up with your primary care physician for a comprehensive check-up.",
        "developer_credit": "Developed by **Oussama SEBROU**",
        "about_title": "About AI NeuroScan",
        "about_text": "AI NeuroScan is a prototype application developed to demonstrate the potential of Teachable Machine models in medical image classification.",
        "references_title": "Professional References",
        "references_text": "This application is based on the standard architecture for Keras models exported from Google's Teachable Machine.",
    },
    "ar": {
        "app_title": "الماسح العصبي بالذكاء الاصطناعي",
        "title": "كشف أورام الدماغ (مدعوم بالذكاء الاصطناعي)",
        "subtitle": "تسخير الذكاء الاصطناعي المتقدم للكشف المبكر وغير الجراحي عن أورام الدماغ في صور الرنين المغناطيسي. دقة في كل بكسل.",
        "sidebar_title": "الإعدادات والمعلومات",
        "language_label": "اختر اللغة",
        "input_mode_label": "اختر طريقة الإدخال",
        "mode_upload": "تحميل صورة",
        "mode_camera": "الكاميرا المباشرة",
        "upload_help": "قم بتحميل صورة رنين مغناطيسي (MRI) للدماغ (JPG, PNG, JPEG)",
        "camera_button": "التقاط الصورة",
        "processing": "جاري معالجة الصورة...",
        "no_file": "الرجاء اختيار طريقة إدخال وتقديم صورة.",
        "error_unclear": "الصورة المحملة غير واضحة أو لم يتم التعرف عليها كصورة رنين مغناطيسي للدماغ. يرجى رفع صورة MRI واضحة للدماغ فقط ليتعرف عليها النظام.",
        "result_header": "نتيجة التحليل",
        "result_yes_title": "تم الكشف عن شذوذ (Anomaly Detected)",
        "result_yes_text": "أفهم تماماً حجم القلق الذي تشعر به الآن، والصراحة المهنية تقتضي أن أخبرك بوجود نمو غير طبيعي تظهره الصور، مما يتطلب تحركاً طبياً دقيقاً. لذلك، سنوجهك إلى فريق مختص يجب أن تتابع معه فوراً، يضم نخبة من جراحي الأعصاب وأطباء الأورام لوضع الخطة العلاجية الأنسب لحالتك.",
        "result_no_title": "المسح سليم (Scan Clear)",
        "result_no_text": "أهنئك من كل قلبي، فنتائج الأشعة والتحاليل جاءت مطمئنة تماماً ولا تظهر أي وجود لورم كما كنت تخشى. الصداع أو الأعراض التي كنت تشعر بها لها أسباب أخرى أبسط بكثير، وسنعمل معاً على معالجتها بهدوء.",
        "developer_credit": "تم التطوير بواسطة **Oussama SEBROU**",
        "about_title": "حول الماسح العصبي بالذكاء الاصطناعي",
        "about_text": "الماسح العصبي بالذكاء الاصطناعي هو تطبيق نموذجي تم تطويره لإظهار إمكانات نماذج 'Teachable Machine' في تصنيف الصور الطبية.",
        "references_title": "المراجع المهنية",
        "references_text": "يعتمد هذا التطبيق على البنية القياسية لنماذج Keras المصدرة من 'Teachable Machine' من Google.",
    }
}

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp { background-color: #ffffff; color: #212529; font-family: 'Arial', sans-serif; }
h1 { color: #007bff; text-align: center; font-weight: 700; font-size: 2.5em; }
.footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f8f9fa; color: #6c757d; text-align: center; padding: 8px; font-size: 0.85em; border-top: 1px solid #e9ecef; z-index: 1000; }
.rtl-text { direction: rtl; text-align: right; }
</style>
"""

@st.cache_resource
def load_model_and_labels():
    try:
        model = tensorflow.keras.models.load_model(MODEL_PATH, compile=False)
        class_names = []
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) > 1:
                    class_names.append(parts[1].strip())
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model/labels: {e}")
        st.stop()

def preprocess_image(image):
    # تحويل الصورة إلى RGB لضمان وجود 3 قنوات (يحل مشكلة الصور الأبيض والأسود)
    image = image.convert("RGB")
    
    # تغيير الحجم وقص الصورة من المركز
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # تحويل إلى مصفوفة numpy
    image_array = np.asarray(image)
    
    # تطبيع القيم (Normalizing)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # تهيئة المصفوفة بالشكل المطلوب (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

def main():
    if 'lang' not in st.session_state: st.session_state.lang = 'en'
    if 'input_mode_key' not in st.session_state: st.session_state.input_mode_key = 'upload'

    lang = st.session_state.lang
    msg = MESSAGES[lang]
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.title(msg["sidebar_title"])
        lang_choice = st.radio(msg["language_label"], ("English", "العربية"), index=0 if lang == 'en' else 1)
        new_lang = 'en' if lang_choice == "English" else 'ar'
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

        input_mode = st.radio(msg["input_mode_label"], [msg["mode_upload"], msg["mode_camera"]])
        st.session_state.input_mode_key = 'upload' if input_mode == msg["mode_upload"] else 'camera'

    st.title(msg["app_title"])
    st.markdown(f'<p style="text-align: center; color: #6c757d;">{msg["subtitle"]}</p>', unsafe_allow_html=True)
    
    if lang == 'ar': st.markdown('<div class="rtl-text">', unsafe_allow_html=True)

    uploaded_file = None
    if st.session_state.input_mode_key == 'upload':
        uploaded_file = st.file_uploader(msg["upload_help"], type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input(msg["camera_button"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        with st.spinner(msg["processing"]):
            try:
                model, class_names = load_model_and_labels()
                data = preprocess_image(image)
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = prediction[0][index]

                # التحقق من صحة الصورة
                is_known_class = any(name in class_name for name in ["No Tumor", "Tumor Detected"])

                if confidence < CONFIDENCE_THRESHOLD or not is_known_class:
                    st.error(msg["error_unclear"])
                else:
                    st.header(msg["result_header"])
                    
                    if "Tumor Detected" in class_name:
                        st.markdown(f'<h3 style="color: #dc3545;">{msg["result_yes_title"]}</h3>', unsafe_allow_html=True)
                        st.write(msg["result_yes_text"])
                    elif "No Tumor" in class_name:
                        st.markdown(f'<h3 style="color: #28a745;">{msg["result_no_title"]}</h3>', unsafe_allow_html=True)
                        st.write(msg["result_no_text"])
                    
                    st.write(f"**Confidence:** {confidence*100:.2f}%")
                    st.write(f"**Class:** {class_name}")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if lang == 'ar': st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="footer">{msg["developer_credit"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
