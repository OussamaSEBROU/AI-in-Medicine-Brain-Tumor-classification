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

# --- Internationalization (i18n) Messages ---
MESSAGES = {
    "en": {
        "app_title": "AI NeuroScan",
        "intro_text": "Welcome to AI NeuroScan. This platform provides an initial automated screening for brain MRI scans using advanced neural networks to assist in early detection.",
        "subtitle": "Leveraging advanced AI for early, non-invasive screening of brain MRI scans. Precision in every pixel.",
        "sidebar_title": "Settings & Information",
        "language_label": "Select Language",
        "input_mode_label": "Select Input Method",
        "mode_upload": "Upload Image",
        "mode_camera": "Live Camera",
        "upload_help": "Upload a brain MRI image (JPG, PNG, JPEG)",
        "camera_button": "Capture Image",
        "processing": "Processing image...",
        "result_header": "Analysis Result",
        "result_yes_title": "Anomaly Detected (Tumor Found)",
        "result_yes_text": "I fully understand the depth of anxiety you are feeling right now. Professional integrity requires me to inform you that the images reveal an abnormal growth, which necessitates precise medical action. Therefore, we will refer you to a specialized team you must follow up with immediately—including elite neurosurgeons and oncologists—to develop the most appropriate treatment plan for your condition. I want to reassure you that modern science has made incredible leaps in this field, and we will be with you every step of the way to provide both medical and psychological support. Rest assured that our early diagnosis is the first step toward recovery, and your psychological strength will be the primary engine for the success of this treatment journey, God willing.",
        "result_no_title": "Scan Clear (No Tumor Found)",
        "result_no_text": "Great news! I offer you my heartfelt congratulations; your scan and lab results are entirely reassuring and show no evidence of a tumor as you had feared. The headaches or symptoms you have been experiencing stem from much simpler causes, which we will work together to address calmly. We will refer you to a specialized team for further follow-up to check your sinuses, vision, or perhaps the impact of daily life stressors, ensuring your complete comfort and well-being. You can go home with a peaceful mind; you are in good health, and that is truly the best news today.",
        "developer_credit": "Developed by **ErinovAIClub**",
        "about_title": "About technology",
        "about_text": "This app uses a Deep Learning model (Convolutional Neural Network) trained on thousands of MRI images to identify structural anomalies in the brain.",
        "how_to_use_title": "How to Use",
        "how_to_use_text": "1. Select 'Upload' or 'Camera'.\n2. Provide a clear MRI image.\n3. Wait for the AI analysis.\n4. Review the confidence score and prediction.",
        "references_title": "Disclaimer",
        "references_text": "This is a prototype for educational and screening purposes. It is NOT a definitive medical diagnosis.",
    },
    "ar": {
        "app_title": "الماسح العصبي بالذكاء الاصطناعي",
        "intro_text": "مرحباً بكم في منصة الماسح العصبي. يوفر هذا التطبيق فحصاً أولياً مؤتمتاً لصور الرنين المغناطيسي للدماغ باستخدام شبكات عصبية متطورة للمساعدة في الكشف المبكر.",
        "subtitle": "تسخير الذكاء الاصطناعي المتقدم للكشف المبكر وغير الجراحي عن أورام الدماغ في صور الرنين المغناطيسي. دقة في كل بكسل.",
        "sidebar_title": "الإعدادات والمعلومات",
        "language_label": "اختر اللغة",
        "input_mode_label": "اختر طريقة الإدخال",
        "mode_upload": "تحميل صورة",
        "mode_camera": "الكاميرا المباشرة",
        "upload_help": "قم بتحميل صورة رنين مغناطيسي (MRI) للدماغ (JPG, PNG, JPEG)",
        "camera_button": "التقاط الصورة",
        "processing": "جاري معالجة الصورة...",
        "result_header": "نتيجة التحليل",
        "result_yes_title": "تم الكشف عن شذوذ (وجود ورم)",
        "result_yes_text": "أفهم تماماً حجم القلق الذي تشعر به الآن، والصراحة المهنية تقتضي أن أخبرك بوجود نمو غير طبيعي تظهره الصور، مما يتطلب تحركاً طبياً دقيقاً. لذلك، سنوجهك إلى فريق مختص يجب أن تتابع معه فوراً، يضم نخبة من جراحي الأعصاب وأطباء الأورام لوضع الخطة العلاجية الأنسب لحالتك. أود أن أطمئنك بأن العلم الحديث قد حقق قفزات هائلة في هذا المجال، وسنكون معك في كل خطوة لتقديم الدعم الطبي والنفسي اللازم. تأكد أن تشخيصنا المبكر هو أولى خطوات الشفاء، وقوتك النفسية ستكون المحرك الأول لنجاح هذه الرحلة العلاجية بإذن الله.",
        "result_no_title": "المسح سليم (لا يوجد ورم)",
        "result_no_text": "أهنئك من كل قلبي، فنتائج الأشعة والتحاليل جاءت مطمئنة تماماً ولا تظهر أي وجود لورم كما كنت تخشى. الصداع أو الأعراض التي كنت تشعر بها لها أسباب أخرى أبسط بكثير، وسنعمل معاً على معالجتها بهدوء. سنقوم بتوجيهك لفريق مختص للمتابعة الإضافية للتأكد من الجيوب الأنفية أو النظر أو ربما ضغوط الحياة اليومية لضمان راحتك الكاملة. يمكنك العودة إلى منزلك ببال مطمئن، فأنت بخير وصحة جيدة، وهذا هو الخبر الأجمل اليوم.",
        "developer_credit": "تم التطوير بواسطة **ErinovAIClub**",
        "about_title": "حول التقنية المستخدمة",
        "about_text": "يعتمد التطبيق على خوارزميات التعلم العميق (Deep Learning) وبالتحديد الشبكات العصبية التلافيفية (CNN) التي تم تدريبها لتمييز الأنماط غير الطبيعية في صور الرنين المغناطيسي.",
        "how_to_use_title": "كيفية الاستخدام",
        "how_to_use_text": "1. اختر 'تحميل صورة' أو 'الكاميرا'.\n2. ارفع صورة MRI واضحة للدماغ.\n3. انتظر معالجة الذكاء الاصطناعي.\n4. راجع النتيجة ونسبة الثقة الظاهرة.",
        "references_title": "تنبيه هام",
        "references_text": "هذا التطبيق هو نموذج أولي للأغراض التعليمية والفحص الأولي فقط، ولا يعتبر تشخيصاً طبياً نهائياً.",
    }
}

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp { background-color: #ffffff; color: #212529; font-family: 'Arial', sans-serif; }
h1 { color: #007bff; text-align: center; font-weight: 700; font-size: 2.5em; }
.intro-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; margin-bottom: 25px; text-align: center; }
.footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f8f9fa; color: #6c757d; text-align: center; padding: 8px; font-size: 0.85em; border-top: 1px solid #e9ecef; z-index: 1000; }
.rtl-text { direction: rtl; text-align: right; }
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
                        class_names.append(parts[1].strip())
        if not class_names:
            class_names = ["No Tumor in Brain Scan", "Yes Have a Tumor in Brain Scan"] 
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def preprocess_image(image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
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

        st.markdown("---")
        st.subheader(msg["how_to_use_title"])
        st.info(msg["how_to_use_text"])
        
        st.subheader(msg["about_title"])
        st.caption(msg["about_text"])

        st.subheader(msg["references_title"])
        st.warning(msg["references_text"])

    st.title(msg["app_title"])
    
    # Introduction Box
    st.markdown(f'<div class="intro-box {"rtl-text" if lang == "ar" else ""}"> {msg["intro_text"]} </div>', unsafe_allow_html=True)
    
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

                st.header(msg["result_header"])
                
                # Updated Logic to match EXACT strings in labels.txt
                if "Yes Have a Tumor" in class_name:
                    st.markdown(f'<h3 style="color: #dc3545;">{msg["result_yes_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_yes_text"])
                elif "No Tumor" in class_name:
                    st.markdown(f'<h3 style="color: #28a745;">{msg["result_no_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_no_text"])
                else:
                    st.warning(f"Classification result: {class_name}")
                
                st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                st.write(f"**Classification Details:** {class_name}")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if lang == 'ar': st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="footer">{msg["developer_credit"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
