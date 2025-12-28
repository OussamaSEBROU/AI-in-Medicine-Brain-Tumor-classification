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
        "intro_text": "Welcome to AI NeuroScan Our advanced neural network provides rapid, automated screening of brain MRI scans Designed to support clinical workflows, this tool identifies structural anomalies with high precision to aid in early detection and proactive health management.",
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
        "result_yes_text": "I want to speak with you with complete transparency and empathy. The AI analysis has detected an abnormal growth in your brain scan that requires immediate medical attention. I understand this news may feel overwhelming, and it's completely natural to feel anxious right now.\n\n**What This Means:**\nThis preliminary screening indicates the presence of tissue that appears different from normal brain structure. However, it's crucial to understand that:\n- This is an AI-assisted screening tool, not a definitive diagnosis\n- Further specialized imaging and expert evaluation are essential\n- Many brain abnormalities are treatable, especially when detected early\n- Modern neurosurgery and oncology have achieved remarkable success rates\n\n**Immediate Next Steps:**\n1. Schedule an urgent consultation with a neurologist or neurosurgeon\n2. Bring this scan to your appointment for professional review\n3. Additional diagnostic tests (contrast MRI, biopsy) may be recommended\n4. Consider seeking a second opinion from a specialized medical center\n\n**Psychological Support:**\nYour mental well-being is as important as your physical health. Consider:\n- Speaking with a counselor or psychologist who specializes in medical diagnoses\n- Connecting with support groups for patients facing similar challenges\n- Leaning on your family and friends during this time\n\nRemember: Early detection significantly improves treatment outcomes. Your proactive approach in getting screened is already a positive step toward your health journey. You are not alone in this",
        "result_no_title": "Scan Clear (No Tumor Found)",
        "result_no_text": "I'm pleased to share positive news with you. The AI analysis of your brain MRI scan shows no evidence of tumor or significant abnormal growth. This is genuinely encouraging and should bring you considerable peace of mind.\n\n**What This Means:**\n- No suspicious masses or lesions were detected in the brain tissue\n- The scan appears consistent with normal brain structure\n- Your symptoms (if any) likely have other, more common causes\n\n**Understanding Your Symptoms:**\nIf you've been experiencing headaches, dizziness, or other neurological symptoms, these could be related to:\n- Tension or migraine headaches (very common and manageable)\n- Sinus issues or allergies\n- Vision problems requiring corrective lenses\n- Sleep disturbances or stress\n- Dehydration or nutritional factors\n\n**Recommended Follow-Up:**\nWhile this screening is reassuring, I recommend:\n1. Discuss this result with your primary care physician\n2. Address any ongoing symptoms with appropriate specialists (ENT, ophthalmologist, neurologist)\n3. Maintain healthy lifestyle habits: adequate sleep, stress management, regular exercise\n4. Keep routine medical check-ups as recommended by your doctor\n\n**Important Reminder:**\nThis AI screening is a valuable tool, but it doesn't replace professional medical evaluation. If your symptoms persist or worsen, don't hesitate to seek medical advice.\n\nYou can move forward with confidence knowing that this aspect of your health appears stable. Focus on addressing any symptoms through appropriate medical channels, and maintain your overall wellness",
        "invalid_image_msg": "⚠️ Invalid Image - Not a Brain MRI Scan",
        "invalid_image_details": "The uploaded/captured image is not clear or is not a brain MRI scan. Please ensure you upload a high-quality MRI brain scan image for accurate analysis",
        "developer_credit": "Developed by Erinov AI Club",
        "about_title": "About technology",
        "about_text": "This app uses a Deep Learning model (Convolutional Neural Network) trained on thousands of MRI images",
        "how_to_use_title": "How to Use",
        "how_to_use_text": "1. Select 'Upload' or 'Camera'.\n2. Provide a clear MRI image.\n3. Wait for the AI analysis",
        "references_title": "Disclaimer",
        "references_text": "This is a prototype for educational and screening purposes. It is NOT a definitive medical diagnosis",
        "developers_title": "Developers Team",
        "supervisor": "Under the supervision of *M. Oussama SEBROU*"
    },
    "ar": {
        "app_title": "AI NeuroScan",
        "intro_text": "مرحباً بكم، توفر منصتنا فحصاً ذكياً وفورياً لصور الرنين المغناطيسي للدماغ باستخدام تقنيات التعلم العميق، يهدف هذا النظام إلى دعم الكشف المبكر عن الأورام وتقديم تحليل أولي دقيق يساعد في اتخاذ قرارات صحية استباقية",
        "subtitle": "تسخير الذكاء الاصطناعي المتقدم للكشف المبكر وغير الجراحي عن أورام الدماغ",
        "sidebar_title": "الإعدادات والمعلومات",
        "language_label": "اختر اللغة",
        "input_mode_label": "اختر طريقة الإدخال",
        "mode_upload": "تحميل صورة",
        "mode_camera": "الكاميرا المباشرة",
        "upload_help": "قم بتحميل صورة رنين مغناطيسي (MRI) للدماغ",
        "camera_button": "التقاط الصورة",
        "processing": "جاري معالجة الصورة...",
        "result_header": "نتيجة التحليل",
        "result_yes_title": "تم الكشف عن شذوذ (وجود ورم)",
        "result_yes_text": "أود أن أتحدث معك بشفافية كاملة وتعاطف. أظهر التحليل بواسطة الذكاء الاصطناعي وجود نمو غير طبيعي في فحص الدماغ الخاص بك يتطلب اهتماماً طبياً فورياً. أدرك أن هذا الخبر قد يكون مربكاً، ومن الطبيعي تماماً أن تشعر بالقلق الآن.\n\n**ماذا يعني هذا:**\nيشير هذا الفحص الأولي إلى وجود نسيج يبدو مختلفاً عن البنية الطبيعية للدماغ. ومع ذلك، من الضروري أن تفهم أن:\n- هذه أداة فحص بمساعدة الذكاء الاصطناعي، وليست تشخيصاً نهائياً\n- التصوير المتخصص الإضافي والتقييم من قبل الخبراء أمر ضروري\n- العديد من الشذوذات الدماغية قابلة للعلاج، خاصة عند اكتشافها مبكراً\n- حققت جراحة الأعصاب الحديثة وعلم الأورام معدلات نجاح ملحوظة\n\n**الخطوات الفورية التالية:**\n1. حدد موعداً عاجلاً مع طبيب أعصاب أو جراح أعصاب\n2. أحضر هذا الفحص إلى موعدك للمراجعة المهنية\n3. قد يُوصى بفحوصات تشخيصية إضافية (رنين مغناطيسي بالصبغة، خزعة)\n4. فكر في الحصول على رأي ثانٍ من مركز طبي متخصص\n\n**الدعم النفسي:**\nرفاهيتك النفسية لا تقل أهمية عن صحتك الجسدية. ننصحك بـ:\n- التحدث مع مستشار نفسي أو أخصائي نفسي متخصص في التشخيصات الطبية\n- التواصل مع مجموعات الدعم للمرضى الذين يواجهون تحديات مماثلة\n- الاعتماد على عائلتك وأصدقائك خلال هذا الوقت\n\nتذكر: الكشف المبكر يحسن بشكل كبير من نتائج العلاج. نهجك الاستباقي في إجراء الفحص هو بالفعل خطوة إيجابية نحو رحلتك الصحية. أنت لست وحدك في هذا",
        "result_no_title": "المسح سليم (لا يوجد ورم)",
        "result_no_text": "يسعدني مشاركة أخبار إيجابية معك. أظهر تحليل الذكاء الاصطناعي لفحص الرنين المغناطيسي للدماغ عدم وجود دليل على ورم أو نمو غير طبيعي كبير. هذا مشجع حقاً ويجب أن يمنحك راحة البال الكبيرة.\n\n**ماذا يعني هذا:**\n- لم يتم اكتشاف كتل أو آفات مشبوهة في أنسجة الدماغ\n- يبدو الفحص متسقاً مع بنية الدماغ الطبيعية\n- من المحتمل أن تكون أعراضك (إن وجدت) ناتجة عن أسباب أخرى أكثر شيوعاً\n\n**فهم أعراضك:**\nإذا كنت تعاني من صداع أو دوخة أو أعراض عصبية أخرى، فقد تكون مرتبطة بـ:\n- صداع التوتر أو الصداع النصفي (شائع جداً وقابل للإدارة)\n- مشاكل الجيوب الأنفية أو الحساسية\n- مشاكل الرؤية التي تتطلب نظارات تصحيحية\n- اضطرابات النوم أو التوتر\n- الجفاف أو عوامل غذائية\n\n**المتابعة الموصى بها:**\nبينما هذا الفحص مطمئن، أوصي بـ:\n1. مناقشة هذه النتيجة مع طبيبك العام\n2. معالجة أي أعراض مستمرة مع المتخصصين المناسبين (أنف وأذن وحنجرة، طبيب عيون، طبيب أعصاب)\n3. الحفاظ على عادات نمط حياة صحية: نوم كافٍ، إدارة التوتر، ممارسة الرياضة بانتظام\n4. الحفاظ على الفحوصات الطبية الروتينية كما يوصي طبيبك\n\n**تذكير مهم:**\nهذا الفحص بالذكاء الاصطناعي هو أداة قيمة، لكنه لا يحل محل التقييم الطبي المهني. إذا استمرت أعراضك أو تفاقمت، لا تتردد في طلب المشورة الطبية.\n\nيمكنك المضي قدماً بثقة مع العلم أن هذا الجانب من صحتك يبدو مستقراً. ركز على معالجة أي أعراض من خلال القنوات الطبية المناسبة، وحافظ على عافيتك العامة",
        "invalid_image_msg": "⚠️ صورة غير صالحة - ليست صورة رنين مغناطيسي",
        "invalid_image_details": "الصورة التي تم رفعها أو التقاطها ليست واضحة أو ليست صورة رنين مغناطيسي للدماغ (MRI Brain Scan). يرجى التأكد من رفع صورة MRI واضحة وعالية الجودة للحصول على تحليل دقيق.",
        "developer_credit": "Developed by Erinov AI Club",
        "about_title": "حول التقنية المستخدمة",
        "about_text": "يعتمد التطبيق على خوارزميات التعلم العميق (Deep Learning) وبالتحديد الشبكات العصبية تلافيفية (CNN)",
        "how_to_use_title": "كيفية الاستخدام",
        "how_to_use_text": "1. اختر 'تحميل صورة' أو 'الكاميرا'.\n2. ارفع صورة MRI واضحة.\n3. انتظر معالجة الذكاء الاصطناعي",
        "references_title": "تنبيه هام",
        "references_text": "هذا التطبيق هو نموذج أولي للأغراض التعليمية فقط، ولا يعتبر تشخيصاً طبياً نهائياً",
        "developers_title": "فريق التطوير",
        "supervisor": "تحت إشراف *M. Oussama SEBROU*"
    }
}

# --- Custom CSS ---
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp { background-color: #ffffff; color: #212529; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
h1 { color: #007bff; text-align: center; font-weight: 800; font-size: 3em; margin-bottom: 10px; }
.intro-box { 
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
    padding: 30px; 
    border-radius: 15px; 
    border-right: 8px solid #007bff; 
    border-left: 8px solid #007bff; 
    margin-bottom: 30px; 
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    line-height: 1.6;
}
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
            class_names = ["Yes Have a Tumor in Brain Scan", "No Tumor in Brain Scan", "Not MRI Scan"]
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

        # --- Developers Section ---
        st.markdown("---")
        st.subheader(msg["developers_title"])
        st.write(msg["supervisor"]) # السطر المائل الجديد
        st.markdown("""
        - Walid Tahkoubit
        - Dalia
        - B. Mohamed 
        - Omar Slimen
        - B. Lokman
        - Omar
        """)

    st.title(msg["app_title"])
    
    # Developed Introduction Box
    st.markdown(f'<div class="intro-box {"rtl-text" if lang == "ar" else ""}"> {msg["intro_text"]} </div>', unsafe_allow_html=True)

    if lang == 'ar': st.markdown('<div class="rtl-text">', unsafe_allow_html=True)
    
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

                st.header(msg["result_header"])

                if "Yes Have a Tumor in Brain Scan" in class_name:
                    st.markdown(f'<h3 style="color: #dc3545;">{msg["result_yes_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_yes_text"])
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                elif "No Tumor in Brain Scan" in class_name:
                    st.markdown(f'<h3 style="color: #28a745;">{msg["result_no_title"]}</h3>', unsafe_allow_html=True)
                    st.write(msg["result_no_text"])
                    st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                elif "Not MRI Scan" in class_name:
                    st.markdown(f'<h3 style="color: #ff9800;">{msg["invalid_image_msg"]}</h3>', unsafe_allow_html=True)
                    st.warning(msg["invalid_image_details"])
                else:
                    st.error(msg["invalid_image_msg"])
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if lang == 'ar': st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="footer">{msg["developer_credit"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
