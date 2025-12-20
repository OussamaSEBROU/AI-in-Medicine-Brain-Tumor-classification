import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Neuro Diagnosis",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Hide standard Streamlit branding for a more professional look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Custom CSS for Styling (Modern & Medical Theme)
# -----------------------------------------------------------------------------
def load_css(lang_code):
    """
    Load custom CSS based on language selection.
    Handles RTL for Arabic and LTR for English.
    """
    align = "right" if lang_code == "ar" else "left"
    direction = "rtl" if lang_code == "ar" else "ltr"
    
    st.markdown(f"""
    <style>
        .main {{
            background-color: #f8f9fa;
        }}
        .report-container {{
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            direction: {direction};
            text-align: {align};
            margin-top: 20px;
            border-left: 5px solid #2c3e50;
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #007bff;
            color: white;
            border: none;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: #0056b3;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .medical-alert {{
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 1.1em;
            line-height: 1.6;
        }}
        .alert-positive {{
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        .alert-negative {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .sidebar-text {{
            font-size: 1.2em;
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Text Assets & Translations
# -----------------------------------------------------------------------------
TEXTS = {
    "ar": {
        "title": "نظام التشخيص الإشعاعي الذكي",
        "subtitle": "تحليل صور الرنين المغناطيسي (MRI) باستخدام الذكاء الاصطناعي",
        "sidebar_title": "الإعدادات",
        "choose_source": "اختر طريقة الفحص",
        "camera": "كاميرا مباشرة",
        "upload": "رفع صورة",
        "camera_label": "التقط صورة للأشعة",
        "upload_label": "ارفع صورة الأشعة هنا (JPG, PNG)",
        "analyze_btn": "بدء التحليل الطبي",
        "error_no_img": "الرجاء رفع صورة أو التقاطها أولاً.",
        "error_not_mri": "تنبيه: الصورة المدخلة لا تبدو كصورة رنين مغناطيسي للدماغ. يرجى التأكد من جودة الصورة.",
        "result_title": "التقرير الطبي",
        "confidence": "نسبة التأكد:",
        # Positive Case (Tumor Found)
        "msg_positive_header": "⚠️ نتيجة التحليل المبدئي",
        "msg_positive_body": """
        أفهم تماماً حجم القلق الذي تشعر به الآن، والصراحة المهنية تقتضي أن أخبرك بوجود نمو غير طبيعي تظهره الصور، مما يتطلب تحركاً طبياً دقيقاً.
        
        لذلك، سنوجهك إلى فريق مختص يجب أن تتابع معه فوراً، يضم نخبة من جراحي الأعصاب وأطباء الأورام لوضع الخطة العلاجية الأنسب لحالتك.
        
        أطمئنك بأن العلم الحديث حقق قفزات مذهلة في هذا المجال، ونحن معك خطوة بخطوة لدعمك طبياً ونفسياً. ثق بأن تشخيصنا المبكر هو أول طريق التعافي، وقوتك النفسية ستكون المحرك الأساسي لنجاح رحلة العلاج بإذن الله.
        """,
        # Negative Case (No Tumor)
        "msg_negative_header": "✅ نتيجة مطمئنة",
        "msg_negative_body": """
        أهنئك من كل قلبي، فنتائج الأشعة والتحاليل جاءت مطمئنة تماماً ولا تظهر أي وجود لورم كما كنت تخشى.
        
        الصداع أو الأعراض التي كنت تشعر بها لها أسباب أخرى أبسط بكثير، وسنعمل معاً على معالجتها بهدوء. سنوجهك إلى فريق مختص يجب أن تتابع معه للتأكد من سلامة الجيوب الأنفية أو النظر أو ربما ضغوط الحياة اليومية، لضمان راحتك التامة.
        
        عد إلى منزلك وأنت مرتاح البال، فصحتك بخير وهذا هو الخبر الأجمل اليوم.
        """
    },
    "en": {
        "title": "AI Neuro-Radiology System",
        "subtitle": "Brain MRI Analysis powered by Artificial Intelligence",
        "sidebar_title": "Settings",
        "choose_source": "Select Input Source",
        "camera": "Live Camera",
        "upload": "Upload Image",
        "camera_label": "Capture MRI Scan",
        "upload_label": "Upload MRI Image (JPG, PNG)",
        "analyze_btn": "Start Medical Analysis",
        "error_no_img": "Please upload or capture an image first.",
        "error_not_mri": "Warning: The input image does not appear to be a clear Brain MRI. Please ensure image quality.",
        "result_title": "Medical Report",
        "confidence": "Confidence Score:",
        # Positive Case (Translation of the Arabic sentiment)
        "msg_positive_header": "⚠️ Analysis Result: Attention Required",
        "msg_positive_body": """
        I completely understand the anxiety you might be feeling right now. Professional honesty requires me to inform you that the scans show abnormal growth, which requires precise medical attention.
        
        Therefore, we advise you to consult immediately with a specialized team of neurosurgeons and oncologists to develop the most appropriate treatment plan.
        
        Rest assured that modern science has made amazing leaps in this field, and we are with you step by step. Trust that early diagnosis is the first step to recovery, and your psychological strength will be the main driver for the success of the treatment journey.
        """,
        # Negative Case (Translation of the Arabic sentiment)
        "msg_negative_header": "✅ Reassuring Result",
        "msg_negative_body": """
        I congratulate you from the bottom of my heart. The scan results are completely reassuring and do not show any presence of a tumor as you feared.
        
        The headaches or symptoms you were feeling likely have much simpler causes. We recommend checking with specialists regarding sinus health, vision, or daily stress factors to ensure your complete comfort.
        
        Go home with peace of mind; your health is fine, and that is the best news today.
        """
    }
}

# -----------------------------------------------------------------------------
# Model Management
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tm_model():
    """
    Load the Keras model.
    NOTE: Ensure 'keras_model.h5' is in the same directory.
    This file is downloaded from Teachable Machine (Export -> Tensorflow -> Keras).
    """
    try:
        # Load the model with compile=False for speed/safety if using custom layers
        model = tf.keras.models.load_model('keras_model.h5', compile=False)
        
        # Load labels
        with open('labels.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
            
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model. Make sure 'keras_model.h5' and 'labels.txt' are in the directory. Error: {e}")
        return None, None

def process_and_predict(image, model):
    """
    Process image to match Teachable Machine's requirements:
    1. Resize to 224x224
    2. Normalize to [-1, 1] range
    """
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Resize and crop
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    score = prediction[0][index]
    
    return index, score

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
def main():
    # 1. Sidebar Language Selection
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80) # Generic Medical Icon
        lang_choice = st.selectbox("Language / اللغة", ["العربية", "English"])
        lang = "ar" if lang_choice == "العربية" else "en"
        
        st.markdown("---")
        st.write("© 2024 Medical AI Solutions")
    
    # Load CSS based on language
    load_css(lang)
    t = TEXTS[lang]

    # 2. Header
    st.title(t["title"])
    st.subheader(t["subtitle"])
    st.markdown("---")

    # 3. Load Model
    model, class_names = load_tm_model()
    
    if model:
        # 4. Input Method
        input_method = st.radio(t["choose_source"], (t["camera"], t["upload"]), horizontal=True)
        
        image_input = None
        
        if input_method == t["camera"]:
            image_input = st.camera_input(t["camera_label"])
        else:
            image_input = st.file_uploader(t["upload_label"], type=["jpg", "png", "jpeg"])

        # 5. Analysis
        if image_input is not None:
            # Display the user image
            image = Image.open(image_input).convert("RGB")
            st.image(image, caption="Source Scan", use_column_width=True)
            
            if st.button(t["analyze_btn"]):
                with st.spinner('Analyzing patterns... / جاري تحليل الأنسجة...'):
                    class_idx, score = process_and_predict(image, model)
                    
                    # Get class name (assuming Teachable Machine export: '0 ClassName', '1 ClassName')
                    # IMPORTANT: You must verify your class names in labels.txt
                    # Here we assume logic based on the class text itself
                    prediction_label = class_names[class_idx]
                    
                    # Logic to determine if Tumor is Yes or No based on label text
                    # Adjust 'yes' or 'tumor' based on how you named your classes in Teachable Machine
                    is_tumor = "yes" in prediction_label.lower() or "tumor" in prediction_label.lower()
                    
                    # --- Presentation ---
                    st.markdown(f"### {t['result_title']}")
                    
                    # Simple heuristic check for non-MRI images (low confidence or pure white/black)
                    # Note: A real robust check needs a 3rd class "Random", but we use confidence threshold here
                    if score < 0.60:
                        st.warning(t["error_not_mri"])
                    else:
                        st.write(f"**{t['confidence']}** {score*100:.2f}%")
                        
                        container_class = "alert-positive" if is_tumor else "alert-negative"
                        header_text = t["msg_positive_header"] if is_tumor else t["msg_negative_header"]
                        body_text = t["msg_positive_body"] if is_tumor else t["msg_negative_body"]
                        
                        st.markdown(f"""
                        <div class="report-container">
                            <div class="medical-alert {container_class}">
                                <h3>{header_text}</h3>
                                <p>{body_text}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Disclaimer
                        st.caption("Disclaimer: This AI tool is for assistance only and does not replace professional medical diagnosis.")

if __name__ == "__main__":
    main()