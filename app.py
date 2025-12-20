import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- Configuration and Setup ---
st.set_page_config(
    page_title="AI Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Language Dictionary ---
# All application text is stored here for easy translation and switching
LANG_DICT = {
    "ar": {
        "title": "تصنيف أورام الدماغ بالذكاء الاصطناعي",
        "sidebar_title": "مصنف أورام الدماغ بالذكاء الاصطناعي",
        "lang_select": "اختيار اللغة",
        "upload_option": "تحميل صورة",
        "camera_option": "التقاط صورة مباشرة",
        "upload_file_prompt": "اختر صورة رنين مغناطيسي للدماغ...",
        "camera_prompt": "التقاط صورة مباشرة (MRI)",
        "uploaded_caption": "صورة الرنين المغناطيسي المحملة",
        "analysis_result": "نتيجة التحليل",
        "analysis_spinner": "جاري تحليل الصورة وتصنيف الورم...",
        "confidence_label": "نسبة الثقة",
        "expert_advice_title": "توجيهات ونصائح طبية",
        "unclassified_error": "النتيجة: {result} - حدث خطأ في التصنيف أو قراءة النتائج.",
        "no_tumor_success": "النتيجة: {result}",
        "tumor_error": "النتيجة: {result}",
        "sidebar_overview_header": "نظرة عامة على النظام",
        "sidebar_overview_info": """
            هذا النظام هو أداة مساعدة تعتمد على **التعلم العميق (Deep Learning)** لتصنيف صور الرنين المغناطيسي (MRI) للدماغ.
            
            *   **النموذج:** تم التدريب باستخدام شبكة عصبية تلافيفية (CNN) عبر مكتبة Keras/TensorFlow.
            *   **الهدف:** المساعدة في الكشف الأولي وتصنيف أنواع أورام الدماغ.
        """,
        "sidebar_usage_header": "إرشادات الاستخدام",
        "sidebar_usage_info": """
            1.  **اختيار الطريقة:** اختر بين تحميل صورة من جهازك أو التقاط صورة مباشرة.
            2.  **التصنيف:** سيقوم الذكاء الاصطناعي بتحليل الصورة وتقديم نتيجة التصنيف ونسبة الثقة.
            3.  **التوجيه الطبي:** ستظهر نصيحة طبية مفصلة ومبنية على النتيجة لتوجيهك نحو الخطوات التالية.
        """,
        "sidebar_disclaimer_header": "إخلاء مسؤولية طبي",
        "sidebar_disclaimer_warning": """
            **هذا النظام ليس بديلاً عن التشخيص الطبي الاحترافي.**
            
            النتائج المقدمة هي لأغراض إعلامية ومساعدة فقط. يجب دائماً استشارة طبيب مختص أو أخصائي أشعة لتأكيد أي تشخيص أو اتخاذ قرارات علاجية.
        """,
        "footer": "Developed by Oussama SEBROU",
        "advice_db": {
            "No Tumor": {
                "title": "نتائج مطمئنة: لا يوجد ورم",
                "advice": """
                **خبر سار ومريح!** تشير نتائج التحليل إلى أن الصورة لا تحمل أي علامات للورم المصنف. هذا يبعث على الاطمئنان.
                
                **توجيهات للحفاظ على الصحة:** تهانينا على هذه النتيجة الإيجابية. للحفاظ على صحتك وقوتك العقلية، نوصي بما يلي:
                *   **الاستمرار في الفحوصات الروتينية** التي يحددها طبيبك.
                *   **الحفاظ على نمط حياة صحي** يشمل التغذية المتوازنة والنشاط البدني والنوم الكافي.
                *   **الدعم النفسي:** استمتع براحة البال، وتذكر أن العناية بالصحة رحلة مستمرة.
                """
            },
            "Tumor Detected": {
                "title": "تنبيه هام: نتيجة تتطلب استشارة طبية عاجلة",
                "advice": """
                **رسالة دعم وتوجيه:** يشير التحليل إلى وجود احتمال لوجود ورم. من الطبيعي أن تشعر بالقلق، ولكن من المهم جداً أن تعلم أن هذا التشخيص هو **الخطوة الأولى نحو العلاج الفعال**.
                
                **توجيهات الطبيب الخبير:**
                *   **التصرف بهدوء:** أهم خطوة الآن هي **مراجعة طبيب مختص (جراح أعصاب أو طبيب أورام عصبية) بأسرع وقت ممكن**.
                *   **الأمل والعلاج:** مع التطور التكنولوجي والطبي الهائل، هناك خيارات علاجية متقدمة وفعالة جداً. **لا تقنط**، فالعلاج متاح، وفريقك الطبي سيضع خطة تناسب حالتك.
                *   **الدعم النفسي:** لا تخض هذه التجربة وحدك. تحدث مع عائلتك وأصدقائك، واطلب الدعم النفسي. القوة الداخلية هي جزء أساسي من رحلة العلاج.
                """
            },
            "Unclassified": {
                "title": "تنبيه: نتيجة غير مصنفة",
                "advice": "تم تصنيف الصورة بنجاح، ولكن النتيجة غير موجودة في قاعدة بيانات النصائح الطبية. **الخطوة التالية:** يرجى مراجعة طبيب مختص على الفور لمناقشة النتيجة: **{result}**."
            }
        }
    },
    "en": {
        "title": "AI Brain Tumor Classification",
        "sidebar_title": "AI Brain Tumor Classifier",
        "lang_select": "Select Language",
        "upload_option": "Upload Image",
        "camera_option": "Capture Live Image",
        "upload_file_prompt": "Choose a brain MRI image...",
        "camera_prompt": "Capture Live Image (MRI)",
        "uploaded_caption": "Uploaded MRI Image",
        "analysis_result": "Analysis Result",
        "analysis_spinner": "Analyzing image and classifying tumor...",
        "confidence_label": "Confidence Score",
        "expert_advice_title": "Guidance and Medical Advice",
        "unclassified_error": "Result: {result} - An error occurred during classification or result reading.",
        "no_tumor_success": "Result: {result}",
        "tumor_error": "Result: {result}",
        "sidebar_overview_header": "System Overview",
        "sidebar_overview_info": """
            This system is an auxiliary tool based on **Deep Learning** to classify brain Magnetic Resonance Imaging (MRI) scans.
            
            *   **Model:** Trained using a Convolutional Neural Network (CNN) via the Keras/TensorFlow library.
            *   **Goal:** To assist in the initial detection and classification of brain tumor types.
        """,
        "sidebar_usage_header": "Usage Instructions",
        "sidebar_usage_info": """
            1.  **Select Method:** Choose between uploading an image from your device or capturing a live image.
            2.  **Classification:** The AI will analyze the image and provide the classification result and confidence score.
            3.  **Medical Guidance:** Detailed medical advice based on the result will appear to guide you on the next steps.
        """,
        "sidebar_disclaimer_header": "Medical Disclaimer",
        "sidebar_disclaimer_warning": """
            **This system is NOT a substitute for professional medical diagnosis.**
            
            The results provided are for informational and assistive purposes only. You must always consult a specialized physician or radiologist to confirm any diagnosis or make treatment decisions.
        """,
        "footer": "Developed by Oussama SEBROU",
        "advice_db": {
            "No Tumor": {
                "title": "Reassuring Results: No Tumor Found",
                "advice": """
                **Great and Reassuring News!** The analysis indicates that the image shows no signs of the classified tumor types. This is a source of relief.
                
                **Guidance for Health Maintenance:** Congratulations on this positive result. To maintain your health and mental strength, we recommend the following:
                *   **Continue with routine check-ups** as determined by your physician.
                *   **Maintain a healthy lifestyle** including balanced nutrition, physical activity, and sufficient sleep.
                *   **Psychological Support:** Enjoy your peace of mind, and remember that health care is an ongoing journey.
                """
            },
            "Tumor Detected": {
                "title": "Important Alert: Result Requires Urgent Medical Consultation",
                "advice": """
                **A Message of Support and Guidance:** The analysis indicates a possibility of a tumor. It is normal to feel anxious, but it is very important to know that this diagnosis is **the first step towards effective treatment**.
                
                **Expert Medical Guidance:**
                *   **Act Calmly:** The most important step now is to **consult a specialized physician (neurosurgeon or neuro-oncologist) as soon as possible**.
                *   **Hope and Treatment:** With the tremendous technological and medical advancements, there are advanced and very effective treatment options. **Do not despair**, treatment is available, and your medical team will develop a plan that suits your condition.
                *   **Psychological Support:** Do not go through this experience alone. Talk to your family and friends, and seek psychological support. Inner strength is an essential part of the treatment journey.
                """
            },
            "Unclassified": {
                "title": "Alert: Unclassified Result",
                "advice": "The image was classified successfully, but the result is not in the medical advice database. **Next Step:** Please consult a specialized physician immediately to discuss the result: **{result}**."
            }
        }
    }
}

# --- Language Selection in Sidebar ---
if 'lang' not in st.session_state:
    st.session_state.lang = "ar" # Default to Arabic

with st.sidebar:
    st.header(LANG_DICT[st.session_state.lang]["lang_select"])
    lang_choice = st.radio(
        "",
        ("العربية", "English"),
        index=0 if st.session_state.lang == "ar" else 1,
        key="lang_radio"
    )
    
    if lang_choice == "العربية":
        st.session_state.lang = "ar"
    else:
        st.session_state.lang = "en"

# Get the current language texts
T = LANG_DICT[st.session_state.lang]

# --- Load Model and Labels ---
@st.cache_resource
def load_model():
    # The model loading is cached, which significantly improves startup speed after the first run.
    model = tf.keras.models.load_model('keras_model.h5')
    return model

@st.cache_data
def load_labels():
    # Labels loading is also cached.
    try:
        with open('labels.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error(T["unclassified_error"].format(result="'labels.txt' not found"))
        # Placeholder labels - USER MUST VERIFY THESE MATCH THEIR MODEL
        labels = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"] 
    return labels

model = load_model()
labels = load_labels()

# --- Expert Medical Advice Function ---
def get_medical_advice(result_class):
    # Map all tumor types to the general "Tumor Detected" advice
    if result_class in ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]:
        return T["advice_db"]["Tumor Detected"]
    elif result_class == "No Tumor":
        return T["advice_db"]["No Tumor"]
    else:
        # Handle unclassified or unknown results
        advice_data = T["advice_db"]["Unclassified"]
        advice_data['advice'] = advice_data['advice'].format(result=result_class)
        return advice_data

# --- Sidebar Content (Dynamic) ---
with st.sidebar:
    st.title(T["sidebar_title"])
    st.markdown("---")
    
    st.header(T["sidebar_overview_header"])
    st.info(T["sidebar_overview_info"])
    
    st.header(T["sidebar_usage_header"])
    st.markdown(T["sidebar_usage_info"])
    
    st.markdown("---")
    st.header(T["sidebar_disclaimer_header"])
    st.warning(T["sidebar_disclaimer_warning"])


# --- Main Application Layout ---
st.title(T["title"])
st.markdown("---")

# --- Image Input Options ---
input_method = st.radio(
    "**1. اختيار طريقة إدخال الصورة:**" if st.session_state.lang == "ar" else "**1. Select Image Input Method:**",
    (T["upload_option"], T["camera_option"]),
    key="input_method_radio"
)

uploaded_file = None
image_data = None

if input_method == T["upload_option"]:
    uploaded_file = st.file_uploader(T["upload_file_prompt"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file).convert('RGB')
elif input_method == T["camera_option"]:
    camera_image = st.camera_input(T["camera_prompt"])
    if camera_image is not None:
        image_data = Image.open(camera_image).convert('RGB')


if image_data is not None:
    # Use columns for a cleaner layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display the uploaded image
        st.image(image_data, caption=T["uploaded_caption"], use_column_width=True)
        
    with col2:
        st.subheader(T["analysis_result"])
        with st.spinner(T["analysis_spinner"]):
            # Preprocess the image
            size = (224, 224)
            image_resized = image_data.resize(size)
            image_array = np.asarray(image_resized)
            
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            
            # Create the array of the right shape
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make prediction
            prediction = model.predict(data)
            
            # Get the index of the highest confidence prediction
            predicted_class_index = np.argmax(prediction)
            
            # Check if the index is valid
            if predicted_class_index < len(labels):
                predicted_class = labels[predicted_class_index]
                confidence_score = prediction[0][predicted_class_index] * 100
            else:
                predicted_class = "Unclassified"
                confidence_score = 0.0

        # Display results with better formatting
        if "No Tumor" in predicted_class:
            st.balloons()
            st.success(T["no_tumor_success"].format(result=predicted_class))
        elif "Unclassified" in predicted_class:
            st.error(T["unclassified_error"].format(result=predicted_class))
        else:
            st.error(T["tumor_error"].format(result=predicted_class))
            
        st.metric(label=T["confidence_label"], value=f"{confidence_score:.2f}%")
        
        st.markdown("---")
        
        # --- Expert Advice Section ---
        st.subheader(T["expert_advice_title"])
        
        # Get advice from the rule-based function
        advice_data = get_medical_advice(predicted_class)
        
        st.markdown(f"#### {advice_data['title']}")
        st.markdown(advice_data['advice'])
        
# --- Footer ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; font-size: 14px;'>{T['footer']}</p>", unsafe_allow_html=True)
