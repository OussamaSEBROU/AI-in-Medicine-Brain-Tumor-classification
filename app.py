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
        "sidebar_title": "🧠 مصنف أورام الدماغ بالذكاء الاصطناعي",
        "lang_select": "اختر اللغة",
        "upload_option": "تحميل صورة",
        "camera_option": "التقاط صورة مباشرة",
        "upload_file_prompt": "🖼️ اختر صورة رنين مغناطيسي للدماغ...",
        "camera_prompt": "📸 التقاط صورة مباشرة (MRI)",
        "uploaded_caption": "صورة الرنين المغناطيسي المحملة",
        "analysis_result": "📊 نتيجة التحليل",
        "analysis_spinner": "جاري تحليل الصورة وتصنيف الورم...",
        "confidence_label": "نسبة الثقة",
        "expert_advice_title": "👨‍⚕️ نصيحة الطبيب الخبير والتوجيه",
        "unclassified_error": "❌ النتيجة: {result} - حدث خطأ في التصنيف أو قراءة النتائج.",
        "no_tumor_success": "✅ النتيجة: {result}",
        "tumor_error": "❌ النتيجة: {result}",
        "sidebar_overview_header": "🔍 نظرة عامة على المشروع",
        "sidebar_overview_info": """
            هذا التطبيق هو أداة مساعدة تعتمد على **التعلم العميق (Deep Learning)** لتصنيف صور الرنين المغناطيسي (MRI) للدماغ.
            
            *   **النموذج:** تم التدريب باستخدام شبكة عصبية تلافيفية (CNN) عبر مكتبة Keras/TensorFlow.
            *   **الهدف:** المساعدة في الكشف الأولي وتصنيف أنواع أورام الدماغ.
        """,
        "sidebar_usage_header": "💡 كيفية الاستخدام",
        "sidebar_usage_info": """
            1.  **اختر الطريقة:** اختر بين تحميل صورة من جهازك أو التقاط صورة مباشرة.
            2.  **التصنيف:** سيقوم الذكاء الاصطناعي بتحليل الصورة وتقديم نتيجة التصنيف ونسبة الثقة.
            3.  **النصيحة الطبية:** ستظهر نصيحة طبية مفصلة ومبنية على النتيجة لتوجيهك نحو الخطوات التالية.
        """,
        "sidebar_disclaimer_header": "⚠️ إخلاء مسؤولية طبي",
        "sidebar_disclaimer_warning": """
            **هذا التطبيق ليس بديلاً عن التشخيص الطبي الاحترافي.**
            
            النتائج المقدمة هي لأغراض إعلامية ومساعدة فقط. يجب دائماً استشارة طبيب مختص أو أخصائي أشعة لتأكيد أي تشخيص أو اتخاذ قرارات علاجية.
        """,
        "footer": "تم التطوير بواسطة **Oussama SEBROU** | مشروع الذكاء الاصطناعي في الطب",
        "advice_db": {
            "No Tumor": {
                "title": "نتائج مطمئنة: لا يوجد ورم",
                "advice": """
                **خبر سار ومريح!** تشير نتائج التحليل إلى أن الصورة لا تحمل أي علامات للورم المصنف. هذا يبعث على الاطمئنان.
                
                **نصيحة الطبيب الخبير والتوجيه النفسي:**
                *   **راحة البال:** استمتع بهذه النتيجة الإيجابية، ولكن تذكر أن العناية بالصحة رحلة مستمرة.
                *   **المتابعة الوقائية:** يُنصح بمتابعة الفحوصات الروتينية التي يحددها طبيبك العام كإجراء وقائي.
                *   **الحياة الصحية:** حافظ على نشاطك، وتغذيتك، ونومك الجيد. صحة الدماغ تبدأ من نمط الحياة.
                *   **تذكير هام:** هذا التطبيق أداة مساعدة، والكلمة الفصل دائماً للطبيب المختص الذي يقرأ الصورة بالكامل.
                """
            },
            "Glioma Tumor": {
                "title": "ورم دبقي (Glioma): خطوة أولى نحو العلاج",
                "advice": """
                **تنبيه هام:** يشير التحليل إلى احتمال وجود ورم دبقي. من الطبيعي أن تشعر بالقلق، ولكن تذكر أن هذا التشخيص هو **الخطوة الأولى نحو العلاج الفعال**.
                
                **نصيحة الطبيب الخبير والتوجيه النفسي:**
                *   **التصرف الفوري بهدوء:** أهم خطوة الآن هي **مراجعة جراح أعصاب أو طبيب أورام عصبية متخصص بأسرع وقت ممكن**. لا تتردد، فالتشخيص المبكر يفتح آفاقاً أوسع للعلاج.
                *   **التركيز على العلاج:** هناك فرق كبير بين التشخيص والحكم النهائي. الطب الحديث يوفر خيارات علاجية متقدمة (جراحة، إشعاع، كيماوي). فريقك الطبي هو أفضل من يحدد الخطة المناسبة لك.
                *   **الدعم النفسي:** لا تخض هذه التجربة وحدك. تحدث مع عائلتك وأصدقائك، واطلب الدعم النفسي. القوة الداخلية هي جزء أساسي من رحلة العلاج.
                """
            },
            "Meningioma Tumor": {
                "title": "ورم سحائي (Meningioma): غالباً حميد وخيارات متعددة",
                "advice": """
                **نتيجة تتطلب المتابعة:** يشير التحليل إلى ورم سحائي محتمل. الخبر الجيد هو أن **الغالبية العظمى من هذه الأورام حميدة** وتنمو ببطء شديد.
                
                **نصيحة الطبيب الخبير والتوجيه النفسي:**
                *   **الهدوء والمراقبة:** في كثير من الحالات، لا يتطلب الورم السحائي علاجاً فورياً، بل "المراقبة اليقظة" مع تصوير دوري.
                *   **استشر خبيراً:** يجب مراجعة طبيب أعصاب أو جراح أعصاب لتأكيد نوع الورم وتحديد ما إذا كان يتطلب تدخلاً جراحياً أو إشعاعياً، أو مجرد متابعة.
                *   **التفاؤل:** احتمالية الشفاء والتعايش مع هذا النوع من الأورام عالية جداً. كن إيجابياً وتابع مع طبيبك.
                """
            },
            "Pituitary Tumor": {
                "title": "ورم الغدة النخامية (Pituitary): تقييم هرموني ضروري",
                "advice": """
                **نتيجة إيجابية:** يشير التحليل إلى ورم محتمل في الغدة النخامية. هذه الأورام غالباً ما تكون حميدة، ولكنها قد تؤثر على التوازن الهرموني في الجسم.
                
                **نصيحة الطبيب الخبير والتوجيه النفسي:**
                *   **التخصص هو الحل:** يجب مراجعة طبيب **غدد صماء** فوراً لتقييم مستويات الهرمونات، وطبيب أعصاب أو جراح أعصاب.
                *   **العلاج غير الجراحي:** العديد من أورام الغدة النخامية تستجيب بشكل ممتاز للعلاج الدوائي دون الحاجة للجراحة.
                *   **التركيز على التوازن:** الهدف هو استعادة التوازن الهرموني. كن واثقاً بأن الأطباء سيجدون الخطة العلاجية التي تناسب حالتك.
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
        "sidebar_title": "🧠 AI Brain Tumor Classifier",
        "lang_select": "Select Language",
        "upload_option": "Upload Image",
        "camera_option": "Capture Live Image",
        "upload_file_prompt": "🖼️ Choose a brain MRI image...",
        "camera_prompt": "📸 Capture Live Image (MRI)",
        "uploaded_caption": "Uploaded MRI Image",
        "analysis_result": "📊 Analysis Result",
        "analysis_spinner": "Analyzing image and classifying tumor...",
        "confidence_label": "Confidence Score",
        "expert_advice_title": "👨‍⚕️ Expert Medical Advice and Guidance",
        "unclassified_error": "❌ Result: {result} - An error occurred during classification or result reading.",
        "no_tumor_success": "✅ Result: {result}",
        "tumor_error": "❌ Result: {result}",
        "sidebar_overview_header": "🔍 Project Overview",
        "sidebar_overview_info": """
            This application is an auxiliary tool based on **Deep Learning** to classify brain Magnetic Resonance Imaging (MRI) scans.
            
            *   **Model:** Trained using a Convolutional Neural Network (CNN) via the Keras/TensorFlow library.
            *   **Goal:** To assist in the initial detection and classification of brain tumor types.
        """,
        "sidebar_usage_header": "💡 How to Use",
        "sidebar_usage_info": """
            1.  **Select Method:** Choose between uploading an image from your device or capturing a live image.
            2.  **Classification:** The AI will analyze the image and provide the classification result and confidence score.
            3.  **Medical Advice:** Detailed medical advice based on the result will appear to guide you on the next steps.
        """,
        "sidebar_disclaimer_header": "⚠️ Medical Disclaimer",
        "sidebar_disclaimer_warning": """
            **This application is NOT a substitute for professional medical diagnosis.**
            
            The results provided are for informational and assistive purposes only. You must always consult a specialized physician or radiologist to confirm any diagnosis or make treatment decisions.
        """,
        "footer": "Developed by **Oussama SEBROU** | AI-in-Medicine Project",
        "advice_db": {
            "No Tumor": {
                "title": "Reassuring Results: No Tumor Found",
                "advice": """
                **Great and Reassuring News!** The analysis indicates that the image shows no signs of the classified tumor types. This is a source of relief.
                
                **Expert Medical Advice and Psychological Guidance:**
                *   **Peace of Mind:** Enjoy this positive result, but remember that health care is an ongoing journey.
                *   **Preventive Follow-up:** Routine check-ups recommended by your general practitioner are advised as a preventive measure.
                *   **Healthy Living:** Maintain your activity, nutrition, and good sleep. Brain health starts with lifestyle.
                *   **Important Reminder:** This application is an auxiliary tool, and the final word always belongs to the specialized physician who reads the entire scan.
                """
            },
            "Glioma Tumor": {
                "title": "Glioma Tumor: The First Step Towards Treatment",
                "advice": """
                **Important Alert:** The analysis indicates a potential Glioma tumor. It is normal to feel anxious, but remember that this diagnosis is **the first step towards effective treatment**.
                
                **Expert Medical Advice and Psychological Guidance:**
                *   **Immediate Action with Calm:** The most important step now is to **consult a neurosurgeon or specialized neuro-oncologist as soon as possible**. Do not hesitate; early diagnosis opens up wider horizons for treatment.
                *   **Focus on Treatment:** There is a big difference between diagnosis and final judgment. Modern medicine offers advanced treatment options (surgery, radiation, chemotherapy). Your medical team is the best to determine the right plan for you.
                *   **Psychological Support:** Do not go through this experience alone. Talk to your family and friends, and seek psychological support. Inner strength is an essential part of the treatment journey.
                """
            },
            "Meningioma Tumor": {
                "title": "Meningioma Tumor: Often Benign with Multiple Options",
                "advice": """
                **A Result Requiring Follow-up:** The analysis indicates a potential Meningioma tumor. The good news is that **the vast majority of these tumors are benign** and grow very slowly.
                
                **Expert Medical Advice and Psychological Guidance:**
                *   **Calm and Monitoring:** In many cases, a Meningioma does not require immediate treatment, but rather "Watchful Waiting" with periodic imaging.
                *   **Consult an Expert:** You must consult a neurologist or neurosurgeon to confirm the tumor type and determine if it requires surgical or radiation intervention, or just monitoring.
                *   **Optimism:** The probability of recovery and living with this type of tumor is very high. Be positive and follow up with your doctor.
                """
            },
            "Pituitary Tumor": {
                "title": "Pituitary Tumor: Hormonal Evaluation is Essential",
                "advice": """
                **Positive Result:** The analysis indicates a potential Pituitary tumor. These tumors are often benign, but they can affect the body's hormonal balance.
                
                **Expert Medical Advice and Psychological Guidance:**
                *   **Specialization is the Key:** You must consult an **endocrinologist** immediately to evaluate hormone levels, and a neurologist or neurosurgeon.
                *   **Non-Surgical Treatment:** Many pituitary tumors respond excellently to medical treatment without the need for surgery.
                *   **Focus on Balance:** The goal is to restore hormonal balance. Be confident that doctors will find the treatment plan that suits your condition.
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
    model = tf.keras.models.load_model('keras_model.h5')
    return model

@st.cache_data
def load_labels():
    try:
        with open('labels.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error(T["unclassified_error"].format(result="'labels.txt' not found"))
        # Placeholder labels based on common datasets - USER MUST VERIFY THESE MATCH THEIR MODEL
        labels = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"] 
    return labels

model = load_model()
labels = load_labels()

# --- Expert Medical Advice Function ---
def get_medical_advice(result_class):
    advice_data = T["advice_db"].get(result_class, T["advice_db"]["Unclassified"])
    
    if advice_data is T["advice_db"]["Unclassified"]:
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
    
    st.markdown("---")
    st.markdown(T["footer"])


# --- Main Application Layout ---
st.title(T["title"])
st.markdown("---")

# --- Image Input Options ---
input_method = st.radio(
    "**1. اختر طريقة إدخال الصورة:**" if st.session_state.lang == "ar" else "**1. Select Image Input Method:**",
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
st.markdown(T["footer"])
