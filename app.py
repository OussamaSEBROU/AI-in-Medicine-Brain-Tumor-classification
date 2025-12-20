import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuration and Setup ---
st.set_page_config(
    page_title="AI Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Labels (from original code) ---
@st.cache_resource
def load_model():
    # Assuming 'keras_model.h5' is available in the root directory
    model = tf.keras.models.load_model('keras_model.h5')
    return model

@st.cache_data
def load_labels():
    # Assuming 'labels.txt' is available in the root directory
    try:
        with open('labels.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        st.error("Error: 'labels.txt' not found. Please ensure it is in the same directory.")
        # Placeholder labels based on common datasets - USER MUST VERIFY THESE MATCH THEIR MODEL
        labels = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"] 
    return labels

model = load_model()
labels = load_labels()

# --- Expert Medical Advice Database (Rule-Based System) ---
# NOTE: The keys in this dictionary MUST exactly match the labels in your 'labels.txt' file.
# The advice is written in Arabic as requested.
MEDICAL_ADVICE_DB = {
    "No Tumor": {
        "title": "نتائج مطمئنة: لا يوجد ورم",
        "advice": """
        **تهانينا!** تشير نتائج التحليل إلى عدم وجود أي ورم دماغي مصنف.
        
        **نصيحة الطبيب الخبير:**
        *   **المتابعة الروتينية:** على الرغم من النتيجة السلبية، يُنصح دائماً بمتابعة الفحوصات الروتينية الدورية التي يحددها طبيبك.
        *   **نمط الحياة الصحي:** حافظ على نمط حياة صحي، بما في ذلك التغذية المتوازنة، وممارسة الرياضة بانتظام، والنوم الكافي، للوقاية العامة.
        *   **إخلاء مسؤولية:** تذكر أن هذا التطبيق هو أداة مساعدة للفرز الأولي، ويجب عليك دائماً استشارة طبيب مختص لتأكيد التشخيص وقراءة صور الرنين المغناطيسي بشكل كامل.
        """
    },
    "Glioma Tumor": {
        "title": "ورم دبقي (Glioma): يتطلب تدخلاً عاجلاً",
        "advice": """
        **تنبيه هام:** يشير التحليل إلى ورم دبقي محتمل. الأورام الدبقية تنشأ من الخلايا الدبقية في الدماغ.
        
        **نصيحة الطبيب الخبير:**
        *   **استشارة فورية:** يجب مراجعة جراح الأعصاب أو طبيب الأورام العصبية فوراً.
        *   **التشخيص النهائي:** يتطلب هذا النوع من الأورام عادةً فحوصات إضافية مثل الخزعة (Biopsy) وتصوير متقدم لتحديد درجة الورم (Grade).
        *   **خيارات العلاج:** قد تشمل الجراحة لإزالة أكبر قدر ممكن من الورم، والعلاج الإشعاعي، والعلاج الكيميائي، اعتماداً على نوع ودرجة الورم.
        """
    },
    "Meningioma Tumor": {
        "title": "ورم سحائي (Meningioma): غالباً حميد",
        "advice": """
        **نتيجة إيجابية:** يشير التحليل إلى ورم سحائي محتمل. الأورام السحائية تنشأ من الأغشية المحيطة بالدماغ والحبل الشوكي، وغالبيتها حميدة.
        
        **نصيحة الطبيب الخبير:**
        *   **المراقبة اليقظة:** العديد من الأورام السحائية تنمو ببطء وقد لا تحتاج إلى علاج فوري، بل إلى "المراقبة اليقظة" (Watchful Waiting) مع تصوير دوري.
        *   **التدخل الجراحي:** إذا كان الورم كبيراً أو يسبب أعراضاً عصبية، فقد تكون الجراحة هي الخيار الأول.
        *   **التأكيد:** يجب تأكيد التشخيص من قبل طبيب مختص لتحديد خطة المتابعة أو العلاج المناسبة لحالتك.
        """
    },
    "Pituitary Tumor": {
        "title": "ورم الغدة النخامية (Pituitary): تقييم هرموني ضروري",
        "advice": """
        **نتيجة إيجابية:** يشير التحليل إلى ورم محتمل في الغدة النخامية. هذه الأورام قد تؤثر على إفراز الهرمونات.
        
        **نصيحة الطبيب الخبير:**
        *   **تقييم الغدد الصماء:** يجب مراجعة طبيب الغدد الصماء لتقييم مستويات الهرمونات في الجسم، حيث أن العديد من هذه الأورام تفرز هرمونات بشكل مفرط.
        *   **التصوير المتقدم:** قد يحتاج الأمر إلى تصوير رنين مغناطيسي متخصص للغدة النخامية.
        *   **العلاج:** قد يشمل العلاج الأدوية (خاصة للأورام المفرزة للبرولاكتين)، أو الجراحة (عبر الأنف في الغالب)، أو العلاج الإشعاعي.
        """
    }
}

def get_medical_advice(result_class):
    # Fallback for any unhandled label
    default_advice = {
        "title": "تنبيه: نتيجة غير مصنفة",
        "advice": """
        تم تصنيف الصورة بنجاح، ولكن النتيجة غير موجودة في قاعدة بيانات النصائح الطبية.
        **الخطوة التالية:** يرجى مراجعة طبيب مختص على الفور لمناقشة النتيجة: **{result_class}**.
        """
    }
    
    advice_data = MEDICAL_ADVICE_DB.get(result_class, default_advice)
    
    # If it's the default advice, format the text to include the result class
    if advice_data is default_advice:
        advice_data['advice'] = advice_data['advice'].format(result_class=result_class)
        
    return advice_data


# --- Sidebar Content (Professional Look) ---
with st.sidebar:
    st.title("🧠 مصنف أورام الدماغ بالذكاء الاصطناعي")
    st.markdown("---")
    
    st.header("🔍 نظرة عامة على المشروع")
    st.info("""
        هذا التطبيق هو أداة مساعدة تعتمد على **التعلم العميق (Deep Learning)** لتصنيف صور الرنين المغناطيسي (MRI) للدماغ.
        
        *   **النموذج:** تم التدريب باستخدام شبكة عصبية تلافيفية (CNN) عبر مكتبة Keras/TensorFlow.
        *   **الهدف:** المساعدة في الكشف الأولي وتصنيف أنواع أورام الدماغ.
    """)
    
    st.header("💡 كيفية الاستخدام")
    st.markdown("""
        1.  **تحميل الصورة:** استخدم زر "اختر صورة رنين مغناطيسي للدماغ..." لتحميل صورة رنين مغناطيسي للدماغ.
        2.  **التصنيف:** سيقوم الذكاء الاصطناعي بتحليل الصورة وتقديم نتيجة التصنيف ونسبة الثقة.
        3.  **النصيحة الطبية:** ستظهر نصيحة طبية مفصلة ومبنية على النتيجة لتوجيهك نحو الخطوات التالية.
    """)
    
    st.markdown("---")
    st.header("⚠️ إخلاء مسؤولية طبي")
    st.warning("""
        **هذا التطبيق ليس بديلاً عن التشخيص الطبي الاحترافي.**
        
        النتائج المقدمة هي لأغراض إعلامية ومساعدة فقط. يجب دائماً استشارة طبيب مختص أو أخصائي أشعة لتأكيد أي تشخيص أو اتخاذ قرارات علاجية.
    """)
    
    st.markdown("---")
    st.markdown("Developed by **Oussama SEBROU** | AI-in-Medicine Project")


# --- Main Application Layout ---
st.title("تصنيف أورام الدماغ (Brain Tumor Classification)")
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("🖼️ اختر صورة رنين مغناطيسي للدماغ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use columns for a cleaner layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='صورة الرنين المغناطيسي المحملة', use_column_width=True)
        
    with col2:
        st.subheader("📊 نتيجة التحليل")
        with st.spinner('جاري تحليل الصورة وتصنيف الورم...'):
            # Preprocess the image
            size = (224, 224)
            image = image.resize(size)
            image_array = np.asarray(image)
            
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
            st.success(f"✅ النتيجة: {predicted_class}")
        elif "Unclassified" in predicted_class:
            st.error(f"❌ النتيجة: {predicted_class} - حدث خطأ في التصنيف أو قراءة النتائج.")
        else:
            st.error(f"❌ النتيجة: {predicted_class}")
            
        st.metric(label="نسبة الثقة", value=f"{confidence_score:.2f}%")
        
        st.markdown("---")
        
        # --- Expert Advice Section ---
        st.subheader("👨‍⚕️ نصيحة الطبيب الخبير")
        
        # Get advice from the rule-based function
        advice_data = get_medical_advice(predicted_class)
        
        st.markdown(f"#### {advice_data['title']}")
        st.markdown(advice_data['advice'])
        
# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Oussama SEBROU** | AI-in-Medicine Project")
