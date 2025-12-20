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
        "title": "Brain Tumor Detection (AI-Powered)",
        "sidebar_title": "Settings & Input",
        "language_label": "Select Language",
        "upload_tab": "Upload Image",
        "camera_tab": "Live Camera",
        "upload_help": "Upload a brain MRI image (JPG, PNG, JPEG)",
        "camera_button": "Capture Image",
        "processing": "Processing image...",
        "no_file": "Please upload an image or capture one from the camera.",
        "error_unclear": "Image unclear or not an MRI scan. Please upload a clear brain MRI.",
        "result_header": "Analysis Result",
        "result_yes_title": "Anomaly Detected",
        "result_yes_text": "We have detected an anomaly. Please consult a specialist immediately. Early detection is key to successful treatment. We recommend contacting a neurosurgeon or oncologist.",
        "result_no_title": "Scan Clear",
        "result_no_text": "Great news! The scan looks clear. No tumor detected. The symptoms you are experiencing may be due to other, less serious causes. Please follow up with your primary care physician for a comprehensive check-up.",
        "developer_credit": "Developed by **Oussama SEBROU**",
    },
    "ar": {
        "title": "كشف أورام الدماغ (مدعوم بالذكاء الاصطناعي)",
        "sidebar_title": "الإعدادات والإدخال",
        "language_label": "اختر اللغة",
        "upload_tab": "تحميل صورة",
        "camera_tab": "الكاميرا المباشرة",
        "upload_help": "قم بتحميل صورة رنين مغناطيسي (MRI) للدماغ (JPG, PNG, JPEG)",
        "camera_button": "التقاط الصورة",
        "processing": "جاري معالجة الصورة...",
        "no_file": "الرجاء تحميل صورة أو التقاط واحدة من الكاميرا.",
        "error_unclear": "الصورة غير واضحة أو ليست مسحاً بالرنين المغناطيسي. يرجى تحميل صورة واضحة.",
        "result_header": "نتيجة التحليل",
        "result_yes_title": "تم الكشف عن شذوذ",
        "result_yes_text": "أفهم تماماً حجم القلق الذي تشعر به الآن، والصراحة المهنية تقتضي أن أخبرك بوجود نمو غير طبيعي تظهره الصور، مما يتطلب تحركاً طبياً دقيقاً. لذلك، سنوجهك إلى فريق مختص يجب أن تتابع معه فوراً، يضم نخبة من جراحي الأعصاب وأطباء الأورام لوضع الخطة العلاجية الأنسب لحالتك. أطمئنك بأن العلم الحديث حقق قفزات مذهلة في هذا المجال، ونحن معك خطوة بخطوة لدعمك طبياً ونفسياً. ثق بأن تشخيصنا المبكر هو أول طريق التعافي، وقوتك النفسية ستكون المحرك الأساسي لنجاح رحلة العلاج بإذن الله.",
        "result_no_title": "المسح سليم",
        "result_no_text": "أهنئك من كل قلبي، فنتائج الأشعة والتحاليل جاءت مطمئنة تماماً ولا تظهر أي وجود لورم كما كنت تخشى. الصداع أو الأعراض التي كنت تشعر بها لها أسباب أخرى أبسط بكثير، وسنعمل معاً على معالجتها بهدوء. سنوجهك إلى فريق مختص يجب أن تتابع معه للتأكد من سلامة الجيوب الأنفية أو النظر أو ربما ضغوط الحياة اليومية، لضمان راحتك التامة. عد إلى منزلك وأنت مرتاح البال، فصحتك بخير وهذا هو الخبر الأجمل اليوم.",
        "developer_credit": "تم التطوير بواسطة **Oussama SEBROU**",
    }
}

# --- Custom CSS for UI/UX and RTL Support ---
CUSTOM_CSS = """
<style>
/* Hide Streamlit header/footer */
#MainMenu, footer {visibility: hidden;}

/* Medical Blue/White Theme */
.stApp {
    background-color: #f0f2f6; /* Light background */
    color: #1c1c1c; /* Dark text */
}
.st-emotion-cache-1cypcdb { /* Main content container */
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    margin: auto; /* Center the content */
}
h1 {
    color: #007bff; /* Medical Blue for titles */
    text-align: center;
    font-weight: 600;
}

/* Developer Credit Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #e9ecef;
    color: #6c757d;
    text-align: center;
    padding: 10px;
    font-size: 0.8em;
    border-top: 1px solid #dee2e6;
    z-index: 1000; /* Ensure it's on top */
}
.footer strong {
    color: #007bff;
}

/* RTL Support for Arabic */
.rtl-text {
    direction: rtl;
    text-align: right;
}
/* Ensure sidebar elements are also styled for RTL when needed */
.st-emotion-cache-1lcbmms { /* Target sidebar elements */
    direction: rtl;
    text-align: right;
}
</style>
"""

# --- Model Loading Function ---
@st.cache_resource
def load_model_and_labels():
    """Loads the Keras model and labels file."""
    try:
        # Load the model
        model = tensorflow.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Load the labels
        class_names = []
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                class_names.append(line.strip()[2:]) # Remove the "0 " or "1 " prefix
        
        return model, class_names
    except FileNotFoundError:
        st.error(f"Model files not found. Please ensure '{MODEL_PATH}' and '{LABELS_PATH}' are in the same directory as 'app.py'.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# --- Preprocessing Function ---
def preprocess_image(image):
    """Preprocesses the image for the model."""
    # Resize the image to be at least 224x224 and then crop from the center
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert the image to a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image (Teachable Machine standard)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    # Create the array of the right shape to feed into the Keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

# --- Prediction Function ---
def predict(image):
    """Runs the prediction and returns the result."""
    model, class_names = load_model_and_labels()
    
    # Preprocess
    data = preprocess_image(image)
    
    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# --- Main Application Logic ---
def main():
    # Initialize session state for language if not set
    if 'lang' not in st.session_state:
        st.session_state.lang = 'en'

    # Get the current language messages
    lang = st.session_state.lang
    msg = MESSAGES[lang]
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # --- Sidebar for Language Selection ---
    with st.sidebar:
        st.title(msg["sidebar_title"], anchor=False)
        
        # Language selector
        lang_option = st.radio(
            msg["language_label"],
            ("English", "العربية"),
            index=0 if lang == 'en' else 1,
            key="lang_selector",
            horizontal=True,
            on_change=lambda: st.session_state.update(lang='en' if st.session_state.lang_selector == 'English' else 'ar')
        )
        
        # Re-run the app if language changes to update all text
        if (lang == 'en' and lang_option == 'العربية') or (lang == 'ar' and lang_option == 'English'):
            st.rerun()

    # --- Main Content ---
    st.title(msg["title"])
    
    # Apply RTL class to main content if Arabic is selected
    if lang == 'ar':
        st.markdown('<div class="rtl-text">', unsafe_allow_html=True)

    # Input Tabs
    tab_upload, tab_camera = st.tabs([msg["upload_tab"], msg["camera_tab"]])
    
    uploaded_file = None
    
    with tab_upload:
        uploaded_file = st.file_uploader(msg["upload_help"], type=["jpg", "png", "jpeg"])
        
    with tab_camera:
        camera_image = st.camera_input(msg["camera_button"])
        if camera_image:
            uploaded_file = camera_image

    # --- Prediction Logic ---
    if uploaded_file is not None:
        st.info(msg["processing"])
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        try:
            # Run prediction
            class_name, confidence_score = predict(image)
            
            # Convert confidence to percentage
            confidence_percent = confidence_score * 100
            
            # --- Non-MRI Validation (Heuristic) ---
            if confidence_percent < CONFIDENCE_THRESHOLD * 100:
                st.error(msg["error_unclear"])
            else:
                # --- Display Results ---
                st.header(msg["result_header"], anchor=False)
                
                # Determine the result scenario
                # Assuming the Teachable Machine model has "No" as class 0 and "Yes" as class 1
                # We will rely on the class_name extracted from labels.txt
                
                # Check if the class name contains "Yes" (or the positive indicator)
                is_tumor = "yes" in class_name.lower() or "tumor" in class_name.lower()
                
                if is_tumor:
                    st.markdown(f'<h3 style="color: #dc3545;">{msg["result_yes_title"]}</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 1.1em; line-height: 1.6;">{msg["result_yes_text"]}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<h3 style="color: #28a745;">{msg["result_no_title"]}</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size: 1.1em; line-height: 1.6;">{msg["result_no_text"]}</p>', unsafe_allow_html=True)
                
                st.markdown(f"**Confidence:** {confidence_percent:.2f}%")
                st.markdown(f"**Predicted Class:** {class_name}")
                
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
    
    elif 'lang_selector' in st.session_state:
        # Only show the initial message if no file is uploaded and the language selector has been initialized
        st.info(msg["no_file"])

    # Close RTL div if Arabic is selected
    if lang == 'ar':
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Developer Credit Footer ---
    st.markdown(f'<div class="footer">{msg["developer_credit"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
