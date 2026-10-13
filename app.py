
import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet Detection AI",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #f2f6fc;
}
[data-testid="stSidebar"] {
    background: #e2edf9;
}
.hero {
    background: linear-gradient(120deg, #153659, #287092);
    padding: 35px;
    border-radius: 0 0 22px 22px;
    color: white;
    margin-bottom: 30px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #e4f1ff;
    font-size: 17px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #d9e5f2;
    text-align: center;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #17466e;
}
.metric-label {
    color: #60758b;
}
div.stButton > button {
    background: #17466e;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 44px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #287092;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🪖 HELMET AI")
    st.caption("Computer Vision • YOLO")
    st.divider()

    st.subheader("Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.divider()
    st.subheader("How to use")

    st.markdown("""
    1. Upload a road image.
    2. Click **Run Detection**.
    3. View detected helmets and violations.
    4. Download the result image.
    """)

    st.divider()
    st.caption("AI-Based Helmet Violation Detection")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>AI Helmet Violation Detection</h1>
    <p>AI-powered helmet and road safety monitoring using YOLO</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
# If best.pt is in the GitHub repository, it will load directly.
# Otherwise, it downloads the model from Hugging Face.

MODEL_PATH = "best.pt"

MODEL_URL = (
    "https://huggingface.co/Bilawalkhan709/"
    "helmet-detection-model/resolve/main/best%20%282%29.pt"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(
            MODEL_URL,
            timeout=300,
            allow_redirects=True
        )
        response.raise_for_status()

        if len(response.content) < 1000:
            raise RuntimeError(
                "Downloaded model file is too small. "
                "Check the Hugging Face model filename."
            )

        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)

    return YOLO(MODEL_PATH)

try:
    with st.spinner("Loading trained YOLO model..."):
        model = load_model()
except Exception as error:
    st.error("Model could not be loaded.")
    st.code(str(error))
    st.info(
        "Check that your Hugging Face repository is public "
        "and MODEL_URL contains the exact model filename."
    )
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
st.subheader("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload an image to detect helmet violations",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Detection shuru karne ke liye road image upload karein.")
    st.caption("Supported formats: JPG, JPEG, PNG")

else:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Run Detection", use_container_width=True):
        with st.spinner("Analyzing image..."):
            image_array = np.array(image)

            results = model.predict(
                source=image_array,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # YOLO plot returns BGR; convert to RGB for Streamlit.
            annotated_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(
                annotated_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.session_state["annotated_image"] = annotated_rgb
            st.session_state["detections"] = [
                {
                    "label": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                }
                for box in result.boxes
            ]

    # Show results after detection
    if "annotated_image" in st.session_state:
        with right:
            st.markdown("### Detection Results")
            st.image(
                st.session_state["annotated_image"],
                use_container_width=True
            )

        detections = st.session_state["detections"]

        st.divider()
        st.subheader("📊 Detection Summary")

        total = len(detections)

        # Change these keywords if your model uses different class names.
        no_helmet = sum(
            1 for item in detections
            if "no_helmet" in item["label"].lower()
            or "without_helmet" in item["label"].lower()
            or "no helmet" in item["label"].lower()
        )

        helmet = sum(
            1 for item in detections
            if "helmet" in item["label"].lower()
            and "no_helmet" not in item["label"].lower()
            and "without_helmet" not in item["label"].lower()
            and "no helmet" not in item["label"].lower()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{helmet}</div>
                <div class="metric-label">Helmet Detections</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{no_helmet}</div>
                <div class="metric-label">Possible Violations</div>
            </div>
            """, unsafe_allow_html=True)

        if detections:
            st.markdown("### Detection Details")

            for index, item in enumerate(detections, start=1):
                st.write(
                    f"{index}. **{item['label']}** — "
                    f"Confidence: {item['confidence']:.1%}"
                )
        else:
            st.warning(
                "No objects detected. Try another image or "
                "adjust the confidence threshold."
            )

        # Download annotated image
        success, encoded_image = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                st.session_state["annotated_image"],
                cv2.COLOR_RGB2BGR
            )
        )

        if success:
            st.download_button(
                "⬇️ Download Detection Image",
                data=encoded_image.tobytes(),
                file_name="helmet_detection_result.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
