
import os
import requests
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Helmet Detection", page_icon="🪖")

st.title("AI-Based Helmet Violation Detection")
st.write("Upload an image to detect helmets and potential violations.")

MODEL_PATH = "best.pt"
MODEL_URL = "https://huggingface.co/Bilawalkhan709/helmet-detection-model/resolve/main/best%20%282%29.pt"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, timeout=300)
        response.raise_for_status()

        if len(response.content) < 1000:
            raise RuntimeError("Downloaded model file is too small.")

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    return YOLO(MODEL_PATH)


try:
    with st.spinner("Loading trained model..."):
        model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error("Could not load the model.")
    st.code(str(e))
    st.info(
        "Check that the Hugging Face repository is public "
        "and the model filename in MODEL_URL is correct."
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model.predict(image, conf=0.25)
        result = results[0]
        output_image = result.plot()

    st.subheader("Detection Results")
    st.image(output_image[:, :, ::-1], use_container_width=True)

    if len(result.boxes) > 0:
        st.write("Detected objects:")

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            confidence = float(box.conf[0])

            st.write(f"- {label}: {confidence:.2%} confidence")
    else:
        st.warning("No objects detected. Try another image.")
