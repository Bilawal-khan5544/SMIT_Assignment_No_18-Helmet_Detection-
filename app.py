import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Helmet Detection")

model = YOLO("best.pt")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    result = model(image)[0]
    st.image(result.plot()[:, :, ::-1])
