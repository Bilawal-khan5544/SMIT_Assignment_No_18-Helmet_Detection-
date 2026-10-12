import os
import requests
from ultralytics import YOLO

MODEL_PATH = "best.pt"
MODEL_URL = "https://huggingface.co/Bilawalkhan709/helmet-detection-model/resolve/main/best.pt"

if not os.path.exists(MODEL_PATH):
response = requests.get(MODEL_URL, timeout=300)
response.raise_for_status()
with open(MODEL_PATH, "wb") as f:
f.write(response.content)

model = YOLO(MODEL_PATH)
