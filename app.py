import streamlit as st
import torch
import numpy as np
import torchvision.models as models
import requests
import cv2
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os
import gdown
from PIL import Image, ImageDraw, ImageFont


MODEL_PATH = "resnet_emotion.pt"
MODEL_URL =  "https://drive.google.com/uc?export=download&id=18Fu7zBTW0vZzJDq9UHVvjugU56QXqhs-"


if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


# Load a Unicode-capable font (DejaVuSans is bundled with many OSes)
font = ImageFont.truetype("DejaVuSans.ttf", 32)
def detect_and_predict_faces(frame, limit_faces=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    if limit_faces:
        faces = faces[:limit_faces]

    predictions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face_img = Image.fromarray(roi_gray).convert("L").resize((48, 48))
        tensor = transform_train(face_img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            label = class_names[predicted.item()]
            emoji = get_emoji(label)
            predictions.append(((x, y, w, h), label, emoji))

    return predictions

device = torch.device("cpu")
model = models.resnet18()
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 7)


model.load_state_dict(torch.load("resnet_emotion.pt", map_location=device))  # No weights_only!
model.to(device)
model.eval()


# Emojis for each class
EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

def get_emoji(label):
    return EMOJI_MAP.get(label.lower(), "")

# Transformation

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Page config
st.set_page_config(page_title="Facial Expression Recognition", layout="centered")

# DARK MODE TOGGLE
dark_mode = st.sidebar.toggle("🌗 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stApp {
            background-color: #121212;
        }
        </style>
    """, unsafe_allow_html=True)

# App title
st.title("Facial Expression Recognition")

# Choose mode
mode = st.sidebar.radio("Choose mode", ["Upload Image", "Webcam"])

# Prediction function
def predict_emotion(img):
    img = transform_train(img).unsqueeze(0)
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    emotion = class_names[predicted.item()]
    return emotion, EMOJI_MAP.get(emotion.lower(), "")

# Upload image mode
if mode == "Webcam":
    st.markdown("⚠️ **Please allow camera access in your browser when prompted.For better results, choose plane background**")
    st.markdown("✅ Uncheck the box to stop the webcam and release it safely.")

    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    run = st.checkbox("Start Webcam", value=st.session_state.run_webcam)

    if run:
        st.session_state.run_webcam = True
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)

        try:
            while st.session_state.run_webcam:
                ret, frame = cap.read()
                if not ret:
                    break

                results = detect_and_predict_faces(frame, limit_faces=3)
                st.image(frame)
                ret, frame = cap.read()
                if ret:
    # Convert OpenCV frame to RGB for PIL
                    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    draw = ImageDraw.Draw(image_pil)
                    font = ImageFont.truetype("DejaVuSans.ttf", 32)

                    for (x, y, w, h), label, emoji in results:
                        draw.rectangle([(x, y), (x+w, y+h)], outline="green", width=2)
                        draw.text((x, y - 30), f"{label.upper()} {emoji}", font=font, fill="green")

    # Convert back to NumPy for Streamlit
                    frame = np.array(image_pil)
                    FRAME_WINDOW.image(frame)
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            cap.release()
            st.session_state.run_webcam = False




# Webcam mode
elif mode == "Upload Image":
    file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if file:
        img = np.array(Image.open(file).convert("RGB"))
        results = detect_and_predict_faces(img)

        for (x, y, w, h), label, emoji in results:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label.upper()} {emoji}"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        st.image(img, caption="Detected Faces with Emotions 🎭")

