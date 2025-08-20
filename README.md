# 🎭 Emotion Detection AI (Facial Expression Recognition)

A **Streamlit** web app that detects facial expressions using a **PyTorch ResNet18** model trained on **FER2013** (48×48 grayscale faces).  
Supports **image uploads** (cloud + local) and **webcam** (local only).  
Draws bounding boxes, labels, and **emojis** on faces.

---

## ✨ Features

- 🧠 ResNet18 backbone (first conv adapted to 1-channel, 7-class head)
- 🖼️ Image upload (works everywhere)
- 🎥 Webcam (local desktops only, not on Streamlit Cloud)
- 😀 Emoji overlay and label above each face
- 🌗 Optional dark mode
- 👥 Multi-face support (3 faces for webcam, all faces for images)
- 🧰 Ready to deploy to **Streamlit Cloud**
- 📦 Large model handled with **Git LFS** (recommended) or **gdown** from Drive

---

## 🗂️ Project Structure
facial_expression_app/

     ├─ app.py
     ├─ data_loader.py # (if you train locally)
     ├─ train.py # (if you train locally)
     ├─ haarcascade_frontalface_default.xml # optional; OpenCV provides path too
     ├─ DejaVuSans.ttf # emoji-capable font (optional; app can fallback)
     │
     ├─ model/
          │ └─ resnet_emotion.pt # tracked via Git LFS (recommended)
     │
     ├─ requirements.txt
     ├─ .gitattributes # Git LFS config
     └─ README.md
      

---

## ⚙️ Prerequisites

- Python **3.10+** (works on 3.13 too)
- Pip / venv (recommended)
- Git (**Git LFS** for large model file)
- (Local webcam only) A camera & desktop OS

---

## 📦 Installation


### 1) Clone

     git clone https://github.com/Osejan/facial_expression_app.git
     cd facial_expression_app

### 2) (Optional) Create & activate venv
     python -m venv .venv
#### Windows
    .venv\Scripts\activate
#### macOS/Linux
    source .venv/bin/activate

### 3) Install deps
    pip install -r requirements.txt

### git lfs install
    git lfs pull   
pulls the .pt file if the pointer exists in repo

## ▶️ Run the App (Local)

1) streamlit run app.py
2) Open the link shown (usually http://localhost:8501)
3) In the sidebar, choose Upload Image or Webcam
(Webcam only works locally; not on Streamlit Cloud)

## 🙏 Acknowledgements

FER2013 Dataset (Kaggle)
PyTorch / TorchVision
OpenCV
Streamlit
