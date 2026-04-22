# 😷 Cyberpunk Face Mask Detector

A neural network-powered Streamlit web application that detects human faces and verifies mask compliance in uploaded images. Featuring a custom, high-fidelity Cyberpunk UI with dark/light mode support, neon animations, and real-time inference statistics.

## ✨ Features
- **Deep Learning Inference:** Uses a highly accurate pre-trained SSD face detector and a custom MobileNetV2 classifier to determine mask status.
- **Premium Cyberpunk UI:** Completely custom CSS featuring Orbitron typography, floating neon particles, glowing orbs, and 3D glassmorphism cards.
- **Dark & Light Modes:** Seamlessly toggle between a neon dark mode and a clean light mode layout.
- **Analytics Dashboard:** Instantly view statistics on total faces detected, safe (mask on) counts, and danger (no mask) counts.

## 🚀 Live Demo
*(You can paste your deployed Streamlit Community Cloud URL here once it's live!)*

## 🛠️ Tech Stack
- **Frontend:** Streamlit, Custom HTML/CSS
- **Backend / ML Core:** TensorFlow/Keras, OpenCV, NumPy, Pillow
- **Model Architecture:** MobileNetV2 (transfer learning)

## 💻 Local Setup
If you want to run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dishaasija315/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---
*Developed with modern UI/UX principles and AI computer vision.*
