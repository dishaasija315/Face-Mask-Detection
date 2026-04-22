import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='collapsed')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_resource
def load_models():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.model")
    return net, model


def process_image(image_path, net, model):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces_detected = 0
    masks_on = 0
    masks_off = 0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            faces_detected += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask":
                masks_on += 1
            else:
                masks_off += 1

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            conf_pct = max(mask, withoutMask) * 100
            display_label = "{}: {:.1f}%".format(label, conf_pct)
            cv2.putText(image, display_label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)

    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RGB_img, faces_detected, masks_on, masks_off


def mask_detection():
    local_css("css/styles.css")
    net, model = load_models()

    # ===== Dark/Light Mode Toggle =====
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    col_spacer, col_toggle = st.columns([6, 1])
    with col_toggle:
        mode_label = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(mode_label, key="theme_toggle", help="Toggle Dark/Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # Inject light-mode class if needed
    if not st.session_state.dark_mode:
        st.markdown('<style>.stApp { background: #f0f2f6 !important; } .stApp::before { content: ""; } </style>', unsafe_allow_html=True)
        st.markdown('<div class="light-mode-active"></div>', unsafe_allow_html=True)
        st.markdown('''<style>
            /* ===== LIGHT MODE OVERRIDES ===== */
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f0 50%, #f0f2f6 100%) !important;
                background-image: 
                    radial-gradient(ellipse at 20% 50%, rgba(0,180,120,0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 20%, rgba(0,140,200,0.06) 0%, transparent 50%) !important;
            }
            .hero-wrap, .hero-wrap * { color: #1a1a2e !important; }
            .hero-title {
                -webkit-text-fill-color: transparent !important;
                background: linear-gradient(135deg, #00805a 0%, #006090 40%, #5a20b0 80%, #c01060 100%) !important;
                -webkit-background-clip: text !important; background-clip: text !important;
            }
            .hero-desc, .hero-desc p { color: #4a5568 !important; }
            .hero-desc strong { color: #00805a !important; }
            .hero-badge {
                background: rgba(0,180,120,0.1) !important; border-color: rgba(0,180,120,0.3) !important;
                color: #00805a !important;
            }
            .hero-badge::before { background: #00805a !important; }

            /* Cards */
            .step-box, .upload-zone, .result-stat {
                background: rgba(255,255,255,0.8) !important;
                border-color: rgba(0,0,0,0.08) !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.06) !important;
            }
            .step-box:hover {
                box-shadow: 0 8px 30px rgba(0,0,0,0.1) !important;
                border-color: rgba(0,180,120,0.3) !important;
            }
            .step-box h4 { color: #1a1a2e !important; }
            .step-box p { color: #6b7280 !important; }
            .step-icon { background: rgba(0,180,120,0.1) !important; border-color: rgba(0,180,120,0.2) !important; }

            /* Section headers */
            .section-hdr .label {
                color: #00805a !important;
                border-color: rgba(0,180,120,0.3) !important;
                background: rgba(0,180,120,0.08) !important;
            }
            .section-hdr .line { background: linear-gradient(90deg, #00805a, transparent) !important; }
            .section-hdr .line.right { background: linear-gradient(90deg, transparent, #006090) !important; }

            /* Upload zone */
            .upload-zone h3 { color: #1a1a2e !important; }
            .upload-hint { color: #6b7280 !important; }
            [data-testid="stFileUploader"] > div {
                background: rgba(0,180,120,0.03) !important;
                border-color: rgba(0,180,120,0.2) !important;
            }
            [data-testid="stFileUploader"] > div:hover {
                border-color: rgba(0,180,120,0.4) !important;
                background: rgba(0,180,120,0.06) !important;
            }

            /* Button */
            .stButton > button {
                color: #00805a !important; border-color: #00805a !important;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #00805a, #006090) !important;
                color: white !important;
            }

            /* Messages */
            .success-msg { color: #00805a !important; background: rgba(0,180,120,0.08) !important; border-color: rgba(0,180,120,0.2) !important; }
            .warn-msg { color: #c01060 !important; background: rgba(192,16,96,0.06) !important; border-color: rgba(192,16,96,0.2) !important; }
            .info-msg { color: #006090 !important; background: rgba(0,96,144,0.06) !important; border-color: rgba(0,96,144,0.2) !important; }

            /* Result stats */
            .result-stat.faces .val { color: #006090 !important; text-shadow: none !important; }
            .result-stat.safe .val { color: #00805a !important; text-shadow: none !important; }
            .result-stat.danger .val { color: #c01060 !important; text-shadow: none !important; }
            .result-stat .lbl { color: #6b7280 !important; }

            /* Image */
            [data-testid="stImage"] img { border-color: rgba(0,0,0,0.08) !important; }

            /* Footer */
            .app-footer p { color: #9ca3af !important; }
            .footer-tech span { color: #6b7280 !important; border-color: rgba(0,0,0,0.08) !important; }

            /* Particles & orbs hidden in light mode */
            .particles, .orb { opacity: 0.3 !important; }

            /* Scrollbar */
            ::-webkit-scrollbar-track { background: #f0f2f6 !important; }
            ::-webkit-scrollbar-thumb { background: rgba(0,180,120,0.3) !important; }
        </style>''', unsafe_allow_html=True)

    # ===== Floating Particles & Orbs =====
    st.markdown('''
    <div class="particles">
        <div class="particle p1"></div><div class="particle p2"></div>
        <div class="particle p3"></div><div class="particle p4"></div>
        <div class="particle p5"></div><div class="particle p6"></div>
        <div class="particle p7"></div><div class="particle p8"></div>
    </div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    ''', unsafe_allow_html=True)

    # ===== Hero Section =====
    st.markdown('''
    <div class="hero-wrap">
        <div class="hero-badge">● NEURAL NETWORK POWERED</div>
        <div class="hero-title">FACE MASK<br>DETECTOR</div>
        <p class="hero-desc">
            Upload any image and our <strong>deep learning AI</strong> will instantly 
            scan every face and determine mask compliance with surgical precision.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # ===== How It Works =====
    st.markdown('''
    <div class="section-hdr">
        <div class="line"></div>
        <div class="label">⚡ HOW IT WORKS</div>
        <div class="line right"></div>
    </div>
    <div class="steps-grid">
        <div class="step-box">
            <div class="step-icon">📤</div>
            <h4>UPLOAD</h4>
            <p>Drop any image with faces — JPG, JPEG or PNG</p>
        </div>
        <div class="step-box">
            <div class="step-icon">🧠</div>
            <h4>ANALYZE</h4>
            <p>MobileNetV2 neural network scans every detected face</p>
        </div>
        <div class="step-box">
            <div class="step-icon">📊</div>
            <h4>RESULTS</h4>
            <p>Get instant verdicts with confidence percentages</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ===== Upload Section =====
    st.markdown('''
    <div class="section-hdr">
        <div class="line"></div>
        <div class="label">🎯 SCAN IMAGE</div>
        <div class="line right"></div>
    </div>
    <div class="upload-zone">
        <h3>DROP YOUR IMAGE</h3>
        <p class="upload-hint">Supported: JPG • JPEG • PNG  |  Max: 200MB</p>
    </div>
    ''', unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if image_file is not None:
        our_image = Image.open(image_file)
        our_image.save('./images/out.jpg')

        st.markdown('''
        <div class="success-msg">✅ Image locked and loaded — hit <strong>INITIATE SCAN</strong> to analyze</div>
        ''', unsafe_allow_html=True)

        st.image(image_file, caption='Uploaded Image', use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_btn = st.button('⚡  INITIATE SCAN', use_container_width=True)

        if process_btn:
            with st.spinner('🧠 Neural network analyzing faces...'):
                result_img, faces, masks_on, masks_off = process_image('./images/out.jpg', net, model)

            # Results Header
            st.markdown('''
            <div class="section-hdr">
                <div class="line"></div>
                <div class="label">📡 SCAN RESULTS</div>
                <div class="line right"></div>
            </div>
            ''', unsafe_allow_html=True)

            # Stats Cards
            st.markdown(f'''
            <div class="results-grid">
                <div class="result-stat faces">
                    <div class="val">{faces}</div>
                    <div class="lbl">Faces Detected</div>
                </div>
                <div class="result-stat safe">
                    <div class="val">{masks_on}</div>
                    <div class="lbl">Mask On ✓</div>
                </div>
                <div class="result-stat danger">
                    <div class="val">{masks_off}</div>
                    <div class="lbl">No Mask ✗</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            # Result Image with scan effect
            st.image(result_img, caption='Detection Result', use_container_width=True)

            # Verdict
            if faces == 0:
                st.markdown('<div class="info-msg">⚠️ No faces detected — try a clearer image with visible faces</div>', unsafe_allow_html=True)
            elif masks_off == 0:
                st.markdown('<div class="success-msg">🛡️ ALL CLEAR — Every face is properly masked. Full compliance detected.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warn-msg">🚨 ALERT — {masks_off} face(s) detected WITHOUT a mask!</div>', unsafe_allow_html=True)

    # ===== Footer =====
    st.markdown('''
    <div class="app-footer">
        <p>Engineered with precision using cutting-edge AI</p>
        <div class="footer-tech">
            <span>TensorFlow</span>
            <span>OpenCV</span>
            <span>MobileNetV2</span>
            <span>Streamlit</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

mask_detection()
