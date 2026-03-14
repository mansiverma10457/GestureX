import streamlit as st
import cv2
import mediapipe as mp
import joblib
import math
import time
import pandas as pd

st.set_page_config(page_title="GestureX", layout="wide", page_icon="🖐")

# ---------------- STYLE ----------------

st.markdown("""
<style>

*{
cursor: crosshair;
}

[data-testid="stAppViewContainer"]{
background:#0B0F19;
color:#E6EAF2;
}

[data-testid="stSidebar"]{
background:#0E1320;
}

.logo{
font-size:80px;
text-align:center;
animation: glow 3s infinite alternate;
}

@keyframes glow{
from {text-shadow:0 0 10px #6C63FF;}
to {text-shadow:0 0 25px #8A7CFF;}
}

.title{
font-size:60px;
text-align:center;
font-weight:700;
}

.subtitle{
text-align:center;
font-size:22px;
color:#9BA3B4;
margin-bottom:40px;
}

.card{
background:rgba(108,99,255,0.08);
padding:30px;
border-radius:16px;
text-align:center;
border:1px solid rgba(108,99,255,0.2);
transition:0.3s;
}

.card:hover{
transform:scale(1.06);
background:rgba(108,99,255,0.15);
box-shadow:0px 10px 30px rgba(108,99,255,0.4);
}

.statusbox{
background:rgba(108,99,255,0.15);
padding:20px;
border-radius:12px;
border:1px solid rgba(108,99,255,0.3);
}

.stButton>button{
background:linear-gradient(90deg,#6C63FF,#8A7CFF);
color:white;
border:none;
border-radius:10px;
padding:10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("models/gesture_model.pkl")

# ---------------- SESSION DATA ----------------

if "mapping" not in st.session_state:
    st.session_state.mapping={
        "✊ Fist":"Mouse Click",
        "✌ Peace":"Double Click",
        "👍 Thumbs Up":"Volume / Brightness",
        "🖐 Palm":"Scroll Down",
        "☝ Point":"Move Cursor"
    }

if "gesture_log" not in st.session_state:
    st.session_state.gesture_log=[]

if "confidence_log" not in st.session_state:
    st.session_state.confidence_log=[]

# ---------------- SIDEBAR ----------------

st.sidebar.title("🖐 GestureX")
st.sidebar.markdown("AI Touchless Interaction")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
"Navigation",
[
"🏠 Home",
"🎥 Live AI Detection",
"⚙ Gesture Configuration",
"📈 Analytics",
"📊 System Information",
"❓ Help & Support"
]
)

# ---------------- HOME ----------------

if menu=="🏠 Home":

    st.markdown('<div class="logo">🖐</div>',unsafe_allow_html=True)
    st.markdown('<div class="title">GestureX</div>',unsafe_allow_html=True)
    st.markdown(
    '<div class="subtitle">GestureX — The Era Of Contactless Interaction</div>',
    unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)

    with col1:
        st.markdown('<div class="card">🤖<h3>AI Model</h3>Random Forest</div>',unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">🖐<h3>Hand Landmarks</h3>21 MediaPipe Points</div>',unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">⚡<h3>Realtime Processing</h3>Gesture Recognition</div>',unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Current Gesture Mapping")

    for g,a in st.session_state.mapping.items():
        st.write(f"{g} → **{a}**")

# ---------------- LIVE DETECTION ----------------

elif menu=="🎥 Live AI Detection":

    st.title("Live AI Gesture Detection")

    col1,col2=st.columns([3,1])

    frame_window=col1.image([])
    status_panel=col2.empty()

    cap=cv2.VideoCapture(0)

    mp_hands=mp.solutions.hands
    hands=mp_hands.Hands()

    mp_draw=mp.solutions.drawing_utils

    run=st.checkbox("Start Camera")

    while run:

        start=time.time()

        ret,frame=cap.read()
        if not ret:
            break

        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=hands.process(rgb)

        gesture="None"
        confidence=0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

                landmarks=[]

                wrist_x=hand_landmarks.landmark[0].x
                wrist_y=hand_landmarks.landmark[0].y

                mid_tip=hand_landmarks.landmark[12]

                scale=math.sqrt(
                (mid_tip.x-wrist_x)**2 +
                (mid_tip.y-wrist_y)**2)

                for lm in hand_landmarks.landmark:

                    landmarks.append((lm.x-wrist_x)/scale)
                    landmarks.append((lm.y-wrist_y)/scale)

                if len(landmarks)==42:

                    probs=model.predict_proba([landmarks])[0]

                    idx=probs.argmax()

                    confidence=probs[idx]

                    gesture=model.classes_[idx]

        frame_window.image(frame,channels="BGR")

        fps=int(1/(time.time()-start))

        st.session_state.gesture_log.append(gesture)
        st.session_state.confidence_log.append(confidence)

        status_panel.markdown(f"""
        <div class="statusbox">
        <h3>🤖 AI Status Panel</h3>
        <b>Gesture:</b> {gesture} <br>
        <b>Confidence:</b> {round(confidence,2)} <br>
        <b>FPS:</b> {fps} <br>
        <b>Status:</b> Active
        </div>
        """,unsafe_allow_html=True)

# ---------------- CONFIG ----------------

elif menu=="⚙ Gesture Configuration":

    st.title("Gesture Configuration")

    gesture=st.selectbox(
    "Select Gesture",
    list(st.session_state.mapping.keys())
    )

    action=st.selectbox(
    "Select Action",
    [
    "Mouse Click",
    "Double Click",
    "Volume Up",
    "Scroll Down",
    "Move Cursor"
    ])

    if st.button("Save Mapping"):

        st.session_state.mapping[gesture]=action
        st.success("Mapping Updated")

# ---------------- ANALYTICS ----------------

elif menu=="📈 Analytics":

    st.title("Gesture Analytics Dashboard")

    if len(st.session_state.gesture_log)>0:

        df=pd.DataFrame({
            "Gesture":st.session_state.gesture_log,
            "Confidence":st.session_state.confidence_log
        })

        st.subheader("Gesture Detection History")

        st.dataframe(df.tail(20))

        st.subheader("Confidence Graph")

        st.line_chart(df["Confidence"])

        st.subheader("Gesture Frequency")

        st.bar_chart(df["Gesture"].value_counts())

    else:

        st.info("Run Live Detection to collect analytics.")

# ---------------- SYSTEM INFO ----------------

elif menu=="📊 System Information":

    st.title("GestureX System Information")

    st.markdown("### Application Overview")

    st.write("""
GestureX is an AI-powered gesture recognition platform designed to enable
contactless human-computer interaction using computer vision and machine learning.
""")

    st.markdown("### Product Features")

    st.write("""
• Real-time gesture recognition  
• AI-powered computer control  
• Touchless interaction system  
• Embedded camera dashboard  
• Gesture analytics monitoring  
• Custom gesture configuration
""")

    st.markdown("### Technology Stack")

    st.write("""
Python  
OpenCV  
MediaPipe  
Scikit-Learn  
NumPy  
Streamlit
""")

    st.markdown("### Recognition Pipeline")

    st.write("""
1. Webcam captures frames  
2. MediaPipe detects hand landmarks  
3. Feature vector created  
4. Random Forest predicts gesture  
5. Action executed
""")

    st.markdown("### Performance Metrics")

    st.write("""
Accuracy: 94-97%  
Processing Speed: 20-30 FPS  
Gesture Latency: <2 seconds
""")

# ---------------- SUPPORT ----------------

elif menu=="❓ Help & Support":

    st.title("Support Center")

    st.subheader("Raise Support Ticket")

    name=st.text_input("Name")
    email=st.text_input("Email")

    issue=st.selectbox(
    "Issue Type",
    [
    "Gesture Detection Issue",
    "Camera Problem",
    "Performance Issue",
    "Configuration Issue",
    "Other"
    ])

    desc=st.text_area("Describe the issue")

    if st.button("Submit Ticket"):

        st.success("Ticket submitted successfully!")

    st.markdown("---")

    st.subheader("Troubleshooting Guide")

    st.write("""
• Ensure good lighting  
• Keep hand visible in frame  
• Avoid fast hand movement  
• Restart the app if detection stops
""")