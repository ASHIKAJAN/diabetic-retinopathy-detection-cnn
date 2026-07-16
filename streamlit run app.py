import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import base64
import os
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie

from datetime import datetime
from PIL import Image

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

BACKGROUND_IMAGE = "image_bg.jpg"
DEVELOPER_IMAGE = "person.jpg"

# ==========================================
# BACKGROUND
# ==========================================

def set_background(image_path):

    if os.path.exists(image_path):

        with open(image_path, "rb") as img:

            encoded = base64.b64encode(
                img.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp{{
                background-image:
                linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)),
                url("data:image/jpg;base64,{encoded}");

                background-size:cover;
                background-position:center;
                background-repeat:no-repeat;
                background-attachment:fixed;
            }}

            [data-testid="metric-container"]{{
                background:rgba(255,255,255,0.12);
                border-radius:20px;
                padding:20px;
                border:1px solid rgba(255,255,255,0.2);
            }}

            div.stButton > button{{
                border-radius:15px;
                height:50px;
                font-size:16px;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

set_background(BACKGROUND_IMAGE)

# ==========================================
# USER DATABASE
# ==========================================

USER_FILE = "users.pkl"

def load_users():

    if os.path.exists(USER_FILE):

        with open(USER_FILE, "rb") as f:
            return pickle.load(f)

    return {}

def save_users(users):

    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "mobilenetv2_best_77.h5",
        compile=False
    )

    return model


# ==========================================
# PDF REPORT
# ==========================================

def create_pdf(
        severity,
        confidence,
        recommendation):

    doc = SimpleDocTemplate(
        "DR_Report.pdf"
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Diabetic Retinopathy Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            f"Date : {datetime.now()}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Prediction : {severity}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Confidence : {confidence:.2f} %",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            recommendation,
            styles["Normal"]
        )
    )

    doc.build(story)

# ==========================================
# SESSION
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ==========================================
# LOGIN PAGE
# ==========================================

if not st.session_state.logged_in:

    users = load_users()

    st.markdown("""
    <h1 style='
        text-align:center;
        color:#00FFFF;
        text-shadow:0px 0px 20px cyan;'>

        👁️ Diabetic Retinopathy Detection System

    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style='text-align:center;color:white;'>

    AI Powered Retinal Disease Screening Platform

    </h4>
    """, unsafe_allow_html=True)

    st.write("")

    # ================= HERO STATS =================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model Accuracy", "77%")

    with c2:
        st.metric("Disease Classes", "5")

    with c3:
        st.metric("Architecture", "MobileNetV2")

    st.success(
        "💡 Early detection can prevent up to 95% of diabetes-related vision loss."
    )

    st.write("")

    # ================= MAIN LOGIN SECTION =================

    left, right = st.columns([1.2, 1])

    # LEFT SIDE
    with left:

        st.markdown("""
        <div style="
            background:rgba(255,255,255,0.10);
            padding:40px;
            border-radius:25px;
            backdrop-filter:blur(15px);
            text-align:center;
            box-shadow:0px 8px 32px rgba(0,0,0,0.35);
        ">

        <h1 style="font-size:120px;">
        👁️
        </h1>

        <h2 style="color:white;">
        AI Retinal Screening System
        </h2>

        <br>

        <p style="color:white;font-size:18px;">

        ✔ Deep Learning Based Detection

        <br><br>

        ✔ Automated Severity Classification

        <br><br>

        ✔ Assists Early Diagnosis

        </p>

        <br>

        <h4 style="color:#00FFAA;">
        🟢 System Ready
        </h4>

        </div>
        """, unsafe_allow_html=True)

    # RIGHT SIDE
    with right:

        st.markdown("""
        <style>

        .login-card{

            background:rgba(255,255,255,0.12);

            padding:35px;

            border-radius:25px;

            backdrop-filter:blur(15px);

            border:1px solid rgba(255,255,255,0.2);

            box-shadow:
                0px 8px 32px rgba(0,0,0,0.35);

        }

        </style>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="login-card">',
            unsafe_allow_html=True
        )

        if st.session_state.page == "login":

            st.markdown("""
            <h2 style='text-align:center;color:white;'>

            👨‍⚕️ Doctor Login

            </h2>

            <p style='text-align:center;color:white;'>

            Sign in to access the system

            </p>
            """, unsafe_allow_html=True)

            username = st.text_input(
                "👤 Username"
            )

            password = st.text_input(
                "🔒 Password",
                type="password"
            )

            if st.button(
                "🚀 Login",
                use_container_width=True
            ):

                if (
                    username in users
                    and users[username] == password
                ):

                    st.success(
                        "Login Successful"
                    )

                    st.session_state.logged_in = True
                    st.rerun()

                else:

                    st.error(
                        "Invalid Username or Password"
                    )

            st.write("")

            if st.button(
                "📝 Create Account",
                use_container_width=True
            ):

                st.session_state.page = "signup"
                st.rerun()

        else:

            st.markdown("""
            <h2 style='text-align:center;color:white;'>

            👨‍⚕️ Doctor Registration

            </h2>
            """, unsafe_allow_html=True)

            new_user = st.text_input(
                "👤 Create Username"
            )

            new_password = st.text_input(
                "🔒 Create Password",
                type="password"
            )

            confirm_password = st.text_input(
                "🔒 Confirm Password",
                type="password"
            )

            if st.button(
                "✅ Register",
                use_container_width=True
            ):

                if new_user in users:

                    st.warning(
                        "Username already exists."
                    )

                elif (
                    new_password
                    != confirm_password
                ):

                    st.warning(
                        "Passwords do not match."
                    )

                else:

                    users[new_user] = new_password

                    save_users(users)

                    st.success(
                        "Registration Successful"
                    )

                    st.session_state.page = "login"
                    st.rerun()

            if st.button(
                "⬅ Back To Login",
                use_container_width=True
            ):

                st.session_state.page = "login"
                st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.info("""
    🟢 AI Model Loaded

    🟢 Hospital Screening Mode Active

    🟢 Ready For Prediction
    """)
# ==========================================
# MAIN APP
# ==========================================

else:

    model = load_model()

    idx_to_class = {

        0:"No DR",
        1:"Mild DR",
        2:"Moderate DR",
        3:"Severe DR",
        4:"Proliferative DR"

    }

    left,right = st.columns([3,1])

    with left:

        st.markdown("""
        <h1 style='color:#00FFFF;'>
        👁️ AI Retinal Screening System
        </h1>
        """,
        unsafe_allow_html=True)

        st.write(
            "📅",
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        )

    with right:

        st.info(
            """
            📞 6235406513

            📧 ashikajankvkl@gmail.com
            """
        )

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Classes","5")
    c2.metric("Model","MobileNetV2")
    c3.metric("Input","224×224")
    

    st.divider()

    col1,col2 = st.columns([2,1])

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Retinal Image",
            type=["jpg","png","jpeg"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                width=500
            )

            img = np.array(image)

            img = cv2.resize(
                img,
                (224,224)
            )

            img = img/255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            with st.spinner(
                    "🔍 AI Analyzing..."
            ):

                progress = st.progress(0)

                for i in range(100):

                    time.sleep(0.01)

                    progress.progress(i+1)

                prediction = model.predict(
                    img
                )

            pred_class = np.argmax(
                prediction
            )

            confidence = (
                    np.max(prediction)
                    *100
            )

            severity = idx_to_class[
                pred_class
            ]

            st.success(
                f"Prediction : {severity}"
            )

            st.info(
                f"Confidence : {confidence:.2f}%"
            )

            if pred_class == 0:

                st.success(
                    "🟢 LOW RISK"
                )

            elif pred_class <= 2:

                st.warning(
                    "🟠 MODERATE RISK"
                )

            else:

                st.error(
                    "🔴 HIGH RISK"
                )

            st.subheader(
                "Prediction Probability"
            )

            probs = prediction[0]*100

            df = pd.DataFrame({

                "Severity":
                list(
                    idx_to_class.values()
                ),

                "Probability":
                probs

            })

            st.bar_chart(
                df.set_index(
                    "Severity"
                )
            )

            feedback = {

                0:
                "No diabetic retinopathy detected.",

                1:
                "Regular monitoring recommended.",

                2:
                "Consult ophthalmologist.",

                3:
                "Immediate medical attention required.",

                4:
                "Urgent treatment recommended."

            }

            st.warning(
                feedback[pred_class]
            )

            report = f"""
Prediction : {severity}

Confidence : {confidence:.2f} %

Recommendation :

{feedback[pred_class]}
"""

            st.download_button(
                "📄 Download Report",
                report,
                file_name="DR_Report.txt"
            )

            create_pdf(
                severity,
                confidence,
                feedback[pred_class]
            )

            with open(
                    "DR_Report.pdf",
                    "rb"
            ) as file:

                st.download_button(
                    "📄 Download PDF Report",
                    file,
                    file_name="DR_Report.pdf"
                )

    with col2:

        st.subheader(
            "Developer"
        )

        if os.path.exists(
                DEVELOPER_IMAGE
        ):

            st.image(
                DEVELOPER_IMAGE,
                width=250
            )

        st.markdown("""
### Ashik Ajan

🎓 B.Tech Computer Science

💻 Deep Learning & AI Enthusiast

✔ Python  
✔ TensorFlow  
✔ Streamlit  
✔ MobileNetV2  
✔ OpenCV
""")

    st.divider()

    st.subheader(
        "MobileNetV2 Architecture"
    )

    if os.path.exists(
            "mobilenet_architecture.png"
    ):

        st.image(
            "mobilenet_architecture.png",
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "About This Website"
    )

    st.write(
        """
This AI application detects diabetic
retinopathy severity using retinal
fundus images and deep learning.
"""
    )

    st.divider()

    if st.button(
            "Logout"
    ):

        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()
