```python
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import base64
import os
import pandas as pd
import time

from PIL import Image

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

BACKGROUND_IMAGE = "image_bg.jpg"
DEVELOPER_IMAGE = "person.jpg"

# ====================================
# BACKGROUND
# ====================================

def set_background(image_path):

    if os.path.exists(image_path):

        with open(image_path, "rb") as img:
            encoded = base64.b64encode(
                img.read()
            ).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{

                background-image:
                linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)),
                url("data:image/jpg;base64,{encoded}");

                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;

            }}

            </style>
            """,
            unsafe_allow_html=True
        )

set_background(BACKGROUND_IMAGE)

# ====================================
# USER DATABASE
# ====================================

USER_FILE = "users.pkl"

def load_users():

    if os.path.exists(USER_FILE):

        with open(USER_FILE, "rb") as f:
            return pickle.load(f)

    return {}

def save_users(users):

    with open(USER_FILE, "wb") as f:
        pickle.dump(users, f)

# ====================================
# LOAD MODEL
# ====================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "mobilenetv2_best_77.h5",
        compile=False
    )

    return model


@st.cache_data
def load_classes():

    with open(
        "class_indices.pkl",
        "rb"
    ) as f:

        return pickle.load(f)

# ====================================
# SESSION STATE
# ====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ====================================
# LOGIN / SIGNUP PAGE
# ====================================

if not st.session_state.logged_in:

    st.markdown(
        """
        <h1 style='text-align:center;color:white;'>
        👁️ Diabetic Retinopathy Detection System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style='text-align:center;color:white;'>
        AI Powered Retinal Disease Screening Platform
        </h4>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1,1,1])

    with c2:

        users = load_users()

        # LOGIN
        if st.session_state.page == "login":

            st.subheader("Doctor Login")

            username = st.text_input(
                "Username"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            if st.button("Login"):

                if (
                    username in users
                    and users[username] == password
                ):

                    st.session_state.logged_in = True
                    st.rerun()

                else:

                    st.error(
                        "Invalid Username or Password"
                    )

            st.write("New User?")

            if st.button(
                "Create Account"
            ):

                st.session_state.page = "signup"
                st.rerun()

        # SIGNUP
        else:

            st.subheader(
                "Doctor Registration"
            )

            new_user = st.text_input(
                "Create Username"
            )

            new_password = st.text_input(
                "Create Password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            if st.button(
                "Register"
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
                        "Account Created Successfully"
                    )

                    st.session_state.page = "login"
                    st.rerun()

            if st.button(
                "Back To Login"
            ):

                st.session_state.page = "login"
                st.rerun()

# ====================================
# MAIN WEBSITE
# ====================================

else:

    model = load_model()

    idx_to_class = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

    # ====================================
    # HEADER
    # ====================================

    left, right = st.columns([3,1])

    with left:

        st.title(
            "🩺 Diabetic Retinopathy Detection"
        )

        st.write(
            """
            AI-powered retinal disease
            screening platform using
            MobileNetV2 Deep Learning Model.
            """
        )

    with right:

        st.markdown(
        """
        <div style="
        background:rgba(255,255,255,0.12);
        padding:15px;
        border-radius:15px;
        color:white;
        text-align:center;
        ">

        <h4>📞 Contact</h4>

        Phone:<br>
        <b>6235406513</b>

        <br><br>

        Email:<br>

        <a href="mailto:ashikajankvkl@gmail.com"
        style="color:#00FFFF;">
        ashikajankvkl@gmail.com
        </a>

        </div>
        """,
        unsafe_allow_html=True
        )

    st.divider()

    # ====================================
    # HERO SECTION
    # ====================================

    st.subheader(
        "Project Highlights"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Classes",
        "5"
    )

    c2.metric(
        "Model",
        "MobileNetV2"
    )

    c3.metric(
        "Input Size",
        "224×224"
    )

    c4.metric(
        "Accuracy",
        "77%"
    )

    st.divider()

    # ====================================
    # MAIN CONTENT
    # ====================================

    col1, col2 = st.columns([2,1])

    # ====================================
    # PREDICTION SECTION
    # ====================================

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Retinal Fundus Image",
            type=["jpg","jpeg","png"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                width=500,
                caption="Uploaded Image"
            )

            img = np.array(image)

            img = cv2.resize(
                img,
                (224,224)
            )

            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            # Progress Animation

            with st.spinner(
                "🔍 AI is analyzing image..."
            ):

                progress = st.progress(0)

                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i+1)

                prediction = model.predict(
                    img
                )

                progress.empty()

            pred_class = np.argmax(
                prediction
            )

            confidence = (
                np.max(prediction)
                * 100
            )

            severity = idx_to_class[
                pred_class
            ]

            st.success(
                f"Severity : {severity}"
            )

            st.info(
                f"Confidence : {confidence:.2f}%"
            )

            # Probability Chart

            st.subheader(
                "Prediction Probability"
            )

            probs = prediction[0] * 100

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

            # Medical Feedback

            st.subheader(
                "Medical Feedback"
            )

            feedback = {

                0:
                "No diabetic retinopathy detected. Continue regular eye examinations.",

                1:
                "Mild diabetic retinopathy detected. Regular monitoring is recommended.",

                2:
                "Moderate diabetic retinopathy detected. Please consult an ophthalmologist.",

                3:
                "Severe diabetic retinopathy detected. Immediate medical attention required.",

                4:
                "Proliferative diabetic retinopathy detected. High risk of blindness. Urgent treatment recommended."

            }

            st.warning(
                feedback[pred_class]
            )

    # ====================================
    # DEVELOPER SECTION
    # ====================================

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

        st.markdown(
        """
        ### Ashik Ajan

        🎓 B.Tech Computer Science

        💻 Deep Learning & AI Enthusiast

        ### Technologies

        ✔ Python

        ✔ TensorFlow

        ✔ Streamlit

        ✔ MobileNetV2

        ✔ OpenCV
        """
        )

    st.divider()

    # ====================================
    # ABOUT
    # ====================================

    st.subheader(
        "About This Website"
    )

    st.write(
        """
        This AI-powered application detects
        diabetic retinopathy severity from
        retinal fundus images.

        It assists doctors in early diagnosis
        and helps prevent vision loss through
        timely treatment.
        """
    )

    st.info(
        """
        Workflow:

        Upload Image ➜ Preprocessing ➜
        MobileNetV2 Prediction ➜
        Severity Classification ➜
        Medical Feedback
        """
    )

    st.divider()

    # ====================================
    # FOOTER
    # ====================================

    st.markdown(
    """
    <div style='text-align:center;color:white;'>

    ❤️ Developed By <b>Ashik Ajan</b>

    <br><br>

    Deep Learning Based Diabetic
    Retinopathy Detection System

    <br><br>

    📞 6235406513

    <br>

    📧 ashikajankvkl@gmail.com

    </div>
    """,
    unsafe_allow_html=True
    )

    st.divider()

    if st.button(
        "Logout"
    ):

        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()
```
