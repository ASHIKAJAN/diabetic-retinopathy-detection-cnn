import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import base64

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁",
    layout="wide"
)

# ==========================
# IMAGE PATHS
# ==========================

BACKGROUND_IMAGE = "image_bg.jpg"
DEVELOPER_IMAGE = "person.jpg"

# ==========================
# BACKGROUND FUNCTION
# ==========================

def set_background(image_path):

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
            rgba(0,0,0,0.55),
            rgba(0,0,0,0.55)),
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

# ==========================
# SESSION
# ==========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ==========================
# LOGIN PAGE
# ==========================

if not st.session_state.logged_in:

    st.markdown(
        """
        <h1 style='text-align:center;
        color:white;'>
        👁 Diabetic Retinopathy Detection System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("##")

    c1, c2, c3 = st.columns([1,1,1])

    with c2:

        st.markdown(
            """
            <div style="
            background:rgba(255,255,255,0.15);
            padding:30px;
            border-radius:20px;">
            """,
            unsafe_allow_html=True
        )

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
                username == "doctor"
                and
                password == "1234"
            ):

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error(
                    "Invalid Username or Password"
                )

# ==========================
# HOME PAGE
# ==========================

else:

    # LOAD MODEL

    model = tf.keras.models.load_model(
        "mobilenetv2_best_77.keras"
    )

    with open(
        "class_indices.pkl",
        "rb"
    ) as f:

        class_indices = pickle.load(f)

    idx_to_class = {

        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

    st.title(
        "🩺 Diabetic Retinopathy Detection"
    )

    st.write(
        """
        Upload a retinal image to predict
        diabetic retinopathy severity.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns([2,1])

    # ==========================
    # IMAGE UPLOAD
    # ==========================

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Retinal Image",
            type=["jpg","jpeg","png"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            )

            st.image(
                image,
                width=500
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

            prediction = model.predict(img)

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
                f"Predicted Severity : {severity}"
            )

            st.info(
                f"Confidence : {confidence:.2f}%"
            )

            st.markdown("### Explanation")

            if pred_class == 0:

                st.success("""
                No diabetic retinopathy detected.

                Continue regular eye checkups.
                """)

            elif pred_class == 1:

                st.warning("""
                Mild diabetic retinopathy detected.

                Annual monitoring is recommended.
                """)

            elif pred_class == 2:

                st.warning("""
                Moderate diabetic retinopathy detected.

                Please consult an ophthalmologist.
                """)

            elif pred_class == 3:

                st.error("""
                Severe diabetic retinopathy detected.

                Immediate medical attention required.
                """)

            else:

                st.error("""
                Proliferative diabetic retinopathy detected.

                High risk of blindness.

                Urgent treatment recommended.
                """)

    # ==========================
    # ABOUT DEVELOPER
    # ==========================

    with col2:

        st.subheader(
            "Developer"
        )

        st.image(
            DEVELOPER_IMAGE,
            width=250
        )

        st.markdown("""
        ### Ashik Ajan

        🎓 B.Tech Computer Science

        💻 Deep Learning &
        Machine Learning Enthusiast

        🩺 Project:

        Deep Learning-Based Diabetic
        Retinopathy Detection and
        Severity Classification Using
        Retinal Images.

        🛠 Technologies Used:

        - Python
        - TensorFlow
        - Streamlit
        - MobileNetV2
        - OpenCV
        """)

    st.markdown("---")

    st.subheader(
        "About This Website"
    )

    st.write(
        """
        This website uses Deep Learning
        techniques to detect diabetic
        retinopathy from retinal fundus images.

        The system assists doctors in
        early diagnosis and severity
        classification, helping prevent
        vision loss through timely treatment.
        """
    )

    st.markdown("---")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.rerun()
