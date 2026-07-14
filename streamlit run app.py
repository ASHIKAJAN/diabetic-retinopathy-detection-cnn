import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
from PIL import Image
import base64
import os


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
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
                rgba(0,0,0,0.60),
                rgba(0,0,0,0.60)
                ),
                url("data:image/jpg;base64,{encoded}");

                background-size: cover;
                background-position:center;
                background-repeat:no-repeat;
                background-attachment:fixed;

            }}

            </style>
            """,
            unsafe_allow_html=True
        )



set_background(BACKGROUND_IMAGE)



# ==========================
# LOAD MODEL
# ==========================


@st.cache_resource
def load_model():

  @st.cache_resource
def load_model():

    model_path = "mobilenetv2_best_77.keras"

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model



# ==========================
# LOAD CLASS FILE
# ==========================


@st.cache_data
def load_classes():

    with open(
        "class_indices.pkl",
        "rb"
    ) as f:

        classes = pickle.load(f)

    return classes



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
        <h1 style="
        text-align:center;
        color:white;
        ">
        👁️ Diabetic Retinopathy Detection System
        </h1>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    col1,col2,col3 = st.columns([1,1,1])


    with col2:


        st.markdown(
            """
            <div style="
            background:rgba(255,255,255,0.15);
            padding:30px;
            border-radius:20px;
            ">
            """,
            unsafe_allow_html=True
        )


        st.subheader(
            "Doctor Login"
        )


        username = st.text_input(
            "Username"
        )


        password = st.text_input(
            "Password",
            type="password"
        )


        if st.button("Login"):


            if username=="doctor" and password=="1234":

                st.session_state.logged_in=True

                st.rerun()


            else:

                st.error(
                    "Invalid Username or Password"
                )



# ==========================
# MAIN WEBSITE
# ==========================


else:


    model = load_model()


    classes = load_classes()



    idx_to_class = {

        0:"No DR",
        1:"Mild DR",
        2:"Moderate DR",
        3:"Severe DR",
        4:"Proliferative DR"

    }



    st.title(
        "🩺 Diabetic Retinopathy Detection"
    )


    st.write(
        """
        Upload a retinal fundus image to detect
        diabetic retinopathy severity.
        """
    )


    st.divider()



    col1,col2 = st.columns([2,1])



    # ======================
    # PREDICTION SECTION
    # ======================


    with col1:


        uploaded_file = st.file_uploader(

            "Upload Retinal Image",

            type=[
                "jpg",
                "jpeg",
                "png"
            ]

        )



        if uploaded_file:


            image = Image.open(
                uploaded_file
            )


            st.image(
                image,
                width=450
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



            prediction = model.predict(
                img
            )



            predicted_class = np.argmax(
                prediction
            )



            confidence = np.max(
                prediction
            )*100



            result = idx_to_class[
                predicted_class
            ]



            st.success(
                f"Prediction : {result}"
            )


            st.info(
                f"Confidence : {confidence:.2f}%"
            )



            st.subheader(
                "Medical Explanation"
            )



            explanations = {


            0:
            """
            No diabetic retinopathy detected.
            Maintain regular eye examinations.
            """,


            1:
            """
            Mild diabetic retinopathy detected.
            Regular monitoring recommended.
            """,


            2:
            """
            Moderate diabetic retinopathy detected.
            Consult an ophthalmologist.
            """,


            3:
            """
            Severe diabetic retinopathy detected.
            Immediate medical attention required.
            """,


            4:
            """
            Proliferative diabetic retinopathy detected.
            High risk condition.
            Urgent treatment recommended.
            """

            }



            st.warning(
                explanations[predicted_class]
            )





    # ======================
    # DEVELOPER SECTION
    # ======================


    with col2:


        st.subheader(
            "Developer"
        )


        if os.path.exists(DEVELOPER_IMAGE):

            st.image(
                DEVELOPER_IMAGE,
                width=220
            )



        st.markdown(
        """

        ### Ashik Ajan

        🎓 B.Tech Computer Science


        💻 Deep Learning & AI Enthusiast


        ### Project

        Deep Learning-Based Diabetic
        Retinopathy Detection and Severity
        Classification Using Retinal Images.


        ### Technologies

        ✔ Python

        ✔ TensorFlow

        ✔ Streamlit

        ✔ MobileNetV2

        ✔ OpenCV

        """
        )



    st.divider()



    st.subheader(
        "About This Website"
    )


    st.write(
        """
        This AI based application detects
        diabetic retinopathy severity from
        retinal fundus images.

        It uses MobileNetV2 deep learning model
        trained on retinal image datasets.
        """
    )



    if st.button("Logout"):

        st.session_state.logged_in=False

        st.rerun()
