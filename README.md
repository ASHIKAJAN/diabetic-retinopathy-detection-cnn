# Deep Learning-Based Diabetic Retinopathy Detection and Severity Classification Using Retinal Images

## Project Overview

Diabetic Retinopathy (DR) is one of the leading causes of blindness among diabetic patients worldwide. Early detection and timely treatment can significantly reduce vision loss.

This project presents an automated deep learning system for detecting and classifying diabetic retinopathy severity from retinal fundus images using transfer learning with MobileNetV2.

The system classifies retinal images into five severity levels:

- Class 0 – No Diabetic Retinopathy
- Class 1 – Mild Non-Proliferative DR
- Class 2 – Moderate Non-Proliferative DR
- Class 3 – Severe Non-Proliferative DR
- Class 4 – Proliferative Diabetic Retinopathy

The developed model can assist ophthalmologists by providing rapid and accurate preliminary screening.

---

# Objectives

- Develop an automated diabetic retinopathy detection system.
- Reduce manual effort in retinal image screening.
- Classify disease severity into five stages.
- Implement transfer learning using MobileNetV2.
- Deploy the trained model using Streamlit.

---

# Dataset

This project uses a combined retinal image dataset collected from multiple publicly available sources:

## Datasets Used

1. EyePACS Dataset
2. APTOS 2019 Blindness Detection Dataset
3. Messidor Dataset

Combined Dataset:

- Total Images: Approximately 90,000+
- Classes: 5
- Image Type: Color Fundus Images

---

# Class Distribution

| Class | Description |
|-------|-------------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

---

# Technologies Used

## Programming Language

- Python 3.12

## Libraries

- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn
- Pillow
- Streamlit

---

# Deep Learning Model

## Transfer Learning Model

### MobileNetV2

Reasons for choosing MobileNetV2:

- Lightweight architecture
- Faster training
- Lower computational cost
- Good performance on medical image classification tasks

---

# Image Preprocessing

The following preprocessing techniques were applied:

1. Image resizing to 224 × 224 pixels
2. RGB conversion
3. Normalization using MobileNetV2 preprocessing
4. Data augmentation:
   - Rotation
   - Horizontal Flip
   - Zoom
   - Width Shift
   - Height Shift

---

# Model Architecture

Input Image (224 × 224 × 3)

↓

MobileNetV2 (Pretrained on ImageNet)

↓

Global Average Pooling Layer

↓

Dropout Layer

↓

Dense Layer (Softmax)

↓

5-Class Prediction

---

# Hyperparameters

| Parameter | Value |
|------------|--------|
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Epochs | 3 + Fine Tuning |
| Activation Function | Softmax |

---

# Performance Results

## Training Performance

| Metric | Value |
|---------|--------|
| Training Accuracy | 76.68% |
| Validation Accuracy | 76.90% |
| Test Accuracy | 77.00% |
| Test Loss | 0.6801 |

---

# Classification Report Summary

The model achieved good performance on majority classes. However, due to class imbalance, prediction performance for minority classes remains comparatively lower.

Weighted Accuracy:

- Test Accuracy: 77%

Limitations:

- Dataset imbalance
- Lower recall in minority DR stages

---

# Confusion Matrix

The confusion matrix demonstrates the classification capability of the model across all five diabetic retinopathy stages.

(Add confusion matrix image here)

```markdown
![Confusion Matrix](images/confusion_matrix.png)
```

---

# Accuracy and Loss Graphs

## Accuracy Graph

```markdown
![Accuracy](images/accuracy.png)
```

## Loss Graph

```markdown
![Loss](images/loss.png)
```

---

# Project Structure

```text
Diabetic-Retinopathy-Detection/

│
├── app.py
├── requirements.txt
├── README.md
├── mobilenetv2_best_77.keras
├── class_indices.pkl
│
├── images/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── notebooks/
│   └── training.ipynb
│
└── report/
    └── Project_Report.pdf
```

---

# Streamlit Application

The trained model is integrated with a Streamlit web application.

Features:

- Upload retinal image
- Automatic preprocessing
- DR stage prediction
- Display prediction confidence
- User-friendly interface

---

# Future Improvements

- Use EfficientNetB0/B3 architectures
- Handle class imbalance using class weights
- Apply advanced image enhancement techniques
- Increase dataset size
- Integrate explainable AI techniques such as Grad-CAM
- Deploy on cloud platforms for real-time screening

---

# Applications

- Hospitals
- Ophthalmology Clinics
- Telemedicine Systems
- Rural Healthcare Screening
- AI-Assisted Eye Disease Diagnosis

---

# Conclusion

This project successfully developed an automated diabetic retinopathy detection system using transfer learning with MobileNetV2.

The model achieved a test accuracy of approximately 77% on a large combined retinal image dataset and demonstrated promising performance for early diabetic retinopathy screening.

The developed system can assist healthcare professionals by providing rapid and cost-effective preliminary diagnosis.

---

# Authors

Ashik Ajan

B.Tech Computer Science and Engineering

Final Year Project

---

# License

This project is developed for educational and research purposes.
