# XPlain

### Explainable Deep Learning Framework for AI-Generated Image Detection

XPlain is a graduation project that addresses the black-box problem in deepfake detection. Instead of relying on a single visual pathway, the model uses a dual-stream architecture. The first stream is based on MobileNetV2 to capture high-level facial structures and semantic information, while the second stream utilizes Spatial Rich Model (SRM) filters to extract forensic noise patterns commonly introduced by generative models.

The extracted features are fused for binary classification, while Grad-CAM provides visual explanations through heatmaps. Additionally, a rule-based explanation module generates textual interpretations to help users understand the reasoning behind each prediction.

The model was trained on the 140K Real and Fake Faces dataset and achieved 99.67% validation accuracy. The system is deployed through an interactive Streamlit application.

---

## Features

* Binary classification of facial images: Real vs AI-Generated
* Dual-stream architecture combining MobileNetV2 and SRM filters
* Visual explainability using Grad-CAM heatmaps
* Automatic textual explanations of model predictions
* Interactive web application built with Streamlit
* Performance dashboard including accuracy, loss curves, confusion matrix, and evaluation metrics

---

## Screenshots

### Image Analysis

<img src="assets/image_analysis.png" width="700">

### Visual Explainability

<img src="assets/visual_explainability.png" width="700">

### Result Interpretation and Confidence Summary

<img src="assets/result_interpretation.png" width="700">

---

## Project Structure

```text
XPlain/
│
├── Code/
│   ├── code.ipynb
│   └── XPlain_test.ipynb
│
├── Streamlit/
│   ├── app.py
│   └── requirements.txt
│
├── assets/
│   ├── image_analysis.png
│   ├── visual_explainability.png
│   └── result_interpretation.png
│
├── Graduation project-final.pdf
└── README.md
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/JoodALbishri/-XPlain-Graduation-Project.git
cd -XPlain-Graduation-Project
```

### 2. Install dependencies

```bash
pip install -r Streamlit/requirements.txt
```

### 3. Run the application

```bash
streamlit run Streamlit/app.py
```

### 4. Open the application

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

---

## Model Architecture

### Stream A (RGB Features)

* MobileNetV2 pretrained on ImageNet
* Extracts a 1280-dimensional semantic feature vector
* Captures facial structures, textures, and visual patterns

### Stream B (Forensic Features)

* Fixed Spatial Rich Model (SRM) filters
* Extracts high-frequency forensic noise residuals
* Dedicated CNN generates a 128-dimensional forensic feature representation

### Feature Fusion

* Concatenation of both feature streams
* Dense(256) bottleneck layer
* Dropout(0.4)
* Sigmoid output layer for binary classification

### Explainability Module

* Grad-CAM applied to the final convolutional layer
* Visual heatmaps highlighting important regions
* Rule-based textual explanations supporting predictions

---

## Dataset

This project uses the 140K Real and Fake Faces dataset from Kaggle, containing approximately 140,000 labeled facial images (real and AI-generated).

Dataset Source:

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

> Note: The dataset is not included in this repository due to its large size.

---

## Model Availability

> The trained model weights are not included in this repository due to GitHub file size limitations.

To reproduce the results, retrain the model using the provided notebooks and the dataset referenced above.

---

## Results

| Metric                 | Score  |
| ---------------------- | ------ |
| Validation Accuracy    | 99.67% |
| Validation Loss        | 0.0107 |
| AI-Generated Precision | 99.68% |
| Real Precision         | 99.66% |
| AUC (ROC)              | 0.9999 |

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* Grad-CAM
* Google Colab

---

## Team

* Jood Musaad Albishri
* Saja Naif Almalki
* Ghadi Hamzah Alhyanie
* Rahaf Alradadi
* Afnan Alsubhi

### Project Advisor

Dr. Mashael Alluhaybi

Computer Science Department
Jamoum University College
Umm Al-Qura University

---

## Limitations & Future Work

* Limited generalization to modern diffusion-based generators
* Currently supports image analysis only
* Textual explanations are rule-based rather than LLM-generated
* Future work includes attention-based fusion mechanisms
* Expansion to larger and more diverse datasets
* Extension toward video deepfake detection

---

## Project Report

The complete project report is available in:

```text
Graduation project-final.pdf
```

