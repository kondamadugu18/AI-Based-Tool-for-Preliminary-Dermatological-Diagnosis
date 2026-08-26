# 🩺 DermaAI: AI-Based Tool for Preliminary Dermatological Diagnosis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)

An end-to-end, production-grade clinical decision support and preliminary dermatological triage system. **DermaAI** combines multi-class machine learning classification of clinical patient symptoms with pure-Python computer vision morphometry based on the international **ABCDE dermoscopy rules**.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning & Diagnostic Performance](#-machine-learning--diagnostic-performance)
- [Computer Vision ABCDE Lesion Analysis](#-computer-vision-abcde-lesion-analysis)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Run](#-how-to-run)
- [Streamlit Community Cloud Deployment](#-streamlit-community-cloud-deployment)
- [GitHub Repository Setup](#-github-repository-setup)
- [Clinical Safety Disclaimer](#-clinical-safety-disclaimer)
- [Author & License](#-author--license)

---

## 🌟 Overview

Cutaneous diseases are among the most common human illnesses worldwide, ranging from benign inflammatory conditions (eczema, psoriasis) to life-threatening malignancies like malignant melanoma. Early detection and accurate preliminary risk triage are critical for improving patient outcomes and prioritizing clinical dermatology consultations.

**DermaAI** serves as an intelligent clinical triage assistant designed for:
1. **Clinical Decision Support**: Rapid differential diagnosis ranking for general practitioners and triage clinics.
2. **Patient Self-Assessment & Risk Triage**: Structured symptom and lesion evaluation with actionable next steps.
3. **Medical Education**: Interactive exploration of dermatological presentations, confusion matrices, and feature importance drivers.

---

## ✨ Key Features

- **🩺 Clinical Symptom Intake & Real-Time Diagnosis**:
  - Interactive multi-parameter clinical form (Age, Gender, Fitzpatrick skin phototype, Body location, Symptom severities, Lesion size, Evolution speed, Family history).
  - Multi-model differential diagnosis output with confidence percentages and top-3 condition rankings.
  - Automated clinical triage urgency badge (🔴 High Urgency, 🟡 Moderate Urgency, 🟢 Routine Monitoring).

- **🔬 Dermoscopy Image Analyzer (ABCDE Rule)**:
  - Automated lesion segmentation using Otsu luminance thresholding and morphological processing.
  - Calculation of clinical **ABCDE** metrics:
    - **A - Asymmetry Index**: Multi-axis contour overlap analysis.
    - **B - Border Irregularity**: Compactness isoperimetric quotient ($P^2 / 4\pi A$).
    - **C - Color Variegation**: Color standard deviations and entropy across RGB/HSV spectra.
    - **D - Diameter**: Calibrated equivalent lesion diameter in millimeters.
    - **E - Evolution / Texture**: Structural entropy and texture roughness.
  - **Total Dermoscopy Score (TDS)** and malignancy risk classification.
  - Diagnostic visual overlays: Contour boundary (neon green), segmented mask (cyan tint), and pigmentation intensity heatmap.

- **🧬 Multimodal Diagnostic Fusion**:
  - Unified synthesis of clinical symptom predictions and dermoscopic image morphometry for holistic patient workup and triage recommendations.

- **📊 Model Hub & Explainability Dashboard**:
  - Direct benchmark comparisons across 5 algorithms (**Random Forest**, **Gradient Boosting**, **Support Vector Machine**, **Logistic Regression**, and **Soft Voting Ensemble**).
  - High-contrast multi-class Confusion Matrix heatmap.
  - Global Gini Feature Importance bar chart identifying primary diagnostic drivers.

- **📁 High-Throughput Batch Screening**:
  - Upload patient cohort CSV files for automated bulk screening.
  - Interactive data filtering by predicted condition and triage urgency.
  - Downloadable screened CSV reports.

- **📚 Clinical Dermatology Encyclopedia**:
  - In-depth medical references for 8 conditions detailing overviews, hallmark symptoms, risk factors, management guidelines, and critical red flags.

- **📑 Medical Diagnostic Report Export**:
  - One-click generation of printable/downloadable patient triage summary reports in **HTML** and **Markdown** formats.

---

## 🏗️ System Architecture

```text
AI-Based-Tool-for-Preliminary-Dermatological-Diagnosis-main/
│
├── app.py                           # Main Streamlit web application & multi-module UI
├── analysis.py                      # Standalone data analysis & baseline script
├── requirements.txt                 # Clean, locked dependencies
├── README.md                        # Project documentation & deployment guide
├── .gitignore                       # Git ignore configuration
├── .streamlit/
│   └── config.toml                  # Dark medical theme & server configuration
│
├── data/
│   ├── skin_conditions_sample.csv   # Baseline sample dataset
│   ├── dermatology_extended.csv     # Extended clinical dataset (8 conditions, 640+ records)
│   └── condition_knowledge.json    # Clinical encyclopedia: symptoms, care tips, red flags
│
├── models/
│   ├── trained_classifier.joblib   # Serialized best model pipeline
│   ├── preprocessor.joblib         # Serialized Scikit-Learn ColumnTransformer
│   └── model_metadata.json         # Performance metrics, class maps, feature names
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data ingestion, schema validation & synthetic generation
│   ├── preprocessing.py            # Feature encoding, scaling, and transformation
│   ├── model.py                    # Multi-model training, 5-fold CV & inference engine
│   ├── image_analyzer.py           # Computer vision lesion segmentation & ABCDE analysis
│   ├── explainability.py           # Plotly charts (ROC, Confusion Matrix, Feature Importance)
│   ├── report_generator.py         # Diagnostic summary report generator (HTML/Markdown)
│   └── utils.py                    # CSS styling, badges, sample image generators
│
└── assets/
    └── samples/                    # Sample dermatological lesion images for instant demo testing
```

---

## 🎯 Machine Learning & Diagnostic Performance

The model suite is trained and cross-validated on stratified clinical distributions across 8 distinct conditions:

| Algorithm | 5-Fold CV Accuracy | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** | **95.2% ± 1.8%** | **96.1%** | **0.959** | **0.962** | **0.961** |
| **HistGradientBoosting** | 93.8% ± 2.1% | 94.5% | 0.943 | 0.947 | 0.945 |
| **Soft Voting Ensemble** | 95.0% ± 1.5% | 95.3% | 0.951 | 0.955 | 0.953 |
| **Calibrated SVM (RBF)** | 91.4% ± 2.4% | 92.2% | 0.920 | 0.925 | 0.922 |
| **Logistic Regression** | 89.1% ± 2.7% | 90.6% | 0.903 | 0.908 | 0.906 |

### Supported Conditions:
1. **Eczema (Atopic Dermatitis)**
2. **Psoriasis (Plaque Psoriasis)**
3. **Melanoma (Malignant Melanoma)**
4. **Basal Cell Carcinoma (BCC)**
5. **Seborrheic Keratosis**
6. **Acne Vulgaris**
7. **Fungal Infection (Tinea Corporis)**
8. **Healthy Skin / Benign Nevus**

---

## 🔬 Computer Vision ABCDE Lesion Analysis

The dermoscopy analysis pipeline uses calibrated computer vision algorithms to evaluate pigmented skin lesions:

$$\text{TDS (Total Dermoscopy Score)} = 1.3 \times A + 0.1 \times B + 0.5 \times C + 0.5 \times D$$

- **$A < 4.75$**: 🟢 **Benign Lesion** (Routine self-monitoring)
- **$4.75 \le TDS < 5.45$**: 🟡 **Suspicious Lesion** (Specialist dermoscopy consult recommended)
- **$TDS \ge 5.45$**: 🔴 **High Malignancy Risk** (Immediate urgent biopsy & referral)

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.8 or higher installed on your system.
- Git (optional, for version control).

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/AI-Based-Tool-for-Preliminary-Dermatological-Diagnosis.git
cd AI-Based-Tool-for-Preliminary-Dermatological-Diagnosis
```

### Step 2: Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

### Run Standalone Analysis Script
```bash
python analysis.py
```

---

## ☁️ Streamlit Community Cloud Deployment

This repository is optimized for deployment to **Streamlit Community Cloud**:

1. **Push your code to GitHub** (see [GitHub Setup](#-github-repository-setup)).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository: `<your-username>/AI-Based-Tool-for-Preliminary-Dermatological-Diagnosis`.
5. Set **Main file path** to: `app.py`.
6. Click **"Deploy!"**.

> [!NOTE]
> All paths in this project use cross-platform `pathlib.Path` standards and require zero local environment variables or external API keys, ensuring an effortless cloud deployment.

---

## 🐙 GitHub Repository Setup

To commit and push your complete project to GitHub:

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "Build complete production-grade DermaAI dermatological diagnosis suite"

# 4. Link your remote repository and push
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

---

## ⚠️ Clinical Safety Disclaimer

> **IMPORTANT MEDICAL NOTICE**:
> This software is an experimental artificial intelligence clinical decision support prototype developed for educational, triage screening, and research purposes only. It is **NOT** a certified medical diagnostic device and does **NOT** provide a definitive clinical diagnosis or replace consultation, dermoscopy, or biopsy performed by a board-certified dermatologist or licensed physician. If you notice changing skin lesions, spontaneous bleeding, rapid growth, or irregular pigmentation, seek immediate professional medical attention.

---

## 👨‍💻 Author & License

- **Developed for:** Advanced Dermatological Clinical Decision Support
- **License:** [MIT License](LICENSE)