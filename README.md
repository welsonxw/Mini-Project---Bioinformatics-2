# Pancreatic Cancer Survival Prediction System

A **Streamlit-based web application** for predicting pancreatic cancer patient survival outcomes using machine learning. This tool analyzes clinical conditions, symptoms, treatment types, and lifestyle factors to provide survival predictions and risk assessments.

https://welsonxw-mini-project---bioinformatics-2-app-0x7v3t.streamlit.app/ 

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Prediction Methodology](#prediction-methodology)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

---

## 🔬 Overview

This application leverages machine learning to predict survival outcomes for pancreatic cancer patients. It combines a trained ML model with a clinical risk scoring system to provide:

- **Survival Status Prediction** (Alive / Deceased)
- **Probability of Survival**
- **Estimated Survival Time** (in months/years)
- **Key Contributing Risk Factors**
- **Interactive Visualizations**

---

## ✨ Features

### Patient Information Input
- **Demographics**: Age, Gender
- **Medical History**: Smoking, Obesity, Diabetes, Chronic Pancreatitis, Family History, Hereditary Conditions
- **Symptoms**: Jaundice, Abdominal Discomfort, Back Pain, Weight Loss, Type 2 Diabetes Development
- **Clinical Information**: Cancer Stage (I-IV), Treatment Type
- **Lifestyle Factors**: Alcohol Consumption, Physical Activity, Diet
- **Socioeconomic Factors**: Healthcare Access, Living Area, Economic Status

### Prediction Outputs
- Survival/Death probability with confidence scores
- Estimated survival time in months and years
- Prognosis classification (Poor/Guarded/Fair/Good)
- Comprehensive risk factor analysis

### Interactive Visualizations
- Survival probability curve over 60 months
- Prediction probability pie chart
- Patient feature overview bar chart
- Risk factor breakdown

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Mini-Project---Bioinformatics-2.git
   cd Mini-Project---Bioinformatics-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   The following files must be in the project directory:
   - `model.pkl` - Trained machine learning model
   - `scaler.pkl` - Feature scaler
   - `feature_columns.pkl` - Feature column names

---

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Application

1. Fill in the patient information form with all required fields
2. Click **"Predict Survival Outcome"**
3. Review the prediction results and visualizations

---

## 📊 Input Parameters

### Demographics
| Parameter | Type | Options/Range |
|-----------|------|---------------|
| Age | Numeric | 0-120 years |
| Gender | Categorical | Male, Female |

### Medical History
| Parameter | Type | Options |
|-----------|------|---------|
| Smoking History | Binary | Yes, No |
| Obesity | Binary | Yes, No |
| Diabetes | Binary | Yes, No |
| Chronic Pancreatitis | Binary | Yes, No |
| Family History of Cancer | Binary | Yes, No |
| Hereditary Condition | Binary | Yes, No |

### Symptoms
| Parameter | Type | Options |
|-----------|------|---------|
| Jaundice | Binary | Yes, No |
| Abdominal Discomfort | Binary | Yes, No |
| Back Pain | Binary | Yes, No |
| Unexplained Weight Loss | Binary | Yes, No |
| Development of Type 2 Diabetes | Binary | Yes, No |

### Clinical Information
| Parameter | Type | Options |
|-----------|------|---------|
| Stage at Diagnosis | Categorical | I, II, III, IV |
| Treatment Type | Categorical | Surgery, Chemotherapy, Radiation, Combination |

### Lifestyle & Socioeconomic Factors
| Parameter | Type | Options |
|-----------|------|---------|
| Alcohol Consumption | Categorical | Low, Medium, High |
| Physical Activity Level | Categorical | Low, Medium, High |
| Processed Food Intake | Categorical | Low, Medium, High |
| Access to Healthcare | Categorical | Low, Medium, High |
| Living Area | Categorical | Urban, Rural |
| Economic Status | Categorical | Low, Medium, High |

---

## 🧮 Prediction Methodology

### Risk Score Calculation

The system calculates a clinical risk score based on weighted factors:

| Factor | Weight |
|--------|--------|
| Stage at Diagnosis | 0.22 |
| Chronic Pancreatitis | 0.12 |
| Smoking History | 0.09 |
| Diabetes | 0.09 |
| Weight Loss | 0.09 |
| Family History | 0.08 |
| Obesity | 0.07 |
| Jaundice | 0.07 |
| Type 2 Diabetes Development | 0.07 |
| Hereditary Condition | 0.06 |

**Age-based risk additions:**
- Age ≥ 70: +0.18
- Age ≥ 60: +0.12
- Age ≥ 50: +0.06

**Protective factors:**
- High Physical Activity: -0.03
- High Healthcare Access: -0.03
- Surgery Treatment: -0.05

### Combined Prediction

The final prediction combines the ML model probability with the clinical risk score using adaptive weighting:

- **High risk (≥0.8)**: 10% model + 90% risk score
- **Moderate-high risk (≥0.6)**: 20% model + 80% risk score
- **Low risk (≤0.2)**: 20% model + 80% risk score
- **Moderate risk**: 40% model + 60% risk score

### Survival Time Estimation

Base survival months by stage:
- Stage I: 60 months
- Stage II: 48 months
- Stage III: 24 months
- Stage IV: 12 months

Adjusted by survival probability multiplier.

---

## 📈 Visualizations

### 1. Survival Probability Curve
- Shows estimated survival probability over 60 months
- Includes 50% survival threshold line
- Marks predicted survival time

### 2. Prediction Probability Pie Chart
- Visual breakdown of survival vs. death probability

### 3. Patient Feature Overview
- Bar chart displaying all encoded input features
- Provides at-a-glance view of patient profile

### 4. Risk Factor Analysis
- Lists active risk factors
- Displays cancer stage and age
- Counts total risk factors present

---

## 📁 Project Structure

```
Mini-Project---Bioinformatics-2/
│
├── app.py                                    # Main Streamlit application
├── model.pkl                                 # Trained ML model
├── scaler.pkl                                # Feature scaler
├── feature_columns.pkl                       # Feature column names
├── pancreatic_cancer_prediction_sample.csv   # Sample data
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
joblib
matplotlib
scikit-learn
```

---

## ⚠️ Disclaimer

> **This system is for academic and research purposes only.**
> 
> It is NOT intended for real clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice and treatment decisions.

---

## 👥 Authors

Bioinformatics 2 - Mini Project

---

## 📄 License

This project is for academic purposes only.
