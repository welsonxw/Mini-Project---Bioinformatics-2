import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load model artifacts
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Pancreatic Cancer Survival Prediction",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.title("Pancreatic Cancer Survival Prediction System")

st.info("""
This application predicts the survival outcome of pancreatic cancer patients using machine learning.  
It analyzes **clinical conditions, symptoms, treatment type, and lifestyle factors** to estimate:

- Survival status (Alive / Deceased)
- Probability of survival
- Estimated survival time
- Key contributing risk factors  
""")

st.success("Disclaimer: This system is for academic and research purposes only. Not for real clinical diagnosis.")

st.divider()

# -------------------------------
# Patient Input Form
# -------------------------------
with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    # ---------------------------
    # Column 1
    # ---------------------------
    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", 0, 120, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("**Medical History**")
        smoking = st.selectbox("Smoking History", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        chronic_pancreatitis = st.selectbox("Chronic Pancreatitis", ["No", "Yes"])
        family_history = st.selectbox("Family History of Cancer", ["No", "Yes"])
        hereditary = st.selectbox("Hereditary Condition", ["No", "Yes"])

    # ---------------------------
    # Column 2
    # ---------------------------
    with col2:
        st.markdown("**Symptoms**")
        jaundice = st.selectbox("Jaundice", ["No", "Yes"])
        abdominal_discomfort = st.selectbox("Abdominal Discomfort", ["No", "Yes"])
        back_pain = st.selectbox("Back Pain", ["No", "Yes"])
        weight_loss = st.selectbox("Unexplained Weight Loss", ["No", "Yes"])
        type2_diabetes = st.selectbox("Development of Type 2 Diabetes", ["No", "Yes"])

        st.markdown("**Clinical Information**")
        stage = st.selectbox("Stage at Diagnosis", ["I", "II", "III", "IV"])
        treatment = st.selectbox(
            "Treatment Type",
            ["Surgery", "Chemotherapy", "Radiation", "Combination"]
        )

    # ---------------------------
    # Column 3
    # ---------------------------
    with col3:
        st.markdown("**Lifestyle Factors**")
        alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"])
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
        diet = st.selectbox("Processed Food Intake", ["Low", "Medium", "High"])

        st.markdown("**Socioeconomic Factors**")
        healthcare_access = st.selectbox("Access to Healthcare", ["Low", "Medium", "High"])
        urban_rural = st.selectbox("Living Area", ["Urban", "Rural"])
        economic_status = st.selectbox("Economic Status", ["Low", "Medium", "High"])

    submit = st.form_submit_button("Predict Survival Outcome")

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_input():
    input_dict = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Smoking_History': 1 if smoking == "Yes" else 0,
        'Obesity': 1 if obesity == "Yes" else 0,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'Chronic_Pancreatitis': 1 if chronic_pancreatitis == "Yes" else 0,
        'Family_History': 1 if family_history == "Yes" else 0,
        'Hereditary_Condition': 1 if hereditary == "Yes" else 0,
        'Jaundice': 1 if jaundice == "Yes" else 0,
        'Abdominal_Discomfort': 1 if abdominal_discomfort == "Yes" else 0,
        'Back_Pain': 1 if back_pain == "Yes" else 0,
        'Weight_Loss': 1 if weight_loss == "Yes" else 0,
        'Development_of_Type2_Diabetes': 1 if type2_diabetes == "Yes" else 0,
        'Stage_at_Diagnosis': {"I": 0, "II": 1, "III": 2, "IV": 3}[stage],
        'Treatment_Type': {"Surgery": 0, "Chemotherapy": 1, "Radiation": 2, "Combination": 3}[treatment],
        'Alcohol_Consumption': {"Low": 0, "Medium": 1, "High": 2}[alcohol],
        'Physical_Activity_Level': {"Low": 0, "Medium": 1, "High": 2}[physical_activity],
        'Diet_Processed_Food': {"Low": 0, "Medium": 1, "High": 2}[diet],
        'Access_to_Healthcare': {"Low": 0, "Medium": 1, "High": 2}[healthcare_access],
        'Urban_vs_Rural': 1 if urban_rural == "Urban" else 0,
        'Economic_Status': {"Low": 0, "Medium": 1, "High": 2}[economic_status]
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    return input_scaled, input_df, input_dict

# -------------------------------
# Risk Score
# -------------------------------
def calculate_risk_score(d):
    """Calculate risk score based on clinical factors - higher score = higher death risk"""
    risk = 0.0
    
    # Critical risk factors with weights
    critical_factors = {
        'Stage_at_Diagnosis': 0.22,
        'Smoking_History': 0.09,
        'Obesity': 0.07,
        'Diabetes': 0.09,
        'Chronic_Pancreatitis': 0.12,
        'Family_History': 0.08,
        'Hereditary_Condition': 0.06,
        'Jaundice': 0.07,
        'Weight_Loss': 0.09,
        'Development_of_Type2_Diabetes': 0.07,
    }

    if d['Age'] >= 70:
        risk += 0.18
    elif d['Age'] >= 60:
        risk += 0.12
    elif d['Age'] >= 50:
        risk += 0.06

    # Stage is the most critical factor
    risk += d['Stage_at_Diagnosis'] * critical_factors['Stage_at_Diagnosis']

    # Add other risk factors
    for factor, weight in critical_factors.items():
        if factor != 'Stage_at_Diagnosis':
            risk += d.get(factor, 0) * weight

    # Protective factors
    if d['Physical_Activity_Level'] == 2:
        risk -= 0.03
    if d['Access_to_Healthcare'] == 2:
        risk -= 0.03
    if d['Treatment_Type'] == 0:
        risk -= 0.05

    return max(0.05, min(0.98, risk))

# -------------------------------
# Prediction & Visualization
# -------------------------------
if submit:
    input_scaled, input_df, input_dict = preprocess_input()
    risk_score = calculate_risk_score(input_dict)

    model_prob = model.predict_proba(input_scaled)[0][1]
    
    # Adaptive weighting based on risk score
    if risk_score >= 0.8:
        combined_death_prob = 0.1 * model_prob + 0.9 * risk_score
    elif risk_score >= 0.6:
        combined_death_prob = 0.2 * model_prob + 0.8 * risk_score
    elif risk_score <= 0.2:
        combined_death_prob = 0.2 * model_prob + 0.8 * risk_score
    else:
        combined_death_prob = 0.4 * model_prob + 0.6 * risk_score

    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)
    
    with col1:
        if combined_death_prob >= 0.5:
            st.error(f"Predicted Status: **Deceased**")
        else:
            st.success(f"Predicted Status: **Alive**")
        confidence = combined_death_prob if combined_death_prob >= 0.5 else (1 - combined_death_prob)
        st.info(f"Confidence: **{confidence*100:.1f}%**")
    
    with col2:
        st.metric("Death Risk Score", f"{combined_death_prob*100:.1f}%")
        st.metric("Survival Chance", f"{(1-combined_death_prob)*100:.1f}%")

    # ---------------------------
    # Survival Time Estimation
    # ---------------------------
    st.subheader("Estimated Survival Time Analysis")

    base_months = {0: 60, 1: 48, 2: 24, 3: 12}
    stage_val = input_dict['Stage_at_Diagnosis']
    survival_multiplier = 2.0 * (1 - combined_death_prob)
    predicted_months = max(1, min(120, base_months[stage_val] * max(0.5, survival_multiplier)))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Survival", f"{predicted_months:.0f} months")
    with col2:
        years = predicted_months / 12
        st.metric("In Years", f"{years:.1f} years")
    with col3:
        if predicted_months <= 6:
            prognosis = "Poor"
        elif predicted_months <= 12:
            prognosis = "Guarded"
        elif predicted_months <= 24:
            prognosis = "Fair"
        else:
            prognosis = "Good"
        st.metric("Prognosis", prognosis)

    # ---------------------------
    # Survival Probability Curve
    # ---------------------------
    st.subheader("Survival Probability Over Time")

    months = list(range(1, 61))
    decay = 0.01 + combined_death_prob * 0.10
    survival_probs = [np.exp(-decay * m) * 100 for m in months]

    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.fill_between(months, survival_probs, alpha=0.3, color='#2196F3')
    ax1.plot(months, survival_probs, color='#1976D2', linewidth=2)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% survival')
    ax1.axvline(x=min(predicted_months, 60), color='green', linestyle='--', alpha=0.7, label=f'Predicted: {predicted_months:.0f} months')
    ax1.set_xlabel("Months")
    ax1.set_ylabel("Survival Probability (%)")
    ax1.set_title("Estimated Survival Probability Curve")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    # ---------------------------
    # Probability Distribution & Risk Factor Analysis
    # ---------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Probability")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie(
            [1 - combined_death_prob, combined_death_prob],
            labels=["Survival", "Death"],
            autopct="%1.1f%%",
            colors=['#4CAF50', '#F44336'],
            startangle=90,
            explode=(0, 0.05)
        )
        ax2.axis("equal")
        plt.tight_layout()
        st.pyplot(fig2)
    
    with col2:
        st.subheader("Risk Factor Analysis")
        risk_factor_names = {
            'Smoking_History': 'Smoking',
            'Obesity': 'Obesity', 
            'Diabetes': 'Diabetes',
            'Chronic_Pancreatitis': 'Chronic Pancreatitis',
            'Family_History': 'Family History',
            'Hereditary_Condition': 'Hereditary Condition',
            'Jaundice': 'Jaundice',
            'Weight_Loss': 'Weight Loss',
            'Development_of_Type2_Diabetes': 'New Type 2 Diabetes'
        }
        
        risk_factors_present = []
        for key, name in risk_factor_names.items():
            if input_dict.get(key, 0) == 1:
                risk_factors_present.append(name)
        
        stage_names = {0: 'Stage I', 1: 'Stage II', 2: 'Stage III', 3: 'Stage IV'}
        current_stage = stage_names[input_dict['Stage_at_Diagnosis']]
        
        st.write(f"**Cancer Stage:** {current_stage}")
        st.write(f"**Age:** {input_dict['Age']} years")
        st.write(f"**Number of Risk Factors:** {len(risk_factors_present)}")
        
        if risk_factors_present:
            st.write("**Active Risk Factors:**")
            for rf in risk_factors_present:
                st.write(f"  - {rf}")
        else:
            st.write("No major risk factors present")

    # ---------------------------
    # Feature Overview
    # ---------------------------
    st.subheader("Patient Feature Overview")

    # Use only the actual input features for visualization (not the full model features)
    visual_dict = {
        'Gender': input_dict['Gender'],
        'Smoking': input_dict['Smoking_History'],
        'Obesity': input_dict['Obesity'],
        'Diabetes': input_dict['Diabetes'],
        'Chronic Pancreatitis': input_dict['Chronic_Pancreatitis'],
        'Family History': input_dict['Family_History'],
        'Hereditary': input_dict['Hereditary_Condition'],
        'Jaundice': input_dict['Jaundice'],
        'Abdominal Discomfort': input_dict['Abdominal_Discomfort'],
        'Back Pain': input_dict['Back_Pain'],
        'Weight Loss': input_dict['Weight_Loss'],
        'Type 2 Diabetes': input_dict['Development_of_Type2_Diabetes'],
        'Stage': input_dict['Stage_at_Diagnosis'],
        'Treatment': input_dict['Treatment_Type'],
        'Alcohol': input_dict['Alcohol_Consumption'],
        'Physical Activity': input_dict['Physical_Activity_Level'],
        'Processed Food': input_dict['Diet_Processed_Food'],
        'Healthcare Access': input_dict['Access_to_Healthcare'],
        'Urban/Rural': input_dict['Urban_vs_Rural'],
        'Economic Status': input_dict['Economic_Status']
    }
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    feature_names = list(visual_dict.keys())
    feature_values = list(visual_dict.values())
    x_positions = range(len(feature_names))
    
    bars = ax3.bar(x_positions, feature_values, color='#2196F3', alpha=0.7, edgecolor='#1976D2')
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, feature_values)):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    str(int(val)), ha='center', va='bottom', fontsize=7)
    
    ax3.set_title("Clinical and Lifestyle Factors")
    ax3.set_ylabel("Encoded Value")
    ax3.set_xlabel("Features")
    ax3.set_ylim(0, 4)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig3)
