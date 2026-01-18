import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Base directory (VERY IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("🚀 Script started")
print("📁 Working directory:", BASE_DIR)

# -----------------------------
# Load data
# -----------------------------
csv_path = os.path.join(
    BASE_DIR,
    "pancreatic_cancer_prediction_sample.csv"
)

df = pd.read_csv(csv_path)
print("✅ Dataset loaded:", df.shape)

# -----------------------------
# Basic cleaning
# -----------------------------
df.columns = df.columns.str.lower()
df['gender'] = df['gender'].str.lower()
df['stage_at_diagnosis'] = df['stage_at_diagnosis'].str.replace(' ', '_')

# -----------------------------
# Encode categorical variables
# -----------------------------
df_encoded = pd.get_dummies(
    df,
    columns=[
        'gender',
        'treatment_type',
        'stage_at_diagnosis',
        'urban_vs_rural',
        'country',
        'physical_activity_level',
        'diet_processed_food',
        'access_to_healthcare',
        'economic_status'
    ],
    drop_first=True
)

# -----------------------------
# Split features & target
# -----------------------------
X = df_encoded.drop(columns=['survival_status'])
y = df_encoded['survival_status']

# Save feature names
feature_path = os.path.join(BASE_DIR, "feature_columns.pkl")
joblib.dump(X.columns.tolist(), feature_path)
print("✅ feature_columns.pkl saved")

# -----------------------------
# Scale data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print("✅ scaler.pkl saved")

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

model_path = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, model_path)
print("✅ model.pkl saved")

print("🎉 Training completed successfully!")
