import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
df = pd.read_csv("NYPD_Shooting_Incident_Data__Historic_.csv")

# 2. Data Preprocessing
cols_to_drop = ["INCIDENT_KEY", "LOC_OF_OCCUR_DESC"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
df.fillna("Unknown", inplace=True)

if "OCCUR_TIME" in df.columns:
    df["OCCUR_HOUR"] = pd.to_datetime(df["OCCUR_TIME"], format="%H:%M:%S").dt.hour
    df = df.drop(columns=["OCCUR_TIME", "OCCUR_DATE"])

# 3. Label Encoding
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    if col != "STATISTICAL_MURDER_FLAG":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

df["STATISTICAL_MURDER_FLAG"] = df["STATISTICAL_MURDER_FLAG"].astype(int)

# 4. Train/Test Split
X = df.drop(columns=["STATISTICAL_MURDER_FLAG"])
y = df["STATISTICAL_MURDER_FLAG"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Model Training
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train_scaled, y_train)

# 7. Save Model Artifacts
joblib.dump(rf_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("Training complete. Artifacts saved successfully.")
