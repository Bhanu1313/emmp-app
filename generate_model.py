import pandas as pd
import joblib
import os
import gdown
from sklearn.ensemble import RandomForestClassifier
from features import FEATURES

# Step 1: Download model.pkl from Google Drive if not already present
model_file = "model.pkl"
google_drive_url = "https://drive.google.com/uc?id=1T06Ndwy7av1-qY_GQ5ukBjKigUktfIN4"

if not os.path.exists(model_file):
    print("Downloading model from Google Drive...")
    gdown.download(google_drive_url, model_file, quiet=False)

# Step 2: Load dataset and process
data = pd.read_csv("adult 3.csv")
data.columns = data.columns.str.strip().str.lower()

df = data[FEATURES]
target = data['income']

# Step 3: Load encoders and transform categorical features
encoders = joblib.load("encoders.pkl")
for col in df.select_dtypes(include='object').columns:
    df[col] = encoders[col].transform(df[col])

# Step 4: Scale the data
scaler = joblib.load("scaler.pkl")
df_scaled = scaler.transform(df)

# Step 5: Load pre-trained model
model = joblib.load(model_file)

# Step 6: (Optional) If you want to retrain and overwrite
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(df_scaled, target)
# joblib.dump(model, "model.pkl")
