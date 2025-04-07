from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import warnings
import requests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)

# Remote model URLs
MODEL_URLS = {
    "diabetes": "https://drive.google.com/uc?export=download&id=14H_wPtW4_W1XPFiiJ3tkmsFFACac_jxS",
    "scaler": "https://drive.google.com/uc?export=download&id=1PnILhtH35yVwG1xfd0bNj7jBquX855bk",
    "multi": "https://drive.google.com/uc?export=download&id=1cnjaKDyR7AiCojKsm0rZYtQZSCiJVWW6"
}

TMP_DIR = "/tmp"

# File paths on disk
DIABETES_MODEL_PATH = os.path.join(TMP_DIR, "diabetes_model.joblib")
SCALER_PATH = os.path.join(TMP_DIR, "scaler.joblib")
MULTI_MODEL_PATH = os.path.join(TMP_DIR, "multi_model.joblib")

# Function to download model if not already cached
def download_model(url, save_path):
    if not os.path.exists(save_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download model from {url}")
            return None
    return joblib.load(save_path)

# Load models
diabetes_model = download_model(MODEL_URLS["diabetes"], DIABETES_MODEL_PATH)
scaler = download_model(MODEL_URLS["scaler"], SCALER_PATH)
multi_condition_model = download_model(MODEL_URLS["multi"], MULTI_MODEL_PATH)

FEATURE_ORDER = ['Glucose', 'Insulin', 'BloodPressure', 'BMI', 
                 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']

def determine_risk(probability):
    if probability >= 75:
        return "HIGH"
    elif probability >= 40:
        return "MODERATE"
    else:
        return "LOW"

def get_multi_condition_predictions(df):
    if multi_condition_model is None:
        return {"error": "Multi-condition model is not available."}
    try:
        predictions = multi_condition_model.predict(df)[0]
        probs_list = multi_condition_model.predict_proba(df)
        if len(probs_list) < 4:
            raise ValueError("Unexpected probability output.")
        return {
            'model': "multi-condition",
            'predictions': {
                'hypertension': bool(predictions[0]),
                'cardiovascular_risk': float(probs_list[1][0][1]),
                'stroke_risk': float(probs_list[2][0][1]),
                'diabetes_risk': float(probs_list[3][0][1])
            }
        }
    except Exception as e:
        return {"error": f"Multi-condition prediction failed: {str(e)}"}

def get_diabetes_prediction(df):
    if diabetes_model is None or scaler is None:
        return {"error": "Diabetes model or scaler is not available."}
    try:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        prediction = diabetes_model.predict(df_scaled)[0]
        probability = diabetes_model.predict_proba(df_scaled)[0][1] * 100
        risk_level = determine_risk(probability)
        return {
            "model": "diabetes",
            "prediction": bool(prediction),
            "probability": float(probability),
            "risk_level": risk_level
        }
    except Exception as e:
        return {"error": f"Diabetes prediction failed: {str(e)}"}

@app.route('/predict', methods=['POST'])
def predict_health():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        required_fields = ["gender", "age", "glucose", "bmi", "systolic", "diastolic"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        gender = data["gender"].strip().lower()
        if gender not in ["male", "female"]:
            return jsonify({"error": "Invalid gender input, must be 'male' or 'female'"}), 400
        data["Gender"] = 1 if gender == "male" else 0

        try:
            systolic = float(data["systolic"])
            diastolic = float(data["diastolic"])
            age = int(data["age"])
            glucose = float(data["glucose"])
            bmi = float(data["bmi"])
        except ValueError:
            return jsonify({"error": "Invalid numerical input"}), 400

        use_multi_condition = systolic < 90 or diastolic < 60

        if use_multi_condition:
            df_multi = pd.DataFrame([{
                "Age": age,
                "Gender": data["Gender"],
                "Systolic_bp": systolic,
                "Diastolic_bp": diastolic,
                "Glucose": glucose,
                "BMI": bmi
            }])
            results = get_multi_condition_predictions(df_multi)
        else:
            df_diabetes = pd.DataFrame([{
                "Pregnancies": data.get("pregnancies", 0),
                "Glucose": glucose,
                "BloodPressure": systolic,
                "Insulin": data.get("insulin", 0),
                "BMI": bmi,
                "Age": age,
                "DiabetesPedigreeFunction": data.get("diabetes_pedigree", 0.0)
            }])
            df_diabetes = df_diabetes[FEATURE_ORDER]
            results = get_diabetes_prediction(df_diabetes)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# This part isn't needed for Vercel (it uses ASGI/WGI adapter)
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000, debug=True)
