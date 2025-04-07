import os
import requests
import joblib
import logging
import pandas as pd
import numpy as np
import warnings
from http.server import BaseHTTPRequestHandler
import json

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
logging.basicConfig(level=logging.INFO)

# Direct Google Drive download links (file IDs)
MODEL_FILES = {
    "DIABETES_MODEL": {
        "id": "14H_wPtW4_W1XPFiiJ3tkmsFFACac_jxS",
        "filename": "finaliseddiabetes_model.joblib"
    },
    "SCALER": {
        "id": "1PnILhtH35yVwG1xfd0bNj7jBquX855bk",
        "filename": "finalisedscaler.joblib"
    },
    "MULTI_MODEL": {
        "id": "1cnjaKDyR7AiCojKsm0rZYtQZSCiJVWW6",
        "filename": "nodiabetes.joblib"
    }
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def gdrive_download(file_id, destination):
    """Download file from Google Drive by file ID."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Downloaded {destination}")
        else:
            raise Exception(f"Failed to download {destination}")
    except Exception as e:
        logging.error(f"Download error for {destination}: {e}")

def load_models():
    models = {}
    for key, info in MODEL_FILES.items():
        file_path = os.path.join(BASE_DIR, info["filename"])
        if not os.path.exists(file_path):
            gdrive_download(info["id"], file_path)
        models[key] = joblib.load(file_path)
    return models

models = load_models()

FEATURE_ORDER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def validate_input(value, input_type=float, min_value=0, max_value=None):
    try:
        value = input_type(value)
        if value < min_value:
            return None
        if max_value and value > max_value:
            return None
        return value
    except:
        return None

def validate_gender(gender):
    return 1 if gender and gender.lower() == 'male' else 0 if gender and gender.lower() == 'female' else None

def calculate_diabetes_pedigree(family, first=0, second=0):
    return min((first * 0.5 + second * 0.25) if family else 0.0, 1.0)

def get_diabetes_prediction(model, df):
    try:
        pred = model.predict(df)[0]
        prob = float(model.predict_proba(df)[0][1]) * 100
        return ('Diabetes' if pred else 'No Diabetes'), prob
    except:
        return None, 0.0

# Vercel's handler
def handler(request):
    if request["method"] == "GET":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "healthy", "message": "Service is running"})
        }

    elif request["method"] == "POST":
        try:
            body = json.loads(request["body"])
            gender = validate_gender(body.get('gender'))
            systolic = validate_input(body.get('systolic'))
            diastolic = validate_input(body.get('diastolic'))
            age = validate_input(body.get('age'))
            glucose = validate_input(body.get('glucose'))
            bmi = validate_input(body.get('bmi'))

            if not all([gender is not None, systolic, diastolic, age, glucose, bmi]):
                return {
                    "statusCode": 400,
                    "body": json.dumps({"status": "error", "error": "Invalid input values"})
                }

            if systolic < 90 or diastolic < 60:
                df_multi = pd.DataFrame([{
                    'Age': age,
                    'Gender': gender,
                    'Systolic_bp': systolic,
                    'Diastolic_bp': diastolic,
                    'Glucose': glucose,
                    'BMI': bmi
                }])
                multi_model = models['MULTI_MODEL']
                preds = multi_model.predict(df_multi)[0]
                probs = multi_model.predict_proba(df_multi)

                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "status": "success",
                        "model": "multi-condition",
                        "predictions": {
                            "hypertension": bool(preds[0]),
                            "cardiovascular_risk": float(probs[1][0][1]),
                            "stroke_risk": float(probs[2][0][1]),
                            "diabetes_risk": float(probs[3][0][1])
                        }
                    })
                }

            pregnancies = validate_input(body.get('pregnancies', 0))
            insulin = validate_input(body.get('insulin'))
            first = validate_input(body.get('first_degree_relatives', 0))
            second = validate_input(body.get('second_degree_relatives', 0))
            family = body.get('family_history', False)

            pedigree = calculate_diabetes_pedigree(family, first, second)

            df = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': systolic,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': pedigree,
                'Age': age
            }])
            df = df[FEATURE_ORDER]
            scaled_df = models['SCALER'].transform(df)
            prediction, prob = get_diabetes_prediction(models['DIABETES_MODEL'], scaled_df)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "status": "success",
                    "model": "diabetes-only",
                    "prediction": prediction,
                    "probability": prob
                })
            }

        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"status": "error", "error": str(e)})
            }

    else:
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Method Not Allowed"})
        }
