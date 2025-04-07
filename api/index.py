import os
import requests
import joblib
import logging
import zipfile
import pandas as pd
import numpy as np
import warnings
from http.server import BaseHTTPRequestHandler
import json
from urllib.parse import parse_qs

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Environment variables
DIABETES_MODEL_URL = os.getenv("DIABETES_MODEL_URL")
SCALER_URL = os.getenv("SCALER_URL")
MULTI_MODEL_URL = os.getenv("MULTI_MODEL_URL")

MODEL_PATHS = {
    "DIABETES_MODEL": "finaliseddiabetes_model.zip",
    "SCALER": "finalisedscaler.zip",
    "MULTI_MODEL": "nodiabetes.zip",
}

EXTRACTED_MODELS = {
    "DIABETES_MODEL": "finaliseddiabetes_model.joblib",
    "SCALER": "finalisedscaler.joblib",
    "MULTI_MODEL": "nodiabetes.joblib",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_model(url, zip_filename):
    zip_path = os.path.join(BASE_DIR, zip_filename)
    if not url:
        logging.error(f"Missing URL for {zip_filename}")
        return False
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        logging.error(f"Error downloading {zip_filename}: {e}")
        return False

def extract_if_needed(zip_filename, extracted_filename):
    zip_path = os.path.join(BASE_DIR, zip_filename)
    extracted_path = os.path.join(BASE_DIR, extracted_filename)
    if os.path.exists(extracted_path):
        return True
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        return True
    except Exception as e:
        return False

def load_model(model_filename):
    path = os.path.join(BASE_DIR, model_filename)
    return joblib.load(path) if os.path.exists(path) else None

def initialize_models():
    models = {}
    for key in MODEL_PATHS:
        zip_file = MODEL_PATHS[key]
        extracted = EXTRACTED_MODELS[key]
        url = globals().get(f"{key}_URL", "")
        if not os.path.exists(os.path.join(BASE_DIR, extracted)):
            download_model(url, zip_file)
            extract_if_needed(zip_file, extracted)
        models[key] = load_model(extracted)
    return models

models = initialize_models()

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
    return 1 if gender.lower() == 'male' else 0 if gender.lower() == 'female' else None

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

            # Use multi-model logic
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

            # Use diabetes-only model
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
