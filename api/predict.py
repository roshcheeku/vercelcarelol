import json
import pandas as pd
import warnings
from download_models import get_models
from starlette.requests import Request
from starlette.responses import JSONResponse

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

diabetes_model, scaler, multi_condition_model = get_models()

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

# ✅ Vercel Serverless Function handler
async def handler(request: Request):
    try:
        data = await request.json()

        required_fields = ["gender", "age", "glucose", "bmi", "systolic", "diastolic"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return JSONResponse({"error": f"Missing fields: {', '.join(missing_fields)}"}, status_code=400)

        gender = data["gender"].strip().lower()
        if gender not in ["male", "female"]:
            return JSONResponse({"error": "Invalid gender input, must be 'male' or 'female'"}, status_code=400)
        data["Gender"] = 1 if gender == "male" else 0

        try:
            systolic = float(data["systolic"])
            diastolic = float(data["diastolic"])
            age = int(data["age"])
            glucose = float(data["glucose"])
            bmi = float(data["bmi"])
        except ValueError:
            return JSONResponse({"error": "Invalid numerical input"}, status_code=400)

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

        return JSONResponse(results)

    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {str(e)}"}, status_code=500)
