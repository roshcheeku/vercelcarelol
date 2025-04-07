import joblib
import os
import requests
import numpy as np
import logging
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tensorflow as tf

router = APIRouter()

# Set up constants
BASE_DIR = "/tmp"  # Only /tmp is writeable in Vercel
logging.basicConfig(level=logging.INFO)

# Google Drive model URLs
MODEL_URLS = {
    "DIABETES_MODEL": "https://drive.google.com/uc?export=download&id=14H_wPtW4_W1XPFiiJ3tkmsFFACac_jxS",
    "SCALER": "https://drive.google.com/uc?export=download&id=1PnILhtH35yVwG1xfd0bNj7jBquX855bk",
    "MULTI_MODEL": "https://drive.google.com/uc?export=download&id=1cnjaKDyR7AiCojKsm0rZYtQZSCiJVWW6"
}

model_cache = {}  # Cache loaded models


def download_model(url, path):
    if os.path.exists(path):
        logging.info(f"Model already exists at {path}")
        return True
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded model to {path}")
            return True
        else:
            logging.error(f"Failed to download from {url}")
            return False
    except Exception as e:
        logging.error(f"Download error: {e}")
        return False


def get_or_load_model(name):
    if name in model_cache:
        return model_cache[name]
    path = os.path.join(BASE_DIR, f"{name}.joblib")
    if download_model(MODEL_URLS[name], path):
        model = joblib.load(path)
        model_cache[name] = model
        return model
    return None


def preprocess_image(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_disease(file: UploadFile):
    model = get_or_load_model("MULTI_MODEL")
    if model is None:
        return "Model not loaded"
    image_array = preprocess_image(file)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = [
        "Acne", "Eczema", "Psoriasis", "Rosacea", "Melanoma", "Lupus",
        "Vitiligo", "Cellulitis", "Hives", "Shingles"
    ]
    return class_names[predicted_class]


def predict_diabetes(bp, age, bmi):
    model = get_or_load_model("DIABETES_MODEL")
    scaler = get_or_load_model("SCALER")
    if model is None or scaler is None:
        return "Model/Scaler not loaded"
    input_data = np.array([[bp, age, bmi]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return "Positive" if prediction == 1 else "Negative"


@router.post("/predict_skin")
async def predict_skin(file: UploadFile = File(...)):
    try:
        result = predict_disease(file)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/predict_diabetes")
async def predict_diabetes_api(
    blood_pressure: float = Form(...),
    age: float = Form(...),
    bmi: float = Form(...)
):
    try:
        result = predict_diabetes(blood_pressure, age, bmi)
        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
