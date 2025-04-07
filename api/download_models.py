import os
import requests
import joblib

TMP_DIR = "/tmp"

MODEL_URLS = {
    "diabetes": "https://drive.google.com/uc?export=download&id=14H_wPtW4_W1XPFiiJ3tkmsFFACac_jxS",
    "scaler": "https://drive.google.com/uc?export=download&id=1PnILhtH35yVwG1xfd0bNj7jBquX855bk",
    "multi": "https://drive.google.com/uc?export=download&id=1cnjaKDyR7AiCojKsm0rZYtQZSCiJVWW6"
}

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

def get_models():
    diabetes_model = download_model(MODEL_URLS["diabetes"], os.path.join(TMP_DIR, "diabetes_model.joblib"))
    scaler = download_model(MODEL_URLS["scaler"], os.path.join(TMP_DIR, "scaler.joblib"))
    multi_condition_model = download_model(MODEL_URLS["multi"], os.path.join(TMP_DIR, "multi_model.joblib"))
    return diabetes_model, scaler, multi_condition_model
