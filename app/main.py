from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import  List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),   # 📁 ملف
        logging.StreamHandler()           # 💻 terminal
    ]
)

class Features(BaseModel):
    features: List[float]

app = FastAPI()
model = joblib.load("model/iris_model.pkl")

@app.get("/")
def home():
    return {"message": "Iris Model API is running"}

@app.post("/predict")
def predict(features: Features):
    logging.info(f"Received input: {features.features}")
    features = np.array(features.features).reshape(1, -1)
    prediction = model.predict(features)
    logging.info(f"Prediction: {prediction[0]}")
    return {"prediction": int(prediction[0])}