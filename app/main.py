import sys
import os

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from app.preprocess import clean_text
#load trained artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
vectorizer = joblib.load(
    os.path.join(BASE_DIR,"models/artifacts/tfidf_vectorizer.pkl")
)

model = joblib.load(
    os.path.join(BASE_DIR, "models/artifacts/svm_classifier.pkl")
)

#FASTAPI app

app= FastAPI(
    title = "CyberShield AI",
    decription ="Real-time scam message detection API",
    version="1.0"
)

#Request Schema
class MessageInput(BaseModel):
    message:str

#prediction endpoint
@app.post("/predict")
def predict(input: MessageInput):
    cleaned = clean_text(input.message)

    vector = vectorizer.transform([cleaned])
    prediction =model.predict(vector)[0]

    confidence =None
    if hasattr(model,"decision_function"):
        scores = model.decision_function(vector)
        confidence = float(np.max(scores))

    return {
        "prediction":prediction,
        "confidence": round(confidence,3) if confidence is not None else None
    }
                           