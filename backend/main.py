import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

print(f"Starting Server. Loading model: {MODEL_NAME}...")

# Simulation einer langen Ladezeit (z.B. großes LLM)
time.sleep(10)

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"FATAL: Could not load model. Error: {e}")
    raise e


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health_check():
    """Einfacher Health-Check für Liveness Probes"""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(request: TextRequest):
    """Führt die Inferenz durch"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = sentiment_pipeline(request.text)
    # Resultat ist z.B. [{'label': 'POSITIVE', 'score': 0.99}]
    return result[0]
