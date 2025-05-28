from fastapi import FastAPI
from src.inference import predict_all
app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    texts = payload["texts"]
    preds = predict_all(texts)
    return {"predictions": preds}