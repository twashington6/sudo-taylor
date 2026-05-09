# fast API backend to serve predictions from the model

# if running from backened folder, use:
# uvicorn main:app --reload 
# and ensure the model is in data/final_model.keras

# if running from project root, use:
# uvicorn backend.main:app --reload
# and ensure the model is in backend/data/final_model.keras

# press ctrl+c in terminal to stop the server when done
# for full app, will need to run both backend and frontend servers:
# 1. in backend folder: uvicorn main:app --reload
# 2. in frontend folder: npm run dev

# http://localhost:8000/health # to check if server is running
# http://localhost:8000/docs # for interactive API docs. upload a 28x28 grayscale image for testing

# imports
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once at startup
model = keras.models.load_model('data/final_model.keras')
print("Model loaded and ready")

@app.post("/predict")
async def predict(file: UploadFile):
    # read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # preprocess to match training format
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # predict
    predictions = model.predict(img)
    digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "digit": digit,
        "confidence": round(confidence, 3)
    }

@app.get("/health")
def health():
    return {"status": "ok"}