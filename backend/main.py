# fast API backend to serve predictions from the model
# Its job: receive an image from the frontend,
# run it through the model, return the predicted digit.
# The frontend never touches the model directly -- it goes through here.

# WHY FastAPI specifically:
# - Extremely fast (built on async Python)
# - Auto-generates documentation at /docs
# - Type hints make the API self-documenting
# - Easy to deploy to cloud services later


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
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile
# UploadFile = handles receiving image files from HTTP requests.
# This is how the canvas drawing will be sent from Next.js.

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

    # CORS = Cross-Origin Resource Sharing.
    # Browsers BLOCK requests between different origins by default.
    # Next.js app (localhost:3000) talking to FastAPI (localhost:8000)
    # = different origins = blocked without this middleware.
    # allow_origins=["http://localhost:3000"] whitelists frontend.
)

# load model once at startup
model = None
# if we're running from backend folder, model path = data/final_model.keras, else if running from project root, model path = backend/data/final_model.keras
if os.path.exists('data/final_model.keras'):
    # we're running from backend folder
    model = keras.models.load_model('data/final_model.keras')
elif os.path.exists('backend/data/final_model.keras'):
    # we're running from project root
    model = keras.models.load_model('backend/data/final_model.keras')
else:
    raise FileNotFoundError("Could not find final_model.keras -- check your working directory")

# Loaded ONCE when the server starts, stays in memory.
# WHY not load it inside the /predict function:
# Loading a model takes ~1-2 seconds. If we did it per-request,
# every prediction would be slow. Loading once = instant predictions.

print("Model loaded and ready")

@app.post("/predict")
async def predict(file: UploadFile):
# @app.post = this function handles POST requests to /predict.
# POST because we're SENDING data (the image) to the server.
# GET requests are for fetching data with no body.
# async = non-blocking. While waiting for file upload,
# the server can handle other requests. Important for performance.

    # read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # preprocess to match training format
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    # CRITICAL: the image must be preprocessed EXACTLY the same way
    # as the training data. Same size, same normalization, same shape.
    # This is one of the most common bugs in ML deployment --
    # training and inference preprocessing don't match.


    # predict
    predictions = model.predict(img)
    # Returns an array of 10 probabilities, one per digit.
    # e.g. [0.01, 0.02, 0.90, 0.03, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01]

    digit = int(np.argmax(predictions))
    # argmax returns the INDEX of the highest value = predicted digit.    

    confidence = float(np.max(predictions))
    # The actual probability of that prediction.
    # Will use this in the frontend to decide whether to
    # show the prediction or ask the user to confirm.
    # e.g. confidence < 0.7 -> show "Did you mean X?" prompt.


    return {
        "digit": digit,
        "confidence": round(confidence, 3)
    }

@app.get("/health")
def health():
    return {"status": "ok"}