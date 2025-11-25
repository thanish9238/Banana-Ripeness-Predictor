# backend/banana_api_fastapi.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
import cv2
import os
from uuid import uuid4

# -----------------------
# App Setup
# -----------------------
app = FastAPI(title="🍌 Banana Ripeness Predictor")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load Model
# -----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Class mapping
id_to_category = {
    0: "Green",
    1: "Yellow",
    2: "Brown Spots",
    3: "Black/Mushy"
}

# Image settings
IMG_SIZE = (224, 224)

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "Banana Predictor API is running 🍌"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        filename = f"{uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Preprocess image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict → your model must output [classification, regression]
        pred_class, pred_days = model.predict(img)
        class_id = int(np.argmax(pred_class[0]))
        category = id_to_category[class_id]
        confidence = float(np.max(pred_class[0])) * 100
        days_left = round(float(pred_days[0][0]))

        return JSONResponse({
            "category": category,
            "confidence": round(confidence, 2),
            "days_left": days_left,
            "image_url": f"/uploads/{filename}"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("banana_api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
