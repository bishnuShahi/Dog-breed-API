"""
FastAPI Application for Image Classification

This application provides an API endpoint for image classification using a pre-trained FastAI model.
It accepts image uploads, validates them, and returns classification predictions.
"""

import os
import tempfile
import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner
from PIL import Image


# Configuration
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_PATH = f"{BASE_DIR}/model.pkl"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# FastAPI app initialization
app = FastAPI(title="Dog Breed Image Classification API",
              description="API for classifying dog breeds using a FastAI model",
              version="1.0.0")

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Check API status", response_description="API status message")
async def root():
    """
    Root endpoint to check if the API is running.

    Returns:
        dict: A dictionary containing the API title and description.
    """
    return {
        "message": "API is running",
        "title": app.title,
        "description": app.description
    }

def load_model(model_path):
    """
    Load the pre-trained FastAI model.

    Args:
        model_path (Path): Path to the model file.

    Returns:
        fastai.learner.Learner: Loaded FastAI model.

    Raises:
        Exception: If model loading fails.
    """
    try:
        model = load_learner(model_path)
        logging.info('Model loaded successfully')
        return model
    except Exception as e:
        logging.error(f'Failed to load model: {e}')
        logging.error(traceback.format_exc())
        raise

# Load the model
learn = load_model(MODEL_PATH)

async def validate_image(file: UploadFile) -> str:
    """
    Validate and save the uploaded image file.

    Args:
        file (UploadFile): The uploaded file.

    Returns:
        str: Path to the temporarily saved image file.

    Raises:
        HTTPException: If the file is invalid or processing fails.
    """
    if file.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid image format. Allowed formats are: png, jpg, jpeg")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        with Image.open(temp_file_path) as img:
            img.verify()

        return temp_file_path

    except Exception as e:
        logging.error(f'Error validating image: {e}')
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def predict(img_path: str) -> dict:
    """
    Make predictions on the given image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        dict: Dictionary containing top 3 predictions and their probabilities.

    Raises:
        ValueError: If prediction fails.
    """
    try:
        pred, _, probs = learn.predict(img_path)
        probs_np = probs.numpy()
        top3_idxs = probs_np.argsort()[-3:]
        top3_probs = probs_np[top3_idxs]
        top3_labels = [learn.dls.vocab[i] for i in top3_idxs]
        probs = [round(float(p), 2) * 100 for p in top3_probs]
        
        return {label: prob for label, prob in zip(top3_labels, probs)}
    
    except Exception as e:
        logging.error(f'Error making prediction: {e}')
        logging.error(traceback.format_exc())
        raise ValueError(f"Error making prediction: {str(e)}") from e

@app.post("/predict", summary="Predict image classification", 
          response_description="Image classification predictions")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict the classification of an uploaded image.

    Args:
        file (UploadFile): The image file to be classified.

    Returns:
        dict: Prediction results.

    Raises:
        HTTPException: For various error conditions.
    """
    temp_file_path = None
    try:
        temp_file_path = await validate_image(file)
        prediction = predict(temp_file_path)
        return {"prediction": prediction}
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)