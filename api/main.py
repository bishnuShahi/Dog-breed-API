"""
FastAPI Application for Image Classification

This application provides an API endpoint for image classification using a pre-trained FastAI model.
It accepts image uploads, validates them, and returns classification predictions.
"""

import os
import tempfile
import logging
import traceback
import pathlib
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner
from PIL import Image

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Configuration
MODEL_PATH = Path("api/model.pkl")
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

async def validate_image_data(image_data: bytes, filename: str) -> str:
    """
    Validate and save the image data to a temporary file.

    Args:
        image_data (bytes): The image data.
        filename (str): The filename of the uploaded image.

    Returns:
        str: Path to the temporarily saved image file.

    Raises:
        HTTPException: If the file is invalid or processing fails.
    """
    extension = filename.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid image format. Allowed formats are: png, jpg, jpeg")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}') as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name

        with Image.open(temp_file_path) as img:
            try:
                img.verify()
            except Exception as e:
                logging.error(f'Error verifying image: {e}')
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

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
        probs = [round(float(p) * 100) for p in top3_probs]
        
        return {label: prob for label, prob in zip(top3_labels, probs)}
    
    except Exception as e:
        logging.error(f'Error making prediction: {e}')
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/", summary="Classify an image", response_description="Image classification results")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint for classifying an image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: Dictionary containing top 3 predictions and their probabilities.

    Raises:
        HTTPException: If the file is invalid or processing fails.
    """
    try:
        # Get the image data from the file object
        image_data = await file.read()

        # Validate and save the image data
        img_path = await validate_image_data(image_data, file.filename)

        # Make predictions on the image
        predictions = predict(img_path)

        # Remove the temporary image file
        os.remove(img_path)

        return predictions

    except HTTPException as e:
        logging.error(f'Error processing image: {e}')
        logging.error(traceback.format_exc())
        raise e

    except Exception as e:
        logging.error(f'Error processing image: {e}')
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")


