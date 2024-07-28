# Dog Breed Image Classification API

This is a FastAPI application that provides an API endpoint for classifying dog breeds from uploaded images using a pre-trained FastAI model.

## Features

- Accepts image uploads in PNG, JPG, or JPEG format
- Validates and processes the uploaded image
- Returns the top 3 predicted dog breeds and their probabilities

## Requirements

- Python 3.7 or higher
- FastAI
- Pillow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/dog-breed-classifier.git
```

Install the required packages:
```bash
pip install -r requirements.txt
```


##Usage
Start the FastAPI server:

```bash
uvicorn app.api.main:app --reload
```

The API will be available at 
```
http://localhost:8000
```

To check if the API is running, send a GET request to the root endpoint:

```
GET http://localhost:8000/
```

To classify a dog breed from an image, send a POST request to the 
/predict
 endpoint with the image file:

```
POST http://localhost:8000/predict
```

Content-Type: multipart/form-data

file: <image_file>


The API will return a JSON response with the top 3 predicted dog breeds and their probabilities:

```json
{
    "prediction": {
        "breed1": 95.23,
        "breed2": 3.12,
        "breed3": 1.65
    }
}
```


#License
This project is licensed under the MIT License.
