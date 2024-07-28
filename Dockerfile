# Use the official Python 3.12 image from the Docker Hub
FROM python:3.12

# Copy the requirements file into the container
COPY ./requirements.txt /api/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /api/requirements.txt

# Copy the rest of the application code into the container
COPY ./api /app

# Specify the command to run the app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
