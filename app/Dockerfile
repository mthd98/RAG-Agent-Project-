# Use the official Python 3.12 slim image as a base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Ensure Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Copy the requirements file into the container
COPY requirements-docker.txt ./requirements.txt

# Install system dependencies needed for some Python packages,
# install Python packages from requirements.txt,
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \   
    tesseract-ocr-eng \  
    libgl1 \           
    libglib2.0-0 \    
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install -v --no-cache-dir -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# Make sure 'main:app' matches your FastAPI entry point (main.py file, app object)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]