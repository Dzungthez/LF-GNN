# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Install unzip utility
RUN apt-get update && apt-get install -y unzip

# Download the dataset from Kaggle
RUN kaggle datasets download -d nhddddz84/lf-gcn-data -p /app/data

# Extract the dataset if necessary (assuming it's a zip file)
RUN unzip /app/data/lf-gcn-data.zip -d /app/data

# Run the training script
CMD ["python", "main.py"]
