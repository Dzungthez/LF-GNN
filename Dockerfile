# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download the dataset from Kaggle
RUN kaggle datasets download -d nhddddz84/lf-gcn-data -p /app/data

# Extract the dataset if necessary (assuming it's a zip file)
RUN unzip /app/data/lf-gcn-data.zip -d /app/data

# Run the training script
CMD ["python3", "main.py"]
