# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip3 install --upgrade pip

# Install torch and torchvision first to resolve dependency issues
RUN pip3 install torch==2.0.1 torchvision==0.15.2

# Install PyTorch Geometric dependencies
# RUN pip3 install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# RUN pip3 install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# RUN pip3 install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# RUN pip3 install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# RUN pip3 install torch-geometric==2.1.0

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install remaining Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download the dataset from Kaggle
RUN kaggle datasets download -d nhddddz84/lf-gcn-data -p /app/data

# Extract the dataset if necessary (assuming it's a zip file)
RUN unzip /app/data/lf-gcn-data.zip -d /app/data

# Ensure the run.sh script is executable
RUN chmod +x run.sh

# Run the training script
CMD ["./run.sh"]
