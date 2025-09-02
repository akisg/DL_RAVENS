# Use Python 3.12 as base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libhdf5-dev \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*


# Clone the mri_pipeline repository
RUN git clone https://github.com/akisg/mri_pipeline.git

# Set working directory
WORKDIR /mri_pipeline

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install core scientific packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    nibabel \
    scipy \
    scikit-learn

# Install SimpleITK
RUN pip install --no-cache-dir SimpleITK

# Install TensorFlow and Keras
RUN pip install --no-cache-dir \
    tensorflow==2.18 \
    keras==3.08

# Install packages from GitHub
RUN pip install --no-cache-dir \
    git+https://github.com/adalca/neurite.git@0776a575eadc3d10d6851a4679b7a305f3a31c65 \
    git+https://github.com/freesurfer/surfa.git@ec5ddb193fd1caf22ec654c457b5678f6bd8e460 \
    git+https://github.com/voxelmorph/voxelmorph.git@2cd706189cb5919c6335f34feec672f05203e36b


# Download the model file
RUN curl -O https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5
# RUN curl -O https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5

# Create directories for inputs and outputs
RUN mkdir -p inputs outputs

# Set environment variables for the pipeline
ENV INPUT_DIR=/mri_pipeline/inputs
ENV OUTPUT_DIR=/mri_pipeline/outputs

# Set default command
CMD ["python", "/mri_pipeline/synthmorph_ravens_pipeline.py"]