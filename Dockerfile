FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app/CKE

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH="/app/CKE:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for Gradio
EXPOSE 8501

CMD ["/bin/bash"]