FROM python:3.10-slim

# Install system dependencies required for OpenCV, PyQt5 (labelImg), and downloading data
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    tar \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add scripts directory to PATH and root to PYTHONPATH so scripts can be executed easily
ENV PATH="/app/scripts:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set the default command to bash for interactive usage
CMD ["/bin/bash"]
