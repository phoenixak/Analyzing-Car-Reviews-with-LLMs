# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV TOKENIZERS_PARALLELISM=false

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies with optimizations for Docker
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Download NLTK data and model data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')" || true

# Pre-download the sentiment model used in config
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')" || true

# Copy the project code into the container
COPY . .

# Create necessary directories
RUN mkdir -p results logs models

# Set proper permissions
RUN chmod +x main.py

# Health check to ensure the container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.pipelines; print('OK')" || exit 1

# Set entrypoint to run the enhanced CLI by default
ENTRYPOINT ["python", "-m", "src.enhanced_cli"]

# Default command line arguments for comprehensive analysis
CMD ["--task", "all", "--visualize", "--save-results", "--verbose"]
