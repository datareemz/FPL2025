FROM python:3.11-slim

# Set home working directory variable
ENV HOME_DIR=/app
WORKDIR $HOME_DIR

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy specific application folders and files
COPY dek/ ./dek/
COPY auth/ ./auth/
COPY api/ ./api/
COPY dataModel/ ./dataModel/
COPY ml/ ./ml/
COPY historical/ ./historical/
COPY constants.py ./
COPY utils.py ./

# Create necessary directories
RUN mkdir -p datastore/current

# Set environment variables
ENV PYTHONPATH=$HOME_DIR
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "dek/dek.py"]