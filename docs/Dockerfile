# Install system dependencies early
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
#FROM pytorch/pytorch:1.13.0-cpu

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps first (cacheable if unchanged)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libpcre2-dev \
    liblzma-dev \
    libbz2-dev \
    zlib1g-dev \
    libicu-dev \
    libtirpc-dev \
    libreadline-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    r-base \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV R_HOME=/usr/lib/R

# Install Python deps early so they're cached
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools packaging && \
    pip install -r requirements.txt


# Copy rest of the code last (changing it doesn't bust earlier caches)
COPY . /app
WORKDIR /app

CMD ["python", "main_cross.py"]
