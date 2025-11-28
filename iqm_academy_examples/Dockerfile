# Use a Python base image. Python 3.12 is a good recent stable version.
FROM python:3.12-slim-buster

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies that might be needed by Python packages.
# 'build-essential' is for compiling certain packages, 'git' for fetching from VCS,
# and 'curl' is used here to download the sdk.txt file.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Define the URL for the sdk.txt file. Using the raw content URL.
ARG SDK_TXT_URL="https://raw.githubusercontent.com/iqm-finland/sdk/main/sdk.txt"

# Download the sdk.txt file from the specified URL.
# This ensures that the Dockerfile always pulls the latest dependencies
# from the GitHub repository at build time.
RUN curl -o sdk.txt "$SDK_TXT_URL"

# Install the Python dependencies listed in sdk.txt.
# We use --no-cache-dir to avoid storing build cache, which reduces image size.
RUN pip install --no-cache-dir -r sdk.txt
RUN pip install qrisp[iqm]
