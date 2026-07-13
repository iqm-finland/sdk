#!/usr/bin/env bash

set -euo pipefail

# Script to copy SDK files to the public directory so they can be accessed by the frontend

echo "Copying SDK files to public directory..."

cd "$(dirname "$0")"

# Create public directory if it doesn't exist
mkdir -p public

# Copy all sdk*.txt files from parent directory to public
cp -v ../sdk*.txt public/ 2>/dev/null && echo "SDK files copied successfully." || echo "No SDK files found in parent directory"
