#!/bin/bash
# Build script for Render deployment

# Install Python dependencies
pip install -r requirements.txt

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Build completed successfully!"
