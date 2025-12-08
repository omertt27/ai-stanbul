#!/bin/bash
# Build script for Render deployment

set -e  # Exit on error

echo "🏗️  Starting build process..."

# Upgrade pip to latest version
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install wheel for faster binary installations
echo "🎡 Installing wheel..."
pip install wheel

# Install numpy first (required by many other packages)
echo "🔢 Installing numpy..."
pip install numpy==1.26.4

# Install Python dependencies
echo "📚 Installing all dependencies..."
pip install -r requirements.txt

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify critical packages
echo "✅ Verifying installations..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo "⚠️ NumPy not installed"
python -c "import scipy; print(f'SciPy installed')" || echo "⚠️ SciPy not installed"  
python -c "import jellyfish; print(f'Jellyfish installed')" || echo "⚠️ Jellyfish not installed"
python -c "import sentence_transformers; print(f'Sentence Transformers installed')" || echo "⚠️ Sentence Transformers not installed"
python -c "import fastapi; print(f'FastAPI installed')"
python -c "import sqlalchemy; print(f'SQLAlchemy installed')"

echo "🎉 Build completed successfully!"
