#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 📦 INSTALL DEPENDENCIES FOR LLM SERVER
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "📦 Installing LLM Server Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Update pip
echo "⬆️  Updating pip..."
pip install --upgrade pip -q

# Install FastAPI and Uvicorn
echo "🚀 Installing FastAPI and Uvicorn..."
pip install fastapi uvicorn[standard] -q

# Install Transformers and Tokenizers
echo "🤗 Installing Transformers..."
pip install transformers tokenizers -q

# Install PyTorch (usually pre-installed on RunPod)
echo "🔥 Checking PyTorch..."
pip install torch -q

# Install Accelerate for multi-GPU support
echo "⚡ Installing Accelerate..."
pip install accelerate -q

# Install bitsandbytes for quantization
echo "🔢 Installing bitsandbytes..."
pip install bitsandbytes -q

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install pydantic requests -q

echo ""
echo "✅ All dependencies installed!"
echo ""

# Verify installations
echo "🔍 Verifying installations..."
python3 -c "import fastapi; print(f'✅ FastAPI: {fastapi.__version__}')"
python3 -c "import uvicorn; print(f'✅ Uvicorn: {uvicorn.__version__}')"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')"
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import accelerate; print(f'✅ Accelerate: {accelerate.__version__}')"
python3 -c "import bitsandbytes; print(f'✅ BitsAndBytes installed')"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Ready to start LLM server!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
