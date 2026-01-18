#!/bin/bash

# RunPod LLM Server Startup Guide
# Use this when SSH'd into your RunPod pod

echo "=============================================="
echo "🚀 RunPod LLM Server Startup & Diagnosis"
echo "=============================================="
echo ""

# Step 1: Check what Python files are available
echo "Step 1: Checking available LLM server files..."
echo "----------------------------------------------"
ls -lh /workspace/*.py 2>/dev/null | grep -E "(llm|server|api)" || echo "No LLM server files found in /workspace"
echo ""

# Step 2: Check the error log
echo "Step 2: Checking error log from previous attempt..."
echo "----------------------------------------------------"
if [ -f "/workspace/server.log" ]; then
    echo "Last 50 lines of server.log:"
    tail -50 /workspace/server.log
else
    echo "No server.log found"
fi
echo ""

# Step 3: Check GPU availability
echo "Step 3: Checking GPU availability..."
echo "-------------------------------------"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Step 4: Check Python and dependencies
echo "Step 4: Checking Python environment..."
echo "---------------------------------------"
python --version
echo ""
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1
echo ""

# Step 5: List all Python files for manual inspection
echo "Step 5: All Python files in /workspace..."
echo "------------------------------------------"
find /workspace -maxdepth 1 -name "*.py" -type f
echo ""

echo "=============================================="
echo "Next steps:"
echo "1. Check server.log output above for errors"
echo "2. Try running the server with different file:"
echo "   python <correct_filename>.py"
echo "3. Common fixes:"
echo "   - Out of memory: Restart the pod"
echo "   - Missing dependencies: pip install -r requirements.txt"
echo "   - Wrong file: Check which .py file is the server"
echo "=============================================="
