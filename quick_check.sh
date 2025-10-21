#!/bin/bash
# Istanbul AI - Quick Environment Check
# Simple version for quick status

echo "=================================================="
echo "Istanbul AI - Quick Environment Check"
echo "=================================================="
echo ""

# Check virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment NOT activated"
    echo ""
    echo "Please activate first:"
    echo "  source venv_gpu_ml/bin/activate"
    echo ""
    echo "Or checking base Python environment..."
    echo ""
fi

# System info
echo "System:"
echo "  • macOS $(sw_vers -productVersion)"
echo "  • $(uname -m) processor"
MEM=$(sysctl hw.memsize | awk '{print int($2/1073741824)}')
echo "  • ${MEM} GB RAM"
echo ""

# Python
echo "Python:"
python3 --version
echo "  Path: $(which python3)"
echo ""

# PyTorch
echo "PyTorch & MPS:"
python3 -c "
import torch
print(f'  • PyTorch: {torch.__version__}')
print(f'  • MPS Available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('  • ✅ Apple GPU acceleration ready!')
" 2>/dev/null || echo "  • ❌ PyTorch not installed"
echo ""

# Key packages
echo "Key Packages:"
for pkg in transformers numpy pandas redis fastapi; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        echo "  • ✅ $pkg ($VERSION)"
    else
        echo "  • ❌ $pkg (not installed)"
    fi
done
echo ""

# TensorFlow check
echo "TensorFlow Status:"
if python3 -c "import tensorflow" 2>/dev/null; then
    TF_VER=$(python3 -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null)
    echo "  • ❌ TensorFlow installed ($TF_VER) - NOT NEEDED"
    echo "  • Recommendation: Remove to save space"
else
    echo "  • ✅ TensorFlow not installed (good!)"
fi
echo ""

# Files check
echo "Project Files:"
for file in gpu_simulator.py LOCAL_DEVELOPMENT_SETUP.md GPU_ML_ENHANCEMENT_PLAN.md; do
    if [[ -f "$file" ]]; then
        echo "  • ✅ $file"
    else
        echo "  • ❌ $file (missing)"
    fi
done
echo ""

# Services
echo "Services:"
if redis-cli ping &>/dev/null; then
    echo "  • ✅ Redis running"
else
    echo "  • ⚠️  Redis not running"
fi

if psql -c "SELECT 1" &>/dev/null; then
    echo "  • ✅ PostgreSQL running"
else
    echo "  • ⚠️  PostgreSQL not running"
fi
echo ""

echo "=================================================="
echo "Quick Actions:"
echo "=================================================="
echo ""
echo "1. Activate virtual environment:"
echo "   source venv_gpu_ml/bin/activate"
echo ""
echo "2. Test GPU simulator:"
echo "   python3 gpu_simulator.py"
echo ""
echo "3. Full environment check:"
echo "   source venv_gpu_ml/bin/activate"
echo "   ./check_environment.sh"
echo ""
echo "4. Clean up unnecessary packages:"
echo "   source venv_gpu_ml/bin/activate"
echo "   ./cleanup_environment.sh"
echo ""
