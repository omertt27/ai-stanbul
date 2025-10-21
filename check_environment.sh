#!/bin/bash
# Istanbul AI - Development Environment Status Check
# Date: October 21, 2025
# Purpose: Comprehensive analysis of current development environment

set -e

echo "=================================================="
echo "Istanbul AI - Environment Status Check"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not activated${NC}"
    echo "Checking system-wide installation..."
    echo ""
else
    echo -e "${GREEN}✅ Virtual environment active:${NC}"
    echo "   $VIRTUAL_ENV"
    echo ""
fi

# ============================================================================
# SECTION 1: System Information
# ============================================================================
echo "=================================================="
echo "1️⃣  System Information"
echo "=================================================="
echo ""

# macOS version
echo -e "${BLUE}Operating System:${NC}"
sw_vers
echo ""

# Processor type
ARCH=$(uname -m)
echo -e "${BLUE}Processor Architecture:${NC} $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}✅ Apple Silicon (M1/M2/M3) - MPS acceleration available!${NC}"
else
    echo -e "${YELLOW}⚠️  Intel Mac - CPU only${NC}"
fi
echo ""

# Memory
TOTAL_MEM=$(sysctl hw.memsize | awk '{print $2/1073741824}')
echo -e "${BLUE}Total Memory:${NC} ${TOTAL_MEM} GB"
if (( $(echo "$TOTAL_MEM >= 16" | bc -l) )); then
    echo -e "${GREEN}✅ Sufficient memory for ML development${NC}"
else
    echo -e "${YELLOW}⚠️  Recommended: 16GB+ for optimal performance${NC}"
fi
echo ""

# Disk space
echo -e "${BLUE}Disk Space:${NC}"
df -h . | awk 'NR==2 {print "   Total: " $2 "\n   Used: " $3 "\n   Available: " $4 "\n   Usage: " $5}'
echo ""

# ============================================================================
# SECTION 2: Python Environment
# ============================================================================
echo "=================================================="
echo "2️⃣  Python Environment"
echo "=================================================="
echo ""

# Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${BLUE}Python Version:${NC} $PYTHON_VERSION"
echo -e "${BLUE}Python Path:${NC} $(which python3)"
echo ""

# pip version
PIP_VERSION=$(pip --version)
echo -e "${BLUE}pip Version:${NC} $PIP_VERSION"
echo ""

# ============================================================================
# SECTION 3: Package Analysis
# ============================================================================
echo "=================================================="
echo "3️⃣  Installed Packages Analysis"
echo "=================================================="
echo ""

TOTAL_PACKAGES=$(pip list | tail -n +3 | wc -l | tr -d ' ')
echo -e "${BLUE}Total Packages:${NC} $TOTAL_PACKAGES"
echo ""

# Core ML/DL packages
echo -e "${BLUE}Core ML/DL Frameworks:${NC}"
declare -A CORE_PACKAGES=(
    ["torch"]="PyTorch (Deep Learning)"
    ["torchvision"]="PyTorch Vision"
    ["tensorflow"]="TensorFlow (NOT NEEDED)"
    ["tensorflow-macos"]="TensorFlow macOS (NOT NEEDED)"
    ["tensorflow-metal"]="TensorFlow Metal (NOT NEEDED)"
)

for package in "${!CORE_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        if [[ "$package" == "tensorflow"* ]]; then
            echo -e "   ${RED}❌ $package ($VERSION) - ${CORE_PACKAGES[$package]}${NC}"
        else
            echo -e "   ${GREEN}✅ $package ($VERSION) - ${CORE_PACKAGES[$package]}${NC}"
        fi
    else
        if [[ "$package" == "tensorflow"* ]]; then
            echo -e "   ${GREEN}✅ $package - Not installed (good!)${NC}"
        else
            echo -e "   ${YELLOW}⚠️  $package - Not installed${NC}"
        fi
    fi
done
echo ""

# NLP & Transformers
echo -e "${BLUE}NLP & Transformers:${NC}"
declare -A NLP_PACKAGES=(
    ["transformers"]="Hugging Face Transformers"
    ["sentence-transformers"]="Sentence Embeddings"
    ["spacy"]="Advanced NLP"
    ["textblob"]="Simple NLP"
)

for package in "${!NLP_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${NLP_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${NLP_PACKAGES[$package]} (NEEDED)${NC}"
    fi
done
echo ""

# Traditional ML
echo -e "${BLUE}Traditional ML:${NC}"
declare -A ML_PACKAGES=(
    ["scikit-learn"]="Scikit-learn"
    ["xgboost"]="XGBoost"
    ["lightgbm"]="LightGBM"
    ["numpy"]="NumPy"
    ["pandas"]="Pandas"
    ["scipy"]="SciPy"
)

for package in "${!ML_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${ML_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${ML_PACKAGES[$package]} (NEEDED)${NC}"
    fi
done
echo ""

# Vector Search & Caching
echo -e "${BLUE}Vector Search & Caching:${NC}"
declare -A CACHE_PACKAGES=(
    ["faiss-cpu"]="FAISS (CPU)"
    ["faiss-gpu"]="FAISS (GPU - Not for Mac)"
    ["redis"]="Redis"
)

for package in "${!CACHE_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        if [[ "$package" == "faiss-gpu" ]]; then
            echo -e "   ${YELLOW}⚠️  $package ($VERSION) - Use faiss-cpu on Mac${NC}"
        else
            echo -e "   ${GREEN}✅ $package ($VERSION) - ${CACHE_PACKAGES[$package]}${NC}"
        fi
    else
        if [[ "$package" == "faiss-gpu" ]]; then
            echo -e "   ${GREEN}✅ $package - Not installed (correct for Mac)${NC}"
        else
            echo -e "   ${YELLOW}⚠️  $package - ${CACHE_PACKAGES[$package]} (NEEDED)${NC}"
        fi
    fi
done
echo ""

# API & Web
echo -e "${BLUE}API & Web:${NC}"
declare -A WEB_PACKAGES=(
    ["fastapi"]="FastAPI"
    ["uvicorn"]="Uvicorn (ASGI server)"
    ["pydantic"]="Pydantic"
    ["aiohttp"]="Async HTTP"
)

for package in "${!WEB_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${WEB_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${WEB_PACKAGES[$package]}${NC}"
    fi
done
echo ""

# MLOps & Monitoring
echo -e "${BLUE}MLOps & Monitoring:${NC}"
declare -A MLOPS_PACKAGES=(
    ["mlflow"]="MLflow"
    ["wandb"]="Weights & Biases"
    ["prometheus-client"]="Prometheus"
)

for package in "${!MLOPS_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${MLOPS_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${MLOPS_PACKAGES[$package]} (RECOMMENDED)${NC}"
    fi
done
echo ""

# Testing
echo -e "${BLUE}Testing:${NC}"
declare -A TEST_PACKAGES=(
    ["pytest"]="Pytest"
    ["pytest-asyncio"]="Pytest Async"
    ["pytest-cov"]="Coverage"
)

for package in "${!TEST_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${TEST_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${TEST_PACKAGES[$package]} (RECOMMENDED)${NC}"
    fi
done
echo ""

# Utilities
echo -e "${BLUE}Utilities:${NC}"
declare -A UTIL_PACKAGES=(
    ["psutil"]="System monitoring"
    ["python-dotenv"]="Environment variables"
    ["pyyaml"]="YAML parsing"
    ["tqdm"]="Progress bars"
)

for package in "${!UTIL_PACKAGES[@]}"; do
    if pip show "$package" &> /dev/null; then
        VERSION=$(pip show "$package" | grep Version | awk '{print $2}')
        echo -e "   ${GREEN}✅ $package ($VERSION) - ${UTIL_PACKAGES[$package]}${NC}"
    else
        echo -e "   ${YELLOW}⚠️  $package - ${UTIL_PACKAGES[$package]}${NC}"
    fi
done
echo ""

# ============================================================================
# SECTION 4: PyTorch & MPS Status
# ============================================================================
echo "=================================================="
echo "4️⃣  PyTorch & Apple MPS Status"
echo "=================================================="
echo ""

python3 << 'EOF'
import sys
try:
    import torch
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("")
    
    # MPS (Metal Performance Shaders) check
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple GPU) Available: YES")
        print("   🍎 Apple Silicon GPU acceleration enabled!")
        print("   Device: mps (Metal Performance Shaders)")
    else:
        print("⚠️  MPS (Apple GPU) Available: NO")
        print("   Using CPU fallback")
        print("   Device: cpu")
    
    print("")
    
    # CUDA check (should be NO on Mac)
    if torch.cuda.is_available():
        print("⚠️  CUDA Available: YES (unexpected on Mac)")
    else:
        print("✅ CUDA Available: NO (expected on Mac)")
    
    print("")
    
    # Test tensor operation
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)
        print(f"✅ Tensor operations working on {device}")
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
    
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)
EOF

echo ""

# ============================================================================
# SECTION 5: Services Status
# ============================================================================
echo "=================================================="
echo "5️⃣  Services Status"
echo "=================================================="
echo ""

# Redis
echo -e "${BLUE}Redis:${NC}"
if brew services list | grep -q "redis.*started"; then
    echo -e "   ${GREEN}✅ Running${NC}"
    if redis-cli ping &> /dev/null; then
        echo -e "   ${GREEN}✅ Connection successful${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Not responding${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Not running${NC}"
    echo -e "   Start with: ${BLUE}brew services start redis${NC}"
fi
echo ""

# PostgreSQL
echo -e "${BLUE}PostgreSQL:${NC}"
if brew services list | grep -q "postgresql.*started"; then
    echo -e "   ${GREEN}✅ Running${NC}"
    if psql -c "SELECT 1" &> /dev/null; then
        echo -e "   ${GREEN}✅ Connection successful${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Not responding${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  Not running${NC}"
    echo -e "   Start with: ${BLUE}brew services start postgresql@15${NC}"
fi
echo ""

# ============================================================================
# SECTION 6: Project Files
# ============================================================================
echo "=================================================="
echo "6️⃣  Project Files Status"
echo "=================================================="
echo ""

# Check for key files
declare -A PROJECT_FILES=(
    ["gpu_simulator.py"]="GPU Simulator"
    ["LOCAL_DEVELOPMENT_SETUP.md"]="Setup Guide"
    ["GPU_ML_ENHANCEMENT_PLAN.md"]="Enhancement Plan"
    ["GPU_ML_IMPLEMENTATION_CHECKLIST.md"]="Implementation Checklist"
    ["requirements.txt"]="Dependencies"
    ["backend/main.py"]="Main Backend"
)

for file in "${!PROJECT_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo -e "   ${GREEN}✅ $file${NC} (${SIZE}) - ${PROJECT_FILES[$file]}"
    else
        echo -e "   ${YELLOW}⚠️  $file${NC} - ${PROJECT_FILES[$file]} (MISSING)"
    fi
done
echo ""

# ============================================================================
# SECTION 7: Disk Usage Analysis
# ============================================================================
echo "=================================================="
echo "7️⃣  Disk Usage Analysis"
echo "=================================================="
echo ""

# Virtual environment size
if [[ -d "venv_gpu_ml" ]]; then
    VENV_SIZE=$(du -sh venv_gpu_ml 2>/dev/null | awk '{print $1}')
    echo -e "${BLUE}Virtual Environment:${NC} $VENV_SIZE"
else
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
fi

# Models directory
if [[ -d "models" ]]; then
    MODELS_SIZE=$(du -sh models 2>/dev/null | awk '{print $1}')
    echo -e "${BLUE}Models Directory:${NC} $MODELS_SIZE"
fi

# Data directory
if [[ -d "data" ]]; then
    DATA_SIZE=$(du -sh data 2>/dev/null | awk '{print $1}')
    echo -e "${BLUE}Data Directory:${NC} $DATA_SIZE"
fi

# Cache directories
CACHE_SIZE=$(du -sh ~/.cache/huggingface 2>/dev/null | awk '{print $1}' || echo "0B")
echo -e "${BLUE}HuggingFace Cache:${NC} $CACHE_SIZE"

echo ""

# ============================================================================
# SECTION 8: Recommendations
# ============================================================================
echo "=================================================="
echo "8️⃣  Recommendations"
echo "=================================================="
echo ""

ISSUES_FOUND=0

# Check for TensorFlow
if pip show tensorflow-macos &> /dev/null || pip show tensorflow &> /dev/null; then
    echo -e "${RED}❌ TensorFlow installed (not needed for this project)${NC}"
    echo -e "   Action: Run ${BLUE}./cleanup_environment.sh${NC} to remove"
    ((ISSUES_FOUND++))
fi

# Check for sentence-transformers
if ! pip show sentence-transformers &> /dev/null; then
    echo -e "${YELLOW}⚠️  sentence-transformers not installed (needed)${NC}"
    echo -e "   Action: ${BLUE}pip install sentence-transformers${NC}"
    ((ISSUES_FOUND++))
fi

# Check for faiss
if ! pip show faiss-cpu &> /dev/null; then
    echo -e "${YELLOW}⚠️  faiss-cpu not installed (needed)${NC}"
    echo -e "   Action: ${BLUE}pip install faiss-cpu${NC}"
    ((ISSUES_FOUND++))
fi

# Check for scikit-learn
if ! pip show scikit-learn &> /dev/null; then
    echo -e "${YELLOW}⚠️  scikit-learn not installed (needed)${NC}"
    echo -e "   Action: ${BLUE}pip install scikit-learn${NC}"
    ((ISSUES_FOUND++))
fi

# Check for xgboost/lightgbm
if ! pip show xgboost &> /dev/null || ! pip show lightgbm &> /dev/null; then
    echo -e "${YELLOW}⚠️  xgboost or lightgbm not installed (needed)${NC}"
    echo -e "   Action: ${BLUE}pip install xgboost lightgbm${NC}"
    ((ISSUES_FOUND++))
fi

if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo -e "${GREEN}✅ No critical issues found!${NC}"
    echo -e "${GREEN}   Your environment is ready for development${NC}"
else
    echo ""
    echo -e "${YELLOW}📝 Total issues found: $ISSUES_FOUND${NC}"
    echo ""
    echo -e "${BLUE}Quick fix:${NC}"
    echo -e "   ${BLUE}./cleanup_environment.sh${NC} - Auto-fix all issues"
fi

echo ""

# ============================================================================
# SECTION 9: Summary
# ============================================================================
echo "=================================================="
echo "9️⃣  Environment Summary"
echo "=================================================="
echo ""

echo -e "${BLUE}System:${NC}"
echo "   • macOS $(sw_vers -productVersion)"
echo "   • $ARCH processor"
echo "   • ${TOTAL_MEM} GB RAM"
echo ""

echo -e "${BLUE}Python:${NC}"
echo "   • $PYTHON_VERSION"
echo "   • $TOTAL_PACKAGES packages installed"
echo ""

echo -e "${BLUE}GPU Acceleration:${NC}"
if [[ "$ARCH" == "arm64" ]]; then
    echo "   • Apple Silicon MPS available ✅"
else
    echo "   • CPU only (Intel Mac) ⚠️"
fi
echo ""

echo -e "${BLUE}Ready for:${NC}"
echo "   • Local development with GPU simulation ✅"
echo "   • Testing on Google Colab (T4 GPU) ✅"
echo "   • Deployment to GCP ✅"
echo ""

echo "=================================================="
echo "✅ Status check complete!"
echo "=================================================="
echo ""

# Save report
REPORT_FILE="environment_status_$(date +%Y%m%d_%H%M%S).txt"
echo "📄 Full report saved to: $REPORT_FILE"
echo ""
