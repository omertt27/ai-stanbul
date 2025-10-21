#!/bin/bash
# Istanbul AI - Phase 1 Quick Start
# Automated setup for local development

set -e

echo "=================================================="
echo "Istanbul AI - Phase 1 Quick Start"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Step 1: Check virtual environment
echo -e "${BLUE}Step 1: Checking virtual environment...${NC}"
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Activating virtual environment...${NC}"
    source venv_gpu_ml/bin/activate
fi
echo -e "${GREEN}✅ Virtual environment active${NC}"
echo ""

# Step 2: Install missing packages
echo -e "${BLUE}Step 2: Installing missing packages...${NC}"
pip install -q sentence-transformers scikit-learn scipy xgboost lightgbm faiss-cpu spacy textblob mlflow pytest pytest-asyncio pytest-cov python-dotenv pyyaml tqdm psutil 2>/dev/null
echo -e "${GREEN}✅ Packages installed${NC}"
echo ""

# Step 3: Download spaCy model
echo -e "${BLUE}Step 3: Downloading spaCy model...${NC}"
python -m spacy download en_core_web_sm --quiet 2>/dev/null || echo "Already downloaded"
echo -e "${GREEN}✅ spaCy model ready${NC}"
echo ""

# Step 4: Check services
echo -e "${BLUE}Step 4: Checking services...${NC}"
if redis-cli ping &>/dev/null; then
    echo -e "${GREEN}✅ Redis running${NC}"
else
    echo -e "${YELLOW}⚠️  Starting Redis...${NC}"
    brew services start redis
fi

if brew services list | grep -q "postgresql.*started"; then
    echo -e "${GREEN}✅ PostgreSQL running${NC}"
else
    echo -e "${YELLOW}⚠️  Starting PostgreSQL...${NC}"
    brew services start postgresql@15
fi
echo ""

# Step 5: Test GPU simulator
echo -e "${BLUE}Step 5: Testing GPU simulator...${NC}"
python3 gpu_simulator.py > /dev/null 2>&1 && echo -e "${GREEN}✅ GPU simulator working${NC}" || echo -e "${YELLOW}⚠️  GPU simulator needs attention${NC}"
echo ""

echo "=================================================="
echo -e "${GREEN}✅ Phase 1 Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review: cat PHASE1_LOCAL_IMPLEMENTATION.md"
echo "  2. Test: python3 gpu_simulator.py"
echo "  3. Develop: Start coding!"
echo ""
