#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 📤 UPLOAD FILES TO RUNPOD
# ═══════════════════════════════════════════════════════════════

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# RunPod Configuration
RUNPOD_HOST="194.68.245.153"
RUNPOD_PORT="22003"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}📤 UPLOADING FILES TO RUNPOD${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Files to upload
FILES=(
    "llm_server.py"
    "start_llm_server_runpod.sh"
    "download_model.sh"
)

# Check if files exist locally
echo -e "${BLUE}Checking local files...${NC}"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        exit 1
    fi
done

echo ""
echo -e "${BLUE}Uploading files to RunPod...${NC}"
echo ""

# Upload each file
for file in "${FILES[@]}"; do
    echo "Uploading $file..."
    scp -P "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$file" "$RUNPOD_USER@$RUNPOD_HOST:/workspace/"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Uploaded: $file${NC}"
    else
        echo -e "${RED}❌ Failed to upload: $file${NC}"
        exit 1
    fi
    echo ""
done

echo -e "${BLUE}Making scripts executable...${NC}"
ssh -p "$RUNPOD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no \
    "$RUNPOD_USER@$RUNPOD_HOST" 'chmod +x /workspace/*.sh'

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Scripts are now executable${NC}"
else
    echo -e "${RED}❌ Failed to make scripts executable${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 ALL FILES UPLOADED SUCCESSFULLY!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. SSH into RunPod:"
echo "   ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_USER@$RUNPOD_HOST"
echo ""
echo "2. Download the model:"
echo "   cd /workspace && ./download_model.sh"
echo ""
echo "3. Start the server:"
echo "   cd /workspace && ./start_llm_server_runpod.sh"
echo ""
