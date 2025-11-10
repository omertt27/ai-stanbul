#!/bin/bash
# CloudShell File Verification Script

echo "🔍 AI Istanbul LLM - CloudShell File Verification"
echo "=================================================="
echo ""

# Check current directory
echo "📁 Current Directory:"
pwd
echo ""

# List all files
echo "📂 Files in Current Directory:"
ls -lh
echo ""

# Check for required files
echo "✅ Required Files Check:"
echo "------------------------"

FILES=(
    "Dockerfile.4bit"
    "llm_api_server_4bit.py"
    "deploy_to_ecs.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✅ $file (Size: $size)"
    else
        echo "❌ $file - NOT FOUND"
    fi
done

echo ""
echo "📋 File Contents Preview:"
echo "========================="

# Check Dockerfile.4bit
if [ -f "Dockerfile.4bit" ]; then
    echo ""
    echo "--- Dockerfile.4bit (First 20 lines) ---"
    head -20 Dockerfile.4bit
    echo ""
    echo "Total lines: $(wc -l < Dockerfile.4bit)"
else
    echo "❌ Dockerfile.4bit not found"
fi

# Check llm_api_server_4bit.py
if [ -f "llm_api_server_4bit.py" ]; then
    echo ""
    echo "--- llm_api_server_4bit.py (First 30 lines) ---"
    head -30 llm_api_server_4bit.py
    echo ""
    echo "Total lines: $(wc -l < llm_api_server_4bit.py)"
else
    echo "❌ llm_api_server_4bit.py not found"
fi

# Check deploy_to_ecs.sh
if [ -f "deploy_to_ecs.sh" ]; then
    echo ""
    echo "--- deploy_to_ecs.sh (First 30 lines) ---"
    head -30 deploy_to_ecs.sh
    echo ""
    echo "Total lines: $(wc -l < deploy_to_ecs.sh)"
    echo "Executable: $([ -x deploy_to_ecs.sh ] && echo 'YES ✅' || echo 'NO ❌')"
else
    echo "❌ deploy_to_ecs.sh not found"
fi

echo ""
echo "🔧 Quick Fix Commands:"
echo "======================"
echo ""
echo "If deploy_to_ecs.sh is not executable:"
echo "  chmod +x deploy_to_ecs.sh"
echo ""
echo "To run deployment:"
echo "  ./deploy_to_ecs.sh"
echo ""
echo "To check AWS credentials:"
echo "  aws sts get-caller-identity"
echo ""
