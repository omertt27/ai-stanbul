#!/bin/bash

# RunPod Environment Check Script
# Run this before starting LLM server to verify you have enough resources

echo "════════════════════════════════════════════════════════"
echo "🔍 RunPod Environment Check"
echo "════════════════════════════════════════════════════════"
echo ""

# Check disk space
echo "💾 Disk Space:"
echo "----------------------------------------"
df -h | grep -E '(Filesystem|/$|/workspace)'
echo ""

# Calculate available space on root and workspace
ROOT_AVAIL=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
WORKSPACE_AVAIL=$(df -BG /workspace 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')

echo "📊 Analysis:"
echo "  Root (/) available: ${ROOT_AVAIL}GB"
if [ ! -z "$WORKSPACE_AVAIL" ]; then
    echo "  Workspace available: ${WORKSPACE_AVAIL}GB"
fi
echo ""

# Recommend model based on space
echo "🎯 Recommended Models:"
if [ "$ROOT_AVAIL" -gt 25 ] || [ "$WORKSPACE_AVAIL" -gt 25 ] 2>/dev/null; then
    echo "  ✅ Llama 3.1 8B - You have enough space!"
    echo "     Needs: 25GB | You have: ${ROOT_AVAIL}GB on /"
elif [ "$ROOT_AVAIL" -gt 15 ] || [ "$WORKSPACE_AVAIL" -gt 15 ] 2>/dev/null; then
    echo "  ⚠️  Use Qwen 2.5 7B instead"
    echo "     Needs: 15GB | You have: ${ROOT_AVAIL}GB on /"
elif [ "$ROOT_AVAIL" -gt 10 ] || [ "$WORKSPACE_AVAIL" -gt 10 ] 2>/dev/null; then
    echo "  ⚠️  Use Llama 3.2 3B (smallest)"
    echo "     Needs: 10GB | You have: ${ROOT_AVAIL}GB on /"
else
    echo "  ❌ Not enough space! Need to clean up or upgrade pod"
    echo "     Available: ${ROOT_AVAIL}GB on /"
fi
echo ""

# Check HuggingFace token
echo "🔑 HuggingFace Token:"
echo "----------------------------------------"
if [ -z "$HF_TOKEN" ]; then
    echo "  ❌ HF_TOKEN not set!"
    echo "  Run: export HF_TOKEN=\"hf_YOUR_TOKEN_HERE\""
else
    echo "  ✅ HF_TOKEN is set"
    echo "  Token: ${HF_TOKEN:0:7}...${HF_TOKEN: -4}"
fi
echo ""

# Check GPU
echo "🎮 GPU Status:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
    echo ""
else
    echo "  ⚠️  nvidia-smi not found"
fi

# Check Python packages
echo "📦 Python Packages:"
echo "----------------------------------------"
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    echo "  ✅ vLLM installed (v${VLLM_VERSION})"
else
    echo "  ❌ vLLM not installed"
    echo "  Run: pip install vllm"
fi

if python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "  ✅ huggingface_hub installed"
else
    echo "  ⚠️  huggingface_hub not installed (optional)"
fi
echo ""

# Check cache directories
echo "📁 Cache Directories:"
echo "----------------------------------------"
if [ -d ~/.cache/huggingface/hub ]; then
    CACHE_SIZE=$(du -sh ~/.cache/huggingface/hub 2>/dev/null | cut -f1)
    echo "  ~/.cache/huggingface/hub: ${CACHE_SIZE}"
fi
if [ -d /root/.cache/huggingface/hub ]; then
    ROOT_CACHE_SIZE=$(du -sh /root/.cache/huggingface/hub 2>/dev/null | cut -f1)
    echo "  /root/.cache/huggingface/hub: ${ROOT_CACHE_SIZE}"
fi
echo ""

# Check if server is already running
echo "🚀 Server Status:"
echo "----------------------------------------"
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "  ⚠️  vLLM server already running!"
    echo "  PID: $(pgrep -f vllm.entrypoints.openai.api_server)"
    echo ""
    echo "  Test with: curl http://localhost:8888/health"
else
    echo "  ❌ No vLLM server running"
fi
echo ""

# Recommendations
echo "════════════════════════════════════════════════════════"
echo "📋 Next Steps:"
echo "════════════════════════════════════════════════════════"

if [ -z "$HF_TOKEN" ]; then
    echo "1. ❌ Set HuggingFace token:"
    echo "   export HF_TOKEN=\"hf_YOUR_TOKEN_HERE\""
    echo ""
fi

if [ "$ROOT_AVAIL" -gt 25 ]; then
    echo "2. ✅ You can use Llama 3.1 8B:"
    echo "   export HF_HOME=/root/.cache/huggingface"
    echo "   python -m vllm.entrypoints.openai.api_server \\"
    echo "     --model meta-llama/Meta-Llama-3.1-8B-Instruct \\"
    echo "     --port 8888 --host 0.0.0.0 --trust-remote-code &"
elif [ "$ROOT_AVAIL" -gt 15 ]; then
    echo "2. ⚠️  Use Qwen 2.5 7B (smaller, equally good):"
    echo "   python -m vllm.entrypoints.openai.api_server \\"
    echo "     --model Qwen/Qwen2.5-7B-Instruct \\"
    echo "     --port 8888 --host 0.0.0.0 --trust-remote-code &"
else
    echo "2. ❌ Not enough space! Options:"
    echo "   a) Clean up: rm -rf ~/.cache/huggingface/hub/*"
    echo "   b) Use smallest model (Llama 3.2 3B)"
    echo "   c) Upgrade RunPod instance"
fi

echo ""
echo "════════════════════════════════════════════════════════"
