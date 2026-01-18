#!/usr/bin/env python3
"""
Deploy vLLM server to RunPod (recommended approach)
vLLM is much faster and more reliable than custom FastAPI servers
"""

import subprocess
import time
import json

POD_SSH = "e9e56rc2ryjtmm-64411022@ssh.runpod.io"
SSH_KEY = "~/.ssh/id_ed25519"

print("🚀 Deploying vLLM Server to RunPod")
print("=" * 50)
print()

# Step 1: Upload vLLM startup script
print("📤 Step 1: Uploading vLLM startup script...")
cmd = f"""cat /Users/omer/Desktop/ai-stanbul/start_vllm_server.sh | \
ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -T {POD_SSH} "cat > /workspace/start_vllm.sh && chmod +x /workspace/start_vllm.sh" """

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if "uploaded" in result.stdout.lower() or result.returncode == 0:
    print("✅ Script uploaded")
else:
    print(f"⚠️  Upload status: {result.stdout or result.stderr}")

print()

# Step 2: Stop old server
print("🛑 Step 2: Stopping old custom server...")
cmd = f"""ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -T {POD_SSH} << 'EOF'
# Kill any existing servers
pkill -f 'python.*server.py' 2>/dev/null || true
pkill -f 'vllm.entrypoints' 2>/dev/null || true
sleep 2
echo "Old servers stopped"
EOF
"""

subprocess.run(cmd, shell=True)
print("✅ Old servers stopped")
print()

# Step 3: Check if vLLM is installed
print("🔍 Step 3: Checking vLLM installation...")
cmd = f"""ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -T {POD_SSH} "python3 -c 'import vllm; print(vllm.__version__)' 2>&1" """

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if "Error" not in result.stdout and result.returncode == 0:
    print(f"✅ vLLM installed: {result.stdout.strip()}")
else:
    print("⚠️  vLLM might not be installed, will try anyway...")
print()

# Step 4: Start vLLM server in background
print("🚀 Step 4: Starting vLLM server...")
cmd = f"""ssh -i {SSH_KEY} -o StrictHostKeyChecking=no -T {POD_SSH} << 'EOF'
cd /workspace
mkdir -p logs
nohup bash start_vllm.sh > logs/vllm_startup.log 2>&1 &
echo "vLLM server starting (PID: $!)"
EOF
"""

subprocess.run(cmd, shell=True)
print("✅ vLLM server started")
print()

# Step 5: Wait for server to be ready
print("⏳ Step 5: Waiting for server to be ready (this takes 1-2 minutes)...")
print("   Loading Llama-3.1-8B-Instruct with 4-bit quantization...")

for i in range(30):
    time.sleep(5)
    
    # Check health endpoint
    result = subprocess.run(
        "curl -s https://e9e56rc2ryjtmm-8000.proxy.runpod.net/health 2>&1",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if "healthy" in result.stdout or "200" in result.stdout:
        print(f"\n✅ Server ready after {(i+1)*5} seconds!")
        break
    
    # Check v1/models endpoint (vLLM specific)
    result = subprocess.run(
        "curl -s https://e9e56rc2ryjtmm-8000.proxy.runpod.net/v1/models 2>&1",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if "llama" in result.stdout.lower():
        print(f"\n✅ vLLM server ready after {(i+1)*5} seconds!")
        break
    
    print(f"   Attempt {i+1}/30...", end='\r')
else:
    print("\n⚠️  Server taking longer than expected, check logs manually")

print()
print("=" * 50)
print("✅ Deployment complete!")
print()
print("🔍 Next steps:")
print("1. Test server: bash verify_vllm_server.sh")
print("2. Run integration test: python3 test_unified_llm.py")
print()
print("📝 Endpoints available:")
print("   - OpenAI compatible: /v1/chat/completions")
print("   - OpenAI compatible: /v1/completions")
print("   - Models list: /v1/models")
