#!/bin/bash

# Quick SSH diagnostic script
# Checks what's running on your RunPod pod

echo "🔍 Checking RunPod server status..."
echo ""
echo "Attempting SSH connection..."
echo ""

ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519 << 'ENDSSH'
echo "✅ SSH connection successful!"
echo ""
echo "=== Current Directory ==="
pwd
echo ""
echo "=== Files in /workspace ==="
ls -lah /workspace | head -20
echo ""
echo "=== Python Processes ==="
ps aux | grep python | grep -v grep
echo ""
echo "=== Network Ports ==="
netstat -tlnp 2>/dev/null | grep LISTEN | head -10 || ss -tlnp 2>/dev/null | grep LISTEN | head -10
echo ""
echo "=== Check if port 8888 is listening ==="
curl -s http://localhost:8888/health 2>&1 | head -20
echo ""
echo "=== Check for LLM server script ==="
ls -la /workspace/*llm* 2>/dev/null | head -10
ls -la /workspace/*api* 2>/dev/null | head -10
ENDSSH

echo ""
echo "✅ Diagnostic complete!"
