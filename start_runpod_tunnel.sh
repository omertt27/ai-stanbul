#!/bin/bash
# SSH Tunnel to RunPod vLLM Server
# This creates a secure tunnel from your Mac (localhost:8000) to RunPod's vLLM (port 8000)

echo "🔐 Starting SSH tunnel to RunPod vLLM server..."
echo "🔌 Connecting to: 194.68.245.13:22162"
echo ""

# RunPod SSH connection details (SSH over exposed TCP)
RUNPOD_HOST="194.68.245.13"
RUNPOD_PORT="22162"
SSH_KEY="~/.ssh/id_ed25519"

echo "Current settings:"
echo "  Host: $RUNPOD_HOST"
echo "  SSH Port: $RUNPOD_PORT"
echo "  SSH Key: $SSH_KEY"
echo "  Tunnel: localhost:8000 → RunPod:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Tunnel active! You can now use http://localhost:8000 to access vLLM"
echo "   Keep this terminal open while using the AI Istanbul backend."
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

# Create the SSH tunnel
# -L 8000:localhost:8000  -> Forward local port 8000 to remote port 8000
# -N                       -> Don't execute remote commands
# -T                       -> Disable pseudo-terminal allocation
# -o ServerAliveInterval=60 -> Keep connection alive
ssh -L 8000:localhost:8000 -N -T \
  -o ServerAliveInterval=60 \
  -i ${SSH_KEY} \
  root@${RUNPOD_HOST} -p ${RUNPOD_PORT}

echo ""
echo "✅ Tunnel closed."
