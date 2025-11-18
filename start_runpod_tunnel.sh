#!/bin/bash
# SSH Tunnel to RunPod vLLM Server
# This creates a secure tunnel from your Mac (localhost:8000) to RunPod's vLLM (port 8000)

echo "🔐 Starting SSH tunnel to RunPod vLLM server..."
echo ""
echo "⚠️  IMPORTANT: Replace the SSH connection details below with YOUR RunPod SSH info!"
echo "   Get it from: https://www.runpod.io/console/pods → Your Pod → Connect → SSH"
echo ""
echo "   Example format:"
echo "   ssh -L 8000:localhost:8000 root@ssh-ca-sjc-1.runpod.io -p 12345"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# RunPod SSH connection details - Using direct TCP connection
# This bypasses RunPod's SSH gateway restrictions
RUNPOD_DIRECT_IP="194.68.245.13"
RUNPOD_DIRECT_PORT="22117"
RUNPOD_SSH_KEY="~/.ssh/id_ed25519"

echo "Current settings:"
echo "  Direct IP: $RUNPOD_DIRECT_IP"
echo "  SSH Port: $RUNPOD_DIRECT_PORT"
echo "  SSH Key: $RUNPOD_SSH_KEY"
echo "  Tunnel: localhost:8000 → RunPod:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Tunnel active! You can now use http://localhost:8000 to access vLLM"
echo "   Keep this terminal open while using the AI Istanbul backend."
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

# Create the SSH tunnel using direct TCP connection
# -L 8000:localhost:8000  -> Forward local port 8000 to remote port 8000
# -N                       -> Don't execute remote commands
# -T                       -> Disable pseudo-terminal allocation
# -o StrictHostKeyChecking=no -> Auto-accept host key
# -o UserKnownHostsFile=/dev/null -> Don't save host key
ssh -L 8000:localhost:8000 -N -T \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ServerAliveInterval=60 \
  root@${RUNPOD_DIRECT_IP} -p ${RUNPOD_DIRECT_PORT} -i ${RUNPOD_SSH_KEY}

echo ""
echo "✅ Tunnel closed."
