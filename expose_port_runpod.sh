#!/bin/bash
# Script to expose port 8000 on RunPod via SSH tunnel (if needed)
# Run this locally to create a tunnel to your pod

POD_ID="pvj233wwhiu6j3-64411542"
SSH_HOST="ssh.runpod.io"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "Creating SSH tunnel to expose port $REMOTE_PORT..."
echo "Local access will be: http://localhost:$LOCAL_PORT"
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

ssh -T -N -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ${POD_ID}@${SSH_HOST} -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no
