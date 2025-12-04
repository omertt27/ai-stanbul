#!/bin/bash
# Quick vLLM Health Check Script

echo "🔍 AI Istanbul - vLLM Health Check"
echo "=================================="
echo ""

# Known pod IDs from previous logs
POD_IDS=("i6c58scsmccj2s" "gbpd35labcq12f")

echo "Testing known RunPod endpoints..."
echo ""

FOUND_WORKING=false

for POD_ID in "${POD_IDS[@]}"; do
  echo "Testing pod: $POD_ID"
  echo "URL: https://${POD_ID}-8888.proxy.runpod.net/health"
  
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" https://${POD_ID}-8888.proxy.runpod.net/health 2>&1)
  
  if [ "$RESPONSE" = "200" ]; then
    echo "✅ SUCCESS! Pod $POD_ID is HEALTHY!"
    echo ""
    echo "Full response:"
    curl -s https://${POD_ID}-8888.proxy.runpod.net/health | python3 -m json.tool 2>/dev/null || curl -s https://${POD_ID}-8888.proxy.runpod.net/health
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ ACTION REQUIRED: Update your .env file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Add this line to your .env file:"
    echo ""
    echo "LLM_API_URL=https://${POD_ID}-8888.proxy.runpod.net"
    echo ""
    FOUND_WORKING=true
    break
  else
    echo "❌ Pod $POD_ID: $RESPONSE"
  fi
  echo ""
done

if [ "$FOUND_WORKING" = false ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ No working pods found"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Possible reasons:"
  echo "1. Pod is stopped - Go to RunPod console and start it"
  echo "2. vLLM service crashed - SSH in and restart it"
  echo "3. Different pod ID - Check RunPod console for active pods"
  echo ""
  echo "Next steps:"
  echo "1. Visit: https://www.runpod.io/console/pods"
  echo "2. Check if your pod is running (🟢 green status)"
  echo "3. If stopped, click 'Start'"
  echo "4. If running but not responding, SSH in and check vLLM:"
  echo "   ssh root@YOUR-POD-ID-ssh.proxy.runpod.net -p PORT"
  echo "   ps aux | grep vllm"
  echo ""
  echo "📖 See CHECK_VLLM_STATUS.md for detailed troubleshooting"
fi

echo ""
echo "Done!"
