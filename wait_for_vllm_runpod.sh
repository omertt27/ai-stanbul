#!/bin/bash

# 🔍 Wait for vLLM to be ready on RunPod
# Run this AFTER starting vLLM

echo "⏳ Waiting for vLLM to start..."
echo "This usually takes 60-90 seconds..."
echo ""

for i in {1..120}; do
    echo -n "."
    sleep 1
    
    if [ $((i % 10)) -eq 0 ]; then
        echo " ${i}s"
    fi
    
    # Test every 5 seconds
    if [ $((i % 5)) -eq 0 ]; then
        RESPONSE=$(curl -s http://localhost:8000/health 2>&1)
        if echo "$RESPONSE" | grep -q "model_name\|vllm"; then
            echo ""
            echo "✅ vLLM is ready!"
            echo ""
            echo "Response:"
            echo "$RESPONSE"
            exit 0
        fi
    fi
done

echo ""
echo "❌ Timeout waiting for vLLM"
echo ""
echo "Check logs on RunPod:"
echo "  tail -50 /root/vllm.log"
exit 1
