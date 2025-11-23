#!/bin/bash

# Quick Check: Is Render Redeploy Complete?

echo "=================================================="
echo "🔍 Checking if Render redeploy is complete..."
echo "=================================================="
echo ""

# Check 1: LLM Health
echo "Test 1: LLM Health Endpoint"
echo "----------------------------"
LLM_HEALTH=$(curl -s https://api.aistanbul.net/api/v1/llm/health)
echo "$LLM_HEALTH" | python3 -m json.tool
echo ""

# Check if newline still present
if echo "$LLM_HEALTH" | grep -q $'wg1sc\n0i37280ah5ajfmm'; then
    echo "❌ NEWLINE STILL PRESENT"
    echo ""
    echo "This means one of:"
    echo "1. Render is still redeploying (wait 1-2 more minutes)"
    echo "2. The environment variable wasn't saved correctly"
    echo "3. Manual redeploy wasn't triggered"
    echo ""
    echo "What to do:"
    echo "1. Go to: https://dashboard.render.com/"
    echo "2. Click your backend service"
    echo "3. Check 'Events' tab - is deployment running?"
    echo "4. If NOT deploying:"
    echo "   - Go to Environment tab"
    echo "   - Verify LLM_API_URL is ONE LINE (no breaks)"
    echo "   - Click 'Manual Deploy' > 'Deploy latest commit'"
    echo ""
elif echo "$LLM_HEALTH" | grep -q '"status": "healthy"'; then
    echo "✅ SUCCESS! LLM is healthy!"
    echo ""
    echo "The fix worked! Running full verification..."
    echo ""
    ./verify_after_newline_fix.sh
    exit 0
else
    echo "⚠️ UNEXPECTED RESPONSE"
    echo "The newline seems gone, but LLM is not healthy."
    echo "Check RunPod server might be down."
    echo ""
fi

# Check 2: When did backend last deploy?
echo "=================================================="
echo "Check: When did backend last update?"
echo "=================================================="
echo ""
echo "Go to Render dashboard and check:"
echo "1. Click your backend service"
echo "2. Look at 'Events' tab"
echo "3. Check latest deployment timestamp"
echo "4. If it's more than 5 minutes old, trigger manual deploy"
echo ""

# Check 3: Environment variable format
echo "=================================================="
echo "What the URL should look like:"
echo "=================================================="
echo ""
echo "❌ WRONG (2 lines):"
echo "https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc"
echo "0i37280ah5ajfmm/"
echo ""
echo "✅ RIGHT (1 line):"
echo "https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/"
echo ""
echo "=================================================="
echo ""
echo "Recommendations:"
echo "1. Wait 2-3 minutes if you just saved"
echo "2. Run this script again: ./check_redeploy_status.sh"
echo "3. If still broken after 5 minutes, check Render logs"
echo ""
