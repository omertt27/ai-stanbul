#!/bin/bash

# Test context-aware follow-up handling
# Scenario: User says "beyoglu" then "places" - should get places in Beyoğlu

SESSION_ID="test_context_session_$(date +%s)"
echo "Testing context-aware follow-up with session: $SESSION_ID"

echo ""
echo "=== Step 1: User mentions 'beyoglu' ==="
RESPONSE1=$(curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_input\": \"beyoglu\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo "User: beyoglu"
echo "AI Response: $(echo $RESPONSE1 | jq -r '.message' | head -c 200)..."

echo ""
echo "=== Step 2: User asks 'places' (should use Beyoğlu context) ==="
RESPONSE2=$(curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_input\": \"places\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo "User: places"
echo "AI Response: $(echo $RESPONSE2 | jq -r '.message' | head -c 300)..."

echo ""
echo "=== Step 3: Check if response mentions Beyoğlu/Galata ==="
if echo "$RESPONSE2" | jq -r '.message' | grep -i "beyoglu\|galata" > /dev/null; then
    echo "✅ SUCCESS: Response correctly includes Beyoğlu/Galata context!"
else
    echo "❌ ISSUE: Response doesn't seem to use Beyoğlu context"
fi

echo ""
echo "=== Step 4: Test 'restaurants' follow-up ==="
RESPONSE3=$(curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{
    \"user_input\": \"restaurants\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo "User: restaurants"
echo "AI Response: $(echo $RESPONSE3 | jq -r '.message' | head -c 300)..."

if echo "$RESPONSE3" | jq -r '.message' | grep -i "beyoglu\|galata" > /dev/null; then
    echo "✅ SUCCESS: Restaurant response correctly includes Beyoğlu context!"
else
    echo "❌ ISSUE: Restaurant response doesn't seem to use Beyoğlu context"
fi
