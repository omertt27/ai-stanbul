#!/bin/bash

echo "Testing context-aware follow-up with session: test_context_session_$(date +%s)"
echo ""

session_id="test_context_session_$(date +%s)"

echo "=== Step 1: User mentions 'beyoglu' ==="
echo "User: beyoglu"
response1=$(timeout 10 curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"beyoglu\", \"session_id\": \"$session_id\"}")

if [ $? -eq 0 ]; then
    echo "AI Response: $(echo "$response1" | python -c "import sys, json; print(json.load(sys.stdin)['message'][:100] + '...')")"
else
    echo "❌ ERROR: First query failed"
    exit 1
fi

echo ""
echo "=== Step 2: User asks 'places' (should use Beyoğlu context) ==="
echo "User: places"
response2=$(timeout 10 curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"places\", \"session_id\": \"$session_id\"}")

if [ $? -eq 0 ]; then
    echo "AI Response: $(echo "$response2" | python -c "import sys, json; print(json.load(sys.stdin)['message'][:150] + '...')")"
else
    echo "❌ ERROR: Places query failed"
    exit 1
fi

echo ""
echo "=== Step 3: Check if response mentions Beyoğlu/Galata ==="
if echo "$response2" | grep -i -E "(beyoğlu|beyoglu|galata|since you were asking)" > /dev/null; then
    echo "✅ SUCCESS: Response correctly includes Beyoğlu/Galata context!"
else
    echo "❌ ISSUE: Places response doesn't seem to use Beyoğlu context"
fi

echo ""
echo "=== Step 4: Test 'restaurants' follow-up ==="
echo "User: restaurants"
response3=$(timeout 10 curl -s -X POST "http://localhost:8001/ai" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"restaurants\", \"session_id\": \"$session_id\"}")

if [ $? -eq 0 ]; then
    echo "AI Response: $(echo "$response3" | python -c "import sys, json; print(json.load(sys.stdin)['message'][:200] + '...')")"
    
    echo ""
    echo "=== Step 5: Check if restaurant response uses Beyoğlu context ==="
    if echo "$response3" | grep -i -E "(beyoğlu|beyoglu|galata|taksim|karakoy|karaköy|istiklal)" > /dev/null; then
        echo "✅ SUCCESS: Restaurant response correctly uses Beyoğlu context!"
    else
        echo "❌ ISSUE: Restaurant response doesn't seem to use Beyoğlu context"
    fi
else
    echo "❌ ERROR: Restaurant query failed"
fi
