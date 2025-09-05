#!/bin/bash

echo "=== COMPREHENSIVE CONTEXT TEST ==="
echo ""

# Generate unique session ID
session_id="test_session_$(date +%s)"
echo "Testing with session: $session_id"
echo ""

# Function to test query and check response
test_query() {
    local query="$1"
    local expected_context="$2"
    local description="$3"
    
    echo "=== $description ==="
    echo "Query: '$query'"
    
    response=$(curl -s -X POST "http://localhost:8000/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$query\", \"session_id\": \"$session_id\"}")
    
    if [ $? -eq 0 ]; then
        echo "Response preview:"
        echo "$response" | jq -r '.message' | head -5 | sed 's/^/  /'
        echo ""
        
        # Check for context usage
        if echo "$response" | grep -i -E "$expected_context" > /dev/null; then
            echo "✅ SUCCESS: Context correctly used ($expected_context)"
        else
            echo "❌ ISSUE: Expected context not found ($expected_context)"
        fi
    else
        echo "❌ ERROR: Failed to get response"
    fi
    echo ""
    echo "---"
    echo ""
}

# Test sequence
test_query "beyoglu" "(beyoğlu|beyoglu|galata|taksim)" "Initial query - mention Beyoğlu"

test_query "places" "(beyoğlu|beyoglu|galata|taksim|since you were asking)" "Follow-up places query"

test_query "restaurants" "(beyoğlu|beyoglu|galata|taksim|karakoy|karaköy|istiklal)" "Follow-up restaurants query"

test_query "museums" "(beyoğlu|beyoglu|galata|pera|modern art)" "Follow-up museums query"

echo "=== CONTEXT TEST COMPLETE ==="
