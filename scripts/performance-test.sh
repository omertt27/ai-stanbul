#!/bin/bash
# 📊 Performance & Load Testing Script for AI Istanbul

set -e

echo "📊 AI Istanbul - Performance & Load Testing"
echo "==========================================="

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8002}"
LANGUAGES=("en" "tr" "fr" "ru" "de" "ar")
CONCURRENT_USERS=(10 50 100 200)
TEST_DURATION=60  # seconds

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v ab &> /dev/null; then
        log_warn "Apache Bench (ab) not found, installing..."
        sudo apt-get install -y apache2-utils
    fi
    
    if ! command -v wrk &> /dev/null; then
        log_warn "wrk not found, installing..."
        sudo apt-get install -y wrk
    fi
    
    if ! command -v jq &> /dev/null; then
        log_warn "jq not found, installing..."
        sudo apt-get install -y jq
    fi
    
    log_info "Prerequisites OK"
}

# Test API health
test_health() {
    log_info "Testing API health..."
    
    if curl -f "$BASE_URL/health" > /dev/null 2>&1; then
        log_info "✅ API is healthy"
    else
        log_error "❌ API health check failed"
        exit 1
    fi
}

# Test single request performance
test_single_request() {
    log_info "Testing single request performance..."
    
    for lang in "${LANGUAGES[@]}"; do
        log_info "Testing language: $lang"
        
        start_time=$(date +%s%N)
        curl -X POST "$BASE_URL/api/chat" \
            -H "Content-Type: application/json" \
            -d "{\"message\":\"Where can I find good restaurants?\",\"language\":\"$lang\"}" \
            -o /dev/null -s
        end_time=$(date +%s%N)
        
        duration_ms=$(( (end_time - start_time) / 1000000 ))
        log_info "  Response time: ${duration_ms}ms"
        
        if [ $duration_ms -lt 5000 ]; then
            echo -e "  ${GREEN}✅ PASS${NC} (< 5s)"
        else
            echo -e "  ${YELLOW}⚠️  SLOW${NC} (> 5s)"
        fi
    done
}

# Load test with Apache Bench
load_test_ab() {
    local concurrent=$1
    local total_requests=$((concurrent * 10))
    
    log_info "Running Apache Bench test: $concurrent concurrent users, $total_requests total requests"
    
    # Create request body file
    cat > /tmp/request-body.json << EOF
{
    "message": "What are the top attractions in Istanbul?",
    "language": "en"
}
EOF
    
    # Run test
    ab -n $total_requests \
       -c $concurrent \
       -p /tmp/request-body.json \
       -T "application/json" \
       -g /tmp/ab-results-$concurrent.tsv \
       "$BASE_URL/api/chat" > /tmp/ab-results-$concurrent.txt
    
    # Parse results
    local rps=$(grep "Requests per second" /tmp/ab-results-$concurrent.txt | awk '{print $4}')
    local mean_time=$(grep "Time per request" /tmp/ab-results-$concurrent.txt | head -1 | awk '{print $4}')
    local failed=$(grep "Failed requests" /tmp/ab-results-$concurrent.txt | awk '{print $3}')
    
    log_info "Results for $concurrent concurrent users:"
    log_info "  Requests/sec: $rps"
    log_info "  Mean time: ${mean_time}ms"
    log_info "  Failed: $failed"
    
    if [ "$failed" = "0" ]; then
        echo -e "  ${GREEN}✅ PASS${NC} (no failures)"
    else
        echo -e "  ${YELLOW}⚠️  WARN${NC} ($failed failures)"
    fi
    
    # Clean up
    rm /tmp/request-body.json
}

# Load test with wrk
load_test_wrk() {
    local concurrent=$1
    local duration=$2
    
    log_info "Running wrk test: $concurrent concurrent connections for ${duration}s"
    
    # Create Lua script for POST requests
    cat > /tmp/wrk-script.lua << 'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"message":"Tell me about Istanbul","language":"en"}'
EOF
    
    # Run test
    wrk -t4 -c$concurrent -d${duration}s \
        -s /tmp/wrk-script.lua \
        "$BASE_URL/api/chat" > /tmp/wrk-results-$concurrent.txt
    
    # Parse results
    cat /tmp/wrk-results-$concurrent.txt
    
    # Clean up
    rm /tmp/wrk-script.lua
}

# Stress test - find breaking point
stress_test() {
    log_info "Running stress test to find breaking point..."
    
    local concurrent=10
    local max_concurrent=500
    local step=20
    
    while [ $concurrent -le $max_concurrent ]; do
        log_info "Testing with $concurrent concurrent users..."
        
        # Run quick test
        ab -n $((concurrent * 5)) \
           -c $concurrent \
           -p /tmp/stress-body.json \
           -T "application/json" \
           "$BASE_URL/api/chat" > /tmp/stress-results.txt 2>&1
        
        failed=$(grep "Failed requests" /tmp/stress-results.txt | awk '{print $3}')
        
        if [ "$failed" != "0" ]; then
            log_warn "Breaking point found at $concurrent concurrent users"
            break
        fi
        
        concurrent=$((concurrent + step))
    done
    
    log_info "System can handle up to $((concurrent - step)) concurrent users"
}

# Test database performance
test_database() {
    log_info "Testing database query performance..."
    
    # This would require database access
    # For now, we'll test through API endpoints that hit the database
    
    log_info "Testing chat history retrieval..."
    start_time=$(date +%s%N)
    curl -f "$BASE_URL/api/chat/history" -o /dev/null -s
    end_time=$(date +%s%N)
    duration_ms=$(( (end_time - start_time) / 1000000 ))
    log_info "  Query time: ${duration_ms}ms"
}

# Test cache effectiveness
test_cache() {
    log_info "Testing cache effectiveness..."
    
    # Same query twice - second should be faster
    query='{"message":"Weather in Istanbul?","language":"en"}'
    
    log_info "First request (cold cache):"
    start_time=$(date +%s%N)
    curl -X POST "$BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "$query" \
        -o /dev/null -s
    end_time=$(date +%s%N)
    cold_time=$(( (end_time - start_time) / 1000000 ))
    log_info "  Time: ${cold_time}ms"
    
    sleep 2
    
    log_info "Second request (warm cache):"
    start_time=$(date +%s%N)
    curl -X POST "$BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "$query" \
        -o /dev/null -s
    end_time=$(date +%s%N)
    warm_time=$(( (end_time - start_time) / 1000000 ))
    log_info "  Time: ${warm_time}ms"
    
    improvement=$(( (cold_time - warm_time) * 100 / cold_time ))
    log_info "  Cache improvement: ${improvement}%"
}

# Generate performance report
generate_report() {
    log_info "Generating performance report..."
    
    cat > performance-report.md << 'EOF'
# 📊 Performance Test Report

**Test Date:** $(date)
**Base URL:** $BASE_URL

## Summary

### Single Request Performance
- Target: < 5 seconds per request
- All languages tested
- Results in logs above

### Load Testing Results

#### Apache Bench Results
- See /tmp/ab-results-*.txt for detailed results

#### wrk Results
- See /tmp/wrk-results-*.txt for detailed results

### Stress Testing
- Breaking point identified
- System capacity documented

### Cache Performance
- Cold cache vs. warm cache comparison
- Cache effectiveness measured

## Recommendations

1. **If response time > 5s**: Optimize LLM prompts or implement caching
2. **If failures > 1%**: Increase server resources or add load balancing
3. **If database queries > 100ms**: Add indexes or optimize queries
4. **If cache improvement < 50%**: Review caching strategy

## Next Steps

- [ ] Review results and identify bottlenecks
- [ ] Implement recommended optimizations
- [ ] Re-run tests to verify improvements
- [ ] Set up continuous performance monitoring
EOF
    
    log_info "Report generated: performance-report.md"
}

# Main execution
main() {
    echo ""
    log_info "Starting performance tests..."
    echo ""
    
    check_prerequisites
    test_health
    
    # Single request tests
    test_single_request
    echo ""
    
    # Load tests
    for concurrent in "${CONCURRENT_USERS[@]}"; do
        load_test_ab $concurrent
        echo ""
    done
    
    # Stress test (optional)
    read -p "Run stress test to find breaking point? (yes/no): " run_stress
    if [ "$run_stress" = "yes" ]; then
        stress_test
        echo ""
    fi
    
    # Cache test
    test_cache
    echo ""
    
    # Database test
    test_database
    echo ""
    
    # Generate report
    generate_report
    
    log_info "🎉 Performance testing complete!"
    log_info "Review the generated report: performance-report.md"
}

# Run main
main
