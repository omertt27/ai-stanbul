#!/bin/bash
echo "🔍 Diagnosing Connection Issue"
echo "================================"
echo ""

# Test 1: DNS Resolution
echo "1️⃣ DNS Resolution:"
dig +short api.asdweq123.org
echo ""

# Test 2: Can we reach Cloudflare IPs?
echo "2️⃣ Testing Cloudflare IPs directly:"
CF_IP=$(dig +short api.asdweq123.org | head -1)
echo "   Cloudflare IP: $CF_IP"
curl -v --connect-timeout 5 https://$CF_IP/health -H "Host: api.asdweq123.org" 2>&1 | grep -E "Connected|HTTP|error"
echo ""

# Test 3: Try with different curl options
echo "3️⃣ Testing with --insecure flag:"
curl -s --insecure --connect-timeout 5 https://api.asdweq123.org/health
echo ""

# Test 4: Check if it's IPv6 issue
echo "4️⃣ Testing with IPv4 only:"
curl -4 -s --connect-timeout 5 https://api.asdweq123.org/health 2>&1
echo ""

# Test 5: Flush DNS cache and retry
echo "5️⃣ Flushing DNS cache..."
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
sleep 2
echo "   Retrying after DNS flush:"
curl -s --connect-timeout 5 https://api.asdweq123.org/health
echo ""

echo "================================"
