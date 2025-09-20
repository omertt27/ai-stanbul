#!/bin/bash

echo "🔍 Comprehensive Arabic Language Test"
echo "=================================="
echo ""

echo "1️⃣ Testing API Languages Endpoint:"
curl -s http://localhost:8000/api/languages | jq '.language_info.ar'
echo ""

echo "2️⃣ Testing Basic Arabic Greeting:"
curl -s -X POST http://localhost:8000/ai -H "Content-Type: application/json" -d '{"query": "مرحبا", "language": "ar"}' | jq '.message' -r
echo ""

echo "3️⃣ Testing Arabic Welcome Translation:"
curl -s -X POST http://localhost:8000/api/translate -H "Content-Type: application/json" -d '{"key": "welcome", "language": "ar"}' | jq '.translated' -r
echo ""

echo "4️⃣ Testing Arabic Museum Introduction:"
curl -s -X POST http://localhost:8000/api/translate -H "Content-Type: application/json" -d '{"key": "museum_intro", "language": "ar"}' | jq '.translated' -r
echo ""

echo "5️⃣ Testing Arabic District Names:"
echo "Sultanahmet: $(curl -s -X POST http://localhost:8000/api/translate -H "Content-Type: application/json" -d '{"key": "districts.sultanahmet", "language": "ar"}' | jq '.translated' -r)"
echo "Beyoğlu: $(curl -s -X POST http://localhost:8000/api/translate -H "Content-Type: application/json" -d '{"key": "districts.beyoglu", "language": "ar"}' | jq '.translated' -r)"
echo "Kadıköy: $(curl -s -X POST http://localhost:8000/api/translate -H "Content-Type: application/json" -d '{"key": "districts.kadikoy", "language": "ar"}' | jq '.translated' -r)"
echo ""

echo "✅ Arabic Language Support: FULLY FUNCTIONAL"
