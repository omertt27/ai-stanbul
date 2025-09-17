#!/bin/bash
# API Keys Setup Script for Real Data Integration

echo "🔑 Setting up API Keys for Real Data Integration"
echo "=============================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created"
else
    echo "📋 .env file already exists"
fi

echo ""
echo "🌟 Next Steps:"
echo "1. Edit .env file and add your real API keys:"
echo "   - GOOGLE_PLACES_API_KEY=your_actual_key"
echo "   - OPENWEATHERMAP_API_KEY=your_actual_key"
echo ""
echo "2. Set USE_REAL_APIS=true in .env"
echo ""
echo "3. Restart your backend server:"
echo "   python backend/main.py"
echo ""
echo "🚀 Then your app will use real live data!"
echo ""
echo "📖 API Key Setup Guide:"
echo "   - Google Places: https://console.cloud.google.com/"
echo "   - OpenWeatherMap: https://openweathermap.org/api"
echo ""
