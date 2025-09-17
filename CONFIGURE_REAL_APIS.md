# 🔑 Real API Key Configuration Guide

## Current Status
Your .env file still contains placeholder values:
```
GOOGLE_PLACES_API_KEY=your_google_places_key_here  ← Replace this
OPENWEATHERMAP_API_KEY=your_openweather_key_here   ← Replace this
```

## ✅ To Activate Your Real APIs:

### Step 1: Edit .env file
```bash
# Open your .env file for editing:
nano /Users/omer/Desktop/ai-stanbul/.env

# Or use any text editor:
code /Users/omer/Desktop/ai-stanbul/.env
```

### Step 2: Replace placeholder values with your real API keys
```bash
# Change these lines:
GOOGLE_PLACES_API_KEY=your_google_places_key_here
OPENWEATHERMAP_API_KEY=your_openweather_key_here

# To your actual keys (example format):
GOOGLE_PLACES_API_KEY=AIzaSyBvOkBo-981BRdMRdr2zA1Q0-h1_YXo0mY
OPENWEATHERMAP_API_KEY=a0f6b1c2d3e4f5a6b7c8d9e0f1a2b3c4

# Make sure USE_REAL_APIS=true (should already be set)
USE_REAL_APIS=true
```

### Step 3: Save and verify
```bash
# Save the file and run verification:
python verify_real_apis.py

# You should see:
# ✅ Google Places API working! Found X restaurants
# ✅ OpenWeatherMap API working! Istanbul: 18°C, clear sky
```

### Step 4: Restart backend to use real data
```bash
# Stop your current backend (Ctrl+C if running)
# Then start it again:
python backend/main.py
```

## 🎯 Expected Results After Configuration:

### Restaurant Queries Will Show:
```
Before: "Sample Turkish Restaurant - Rating: 4.5"
After:  "Pandeli Restaurant - Rating: 4.3 ⭐ (1,247 reviews)
         📍 Eminönü Meydanı, Historic Spice Bazaar
         🕒 Open now until 22:00"
```

### Weather Queries Will Show:
```
Before: "15°C, clouds sky (mock data)"
After:  "18°C, broken clouds (real-time)
         💡 Perfect weather for outdoor dining terraces!"
```

## 🔧 Quick Configuration Commands:

```bash
# 1. Backup current .env
cp .env .env.backup

# 2. Edit .env file (replace with your real keys)
nano .env

# 3. Verify APIs are working
python verify_real_apis.py

# 4. Test with real data
python test_chatbot_quick.py
```

## 🚨 Security Reminder:
- Never share your API keys
- Never commit .env file to git
- Monitor your API usage in provider dashboards
- Set usage limits to prevent unexpected charges

Ready to add your real API keys? 🚀
