# 📁 API Storage Location Guide

## 🗂️ Where Your APIs Are Stored

Your AI Istanbul project has **two types** of API storage:

### 1. 🔑 **API Keys Storage** (Environment Variables)
**Location**: `/Users/omer/Desktop/ai-stanbul/.env`

**Current contents**:
```bash
# Your API keys are stored here (currently with placeholder values)
GOOGLE_PLACES_API_KEY=your_google_places_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
OPENWEATHERMAP_API_KEY=your_openweather_key_here
TRIPADVISOR_API_KEY=your_tripadvisor_key_here

# Configuration settings
USE_REAL_APIS=true
ENABLE_CACHING=true
CACHE_DURATION_MINUTES=30
GOOGLE_PLACES_RATE_LIMIT=100
```

**To add real API keys**, edit this file:
```bash
# Replace these lines with your actual keys:
GOOGLE_PLACES_API_KEY=AIzaSyBvOkBo-981BRdMRdr2zA1Q0-h1_YXo0mY
OPENWEATHERMAP_API_KEY=a0f6b1c2d3e4f5a6b7c8d9e0f1a2b3c4
```

### 2. 🏗️ **API Client Code** (Implementation Files)
**Location**: `/Users/omer/Desktop/ai-stanbul/backend/api_clients/`

**Your API clients**:
```
📂 backend/api_clients/
├── 🆕 enhanced_google_places.py    # NEW: Real Google Places integration
├── 🆕 enhanced_weather.py          # NEW: Real OpenWeatherMap integration  
├── 🆕 istanbul_transport.py        # NEW: Istanbul transport data
├── 🆕 enhanced_api_service.py      # NEW: Unified API service
├── google_places.py               # Original (fallback)
├── weather_enhanced.py            # Original weather client
├── language_processing.py         # Advanced NLP processing
├── multimodal_ai.py               # Multi-modal AI features
├── predictive_analytics.py        # Predictive models
├── realtime_data.py               # Real-time data processing
└── __init__.py                    # Package initialization
```

---

## 🔄 How APIs Are Currently Working

### Current State (Without Real API Keys):
```python
# In enhanced_google_places.py
class EnhancedGooglePlacesClient:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_PLACES_API_KEY")  # Reads from .env
        self.has_api_key = bool(self.api_key)              # Currently False
        
        if not self.has_api_key:
            # Using enhanced mock data (what you see now)
            logger.warning("Using fallback mode with enhanced mock data")
```

### After You Add Real Keys:
```python
# Same code, but now:
self.api_key = "AIzaSyBvOkBo-981BRdMRdr2zA1Q0-h1_YXo0mY"  # Your real key
self.has_api_key = True                                    # Now True!

# Will use real Google Places API for live data
logger.info("Ready for live data integration!")
```

---

## 📋 API Configuration Details

### Where Each API Reads Its Configuration:

#### Google Places API:
```python
# File: backend/api_clients/enhanced_google_places.py
# Reads from .env:
self.api_key = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
self.rate_limit = int(os.getenv("GOOGLE_PLACES_RATE_LIMIT", "100"))
self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
```

#### Weather API:
```python
# File: backend/api_clients/enhanced_weather.py  
# Reads from .env:
self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
```

#### Transport API:
```python
# File: backend/api_clients/istanbul_transport.py
# Reads from .env:
self.base_url = os.getenv("ISTANBUL_TRANSPORT_BASE_URL", "http://api.iett.istanbul/")
self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
```

---

## 🔧 How to Update Your API Keys

### Method 1: Edit .env file directly
```bash
# Navigate to your project
cd /Users/omer/Desktop/ai-stanbul

# Edit the .env file
nano .env

# Replace the placeholder values:
GOOGLE_PLACES_API_KEY=your_actual_google_key_here
OPENWEATHERMAP_API_KEY=your_actual_weather_key_here
USE_REAL_APIS=true

# Save and exit (Ctrl+X, then Y, then Enter)
```

### Method 2: Use the setup script
```bash
# Run our setup script
./setup_api_keys.sh

# It will guide you through the process
```

### Method 3: Check current values
```bash
# See what's currently in your .env file
cat .env
```

---

## 🔍 How to Verify API Storage

### Check if APIs are loaded:
```bash
# Run the integration test
python real_api_integration.py

# Look for these status messages:
# "Google Places API: Using fallback mode" = No real key
# "Google Places API: Ready for live data" = Real key loaded
```

### Check environment variables:
```python
# You can check in Python:
import os
print("Google Places Key:", os.getenv("GOOGLE_PLACES_API_KEY"))
print("Weather Key:", os.getenv("OPENWEATHERMAP_API_KEY"))
print("Use Real APIs:", os.getenv("USE_REAL_APIS"))
```

---

## 🚨 Security Notes

### API Key Security:
```bash
✅ .env file is in .gitignore (won't be committed to git)
✅ API keys are only stored locally
✅ No API keys in source code
✅ Environment variables used for configuration
```

### Best Practices:
```bash
🔒 Never commit .env file to version control
🔒 Use API key restrictions (limit to specific APIs only)
🔒 Monitor API usage in provider dashboards
🔒 Rotate keys periodically for security
```

---

## 📊 Current API Status Summary

| API Service | File Location | Config Location | Status |
|-------------|---------------|-----------------|---------|
| **Google Places** | `backend/api_clients/enhanced_google_places.py` | `.env` → `GOOGLE_PLACES_API_KEY` | 🔄 Ready for real key |
| **Weather** | `backend/api_clients/enhanced_weather.py` | `.env` → `OPENWEATHERMAP_API_KEY` | 🔄 Ready for real key |
| **Transport** | `backend/api_clients/istanbul_transport.py` | `.env` → `ISTANBUL_TRANSPORT_BASE_URL` | ✅ Working with public data |
| **Unified Service** | `backend/api_clients/enhanced_api_service.py` | Combines all above | 🔄 Ready for enhancement |

---

## 🎯 Next Action

**To activate real APIs**:
1. **Edit** `/Users/omer/Desktop/ai-stanbul/.env`
2. **Replace** `your_google_places_key_here` with real Google Places API key
3. **Replace** `your_openweather_key_here` with real OpenWeatherMap API key  
4. **Save** the file
5. **Restart** your backend server
6. **Enjoy** real-time data! 🚀

Your API infrastructure is perfectly organized and ready for real keys!
