# Advanced AI Features Documentation

## Overview

The AI Istanbul travel guide backend has been enhanced with advanced intelligence features to provide more personalized, real-time, and contextually aware recommendations.

## New Features

### 1. Real-time Data Integration (`/ai/real-time-data`)

**Endpoint**: `GET /ai/real-time-data`

**Parameters**:
- `include_events`: bool (default: True) - Include live events data
- `include_crowds`: bool (default: True) - Include crowd level information
- `include_traffic`: bool (default: False) - Include traffic data
- `origin`: Optional[str] - Origin location for traffic routing
- `destination`: Optional[str] - Destination location for traffic routing

**Features**:
- Live events in Istanbul (concerts, festivals, exhibitions)
- Real-time crowd levels at popular attractions
- Traffic-aware routing recommendations
- Integration with Google Maps and event APIs

**Example Response**:
```json
{
  "success": true,
  "real_time_data": {
    "events": [
      {
        "name": "Istanbul Music Festival",
        "location": "Cemil Topuzlu Park",
        "start_time": "2024-01-15T19:00:00",
        "category": "music",
        "description": "Annual classical music festival"
      }
    ],
    "crowd_levels": [
      {
        "location_name": "Hagia Sophia",
        "current_crowd_level": "moderate",
        "estimated_wait_time": "15-20 minutes",
        "best_visit_time": "early_morning"
      }
    ],
    "traffic_info": {
      "current_conditions": "moderate",
      "recommended_routes": []
    }
  }
}
```

### 2. Multimodal AI (`/ai/analyze-image`, `/ai/analyze-menu`)

#### Image Analysis Endpoint

**Endpoint**: `POST /ai/analyze-image`

**Parameters**:
- `image`: UploadFile - Image file to analyze
- `context`: str (form data) - Additional context for analysis

**Features**:
- Location identification from photos
- Landmark recognition specific to Istanbul
- Scene understanding and description
- Recommendation generation based on visual content

#### Menu Analysis Endpoint

**Endpoint**: `POST /ai/analyze-menu`

**Parameters**:
- `image`: UploadFile - Menu image to analyze
- `dietary_restrictions`: Optional[str] (form data) - Dietary preferences

**Features**:
- Menu item recognition and description
- Dietary restriction analysis (vegetarian, vegan, halal, etc.)
- Cuisine type identification
- Price range estimation
- Personalized recommendations

**Example Response**:
```json
{
  "success": true,
  "menu_analysis": {
    "detected_items": [
      "Kebab", "Baklava", "Turkish Tea"
    ],
    "cuisine_type": "Turkish",
    "price_range": "moderate",
    "dietary_info": {
      "vegetarian_options": ["Baklava", "Turkish Tea"],
      "vegan_options": ["Turkish Tea"],
      "halal_certified": true
    },
    "recommendations": [
      "Try the traditional kebab for authentic Turkish flavor"
    ],
    "confidence_score": 0.92
  }
}
```

### 3. Predictive Analytics (`/ai/predictive-analytics`)

**Endpoint**: `GET /ai/predictive-analytics`

**Parameters**:
- `locations`: Optional[str] - Comma-separated list of locations
- `user_preferences`: Optional[str] - JSON string of user preferences

**Features**:
- Weather-based activity recommendations
- Seasonal adjustments for attractions
- Peak time predictions for popular sites
- Dynamic pricing insights

**Example Response**:
```json
{
  "success": true,
  "predictions": {
    "weather_prediction": {
      "recommended_activities": [
        "Indoor museum visits recommended due to rain",
        "Visit covered bazaars and shopping areas"
      ],
      "seasonal_adjustments": {
        "winter_recommendations": ["Hot Turkish tea", "Indoor attractions"]
      }
    },
    "crowd_predictions": {
      "hagia_sophia": {
        "peak_hours": ["10:00-12:00", "14:00-16:00"],
        "recommended_visit_time": "08:00-09:00"
      }
    },
    "dynamic_insights": {
      "trending_locations": ["Galata Tower", "Karakoy District"],
      "seasonal_events": ["Winter Festival", "New Year Celebrations"]
    }
  }
}
```

### 4. Enhanced Recommendations (`/ai/enhanced-recommendations`)

**Endpoint**: `GET /ai/enhanced-recommendations`

**Parameters**:
- `query`: str - User's travel query
- `include_realtime`: bool (default: True) - Include real-time data
- `include_predictions`: bool (default: True) - Include predictive analytics
- `session_id`: Optional[str] - Session ID for personalization

**Features**:
- Combines all AI features for comprehensive recommendations
- Session-based personalization
- Context-aware suggestions
- Multi-factor optimization (weather, crowds, preferences)

**Example Response**:
```json
{
  "success": true,
  "enhanced_data": {
    "current_weather": {
      "temperature": 15,
      "description": "Light rain",
      "humidity": 70
    },
    "real_time_info": {
      "events": [...],
      "crowd_levels": [...]
    },
    "predictions": {
      "weather_prediction": {...},
      "crowd_predictions": {...}
    },
    "enhanced_recommendations": [
      "Visit the Grand Bazaar - covered area perfect for rainy weather",
      "Low crowds at Topkapi Palace - great time to visit!",
      "Live event: Jazz Concert at Galata Tower - happening tonight"
    ],
    "user_preferences_applied": true,
    "ai_features_status": {
      "ai_intelligence": true,
      "advanced_ai": true
    }
  }
}
```

## Implementation Details

### Backend Architecture

The new features are implemented in three main modules:

1. **`api_clients/realtime_data.py`**
   - Real-time event aggregation
   - Crowd level monitoring
   - Traffic data integration
   - Async API clients with fallback support

2. **`api_clients/multimodal_ai.py`**
   - OpenAI Vision API integration
   - Google Vision API support
   - OCR capabilities
   - Istanbul-specific landmark recognition

3. **`api_clients/predictive_analytics.py`**
   - Weather-based recommendation engine
   - Seasonal activity scoring
   - Peak time modeling
   - Dynamic pricing analysis

### Session Management

All new features integrate with the existing session management system:
- User preferences are maintained across requests
- Learning from user interactions
- Contextual recommendations based on history

### Error Handling

Robust fallback mechanisms ensure the system remains functional:
- Graceful degradation when AI services are unavailable
- Dummy implementations for development/testing
- Comprehensive error logging

## Testing

Use the provided test script to verify all endpoints:

```bash
python test_advanced_endpoints.py
```

## Configuration

Set up environment variables for full functionality:

```bash
# OpenAI API Key (for multimodal AI)
OPENAI_API_KEY=your_openai_key

# Google Cloud API Key (for enhanced location services)
GOOGLE_CLOUD_API_KEY=your_google_key

# Weather API Key
WEATHER_API_KEY=your_weather_key
```

## Future Enhancements

- Integration with more real-time data sources
- Advanced computer vision for architectural analysis
- Machine learning models for personalized predictions
- Integration with Istanbul public transportation APIs
- Social media sentiment analysis for location popularity

## Usage Examples

### 1. Getting Real-time Data for Trip Planning

```python
import requests

response = requests.get("http://localhost:8000/ai/real-time-data", params={
    "include_events": True,
    "include_crowds": True
})

data = response.json()
print("Current events:", data["real_time_data"]["events"])
print("Crowd levels:", data["real_time_data"]["crowd_levels"])
```

### 2. Analyzing a Restaurant Menu

```python
import requests

with open("menu_photo.jpg", "rb") as f:
    files = {"image": f}
    data = {"dietary_restrictions": "vegetarian"}
    
    response = requests.post(
        "http://localhost:8000/ai/analyze-menu",
        files=files,
        data=data
    )

result = response.json()
print("Vegetarian options:", result["menu_analysis"]["dietary_info"]["vegetarian_options"])
```

### 3. Getting Comprehensive Recommendations

```python
import requests

response = requests.get("http://localhost:8000/ai/enhanced-recommendations", params={
    "query": "I want to visit historical places but avoid crowds",
    "session_id": "user_123"
})

recommendations = response.json()["enhanced_data"]["enhanced_recommendations"]
for rec in recommendations:
    print(f"- {rec}")
```

This enhanced AI system transforms the Istanbul travel guide into an intelligent, responsive platform that adapts to real-time conditions and user preferences for optimal travel experiences.
