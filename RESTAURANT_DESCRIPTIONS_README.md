# 🍽️ Restaurant Descriptions from Google Maps

This feature provides comprehensive restaurant information with detailed descriptions using the Google Places API. Get rich data about Istanbul restaurants including descriptions, reviews, photos, and more.

## 🚀 Features

### Backend API Endpoints

- **`GET /restaurants/search`** - Search restaurants with descriptions
- **`GET /restaurants/istanbul/{district}`** - Get restaurants from specific Istanbul districts
- **`GET /restaurants/popular`** - Get highly-rated restaurants
- **`GET /restaurants/details/{place_id}`** - Get detailed info for a specific restaurant
- **`POST /restaurants/save`** - Save a restaurant to the local database

### Frontend Component

- Interactive restaurant search interface
- District-based filtering
- Beautiful restaurant cards with descriptions
- Photo galleries and review summaries
- Responsive design for mobile and desktop

### Data Included

Each restaurant includes:
- ✅ Name and address
- ⭐ Ratings and review counts
- 📝 AI-generated descriptions
- 🍴 Cuisine types
- 📞 Contact information
- 🌐 Website links
- 🕒 Opening hours
- 💬 Recent reviews
- 📸 Photos
- 💰 Price levels

## 🛠️ Setup & Installation

### Prerequisites

1. **Google Places API Key**
   - Get an API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Places API, Geocoding API, and Places Photos API
   - Set your API key in the environment or directly in the code

2. **Python Dependencies**
   ```bash
   pip install fastapi uvicorn requests sqlalchemy python-dotenv
   ```

3. **Node.js Dependencies**
   ```bash
   npm install react
   ```

### Backend Setup

1. **Configure API Key**
   ```python
   # Option 1: Environment variable
   export GOOGLE_PLACES_API_KEY="your_api_key_here"
   
   # Option 2: Direct in code (api_clients/google_places.py)
   self.api_key = "your_api_key_here"
   ```

2. **Start the FastAPI Server**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

3. **Test the API**
   ```bash
   # Run the test script
   python test_restaurants_descriptions.py
   
   # Or test endpoints directly
   curl "http://localhost:8000/restaurants/search?district=Beyoğlu&limit=5"
   ```

### Frontend Setup

1. **Add Component to Your App**
   ```jsx
   import RestaurantDescriptions from './components/RestaurantDescriptions';
   
   function App() {
     return (
       <div className="App">
         <RestaurantDescriptions />
       </div>
     );
   }
   ```

2. **Start the React App**
   ```bash
   cd frontend
   npm start
   ```

## 📚 API Usage Examples

### Basic Search
```bash
# Search restaurants in Istanbul
curl "http://localhost:8000/restaurants/search?limit=10"

# Search in specific district
curl "http://localhost:8000/restaurants/search?district=Beyoğlu&limit=5"

# Search with keyword
curl "http://localhost:8000/restaurants/search?keyword=cafe&limit=8"
```

### District-Specific Search
```bash
# Get restaurants from Sultanahmet
curl "http://localhost:8000/restaurants/istanbul/Sultanahmet"

# Get restaurants from Kadıköy
curl "http://localhost:8000/restaurants/istanbul/Kadıköy?limit=15"
```

### Popular Restaurants
```bash
# Get highly-rated restaurants
curl "http://localhost:8000/restaurants/popular?min_rating=4.5&limit=20"
```

### Restaurant Details
```bash
# Get detailed information for a specific place
curl "http://localhost:8000/restaurants/details/ChIJN1t_tDeuEmsRUsoyG83frY4"
```

## 🗄️ Database Integration

### Restaurant Model
```python
class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    cuisine = Column(String)
    location = Column(String)
    rating = Column(Float)
    source = Column(String)
    description = Column(String)  # New field for descriptions
    place_id = Column(String, unique=True)  # Google Places ID
    phone = Column(String)
    website = Column(String)
    price_level = Column(Integer)
```

### Save Restaurant to Database
```bash
# Save a restaurant from Google Places to local DB
curl -X POST "http://localhost:8000/restaurants/save?place_id=ChIJN1t_tDeuEmsRUsoyG83frY4"
```

## 🎨 Frontend Usage

### Basic Implementation
```jsx
import RestaurantDescriptions from './components/RestaurantDescriptions';

// Use the component
<RestaurantDescriptions />
```

### Custom API Base URL
```jsx
// Modify the component to use your API base URL
const API_BASE = 'https://your-api-domain.com';
```

## 🔧 Customization

### Adding More Fields
To include additional fields from Google Places:

1. **Update the GooglePlacesClient**
   ```python
   # In api_clients/google_places.py
   default_fields = [
       "place_id", "name", "formatted_address",
       "your_new_field_here"  # Add your field
   ]
   ```

2. **Update the Restaurant Model**
   ```python
   # In models.py
   class Restaurant(Base):
       # ... existing fields ...
       your_new_field = Column(String)  # Add database field
   ```

### Filtering Options
Add custom filters in the API:

```python
@router.get("/search")
def search_restaurants_with_descriptions(
    min_rating: float = Query(None, ge=1.0, le=5.0),
    max_price: int = Query(None, ge=1, le=4),
    open_now: bool = Query(None),
    # Add your custom filters here
):
    # Implementation
```

## 🌍 Popular Istanbul Districts

The system includes support for these popular districts:
- **Beyoğlu** - Trendy restaurants and cafes
- **Sultanahmet** - Traditional Turkish cuisine
- **Beşiktaş** - Modern dining options
- **Kadıköy** - Hip food scene
- **Şişli** - Upscale restaurants
- **Fatih** - Authentic local eateries
- **Üsküdar** - Waterfront dining
- **Bakırköy** - Family restaurants
- **Zeytinburnu** - Local neighborhood spots

## 🔍 Search Tips

1. **Use specific keywords**: "kebab", "seafood", "vegetarian", "cafe"
2. **Try district names**: More accurate results for specific areas
3. **Adjust radius**: Smaller radius for precise area search
4. **Filter by rating**: Get only highly-rated establishments

## 🚨 Troubleshooting

### Common Issues

1. **No restaurants found**
   - Check your Google Places API key
   - Verify API key has proper permissions
   - Try increasing the search radius
   - Check if the location name is spelled correctly

2. **API Key Errors**
   - Ensure Places API is enabled in Google Cloud Console
   - Check billing is set up if using a lot of requests
   - Verify the API key is not restricted inappropriately

3. **Database Errors**
   - Make sure SQLAlchemy is properly configured
   - Check database connection
   - Ensure tables are created (`Base.metadata.create_all(engine)`)

### Rate Limits

Google Places API has usage limits:
- **Nearby Search**: $32 per 1000 requests
- **Place Details**: $17 per 1000 requests
- **Photos**: $7 per 1000 requests

Consider implementing caching to reduce API calls.

## 📈 Future Enhancements

- [ ] Add restaurant reservation integration
- [ ] Implement user reviews and ratings
- [ ] Add menu photo recognition
- [ ] Include dietary restriction filters
- [ ] Add distance-based sorting
- [ ] Implement favorite restaurants feature
- [ ] Add restaurant comparison tool

## 📄 License

This project is part of the AI Istanbul application. Please refer to the main project license.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

---

**🍽️ Happy Dining! Enjoy discovering Istanbul's amazing restaurants with detailed descriptions!**
