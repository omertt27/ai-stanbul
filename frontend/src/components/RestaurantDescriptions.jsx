import React, { useState, useEffect } from 'react';
import './RestaurantDescriptions.css';

const RestaurantDescriptions = () => {
  const [restaurants, setRestaurants] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchParams, setSearchParams] = useState({
    district: '',
    keyword: '',
    limit: 10
  });

  const istanbulDistricts = [
    'Beyoğlu', 'Sultanahmet', 'Beşiktaş', 'Kadıköy', 
    'Şişli', 'Fatih', 'Üsküdar', 'Bakırköy', 'Zeytinburnu'
  ];

  const fetchRestaurants = async () => {
    setLoading(true);
    setError('');
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
      const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
      let url = `${cleanApiUrl}/restaurants/search`;
      const params = new URLSearchParams();
      
      if (searchParams.district) {
        params.append('district', searchParams.district);
      }
      if (searchParams.keyword) {
        params.append('keyword', searchParams.keyword);
      }
      params.append('limit', searchParams.limit.toString());

      const response = await fetch(`${url}?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setRestaurants(data.restaurants);
      } else {
        setError(`API Error: ${data.status || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Restaurant search error:', err);
      setError(`Connection failed: ${err.message}. Make sure backend is running on ${apiUrl}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchPopularRestaurants = async () => {
    setLoading(true);
    setError('');
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
      const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
      const response = await fetch(`${cleanApiUrl}/restaurants/popular?min_rating=4.0&limit=12`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setRestaurants(data.restaurants);
      } else {
        setError(`API Error: ${data.status || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Popular restaurants error:', err);
      setError(`Connection failed: ${err.message}. Make sure backend is running on ${apiUrl}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSearchParams(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    fetchRestaurants();
  };

  const RestaurantCard = ({ restaurant }) => (
    <div className="restaurant-card">
      <div className="restaurant-header">
        <h3 className="restaurant-name">{restaurant.name}</h3>
        <div className="restaurant-rating">
          <span className="stars">{'⭐'.repeat(Math.floor(restaurant.rating || 0))}</span>
          <span className="rating-text">{restaurant.rating}/5</span>
          <span className="review-count">({restaurant.user_ratings_total} reviews)</span>
        </div>
      </div>

      <div className="restaurant-info">
        <p className="address">📍 {restaurant.address}</p>
        {restaurant.phone && <p className="phone">📞 {restaurant.phone}</p>}
        <p className="cuisine">🍴 {restaurant.cuisine_types}</p>
        {restaurant.price_level && (
          <p className="price">💰 {'$'.repeat(restaurant.price_level)} Price Level</p>
        )}
      </div>

      <div className="restaurant-description">
        <h4>Description</h4>
        <p>{restaurant.description}</p>
      </div>

      {restaurant.opening_hours?.weekday_text && (
        <div className="opening-hours">
          <h4>Hours</h4>
          <div className="hours-list">
            {restaurant.opening_hours.weekday_text.slice(0, 3).map((hours, index) => (
              <p key={index} className="hours-item">{hours}</p>
            ))}
          </div>
        </div>
      )}

      {restaurant.reviews_summary?.recent_review_snippet && (
        <div className="recent-review">
          <h4>Recent Review</h4>
          <p className="review-text">"{restaurant.reviews_summary.recent_review_snippet}"</p>
        </div>
      )}

      {restaurant.photos && restaurant.photos.length > 0 && (
        <div className="restaurant-photos">
          <p>📸 {restaurant.photos.length} photos available</p>
        </div>
      )}

      <div className="restaurant-actions">
        {restaurant.website && (
          <a 
            href={restaurant.website} 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn btn-website"
          >
            Visit Website
          </a>
        )}
        <button 
          className="btn btn-details"
          onClick={() => window.open(`https://www.google.com/maps/place/?q=place_id:${restaurant.place_id}`, '_blank')}
        >
          View on Maps
        </button>
      </div>
    </div>
  );

  useEffect(() => {
    // Load popular restaurants on component mount
    fetchPopularRestaurants();
  }, []);

  return (
    <div className="restaurant-descriptions-container">
      <header className="header">
        <h1>🍽️ Istanbul Restaurant Descriptions</h1>
        <p>Discover amazing restaurants with detailed descriptions from Google Maps</p>
      </header>

      <div className="search-section">
        <form onSubmit={handleSubmit} className="search-form">
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="district">District</label>
              <select
                id="district"
                name="district"
                value={searchParams.district}
                onChange={handleInputChange}
                className="form-control"
              >
                <option value="">All Istanbul</option>
                {istanbulDistricts.map(district => (
                  <option key={district} value={district}>{district}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="keyword">Keyword</label>
              <input
                type="text"
                id="keyword"
                name="keyword"
                value={searchParams.keyword}
                onChange={handleInputChange}
                placeholder="e.g., cafe, kebab, seafood"
                className="form-control"
              />
            </div>

            <div className="form-group">
              <label htmlFor="limit">Limit</label>
              <select
                id="limit"
                name="limit"
                value={searchParams.limit}
                onChange={handleInputChange}
                className="form-control"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
              </select>
            </div>
          </div>

          <div className="button-group">
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Searching...' : 'Search Restaurants'}
            </button>
            <button 
              type="button" 
              className="btn btn-secondary" 
              onClick={fetchPopularRestaurants}
              disabled={loading}
            >
              Popular Restaurants
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="error-message">
          <p>❌ {error}</p>
        </div>
      )}

      {loading && (
        <div className="loading-spinner">
          <p>🔍 Loading restaurants...</p>
        </div>
      )}

      <div className="restaurants-grid">
        {restaurants.map((restaurant, index) => (
          <RestaurantCard key={restaurant.place_id || index} restaurant={restaurant} />
        ))}
      </div>

      {!loading && restaurants.length === 0 && !error && (
        <div className="no-results">
          <p>No restaurants found. Try adjusting your search criteria.</p>
        </div>
      )}
    </div>
  );
};

export default RestaurantDescriptions;
