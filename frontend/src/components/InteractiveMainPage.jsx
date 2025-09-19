import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './InteractiveMainPage.css';

const InteractiveMainPage = ({ onQuickStart }) => {
  const navigate = useNavigate();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [hoveredDistrict, setHoveredDistrict] = useState(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const getTimeBasedSuggestions = () => {
    const hour = currentTime.getHours();
    if (hour >= 6 && hour < 11) {
      return [
        { icon: '☕', text: 'Best Turkish Breakfast', query: 'best Turkish breakfast places' },
        { icon: '🌅', text: 'Morning Views', query: 'best sunrise spots in Istanbul' },
        { icon: '🥖', text: 'Fresh Bakeries', query: 'traditional Turkish bakeries' }
      ];
    } else if (hour >= 11 && hour < 17) {
      return [
        { icon: '🏛️', text: 'Museums & Culture', query: 'must visit museums in Istanbul' },
        { icon: '🛍️', text: 'Shopping Districts', query: 'best shopping areas in Istanbul' },
        { icon: '🍽️', text: 'Lunch Spots', query: 'good lunch restaurants' }
      ];
    } else {
      return [
        { icon: '🌆', text: 'Sunset Views', query: 'best sunset spots Istanbul' },
        { icon: '🍷', text: 'Evening Dining', query: 'romantic dinner restaurants' },
        { icon: '🎭', text: 'Nightlife', query: 'Istanbul nightlife recommendations' }
      ];
    }
  };

  const istanbulDistricts = [
    {
      id: 'sultanahmet',
      name: 'Sultanahmet',
      color: '#e74c3c',
      description: 'Historic peninsula with Ottoman & Byzantine treasures',
      highlights: ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace'],
      vibe: '🏛️ Imperial',
      specialty: 'Ottoman Heritage',
      population: '65,000',
      sideNote: 'European Side • Historic Peninsula',
      backgroundImage: 'https://images.unsplash.com/photo-1541432901042-2d8bd64b4a9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
    },
    {
      id: 'beyoglu',
      name: 'Beyoğlu',
      color: '#9b59b6',
      description: 'Cultural hub with vibrant nightlife & arts',
      highlights: ['Istiklal Street', 'Art Galleries', 'Rooftop Bars'],
      vibe: '🎭 Artistic',
      specialty: 'Modern Culture',
      population: '240,000',
      sideNote: 'European Side • Cultural Heart',
      backgroundImage: '/districts/beyoglu.jpg'
    },
    {
      id: 'kadikoy',
      name: 'Kadıköy',
      color: '#27ae60',
      description: 'Asian side bohemian heart with authentic vibes',
      highlights: ['Moda District', 'Local Markets', 'Street Food'],
      vibe: '🍜 Foodie',
      specialty: 'Authentic Local Life',
      population: '460,000',
      sideNote: 'Asian Side • Bohemian Quarter',
      backgroundImage: '/districts/kadikoy.jpg'
    },
    {
      id: 'besiktas',
      name: 'Beşiktaş',
      color: '#3498db',
      description: 'Upscale waterfront district with modern attractions',
      highlights: ['Dolmabahçe Palace', 'Bosphorus Views', 'Vodafone Park'],
      vibe: '⚽ Dynamic',
      specialty: 'Luxury & Sports',
      population: '190,000',
      sideNote: 'European Side • Bosphorus Coast',
      backgroundImage: '/districts/besiktas.jpeg'
    },
    {
      id: 'uskudar',
      name: 'Üsküdar',
      color: '#f39c12',
      description: 'Traditional Asian side with stunning city views',
      highlights: ['Maiden\'s Tower', 'Çamlıca Hill', 'Historic Mosques'],
      vibe: '🕌 Spiritual',
      specialty: 'Traditional Life',
      population: '530,000',
      sideNote: 'Asian Side • Historic Center',
      backgroundImage: '/districts/uskudar.jpg'
    },
    {
      id: 'sisli',
      name: 'Şişli',
      color: '#1abc9c',
      description: 'Business district with luxury shopping & hotels',
      highlights: ['Cevahir Mall', 'Business Centers', 'Luxury Hotels'],
      vibe: '💼 Business',
      specialty: 'Commerce & Shopping',
      population: '270,000',
      sideNote: 'European Side • Business Hub',
      backgroundImage: '/districts/Sisli.jpeg'
    }
  ];

  const handleDistrictClick = (district) => {
    const query = `tell me about ${district.name} district in Istanbul`;
    onQuickStart(query);
    navigate('/chatbot');
  };

  const handleQuickAction = (suggestion) => {
    onQuickStart(suggestion.query);
    navigate('/chatbot');
  };

  return (
    <div className="interactive-main-page">
      <div className="time-based-suggestions">
        <div className="current-time">
          <span className="time-display">
            {currentTime.toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit',
              timeZone: 'Europe/Istanbul' 
            })}
          </span>
          <span className="time-zone">Istanbul Time</span>
        </div>
        
        <div className="suggestion-bubbles">
          {getTimeBasedSuggestions().map((suggestion, index) => (
            <button
              key={index}
              className="suggestion-bubble"
              onClick={() => handleQuickAction(suggestion)}
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <span className="suggestion-icon">{suggestion.icon}</span>
              <span className="suggestion-text">{suggestion.text}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="districts-gallery-container">
        <div className="gallery-title">
          <h3>🏙️ Explore Istanbul Districts & Islands</h3>
          <p>Discover {istanbulDistricts.length} unique neighborhoods and islands - each with its own character and charm</p>
        </div>
        
        <div className="districts-gallery">
          {istanbulDistricts.map((district) => (
            <div
              key={district.id}
              className={`district-card ${hoveredDistrict === district.id ? 'hovered' : ''}`}
              onMouseEnter={() => setHoveredDistrict(district.id)}
              onMouseLeave={() => setHoveredDistrict(null)}
              onClick={() => handleDistrictClick(district)}
              style={{
                borderColor: district.color,
                backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.7)), url(${district.backgroundImage})`,
                backgroundSize: 'cover',
                backgroundPosition: district.id === 'uskudar' ? 'left center' : 'center',
                backgroundRepeat: 'no-repeat'
              }}
            >
              <div className="district-card-header">
                <div 
                  className="district-color-indicator"
                  style={{ backgroundColor: district.color }}
                ></div>
                <h4 className="district-name">{district.name}</h4>
                <span className="district-vibe">{district.vibe}</span>
              </div>
              
              <div className="district-card-content">
                <p className="district-description">{district.description}</p>
                
                <div className="district-info">
                  <div className="district-specialty">
                    <span className="specialty-label">Specialty:</span>
                    <span className="specialty-value">{district.specialty}</span>
                  </div>
                  <div className="district-location">
                    <span className="location-value">{district.sideNote}</span>
                  </div>
                  <div className="district-population">
                    <span className="population-label">Population:</span>
                    <span className="population-value">{district.population}</span>
                  </div>
                </div>
                
                <div className="district-highlights">
                  <span className="highlights-label">Must Experience:</span>
                  <div className="highlights-list">
                    {district.highlights.map((highlight, idx) => (
                      <span key={idx} className="highlight-tag">{highlight}</span>
                    ))}
                  </div>
                </div>
                
                <button 
                  className="explore-district-btn"
                  style={{ 
                    backgroundColor: district.color,
                    borderColor: district.color 
                  }}
                >
                  Explore {district.name} →
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default InteractiveMainPage;
