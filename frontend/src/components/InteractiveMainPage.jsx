import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

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
      sideNote: 'European Side • Historic Peninsula'
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
      sideNote: 'European Side • Cultural Heart'
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
      sideNote: 'Asian Side • Bohemian Quarter'
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
      sideNote: 'European Side • Bosphorus Coast'
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
      sideNote: 'Asian Side • Historic Center'
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
      sideNote: 'European Side • Business Hub'
    },
    {
      id: 'buyukada',
      name: 'Büyükada',
      color: '#2ecc71',
      description: 'Largest of the Prince Islands, car-free paradise',
      highlights: ['Horse Carriages', 'Historic Mansions', 'Pine Forests'],
      vibe: '🏝️ Peaceful',
      specialty: 'Island Life',
      population: '7,500',
      sideNote: 'Prince Islands • Largest Island'
    },
    {
      id: 'heybeliada',
      name: 'Heybeliada',
      color: '#16a085',
      description: 'Second largest Prince Island with naval school',
      highlights: ['Naval Academy', 'Beaches', 'Island Walks'],
      vibe: '🌊 Serene',
      specialty: 'Maritime Heritage',
      population: '4,000',
      sideNote: 'Prince Islands • Naval Heritage'
    },
    {
      id: 'burgazada',
      name: 'Burgazada',
      color: '#27ae60',
      description: 'Third largest Prince Island, writers\' retreat',
      highlights: ['Sait Faik Museum', 'Quiet Beaches', 'Writers\' Houses'],
      vibe: '📚 Literary',
      specialty: 'Literature & Arts',
      population: '1,500',
      sideNote: 'Prince Islands • Literary Haven'
    },
    {
      id: 'kinaliada',
      name: 'Kınalıada',
      color: '#e67e22',
      description: 'Smallest inhabited Prince Island with red cliffs',
      highlights: ['Red Cliffs', 'Small Beaches', 'Traditional Houses'],
      vibe: '🏖️ Intimate',
      specialty: 'Natural Beauty',
      population: '1,200',
      sideNote: 'Prince Islands • Hidden Gem'
    },
    {
      id: 'galata',
      name: 'Galata',
      color: '#8e44ad',
      description: 'Historic tower district with panoramic views',
      highlights: ['Galata Tower', 'Golden Horn Views', 'Historic Streets'],
      vibe: '🗼 Iconic',
      specialty: 'Historic Views',
      population: '45,000',
      sideNote: 'European Side • Historic Quarter'
    },
    {
      id: 'ortakoy',
      name: 'Ortaköy',
      color: '#e91e63',
      description: 'Charming Bosphorus village with mosque & bridge views',
      highlights: ['Ortaköy Mosque', 'Bridge Views', 'Artisan Markets'],
      vibe: '🌉 Picturesque',
      specialty: 'Bosphorus Beauty',
      population: '25,000',
      sideNote: 'European Side • Bosphorus Village'
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
                background: `linear-gradient(135deg, ${district.color}10, ${district.color}20)`
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
