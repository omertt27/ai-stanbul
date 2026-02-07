import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import './InteractiveMainPage.css';

const InteractiveMainPage = ({ onQuickStart }) => {
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
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
        { icon: '☕', text: t('homepage.suggestions.morning.breakfast'), query: t('homepage.queries.breakfastPlaces') },
        { icon: '🌅', text: t('homepage.suggestions.morning.views'), query: t('homepage.queries.sunriseSpots') },
        { icon: '🥖', text: t('homepage.suggestions.morning.bakeries'), query: t('homepage.queries.turkishBakeries') }
      ];
    } else if (hour >= 11 && hour < 17) {
      return [
        { icon: '🏛️', text: t('homepage.suggestions.afternoon.museums'), query: t('homepage.queries.museums') },
        { icon: '🛍️', text: t('homepage.suggestions.afternoon.shopping'), query: t('homepage.queries.shoppingAreas') },
        { icon: '🍽️', text: t('homepage.suggestions.afternoon.lunch'), query: t('homepage.queries.lunchRestaurants') }
      ];
    } else {
      return [
        { icon: '🌆', text: t('homepage.suggestions.evening.sunset'), query: t('homepage.queries.sunsetSpots') },
        { icon: '🍷', text: t('homepage.suggestions.evening.dining'), query: t('homepage.queries.romanticDinner') },
        { icon: '�', text: t('homepage.suggestions.evening.culturalShows'), query: t('homepage.queries.culturalShows') }
      ];
    }
  };

  const istanbulDistricts = [
    {
      id: 'sultanahmet',
      name: t('homepage.districts.sultanahmet.name'),
      color: '#e74c3c',
      description: t('homepage.districts.sultanahmet.description'),
      highlights: t('homepage.districts.sultanahmet.highlights', { returnObjects: true }) || ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace'],
      vibe: `🏛️ ${t('homepage.districts.sultanahmet.vibe')}`,
      specialty: t('homepage.districts.sultanahmet.specialty'),
      population: '65,000',
      sideNote: t('homepage.districts.sultanahmet.sideNote'),
      backgroundImage: 'https://images.unsplash.com/photo-1541432901042-2d8bd64b4a9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
    },
    {
      id: 'beyoglu',
      name: t('homepage.districts.beyoglu.name'),
      color: '#9b59b6',
      description: t('homepage.districts.beyoglu.description'),
      highlights: t('homepage.districts.beyoglu.highlights', { returnObjects: true }) || ['Istiklal Street', 'Art Galleries', 'Rooftop Bars'],
      vibe: `🎭 ${t('homepage.districts.beyoglu.vibe')}`,
      specialty: t('homepage.districts.beyoglu.specialty'),
      population: '240,000',
      sideNote: t('homepage.districts.beyoglu.sideNote'),
      backgroundImage: '/districts/beyoglu.jpg'
    },
    {
      id: 'kadikoy',
      name: t('homepage.districts.kadikoy.name'),
      color: '#27ae60',
      description: t('homepage.districts.kadikoy.description'),
      highlights: t('homepage.districts.kadikoy.highlights', { returnObjects: true }) || ['Moda District', 'Local Markets', 'Street Food'],
      vibe: `🍜 ${t('homepage.districts.kadikoy.vibe')}`,
      specialty: t('homepage.districts.kadikoy.specialty'),
      population: '460,000',
      sideNote: t('homepage.districts.kadikoy.sideNote'),
      backgroundImage: '/districts/kadikoy.jpg'
    },
    {
      id: 'besiktas',
      name: t('homepage.districts.besiktas.name'),
      color: '#3498db',
      description: t('homepage.districts.besiktas.description'),
      highlights: t('homepage.districts.besiktas.highlights', { returnObjects: true }) || ['Dolmabahçe Palace', 'Bosphorus Views', 'Vodafone Park'],
      vibe: `⚽ ${t('homepage.districts.besiktas.vibe')}`,
      specialty: t('homepage.districts.besiktas.specialty'),
      population: '190,000',
      sideNote: t('homepage.districts.besiktas.sideNote'),
      backgroundImage: '/districts/besiktas.jpeg'
    },
    {
      id: 'uskudar',
      name: t('homepage.districts.uskudar.name'),
      color: '#f39c12',
      description: t('homepage.districts.uskudar.description'),
      highlights: t('homepage.districts.uskudar.highlights', { returnObjects: true }) || ['Maiden\'s Tower', 'Çamlıca Hill', 'Historic Mosques'],
      vibe: `🕌 ${t('homepage.districts.uskudar.vibe')}`,
      specialty: t('homepage.districts.uskudar.specialty'),
      population: '530,000',
      sideNote: t('homepage.districts.uskudar.sideNote'),
      backgroundImage: '/districts/uskudar.jpg'
    },
    {
      id: 'sisli',
      name: t('homepage.districts.sisli.name'),
      color: '#1abc9c',
      description: t('homepage.districts.sisli.description'),
      highlights: t('homepage.districts.sisli.highlights', { returnObjects: true }) || ['Cevahir Mall', 'Business Centers', 'Luxury Hotels'],
      vibe: `💼 ${t('homepage.districts.sisli.vibe')}`,
      specialty: t('homepage.districts.sisli.specialty'),
      population: '270,000',
      sideNote: t('homepage.districts.sisli.sideNote'),
      backgroundImage: '/districts/Sisli.jpeg'
    },
    {
      id: 'fatih',
      name: t('homepage.districts.fatih.name'),
      color: '#c0392b',
      description: t('homepage.districts.fatih.description'),
      highlights: t('homepage.districts.fatih.highlights', { returnObjects: true }) || ['Grand Bazaar', 'Süleymaniye Mosque', 'Eminönü'],
      vibe: `🕌 ${t('homepage.districts.fatih.vibe')}`,
      specialty: t('homepage.districts.fatih.specialty'),
      population: '430,000',
      sideNote: t('homepage.districts.fatih.sideNote'),
      backgroundImage: 'https://images.unsplash.com/photo-1541432901042-2d8bd64b4a9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80' // Sultanahmet Square
    },
    {
      id: 'galata',
      name: t('homepage.districts.galata.name'),
      color: '#8e44ad',
      description: t('homepage.districts.galata.description'),
      highlights: t('homepage.districts.galata.highlights', { returnObjects: true }) || ['Galata Tower', 'Taksim Square', 'Karaköy Cafes'],
      vibe: `🗼 ${t('homepage.districts.galata.vibe')}`,
      specialty: t('homepage.districts.galata.specialty'),
      population: '180,000',
      sideNote: t('homepage.districts.galata.sideNote'),
      backgroundImage: 'https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80' // Galata Tower
    },
    {
      id: 'ortakoy',
      name: t('homepage.districts.ortakoy.name'),
      color: '#16a085',
      description: t('homepage.districts.ortakoy.description'),
      highlights: t('homepage.districts.ortakoy.highlights', { returnObjects: true }) || ['Ortaköy Mosque', 'Bosphorus Bridge', 'Kumpir Stalls'],
      vibe: `🌉 ${t('homepage.districts.ortakoy.vibe')}`,
      specialty: t('homepage.districts.ortakoy.specialty'),
      population: '45,000',
      sideNote: t('homepage.districts.ortakoy.sideNote'),
      backgroundImage: '/districts/ortakoy_neighborhood_222.jpg'
    },
    {
      id: 'balat',
      name: t('homepage.districts.balat.name'),
      color: '#d35400',
      description: t('homepage.districts.balat.description'),
      highlights: t('homepage.districts.balat.highlights', { returnObjects: true }) || ['Colorful Houses', 'Vintage Cafes', 'Greek Orthodox Church'],
      vibe: `🎨 ${t('homepage.districts.balat.vibe')}`,
      specialty: t('homepage.districts.balat.specialty'),
      population: '25,000',
      sideNote: t('homepage.districts.balat.sideNote'),
      backgroundImage: '/districts/Balat_houses.jpg'
    }
  ];

  const handleDistrictClick = (district) => {
    // Create a query in the user's selected language using translation
    const query = t('homepage.districtQuery', { districtName: district.name });
    
    console.log('🗺️ District clicked, navigating with query:', query, '(lang:', i18n.language, ')');
    
    // Pass query in navigation state for auto-send
    navigate('/chat', { state: { initialQuery: query } });
  };

  const handleQuickAction = (suggestion) => {
    console.log('⚡ Quick action clicked, navigating with query:', suggestion.query);
    
    // Pass query in navigation state for auto-send
    navigate('/chat', { state: { initialQuery: suggestion.query } });
  };

  return (
    <div className="interactive-main-page">
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">{t('homepage.title')}</h1>
          <p className="hero-subtitle">{t('homepage.subtitle')}</p>
        </div>
      </div>
      
      <div className="time-based-suggestions">
        <div className="current-time">
          <span className="time-display">
            {currentTime.toLocaleTimeString('en-US', { 
              hour: '2-digit', 
              minute: '2-digit',
              timeZone: 'Europe/Istanbul' 
            })}
          </span>
          <span className="time-zone">{t('homepage.istanbulTime')}</span>
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
          <h3>🏙️ {t('homepage.exploreMore')}</h3>
          <p>{t('homepage.discoverNeighborhoods', { count: istanbulDistricts.length })}</p>
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
                    <span className="specialty-label">{t('homepage.labels.specialty')}</span>
                    <span className="specialty-value">{district.specialty}</span>
                  </div>
                  <div className="district-location">
                    <span className="location-value">{district.sideNote}</span>
                  </div>
                  <div className="district-population">
                    <span className="population-label">{t('homepage.labels.population')}</span>
                    <span className="population-value">{district.population}</span>
                  </div>
                </div>
                
                <div className="district-highlights">
                  <span className="highlights-label">{t('homepage.labels.mustExperience')}</span>
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
                  {t('homepage.exploreMore')} {district.name} →
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
