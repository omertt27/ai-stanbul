import React, { useState, useContext } from 'react';
import { useTranslation } from 'react-i18next';
import LocationContext from '../contexts/LocationContext';

const LocationBasedButtons = ({ onFeatureSelect }) => {
  const { t } = useTranslation();
  const context = useContext(LocationContext);
  const { currentLocation: userLocation } = context || {};
  const [activeFeature, setActiveFeature] = useState(null);

  const handleRestaurantSearch = () => {
    setActiveFeature('restaurants');
    if (onFeatureSelect) {
      onFeatureSelect('restaurants', userLocation);
    }
  };

  const handleRoutePlanning = () => {
    setActiveFeature('route');
    if (onFeatureSelect) {
      onFeatureSelect('route', userLocation);
    }
  };

  const handleAttractionsSearch = () => {
    setActiveFeature('attractions');
    if (onFeatureSelect) {
      onFeatureSelect('attractions', userLocation);
    }
  };

  const buttonStyle = (isActive) => ({
    background: isActive ? '#10a37f' : 'rgba(255, 255, 255, 0.1)',
    color: isActive ? '#ffffff' : '#e5e7eb',
    border: isActive ? '1px solid #10a37f' : '1px solid rgba(255, 255, 255, 0.2)',
    borderRadius: '12px',
    padding: '12px 20px',
    fontSize: '14px',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    minWidth: '120px',
    justifyContent: 'center',
    backdropFilter: 'blur(10px)',
    boxShadow: isActive ? '0 4px 12px rgba(16, 163, 127, 0.3)' : 'none'
  });

  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: '12px',
      justifyContent: 'center',
      margin: '20px 0',
      padding: '0 20px'
    }}>
      {/* Restaurants Button */}
      <button
        data-testid="restaurants"
        onClick={handleRestaurantSearch}
        style={buttonStyle(activeFeature === 'restaurants')}
        onMouseOver={(e) => {
          if (activeFeature !== 'restaurants') {
            e.target.style.background = 'rgba(255, 255, 255, 0.15)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.3)';
          }
        }}
        onMouseOut={(e) => {
          if (activeFeature !== 'restaurants') {
            e.target.style.background = 'rgba(255, 255, 255, 0.1)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)';
          }
        }}
      >
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8.1 13.34l2.83-2.83L3.91 3.5c-1.56 1.56-1.56 4.09 0 5.66l4.19 4.18zm6.78-1.81c1.53.71 3.68.21 5.27-1.38 1.91-1.91 2.28-4.65.81-6.12-1.46-1.46-4.20-1.10-6.12.81-1.59 1.59-2.09 3.74-1.38 5.27L3.7 19.87l1.41 1.41L12 14.41l6.88 6.88 1.41-1.41-6.88-6.88 1.37-1.37z"/>
        </svg>
        {t('features.restaurants', 'Restaurants')}
      </button>

      {/* Route Planning Button */}
      <button
        data-testid="route-planning"
        onClick={handleRoutePlanning}
        style={buttonStyle(activeFeature === 'route')}
        onMouseOver={(e) => {
          if (activeFeature !== 'route') {
            e.target.style.background = 'rgba(255, 255, 255, 0.15)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.3)';
          }
        }}
        onMouseOut={(e) => {
          if (activeFeature !== 'route') {
            e.target.style.background = 'rgba(255, 255, 255, 0.1)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)';
          }
        }}
      >
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
          <path d="M9.78 18.65l.28-4.23 7.68-6.92c.34-.31-.07-.46-.52-.19L8.65 13.7l-4.79-1.51c-1.03-.33-1.05-.84.21-1.21L21.74 5.1c.75-.38 1.48.2 1.21 1.02L18.5 20.28c-.21.78-.81.78-1.2.02l-3.26-5.81-2.26 2.16z"/>
        </svg>
        {t('features.route', 'Directions')}
      </button>

      {/* Attractions Button */}
      <button
        data-testid="attractions"
        onClick={handleAttractionsSearch}
        style={buttonStyle(activeFeature === 'attractions')}
        onMouseOver={(e) => {
          if (activeFeature !== 'attractions') {
            e.target.style.background = 'rgba(255, 255, 255, 0.15)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.3)';
          }
        }}
        onMouseOut={(e) => {
          if (activeFeature !== 'attractions') {
            e.target.style.background = 'rgba(255, 255, 255, 0.1)';
            e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)';
          }
        }}
      >
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
        </svg>
        {t('features.attractions', 'Attractions')}
      </button>

      {/* Location Status */}
      {userLocation && (
        <div style={{
          width: '100%',
          textAlign: 'center',
          fontSize: '12px',
          color: 'rgba(255, 255, 255, 0.6)',
          marginTop: '8px'
        }}>
          📍 {userLocation.name || `${userLocation.latitude?.toFixed(4)}, ${userLocation.longitude?.toFixed(4)}`}
        </div>
      )}
    </div>
  );
};

export default LocationBasedButtons;
