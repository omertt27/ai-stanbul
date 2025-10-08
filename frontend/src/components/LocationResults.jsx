import React from 'react';
import { useTranslation } from 'react-i18next';

export const RestaurantCard = ({ restaurant, distance }) => {
  const { t } = useTranslation();
  
  return (
    <div 
      className="restaurant-card"
      style={{
        background: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '12px',
        padding: '16px',
        margin: '8px 0',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(0, 0, 0, 0.05)'
      }}
    >
      <h4 style={{ 
        margin: '0 0 8px 0', 
        color: '#1f2937',
        fontSize: '16px',
        fontWeight: '600'
      }}>
        {restaurant.name}
      </h4>
      
      <p style={{ 
        margin: '0 0 8px 0', 
        color: '#6b7280',
        fontSize: '14px',
        lineHeight: '1.4'
      }}>
        {restaurant.description || restaurant.type}
      </p>
      
      {distance && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          color: '#10a37f',
          fontSize: '12px',
          fontWeight: '500'
        }}>
          <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
          </svg>
          {distance}
        </div>
      )}
      
      {restaurant.rating && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          marginTop: '4px'
        }}>
          {[...Array(5)].map((_, i) => (
            <span 
              key={i}
              style={{
                color: i < Math.floor(restaurant.rating) ? '#fbbf24' : '#d1d5db',
                fontSize: '12px'
              }}
            >
              ★
            </span>
          ))}
          <span style={{ fontSize: '12px', color: '#6b7280', marginLeft: '4px' }}>
            {restaurant.rating}
          </span>
        </div>
      )}
    </div>
  );
};

export const RouteCard = ({ route }) => {
  const { t } = useTranslation();
  
  return (
    <div 
      className="route-card"
      style={{
        background: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '12px',
        padding: '16px',
        margin: '8px 0',
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(0, 0, 0, 0.05)'
      }}
    >
      <div className="route-info">
        <h4 style={{ 
          margin: '0 0 12px 0', 
          color: '#1f2937',
          fontSize: '16px',
          fontWeight: '600'
        }}>
          {t('route.toDestination', 'Route to Destination')}
        </h4>
        
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '12px'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#10a37f',
            fontSize: '14px',
            fontWeight: '500'
          }}>
            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
            {route.distance || '1.2 km'}
          </div>
          
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#6b7280',
            fontSize: '14px'
          }}>
            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
              <path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z"/>
            </svg>
            {route.duration || '15 min walk'}
          </div>
        </div>
        
        {route.steps && (
          <div className="directions" style={{ marginTop: '12px' }}>
            <p style={{ 
              fontSize: '12px', 
              color: '#6b7280',
              marginBottom: '8px',
              fontWeight: '500'
            }}>
              {t('route.directions', 'Directions:')}
            </p>
            {route.steps.slice(0, 3).map((step, index) => (
              <div key={index} style={{
                fontSize: '12px',
                color: '#4b5563',
                marginBottom: '4px',
                paddingLeft: '12px',
                position: 'relative'
              }}>
                <span style={{
                  position: 'absolute',
                  left: '0',
                  color: '#10a37f',
                  fontWeight: '600'
                }}>
                  {index + 1}.
                </span>
                {step}
              </div>
            ))}
            {route.steps && route.steps.length > 3 && (
              <div style={{
                fontSize: '12px',
                color: '#9ca3af',
                fontStyle: 'italic',
                marginTop: '4px'
              }}>
                {t('route.moreSteps', `+${route.steps.length - 3} more steps...`)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export const LocationResultsContainer = ({ results, type, userLocation }) => {
  const { t } = useTranslation();
  
  if (!results || results.length === 0) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '20px',
        color: 'rgba(255, 255, 255, 0.6)',
        fontSize: '14px'
      }}>
        {t('results.noResults', 'No results found nearby.')}
      </div>
    );
  }
  
  return (
    <div style={{
      maxHeight: '400px',
      overflowY: 'auto',
      padding: '0 20px',
      margin: '16px 0'
    }}>
      {type === 'restaurants' && results.map((restaurant, index) => (
        <RestaurantCard 
          key={index} 
          restaurant={restaurant} 
          distance={restaurant.distance || `${(Math.random() * 2 + 0.1).toFixed(1)} km`}
        />
      ))}
      
      {type === 'route' && results.map((route, index) => (
        <RouteCard key={index} route={route} />
      ))}
      
      {type === 'attractions' && results.map((attraction, index) => (
        <RestaurantCard 
          key={index} 
          restaurant={attraction} 
          distance={attraction.distance || `${(Math.random() * 3 + 0.2).toFixed(1)} km`}
        />
      ))}
    </div>
  );
};
