import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const RoutePlanningForm = ({ isOpen, onClose, onRouteRequest, userLocation }) => {
  const { t } = useTranslation();
  const [destination, setDestination] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!destination.trim()) return;

    setIsLoading(true);
    try {
      await onRouteRequest({
        from: userLocation,
        to: destination,
        source: 'manual'
      });
    } catch (error) {
      console.error('Route planning error:', error);
    } finally {
      setIsLoading(false);
      setDestination('');
      onClose();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && destination.trim() && !isLoading) {
      handleSubmit(e);
    }
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      padding: '20px'
    }}>
      <div style={{
        backgroundColor: '#ffffff',
        borderRadius: '16px',
        padding: '24px',
        maxWidth: '400px',
        width: '100%',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        position: 'relative'
      }}>
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'transparent',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            color: '#6b7280',
            padding: '4px',
            borderRadius: '4px'
          }}
        >
          ×
        </button>

        <h3 style={{
          margin: '0 0 16px 0',
          fontSize: '20px',
          fontWeight: '600',
          color: '#111827',
          textAlign: 'center'
        }}>
          {t('route.planRoute', 'Plan Your Route')}
        </h3>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              marginBottom: '8px',
              fontSize: '14px',
              fontWeight: '500',
              color: '#374151'
            }}>
              {t('route.destination', 'Where do you want to go?')}
            </label>
            
            <input
              type="text"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('route.destinationPlaceholder', 'e.g., Galata Tower, Hagia Sophia')}
              data-testid="destination-input"
              autoFocus
              disabled={isLoading}
              style={{
                width: '100%',
                padding: '12px 16px',
                border: '1px solid #d1d5db',
                borderRadius: '8px',
                fontSize: '16px',
                outline: 'none',
                transition: 'border-color 0.2s ease',
                boxSizing: 'border-box'
              }}
              onFocus={(e) => {
                e.target.style.borderColor = '#6366f1';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = '#d1d5db';
              }}
            />
          </div>

          {userLocation && (
            <div style={{
              backgroundColor: '#f9fafb',
              padding: '12px',
              borderRadius: '8px',
              marginBottom: '16px',
              fontSize: '14px',
              color: '#6b7280'
            }}>
              <strong style={{ color: '#374151' }}>
                {t('route.from', 'From:')}
              </strong>{' '}
              {userLocation.name || `${userLocation.latitude?.toFixed(4)}, ${userLocation.longitude?.toFixed(4)}`}
            </div>
          )}

          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              type="submit"
              disabled={!destination.trim() || isLoading}
              data-testid="plan-route-submit"
              style={{
                flex: 1,
                backgroundColor: (destination.trim() && !isLoading) ? '#10a37f' : '#d1d5db',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                padding: '12px 20px',
                fontSize: '16px',
                fontWeight: '500',
                cursor: (destination.trim() && !isLoading) ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
            >
              {isLoading ? (
                <>
                  <div style={{
                    width: '16px',
                    height: '16px',
                    border: '2px solid #ffffff',
                    borderTop: '2px solid transparent',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }} />
                  {t('route.planning', 'Planning...')}
                </>
              ) : (
                <>
                  <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9.78 18.65l.28-4.23 7.68-6.92c.34-.31-.07-.46-.52-.19L8.65 13.7l-4.79-1.51c-1.03-.33-1.05-.84.21-1.21L21.74 5.1c.75-.38 1.48.2 1.21 1.02L18.5 20.28c-.21.78-.81.78-1.2.02l-3.26-5.81-2.26 2.16z"/>
                  </svg>
                  {t('route.getDirections', 'Get Directions')}
                </>
              )}
            </button>
            
            <button
              type="button"
              onClick={onClose}
              style={{
                backgroundColor: 'transparent',
                color: '#6b7280',
                border: '1px solid #d1d5db',
                borderRadius: '8px',
                padding: '12px 20px',
                fontSize: '16px',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {t('common.cancel', 'Cancel')}
            </button>
          </div>
        </form>

        {/* Spinner Animation */}
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    </div>
  );
};

export default RoutePlanningForm;
