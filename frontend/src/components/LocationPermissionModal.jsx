import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import gpsLocationService from '../services/gpsLocationService';

const LocationPermissionModal = ({ isOpen, onClose, onLocationSet }) => {
  const { t } = useTranslation();
  const [manualLocation, setManualLocation] = useState('');
  const [showManualEntry, setShowManualEntry] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [permissionStatus, setPermissionStatus] = useState('unknown');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      checkPermissionStatus();
    }
  }, [isOpen]);

  const checkPermissionStatus = async () => {
    try {
      const status = await gpsLocationService.getLocationPermissionStatus();
      setPermissionStatus(status);
      
      // If already granted, try to get last known position
      if (status === 'granted') {
        const lastKnown = gpsLocationService.getLastKnownPosition();
        if (lastKnown) {
          const neighborhood = await gpsLocationService.getNeighborhoodFromCoordinates(lastKnown);
          onLocationSet({
            latitude: lastKnown.lat,
            longitude: lastKnown.lng,
            source: 'gps',
            name: neighborhood || 'Current Location',
            accuracy: lastKnown.accuracy,
            timestamp: lastKnown.timestamp
          });
          onClose();
        }
      }
    } catch (error) {
      console.error('Error checking permission status:', error);
    }
  };

  const handleUseMyLocation = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const position = await gpsLocationService.requestLocationPermission();
      const neighborhood = await gpsLocationService.getNeighborhoodFromCoordinates(position);
      
      onLocationSet({
        latitude: position.lat,
        longitude: position.lng,
        source: 'gps',
        name: neighborhood || 'Current Location',
        accuracy: position.accuracy,
        timestamp: position.timestamp
      });
      
      onClose();
    } catch (error) {
      console.error('GPS location error:', error);
      setError(error.message);
      setShowManualEntry(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualSubmit = () => {
    if (manualLocation.trim()) {
      onLocationSet({
        address: manualLocation,
        source: 'manual',
        name: manualLocation
      });
      setManualLocation('');
      onClose();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && manualLocation.trim()) {
      handleManualSubmit();
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="location-modal" 
      data-testid="manual-location"
      style={{
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
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div 
        className="location-popup"
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          padding: '24px',
          maxWidth: '400px',
          width: '100%',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          position: 'relative'
        }}
      >
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
          onMouseOver={(e) => {
            e.target.style.backgroundColor = '#f3f4f6';
          }}
          onMouseOut={(e) => {
            e.target.style.backgroundColor = 'transparent';
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
          {t('location.enableServices', 'Enable Location Services')}
        </h3>
        
        <p style={{
          margin: '0 0 24px 0',
          color: '#6b7280',
          fontSize: '14px',
          textAlign: 'center',
          lineHeight: '1.5'
        }}>
          {t('location.description', 'We need your location to provide better recommendations about Istanbul attractions, restaurants, and routes.')}
        </p>
        
        {/* Error Display */}
        {error && (
          <div style={{
            margin: '0 0 16px 0',
            padding: '12px',
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '8px',
            color: '#dc2626',
            fontSize: '14px',
            textAlign: 'center'
          }}>
            {error.includes('permissions policy') 
              ? 'Location access is restricted. Please enable location services in your browser settings or enter your location manually.'
              : error}
          </div>
        )}
        
        {!showManualEntry ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <button
              onClick={handleUseMyLocation}
              data-testid="location-btn"
              disabled={isLoading}
              style={{
                backgroundColor: isLoading ? '#d1d5db' : '#10a37f',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                padding: '12px 20px',
                fontSize: '16px',
                fontWeight: '500',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}
              onMouseOver={(e) => {
                if (!isLoading) {
                  e.target.style.backgroundColor = '#0d8a6b';
                }
              }}
              onMouseOut={(e) => {
                if (!isLoading) {
                  e.target.style.backgroundColor = '#10a37f';
                }
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
                  {t('location.detecting', 'Detecting Location...')}
                </>
              ) : (
                <>
                  <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                  </svg>
                  {t('location.useMyLocation', 'Use My Location')}
                </>
              )}
            </button>
            
            <button
              onClick={() => setShowManualEntry(true)}
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
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#f9fafb';
                e.target.style.borderColor = '#9ca3af';
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = 'transparent';
                e.target.style.borderColor = '#d1d5db';
              }}
            >
              {t('location.enterManually', 'Enter Location Manually')}
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <label style={{
                display: 'block',
                marginBottom: '8px',
                fontSize: '14px',
                fontWeight: '500',
                color: '#374151'
              }}>
                {t('location.enterLocation', 'Enter your location:')}
              </label>
              <input
                type="text"
                value={manualLocation}
                onChange={(e) => setManualLocation(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={t('location.placeholder', 'e.g., Sultanahmet, Istanbul, Turkey')}
                data-testid="location-input"
                autoFocus
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
            
            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={handleManualSubmit}
                data-testid="location-submit"
                disabled={!manualLocation.trim()}
                style={{
                  flex: 1,
                  backgroundColor: manualLocation.trim() ? '#10a37f' : '#d1d5db',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '12px 20px',
                  fontSize: '16px',
                  fontWeight: '500',
                  cursor: manualLocation.trim() ? 'pointer' : 'not-allowed',
                  transition: 'all 0.2s ease'
                }}
                onMouseOver={(e) => {
                  if (manualLocation.trim()) {
                    e.target.style.backgroundColor = '#0d8a6b';
                  }
                }}
                onMouseOut={(e) => {
                  if (manualLocation.trim()) {
                    e.target.style.backgroundColor = '#10a37f';
                  }
                }}
              >
                {t('location.confirm', 'Set Location')}
              </button>
              
              <button
                onClick={() => setShowManualEntry(false)}
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
                onMouseOver={(e) => {
                  e.target.style.backgroundColor = '#f9fafb';
                }}
                onMouseOut={(e) => {
                  e.target.style.backgroundColor = 'transparent';
                }}
              >
                {t('location.back', 'Back')}
              </button>
            </div>
          </div>
        )}
        
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

export default LocationPermissionModal;
