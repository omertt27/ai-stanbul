/**
 * Interactive Route Map Component
 * Phase 3: Leaflet.js Integration for Route Visualization
 */

import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom markers for different types
const createCustomIcon = (type, number = null) => {
  const colors = {
    start: '#22c55e',
    end: '#ef4444', 
    attraction: '#3b82f6',
    food: '#f59e0b',
    hotel: '#8b5cf6'
  };
  
  const iconHtml = number 
    ? `<div style="background-color: ${colors[type]}; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${number}</div>`
    : `<div style="background-color: ${colors[type]}; width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-marker',
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15]
  });
};

const RouteMap = ({ 
  route = null,
  attractions = [],
  center = [41.0082, 28.9784], // Istanbul center
  zoom = 13,
  onAttractionClick = null,
  onRouteSelect = null,
  showControls = true,
  interactive = true,
  className = "",
  style = { height: '400px', width: '100%' }
}) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const routeLineRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mapError, setMapError] = useState(null);

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    try {
      const map = L.map(mapRef.current, {
        center: center,
        zoom: zoom,
        zoomControl: showControls,
        attributionControl: true,
        interactive: interactive
      });

      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(map);

      // Add alternative free tile layer options
      const cartoLayer = L.tileLayer('https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png', {
        attribution: '© <a href="https://carto.com/">CARTO</a> © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        maxZoom: 20
      });

      const esriLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© <a href="https://www.esri.com/">Esri</a>',
        maxZoom: 20
      });

      // Layer control with free alternatives only
      if (showControls) {
        const baseMaps = {
          "OpenStreetMap": map._layers[Object.keys(map._layers)[0]],
          "CartoDB Light": cartoLayer,
          "Esri Streets": esriLayer
        };
        L.control.layers(baseMaps).addTo(map);
      }

      mapInstanceRef.current = map;
      setMapError(null);

    } catch (error) {
      console.error('Error initializing map:', error);
      setMapError('Failed to initialize map. Please refresh the page.');
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Clear all markers and routes
  const clearMap = () => {
    if (!mapInstanceRef.current) return;
    
    // Remove existing markers
    markersRef.current.forEach(marker => {
      mapInstanceRef.current.removeLayer(marker);
    });
    markersRef.current = [];

    // Remove existing route line
    if (routeLineRef.current) {
      mapInstanceRef.current.removeLayer(routeLineRef.current);
      routeLineRef.current = null;
    }
  };

  // Add route to map  
  useEffect(() => {
    if (!mapInstanceRef.current || !route) return;
    
    setIsLoading(true);
    clearMap();

    try {
      const { points } = route;
      if (!points || points.length === 0) return;

      // Calculate bounds for auto-fit
      const bounds = L.latLngBounds(points.map(p => [p.lat, p.lng]));
      
      // Add route line
      const routeCoords = points.map(p => [p.lat, p.lng]);
      routeLineRef.current = L.polyline(routeCoords, {
        color: '#ef4444',
        weight: 4,
        opacity: 0.7,
        dashArray: '5, 10'
      }).addTo(mapInstanceRef.current);

      // Add markers for each point
      points.forEach((point, index) => {
        let markerType = 'attraction';
        if (index === 0) markerType = 'start';
        else if (index === points.length - 1) markerType = 'end';
        else if (point.category?.toLowerCase().includes('food') || 
                 point.category?.toLowerCase().includes('restaurant')) markerType = 'food';

        const marker = L.marker([point.lat, point.lng], {
          icon: createCustomIcon(markerType, index + 1)
        }).addTo(mapInstanceRef.current);

        // Create detailed popup
        const popupContent = `
          <div class="route-popup">
            <h4 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px;">${point.name}</h4>
            <div style="margin-bottom: 4px;"><strong>Category:</strong> ${point.category}</div>
            <div style="margin-bottom: 4px;"><strong>Duration:</strong> ${point.estimated_duration_minutes} min</div>
            <div style="margin-bottom: 4px;"><strong>Score:</strong> ${point.score.toFixed(1)}/10</div>
            ${point.arrival_time 
              ? `<div style="margin-bottom: 4px;"><strong>Arrival:</strong> ${point.arrival_time}</div>` 
              : ''}
            ${point.notes 
              ? `<div style="margin-top: 8px; font-style: italic; color: #6b7280;">${point.notes}</div>` 
              : ''}
            <div style="margin-top: 8px; font-size: 12px; color: #9ca3af;">
              Stop ${index + 1} of ${points.length}
            </div>
          </div>
        `;

        marker.bindPopup(popupContent, { maxWidth: 300 });

        // Handle click events
        if (onAttractionClick && point.attraction_id) {
          marker.on('click', () => {
            onAttractionClick(point.attraction_id, point);
          });
        }

        markersRef.current.push(marker);
      });

      // Fit bounds with padding
      mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });

      // Add route info control
      if (showControls) {
        const routeInfo = L.control({ position: 'topright' });
        routeInfo.onAdd = function() {
          const div = L.DomUtil.create('div', 'route-info-control');
          div.style = `
            background: white; 
            padding: 12px; 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            font-size: 14px;
            max-width: 200px;
          `;
          div.innerHTML = `
            <h5 style="margin: 0 0 8px 0; color: #1f2937;">${route.name}</h5>
            <div><strong>Distance:</strong> ${route.total_distance_km.toFixed(1)} km</div>
            <div><strong>Duration:</strong> ${route.estimated_duration_hours.toFixed(1)}h</div>
            <div><strong>Stops:</strong> ${points.length}</div>
            <div><strong>Score:</strong> ${route.overall_score.toFixed(1)}/10</div>
          `;
          return div;
        };
        routeInfo.addTo(mapInstanceRef.current);
      }

    } catch (error) {
      console.error('Error adding route to map:', error);
      setMapError('Failed to display route on map.');
    } finally {
      setIsLoading(false);
    }
  }, [route]);

  // Add individual attractions (without route)
  useEffect(() => {
    if (!mapInstanceRef.current || !attractions.length || route) return;
    
    clearMap();
    
    try {
      const bounds = L.latLngBounds();
      
      attractions.forEach((attraction, index) => {
        const marker = L.marker([attraction.lat, attraction.lng], {
          icon: createCustomIcon('attraction', index + 1)
        }).addTo(mapInstanceRef.current);

        const popupContent = `
          <div class="attraction-popup">
            <h4 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px;">${attraction.name}</h4>
            <div style="margin-bottom: 4px;"><strong>Category:</strong> ${attraction.category}</div>
            <div style="margin-bottom: 4px;"><strong>District:</strong> ${attraction.district}</div>
            <div style="margin-bottom: 4px;"><strong>Popularity:</strong> ${attraction.popularity_score}/10</div>
            ${attraction.description 
              ? `<div style="margin-top: 8px; color: #6b7280;">${attraction.description}</div>` 
              : ''}
          </div>
        `;

        marker.bindPopup(popupContent, { maxWidth: 300 });

        if (onAttractionClick) {
          marker.on('click', () => {
            onAttractionClick(attraction.id, attraction);
          });
        }

        bounds.extend([attraction.lat, attraction.lng]);
        markersRef.current.push(marker);
      });

      if (bounds.isValid()) {
        mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
      }

    } catch (error) {
      console.error('Error adding attractions to map:', error);
      setMapError('Failed to display attractions on map.');
    }
  }, [attractions]);

  // Loading overlay
  const LoadingOverlay = () => (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(255, 255, 255, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      borderRadius: '8px'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '4px solid #f3f4f6',
          borderTop: '4px solid #3b82f6',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 12px'
        }}></div>
        <div style={{ color: '#6b7280', fontSize: '14px' }}>Loading map...</div>
      </div>
    </div>
  );

  // Error overlay
  const ErrorOverlay = () => (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      borderRadius: '8px'
    }}>
      <div style={{ textAlign: 'center', padding: '20px' }}>
        <div style={{ fontSize: '48px', marginBottom: '12px' }}>🗺️</div>
        <div style={{ color: '#ef4444', fontSize: '16px', marginBottom: '8px' }}>Map Error</div>
        <div style={{ color: '#6b7280', fontSize: '14px' }}>{mapError}</div>
      </div>
    </div>
  );

  return (
    <div className={`route-map-container ${className}`} style={{ position: 'relative', ...style }}>
      <div 
        ref={mapRef} 
        style={{ height: '100%', width: '100%', borderRadius: '8px' }}
      />
      {isLoading && <LoadingOverlay />}
      {mapError && <ErrorOverlay />}
      
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .route-info-control {
          pointer-events: auto;
        }
        
        .custom-marker {
          background: transparent !important;
          border: none !important;
        }
        
        .route-popup h4,
        .attraction-popup h4 {
          border-bottom: 1px solid #e5e7eb;
          padding-bottom: 4px;
        }
      `}</style>
    </div>
  );
};

export default RouteMap;
