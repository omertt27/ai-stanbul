import React from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

/**
 * RouteCard Component - Displays route information with map visualization
 * Similar to mobile app route cards with functional CTAs
 */
const RouteCard = ({ routeData }) => {
  // 🔍 DEBUG: Log what we're receiving
  console.log('🗺️ RouteCard received routeData:', routeData);
  console.log('🗺️ RouteCard keys:', routeData ? Object.keys(routeData) : 'NO DATA');
  if (routeData) {
    console.log('🗺️ Has map_data:', !!routeData.map_data);
    console.log('🗺️ Has mapData:', !!routeData.mapData);
    console.log('🗺️ Has route_info:', !!routeData.route_info);
    console.log('🗺️ Has routeData:', !!routeData.routeData);
    console.log('🗺️ Has route_data:', !!routeData.route_data);
    console.log('🗺️ Has data:', !!routeData.data);
  }
  
  if (!routeData) return null;

  // Handle both camelCase (frontend) and snake_case (backend) field names
  const map_data = routeData.map_data || routeData.mapData;
  const route_info = routeData.route_info || routeData.routeData || routeData.route_data;
  const message = routeData.message || routeData.text;
  
  // 🔍 DEBUG: Log what we extracted
  console.log('🗺️ Extracted map_data:', map_data ? Object.keys(map_data) : 'NO MAP_DATA');
  console.log('🗺️ Extracted route_info:', route_info ? Object.keys(route_info) : 'NO ROUTE_INFO');

  // If map_data contains route_data, use that for route_info
  const actualRouteInfo = route_info || map_data?.route_data || map_data?.metadata?.route_data;
  
  console.log('🗺️ Final route info:', actualRouteInfo ? Object.keys(actualRouteInfo) : 'NO ROUTE INFO');

  // Extract route information - handle different field name variations
  const origin = actualRouteInfo?.start_location || actualRouteInfo?.origin || 'Starting point';
  const destination = actualRouteInfo?.end_location || actualRouteInfo?.destination || 'Destination';
  const distance = actualRouteInfo?.total_distance 
    ? (actualRouteInfo.total_distance / 1000).toFixed(1) 
    : '0.0';
  const duration = actualRouteInfo?.total_time 
    ? Math.round(actualRouteInfo.total_time / 60) 
    : 0;
  const transfers = actualRouteInfo?.transfer_count || actualRouteInfo?.transfers || 0;
  const confidence = actualRouteInfo?.confidence || 'Medium';
  const lines = actualRouteInfo?.transit_lines || actualRouteInfo?.lines_used || [];
  const steps = actualRouteInfo?.steps || [];

  // Map visualization data
  const hasMapData = map_data && (map_data.routes || map_data.markers);
  const routes = map_data?.routes || [];
  const markers = map_data?.markers || [];
  const center = markers.length > 0 
    ? [markers[0].lat, markers[0].lon] 
    : [41.0082, 28.9784];

  // CTA Handlers
  const handleStartNavigation = () => {
    // Open Google Maps with directions
    const googleMapsUrl = `https://www.google.com/maps/dir/?api=1&origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}&travelmode=transit`;
    window.open(googleMapsUrl, '_blank');
  };

  const handleCopyRoute = async () => {
    const routeText = `
📍 Route: ${origin} → ${destination}
⏱️ Duration: ${duration} min
📏 Distance: ${distance} km
${transfers > 0 ? `🔄 Transfers: ${transfers}\n` : ''}${lines.length > 0 ? `🚇 Lines: ${lines.join(', ')}\n` : ''}
${steps.map((step, idx) => `${idx + 1}. ${step.instruction || step.description}`).join('\n')}
    `.trim();

    try {
      await navigator.clipboard.writeText(routeText);
      // Show feedback (could enhance with toast notification)
      const btn = document.querySelector('.copy-route-btn');
      if (btn) {
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span>✓</span><span class="hidden sm:inline">Copied!</span>';
        setTimeout(() => {
          btn.innerHTML = originalText;
        }, 2000);
      }
    } catch (err) {
      console.error('Failed to copy route:', err);
    }
  };

  const handleShareRoute = async () => {
    const shareData = {
      title: `Route: ${origin} to ${destination}`,
      text: `Transit route from ${origin} to ${destination} - ${duration} min, ${distance} km`,
      url: window.location.href
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        // Fallback: copy link to clipboard
        await navigator.clipboard.writeText(window.location.href);
        const btn = document.querySelector('.share-route-btn');
        if (btn) {
          const originalText = btn.innerHTML;
          btn.innerHTML = '<span>✓</span><span class="hidden sm:inline">Link Copied!</span>';
          setTimeout(() => {
            btn.innerHTML = originalText;
          }, 2000);
        }
      }
    } catch (err) {
      console.error('Failed to share route:', err);
    }
  };

  // Confidence color and tooltip text
  const confidenceInfo = {
    High: {
      color: 'text-green-600 bg-green-50',
      tooltip: 'Based on confirmed schedules and real-time data'
    },
    Medium: {
      color: 'text-yellow-600 bg-yellow-50',
      tooltip: 'Based on typical schedules and average wait times'
    },
    Low: {
      color: 'text-red-600 bg-red-50',
      tooltip: 'Limited data available - times may vary significantly'
    }
  };
  
  const currentConfidence = confidenceInfo[confidence] || confidenceInfo.Medium;

  return (
    <div className="route-card bg-white rounded-lg shadow-md overflow-hidden mb-4">
      {/* Header */}
      <div className="bg-indigo-600 text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold flex items-center">
              <span className="mr-2">📍</span>
              {origin} → {destination}
            </h3>
          </div>
          <div 
            className={`px-3 py-1 rounded-full text-sm font-medium ${currentConfidence.color}`}
            title={currentConfidence.tooltip}
          >
            Confidence: {confidence}
          </div>
        </div>
        
        {/* Route Stats */}
        <div className="flex items-center space-x-6 mt-3 text-sm">
          <div className="flex items-center">
            <span className="mr-1">⏱️</span>
            <span className="font-medium">{duration} min</span>
          </div>
          <div className="flex items-center">
            <span className="mr-1">📏</span>
            <span className="font-medium">{distance} km</span>
          </div>
          {transfers > 0 && (
            <div className="flex items-center">
              <span className="mr-1">🔄</span>
              <span className="font-medium">{transfers} transfer{transfers > 1 ? 's' : ''}</span>
            </div>
          )}
        </div>

        {/* Transit Lines */}
        {lines.length > 0 && (
          <div className="mt-3 flex items-center space-x-2">
            <span className="text-sm">Lines:</span>
            {lines.map((line, idx) => (
              <span
                key={idx}
                className="px-3 py-1 bg-white text-indigo-600 rounded-md text-sm font-bold"
              >
                {line}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Map Visualization */}
      {hasMapData && (
        <div className="map-container" style={{ height: '300px' }}>
          <MapContainer 
            center={center} 
            zoom={13} 
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom={false}
          >
            <TileLayer
              attribution='© OpenStreetMap contributors'
              url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
            />
            
            {/* Route Lines */}
            {routes.map((route, idx) => (
              <Polyline
                key={`route-${idx}`}
                positions={route.coordinates?.map(coord => [coord.lat, coord.lon]) || []}
                color={route.color || '#4F46E5'}
                weight={4}
                opacity={0.7}
              />
            ))}
            
            {/* Markers */}
            {markers.map((marker, idx) => (
              <Marker 
                key={`marker-${idx}`} 
                position={[marker.lat, marker.lon]}
              >
                <Popup>
                  <div>
                    <strong>{marker.label || marker.name}</strong>
                    {marker.type && <div className="text-sm text-gray-600">{marker.type}</div>}
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      )}

      {/* Step-by-Step Directions */}
      {steps.length > 0 && (
        <div className="p-4 border-t">
          <h4 className="font-semibold text-gray-800 mb-3 flex items-center">
            <span className="mr-2">📋</span>
            Step-by-Step Directions:
          </h4>
          <div className="space-y-3">
            {steps.map((step, idx) => (
              <div key={idx} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center font-semibold text-sm">
                  {idx + 1}
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    {getStepIcon(step.mode)}
                    <span className="font-medium text-gray-800">
                      {step.instruction || step.description}
                    </span>
                  </div>
                  {step.details && (
                    <div className="text-sm text-gray-600 mt-1 ml-6">
                      {step.details}
                    </div>
                  )}
                  <div className="flex items-center space-x-3 text-xs text-gray-500 mt-1 ml-6">
                    {step.duration && (
                      <span>~{Math.round(step.duration / 60)} min</span>
                    )}
                    {step.mode === 'transfer' && step.accessibility && (
                      <span className="text-blue-600">♿ {step.accessibility}</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions - Primary CTA with functional handlers */}
      <div className="p-4 bg-gray-50 border-t">
        <div className="flex items-center space-x-3">
          {/* Primary Action - Start Navigation */}
          <button 
            onClick={handleStartNavigation}
            className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors shadow-sm flex items-center justify-center space-x-2"
            title="Open in Google Maps"
          >
            <span>🧭</span>
            <span>Start Navigation</span>
          </button>
          
          {/* Secondary Actions - Copy */}
          <button 
            onClick={handleCopyRoute}
            className="copy-route-btn px-4 py-3 border-2 border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700 font-medium rounded-lg transition-colors flex items-center space-x-1"
            title="Copy route details"
          >
            <span>📋</span>
            <span className="hidden sm:inline">Copy</span>
          </button>
          
          {/* Secondary Actions - Share */}
          <button 
            onClick={handleShareRoute}
            className="share-route-btn px-4 py-3 border-2 border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700 font-medium rounded-lg transition-colors flex items-center space-x-1"
            title="Share route"
          >
            <span>🔗</span>
            <span className="hidden sm:inline">Share</span>
          </button>
        </div>
      </div>
    </div>
  );
};

// Helper function to get appropriate icon for each step mode
function getStepIcon(mode) {
  const icons = {
    walk: '🚶',
    metro: '🚇',
    bus: '🚌',
    tram: '🚋',
    ferry: '⛴️',
    transfer: '🔄',
    funicular: '🚡',
    default: '➡️'
  };
  return <span className="text-lg">{icons[mode?.toLowerCase()] || icons.default}</span>;
}

export default RouteCard;
