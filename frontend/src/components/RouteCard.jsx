import React, { useState, useRef, useEffect, lazy, Suspense } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup, useMap } from 'react-leaflet';
import { useTranslation } from 'react-i18next';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Lazy load the map component for better performance
const LazyRouteMap = lazy(() => import('./LazyRouteMap'));

// Import mobile components
import RouteBottomSheet from './RouteBottomSheet';
import SwipeableStepNavigation from './SwipeableStepNavigation';
import MultiRouteComparison from './MultiRouteComparison';

// Fix default markers for React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

/**
 * MapLoadingSkeleton - Loading state for map
 * @param {Object} props
 * @param {Function} props.t - Translation function (required)
 */
const MapLoadingSkeleton = ({ t }) => {
  // t is required - if not passed, this will fail fast to catch bugs early
  const loadingText = t('routeCard.loadingMap');
  
  return (
    <div className="absolute inset-0 z-[999] bg-gray-100 animate-pulse flex items-center justify-center">
      <div className="text-center">
        <div className="inline-block w-12 h-12 border-4 border-gray-300 border-t-indigo-600 rounded-full animate-spin"></div>
        <div className="mt-3 text-gray-500 font-medium">{loadingText}</div>
      </div>
    </div>
  );
};

/**
 * AnimatedPolyline - Animated route line drawing
 */
const AnimatedPolyline = ({ positions, color, weight = 5, opacity = 0.8, speed = 50 }) => {
  const [visiblePoints, setVisiblePoints] = useState(1);
  
  useEffect(() => {
    if (positions.length <= 1) {
      setVisiblePoints(positions.length);
      return;
    }
    
    const interval = setInterval(() => {
      setVisiblePoints(prev => {
        if (prev >= positions.length) {
          clearInterval(interval);
          return positions.length;
        }
        return prev + 1;
      });
    }, speed);
    
    return () => clearInterval(interval);
  }, [positions.length, speed]);
  
  if (positions.length === 0) return null;
  
  return (
    <Polyline 
      positions={positions.slice(0, visiblePoints)}
      color={color}
      weight={weight}
      opacity={opacity}
    />
  );
};

/**
 * PulsingMarker - Pulsing animation for transfer points
 */
const PulsingMarker = ({ position, label, isPulse = false }) => {
  const pulseIcon = L.divIcon({
    className: 'pulsing-marker',
    html: `
      <div style="position: relative;">
        <div style="
          width: 20px; 
          height: 20px; 
          background: #4F46E5; 
          border-radius: 50%; 
          border: 3px solid white; 
          box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
          ${isPulse ? 'animation: pulse 2s ease-in-out infinite;' : ''}
        "></div>
        ${isPulse ? `
          <div style="
            position: absolute;
            top: 0;
            left: 0;
            width: 20px;
            height: 20px;
            background: #4F46E5;
            border-radius: 50%;
            opacity: 0;
            animation: pulse-ring 2s ease-in-out infinite;
          "></div>
        ` : ''}
      </div>
      <style>
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 0.5; }
          100% { transform: scale(2.5); opacity: 0; }
        }
      </style>
    `,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });
  
  return (
    <Marker position={position} icon={pulseIcon}>
      {label && <Popup><strong>{label}</strong></Popup>}
    </Marker>
  );
};

/**
 * MapControls Component - Interactive controls for map
 */
const MapControls = ({ center, onGeolocationFound }) => {
  const map = useMap();
  const [userLocation, setUserLocation] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleZoomIn = () => {
    map.zoomIn();
  };

  const handleZoomOut = () => {
    map.zoomOut();
  };

  const handleRecenter = () => {
    map.setView(center, 13, { animate: true });
  };

  const handleGeolocation = () => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          const userPos = [latitude, longitude];
          setUserLocation(userPos);
          map.setView(userPos, 15, { animate: true });
          
          // Add user location marker
          if (onGeolocationFound) {
            onGeolocationFound(userPos);
          }
        },
        (error) => {
          console.error('Geolocation error:', error);
          alert('Unable to get your location. Please enable location services.');
        }
      );
    } else {
      alert('Geolocation is not supported by your browser.');
    }
  };

  const handleFullscreen = () => {
    const container = map.getContainer();
    
    if (!isFullscreen) {
      // Enter fullscreen
      if (container.requestFullscreen) {
        container.requestFullscreen();
      } else if (container.webkitRequestFullscreen) {
        container.webkitRequestFullscreen();
      } else if (container.msRequestFullscreen) {
        container.msRequestFullscreen();
      }
      setIsFullscreen(true);
    } else {
      // Exit fullscreen
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      }
      setIsFullscreen(false);
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
    };
  }, []);

  return (
    <div className="absolute top-2 right-2 z-[1000] flex flex-col gap-2">
      <button
        onClick={handleZoomIn}
        className="bg-white hover:bg-gray-100 p-2 rounded shadow-md transition-colors w-8 h-8 flex items-center justify-center"
        title="Zoom in"
        aria-label="Zoom in"
      >
        <span className="text-lg font-bold">+</span>
      </button>
      <button
        onClick={handleZoomOut}
        className="bg-white hover:bg-gray-100 p-2 rounded shadow-md transition-colors w-8 h-8 flex items-center justify-center"
        title="Zoom out"
        aria-label="Zoom out"
      >
        <span className="text-lg font-bold">−</span>
      </button>
      <button
        onClick={handleRecenter}
        className="bg-white hover:bg-gray-100 p-2 rounded shadow-md transition-colors w-8 h-8 flex items-center justify-center"
        title="Recenter map"
        aria-label="Recenter map"
      >
        <span className="text-base">🎯</span>
      </button>
      <button
        onClick={handleGeolocation}
        className="bg-white hover:bg-gray-100 p-2 rounded shadow-md transition-colors w-8 h-8 flex items-center justify-center"
        title="Show my location"
        aria-label="Show my location"
      >
        <span className="text-base">📍</span>
      </button>
      <button
        onClick={handleFullscreen}
        className="bg-white hover:bg-gray-100 p-2 rounded shadow-md transition-colors w-8 h-8 flex items-center justify-center"
        title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
      >
        <span className="text-base">{isFullscreen ? '⊗' : '⛶'}</span>
      </button>
    </div>
  );
};

/**
 * StepByStepNavigation - Navigation mode with current step highlighted
 */
const StepByStepNavigation = ({ steps, onClose }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  
  const currentStep = steps[currentStepIndex];
  const totalSteps = steps.length;
  const progress = ((currentStepIndex + 1) / totalSteps) * 100;
  
  // Voice instruction
  useEffect(() => {
    if (isVoiceEnabled && currentStep && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(
        currentStep.instruction || currentStep.description
      );
      utterance.lang = 'en-US';
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  }, [currentStepIndex, isVoiceEnabled, currentStep]);
  
  const handleNext = () => {
    if (currentStepIndex < totalSteps - 1) {
      setCurrentStepIndex(prev => prev + 1);
    }
  };
  
  const handlePrevious = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex(prev => prev - 1);
    }
  };
  
  const toggleVoice = () => {
    setIsVoiceEnabled(prev => !prev);
    if (isVoiceEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  };
  
  if (!currentStep) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-[2000] flex items-end sm:items-center justify-center p-4">
      <div className="bg-white rounded-t-3xl sm:rounded-2xl w-full sm:max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="bg-indigo-600 text-white p-4 rounded-t-3xl sm:rounded-t-2xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-bold">{t('routeCard.navigationMode')}</h3>
            <button 
              onClick={onClose}
              className="w-8 h-8 bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full flex items-center justify-center transition-colors"
              aria-label="Close navigation"
            >
              <span className="text-xl">×</span>
            </button>
          </div>
          
          {/* Progress Bar */}
          <div className="mb-2">
            <div className="flex items-center justify-between text-sm mb-1">
              <span>{t('routeCard.stepOf', { current: currentStepIndex + 1, total: totalSteps })}</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-white bg-opacity-20 rounded-full h-2">
              <div 
                className="bg-white h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        </div>
        
        {/* Current Step - Large Display */}
        <div className="p-6 bg-gradient-to-b from-indigo-50 to-white">
          <div className="flex items-start space-x-4">
            <div className="flex-shrink-0 w-16 h-16 bg-indigo-600 text-white rounded-full flex items-center justify-center text-2xl font-bold shadow-lg">
              {currentStepIndex + 1}
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-3xl">{getStepIcon(currentStep.mode)}</span>
                {currentStep.mode && (
                  <span className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                    {currentStep.mode}
                  </span>
                )}
              </div>
              <p className="text-xl font-bold text-gray-900 leading-relaxed">
                {currentStep.instruction || currentStep.description}
              </p>
              {currentStep.details && (
                <p className="text-base text-gray-600 mt-2">
                  {currentStep.details}
                </p>
              )}
              <div className="flex items-center space-x-4 mt-3 text-sm text-gray-500">
                {currentStep.duration && (
                  <span className="flex items-center">
                    <span className="mr-1">⏱️</span>
                    ~{Math.round(currentStep.duration / 60)} min
                  </span>
                )}
                {currentStep.mode === 'transfer' && currentStep.accessibility && (
                  <span className="flex items-center text-blue-600">
                    <span className="mr-1">♿</span>
                    {currentStep.accessibility}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
        
        {/* Navigation Controls */}
        <div className="p-4 border-t bg-gray-50">
          <div className="flex items-center space-x-3 mb-3">
            <button
              onClick={handlePrevious}
              disabled={currentStepIndex === 0}
              className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                currentStepIndex === 0
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-white border-2 border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700'
              }`}
            >
              ← Previous
            </button>
            <button
              onClick={handleNext}
              disabled={currentStepIndex === totalSteps - 1}
              className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                currentStepIndex === totalSteps - 1
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-md'
              }`}
            >
              {currentStepIndex === totalSteps - 1 ? '✓ Complete' : 'Next →'}
            </button>
          </div>
          
          {/* Voice Toggle */}
          {'speechSynthesis' in window && (
            <button
              onClick={toggleVoice}
              className={`w-full py-2 px-4 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${
                isVoiceEnabled
                  ? 'bg-indigo-100 border-2 border-indigo-400 text-indigo-700'
                  : 'bg-white border-2 border-gray-300 hover:border-indigo-400 text-gray-700'
              }`}
            >
              <span>{isVoiceEnabled ? '🔊' : '🔇'}</span>
              <span>{isVoiceEnabled ? 'Voice On' : 'Voice Off'}</span>
            </button>
          )}
        </div>
        
        {/* All Steps Preview */}
        <div className="p-4 border-t max-h-48 overflow-y-auto">
          <h4 className="text-sm font-semibold text-gray-500 uppercase mb-2">{t('routeCard.allSteps')}</h4>
          <div className="space-y-1">
            {steps.map((step, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentStepIndex(idx)}
                className={`w-full text-left p-2 rounded-lg transition-all ${
                  idx === currentStepIndex
                    ? 'bg-indigo-100 border-2 border-indigo-400'
                    : idx < currentStepIndex
                    ? 'bg-green-50 border border-green-200 opacity-60'
                    : 'bg-gray-50 border border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span className={`text-xs font-bold ${
                    idx === currentStepIndex ? 'text-indigo-600' : 'text-gray-500'
                  }`}>
                    {idx + 1}
                  </span>
                  <span className="text-sm">{getStepIcon(step.mode)}</span>
                  <span className={`text-sm flex-1 ${
                    idx === currentStepIndex ? 'font-semibold' : ''
                  }`}>
                    {step.instruction || step.description}
                  </span>
                  {idx < currentStepIndex && <span className="text-green-600">✓</span>}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * RouteCard Component - Displays route information with map visualization
 * Similar to mobile app route cards with functional CTAs
 */
const RouteCard = ({ routeData }) => {
  // i18n translation hook
  const { t } = useTranslation();
  
  // State for share button feedback
  const [shareStatus, setShareStatus] = useState({ state: 'idle', message: '' }); // idle, success, error
  
  // State for save route feature
  const [isSaved, setIsSaved] = useState(false);
  
  // State for map loading
  const [isMapLoading, setIsMapLoading] = useState(true);
  
  // State for dark mode (can be passed from parent or use system preference)
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // State for user location marker
  const [userLocation, setUserLocation] = useState(null);
  
  // State for step-by-step navigation mode
  const [isNavigationMode, setIsNavigationMode] = useState(false);
  
  // State for animated routes
  const [enableAnimations, setEnableAnimations] = useState(true);
  
  // State for lazy loading maps (default true for better performance)
  const [useLazyMap, setUseLazyMap] = useState(true);
  
  // Mobile-specific states
  const [isMobile, setIsMobile] = useState(false);
  const [showMobileBottomSheet, setShowMobileBottomSheet] = useState(true);
  const [useMobileNavigation, setUseMobileNavigation] = useState(false);
  
  // Multi-route comparison states
  const [showAlternativeRoutes, setShowAlternativeRoutes] = useState(false);
  const [selectedAlternativeIndex, setSelectedAlternativeIndex] = useState(0);
  const [alternativeRoutes, setAlternativeRoutes] = useState([]);
  
  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768; // md breakpoint
      setIsMobile(mobile);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);
  
  // Check system dark mode preference
  useEffect(() => {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(isDark);
    
    // Listen for changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => setIsDarkMode(e.matches);
    mediaQuery.addEventListener('change', handleChange);
    
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);
  
  // Check if route is already saved
  // Early return if no data
  if (!routeData) return null;

  // Handle both camelCase (frontend) and snake_case (backend) field names
  // Use useMemo to prevent unnecessary recalculations
  const map_data = routeData.map_data || routeData.mapData;
  const route_info = routeData.route_info || routeData.routeData || routeData.route_data;
  const message = routeData.message || routeData.text;

  // If map_data contains route_data, use that for route_info
  const actualRouteInfo = route_info || map_data?.route_data || map_data?.metadata?.route_data;

  // Extract route information - handle different field name variations
  const origin = actualRouteInfo?.start_location || actualRouteInfo?.origin || 'Starting point';
  const destination = actualRouteInfo?.end_location || actualRouteInfo?.destination || 'Destination';
  
  // Handle both time formats: seconds (total_time) and minutes (duration_min)
  let duration;
  if (actualRouteInfo?.duration_min) {
    duration = actualRouteInfo.duration_min; // Already in minutes
  } else if (actualRouteInfo?.total_time) {
    duration = Math.round(actualRouteInfo.total_time / 60); // Convert seconds to minutes
  } else {
    duration = 0;
  }
  
  // Handle both distance formats: meters (total_distance) and km (distance_km)
  let distance;
  if (actualRouteInfo?.distance_km) {
    distance = actualRouteInfo.distance_km.toFixed(1); // Already in km
  } else if (actualRouteInfo?.total_distance) {
    distance = (actualRouteInfo.total_distance / 1000).toFixed(1); // Convert meters to km
  } else {
    distance = '0.0';
  }
  const transfers = actualRouteInfo?.transfer_count || actualRouteInfo?.transfers || 0;
  const confidence = actualRouteInfo?.confidence || 'Medium';
  const lines = actualRouteInfo?.transit_lines || actualRouteInfo?.lines_used || [];
  const steps = actualRouteInfo?.steps || [];

  // Check if route is already saved (moved after early return)
  useEffect(() => {
    const savedRoutes = JSON.parse(localStorage.getItem('savedRoutes') || '[]');
    const routeKey = `${origin}-${destination}`;
    const isAlreadySaved = savedRoutes.some(r => `${r.origin}-${r.destination}` === routeKey);
    setIsSaved(isAlreadySaved);
  }, [origin, destination]);

  // Transform route data for MultiRouteComparison component
  const transformRouteForComparison = (route, routeInfo, index = 0) => {
    const durationMin = route.duration_minutes || route.duration ? Math.round((route.duration || route.duration_minutes * 60) / 60) : duration;
    const distanceMeters = route.walking_meters || route.distance || parseFloat(distance) * 1000;
    const numTransfers = route.num_transfers || route.transfers || transfers;
    const walkingMeters = route.walking_meters || distanceMeters;
    
    return {
      origin: route.origin || origin,
      destination: route.destination || destination,
      duration_minutes: durationMin,
      distance: distanceMeters,
      num_transfers: numTransfers,
      walking_meters: walkingMeters,
      transit_lines: route.transit_lines || route.lines || lines,
      steps: route.steps || steps,
      
      // Comfort scoring (mock for now, should come from backend)
      comfort_score: route.comfort_score || {
        overall_comfort: Math.max(30, 100 - (numTransfers * 15) - (walkingMeters / 50)),
        crowding_comfort: 70,
        transfer_comfort: Math.max(20, 100 - (numTransfers * 25)),
        walking_comfort: Math.max(20, 100 - (walkingMeters / 30)),
        waiting_comfort: 75
      },
      
      // Route preference determination
      preference: route.preference || (index === 0 ? 'fastest' : index === 1 ? 'fewest-transfers' : 'balanced'),
      
      // Highlights
      highlights: route.highlights || [],
      
      // LLM summary
      llm_summary: route.llm_summary || `${durationMin} min journey with ${numTransfers} transfer${numTransfers !== 1 ? 's' : ''}`,
      
      // Overall score (weighted)
      overall_score: route.overall_score || Math.round(
        (100 - durationMin) * 0.4 + // Time weight
        (100 - numTransfers * 20) * 0.3 + // Transfer weight
        (100 - walkingMeters / 50) * 0.3 // Walking weight
      ),
      
      // Keep original data
      route_info: routeInfo,
      map_data: route.map_data || map_data
    };
  };

  // Generate route comparison summary
  const generateRouteComparison = (routes) => {
    if (!routes || routes.length === 0) return {};
    
    let fastestIdx = 0, fewestTransfersIdx = 0, leastWalkingIdx = 0, mostComfortableIdx = 0;
    
    routes.forEach((route, idx) => {
      if (route.duration_minutes < routes[fastestIdx].duration_minutes) fastestIdx = idx;
      if (route.num_transfers < routes[fewestTransfersIdx].num_transfers) fewestTransfersIdx = idx;
      if (route.walking_meters < routes[leastWalkingIdx].walking_meters) leastWalkingIdx = idx;
      if ((route.comfort_score?.overall_comfort || 0) > (routes[mostComfortableIdx].comfort_score?.overall_comfort || 0)) {
        mostComfortableIdx = idx;
      }
    });
    
    return {
      fastest_route: fastestIdx,
      fewest_transfers: fewestTransfersIdx,
      least_walking: leastWalkingIdx,
      most_comfortable: mostComfortableIdx
    };
  };

  // Extract alternative routes if available
  // Use JSON.stringify for stable dependency on array contents
  const linesKey = JSON.stringify(lines);
  const stepsKey = JSON.stringify(steps.map(s => s.instruction || s.description || ''));
  const routeDataId = routeData?.id || routeData?.timestamp || JSON.stringify({origin, destination, duration, distance});
  
  useEffect(() => {
    const alternatives = routeData?.alternatives || routeData?.alternative_routes || [];
    
    // Parse lines and steps from stringified keys
    const currentLines = JSON.parse(linesKey);
    const currentSteps = steps;
    
    // Transform current route
    const currentRoute = transformRouteForComparison({
      duration_minutes: duration,
      distance: parseFloat(distance) * 1000,
      num_transfers: transfers,
      walking_meters: parseFloat(distance) * 1000 * 0.3, // Estimate 30% walking
      transit_lines: currentLines,
      steps: currentSteps
    }, actualRouteInfo, 0);
    
    if (alternatives.length > 0) {
      // Transform alternative routes
      const transformedAlternatives = alternatives.map((alt, idx) => 
        transformRouteForComparison(alt, alt.route_info, idx + 1)
      );
      setAlternativeRoutes([currentRoute, ...transformedAlternatives]);
    } else {
      setAlternativeRoutes([currentRoute]);
    }
  }, [routeDataId, duration, distance, transfers, linesKey, stepsKey]);

  // Map visualization data
  const hasMapData = map_data && (map_data.routes || map_data.markers);
  const routes = map_data?.routes || [];
  const markers = map_data?.markers || [];
  
  // Calculate center - handle both 'lon' and 'lng' field names
  const firstMarker = markers.length > 0 ? markers[0] : null;
  const center = firstMarker
    ? [
        firstMarker.lat || firstMarker.latitude, 
        firstMarker.lon || firstMarker.lng || firstMarker.longitude
      ]
    : [41.0082, 28.9784]; // Default to Istanbul center

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
    const shareText = `Transit route from ${origin} to ${destination}\n⏱️ ${duration} min • 📍 ${distance} km\n${transfers > 0 ? `🔄 ${transfers} transfer${transfers > 1 ? 's' : ''}\n` : ''}🚇 Lines: ${linesUsed.join(', ')}\n\nView route: ${window.location.href}`;
    
    const shareData = {
      title: `Route: ${origin} to ${destination}`,
      text: shareText,
      url: window.location.href
    };

    try {
      // Check if Web Share API is available (mobile devices)
      if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
        await navigator.share(shareData);
        setShareStatus({ state: 'success', message: 'Shared!' });
      } else {
        // Fallback: copy to clipboard with enhanced formatting
        const textToCopy = `${shareData.title}\n\n${shareText}`;
        await navigator.clipboard.writeText(textToCopy);
        setShareStatus({ state: 'success', message: 'Copied!' });
      }
      
      // Reset status after 2 seconds
      setTimeout(() => {
        setShareStatus({ state: 'idle', message: '' });
      }, 2000);
      
    } catch (err) {
      console.error('❌ Failed to share route:', err);
      setShareStatus({ state: 'error', message: 'Failed' });
      
      // Try simple clipboard as last resort
      try {
        await navigator.clipboard.writeText(window.location.href);
        setShareStatus({ state: 'success', message: 'URL Copied!' });
      } catch (clipboardErr) {
        // Even clipboard fallback failed - user will see error state
      }
      
      // Reset status after 2 seconds
      setTimeout(() => {
        setShareStatus({ state: 'idle', message: '' });
      }, 2000);
    }
  };

  const handleSaveRoute = async () => {
    const routeToSave = {
      id: Date.now(),
      origin,
      destination,
      duration,
      distance,
      transfers,
      lines,
      steps,
      map_data,
      savedAt: new Date().toISOString()
    };
    
    try {
      const savedRoutes = JSON.parse(localStorage.getItem('savedRoutes') || '[]');
      
      if (isSaved) {
        // Remove from saved routes
        const routeKey = `${origin}-${destination}`;
        const filteredRoutes = savedRoutes.filter(
          r => `${r.origin}-${r.destination}` !== routeKey
        );
        localStorage.setItem('savedRoutes', JSON.stringify(filteredRoutes));
        setIsSaved(false);
      } else {
        // Add to saved routes
        savedRoutes.push(routeToSave);
        localStorage.setItem('savedRoutes', JSON.stringify(savedRoutes));
        setIsSaved(true);
        
        // Optional: Send to backend for cloud sync
        // await fetch('/api/routes/save', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify(routeToSave)
        // });
      }
    } catch (err) {
      console.error('❌ Failed to save route:', err);
      alert('Failed to save route. Please try again.');
    }
  };

  const handleGeolocationFound = (position) => {
    setUserLocation(position);
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

  // Mobile View - Full-screen map with bottom sheet
  if (isMobile && showMobileBottomSheet) {
    return (
      <>
        {/* Mobile: Full-screen map view */}
        <div className="fixed inset-0 z-40 bg-white">
          {/* Full-Screen Map */}
          <div className="absolute inset-0">
            {hasMapData && (
              <div className="map-container relative h-full w-full">
                {isMapLoading && <MapLoadingSkeleton t={t} />}
                <MapContainer 
                  center={center} 
                  zoom={13} 
                  style={{ height: '100%', width: '100%' }}
                  scrollWheelZoom={true}
                  whenReady={() => {
                    setTimeout(() => setIsMapLoading(false), 300);
                  }}
                >
                  <TileLayer
                    attribution='© CARTO © OpenStreetMap contributors'
                    url='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
                    maxZoom={19}
                    keepBuffer={4}
                    updateWhenIdle={true}
                    updateWhenZooming={false}
                  />
                  
                  {/* Interactive Map Controls */}
                  <MapControls center={center} onGeolocationFound={handleGeolocationFound} />
                  
                  {/* Route Lines */}
                  {routes.map((route, idx) => {
                    const positions = route.coordinates?.map(coord => [
                      coord.lat, 
                      coord.lon || coord.lng
                    ]).filter(pos => pos[0] !== undefined && pos[1] !== undefined) || [];
                    
                    if (positions.length === 0) return null;
                    
                    const getLineStyle = (mode) => {
                      const styles = {
                        walk: { weight: 3, color: '#9CA3AF', opacity: 0.6 },
                        metro: { weight: 5, color: '#DC2626', opacity: 0.8 },
                        bus: { weight: 4, color: '#2563EB', opacity: 0.8 },
                        tram: { weight: 4, color: '#16A34A', opacity: 0.8 },
                        ferry: { weight: 4, color: '#0891B2', opacity: 0.8 },
                        default: { weight: 4, color: '#4F46E5', opacity: 0.7 }
                      };
                      return styles[mode?.toLowerCase()] || styles.default;
                    };
                    
                    const lineStyle = getLineStyle(route.mode);
                    
                    return (
                      <AnimatedPolyline
                        key={`route-${idx}`}
                        positions={positions}
                        color={route.color || lineStyle.color}
                        weight={lineStyle.weight}
                        opacity={lineStyle.opacity}
                        speed={enableAnimations ? 30 : 0}
                      />
                    );
                  })}
                  
                  {/* Transfer Point Markers */}
                  {steps.filter(step => step.mode === 'transfer').map((step, idx) => {
                    if (!step.lat || !step.lon) return null;
                    return (
                      <PulsingMarker
                        key={`transfer-${idx}`}
                        position={[step.lat, step.lon]}
                        label={`Transfer: ${step.instruction || step.description}`}
                        isPulse={true}
                      />
                    );
                  })}
                  
                  {/* Markers */}
                  {markers.map((marker, idx) => {
                    const markerLat = marker.lat || marker.latitude;
                    const markerLon = marker.lon || marker.lng || marker.longitude;
                    
                    if (!markerLat || !markerLon) return null;
                    
                    return (
                      <Marker 
                        key={`marker-${idx}`} 
                        position={[markerLat, markerLon]}
                      >
                        <Popup>
                          <div>
                            <strong>{marker.label || marker.name}</strong>
                            {marker.type && <div className="text-sm text-gray-600">{marker.type}</div>}
                          </div>
                        </Popup>
                      </Marker>
                    );
                  })}
                  
                  {/* User Location Marker */}
                  {userLocation && (
                    <Marker 
                      position={userLocation}
                      icon={L.divIcon({
                        className: 'user-location-marker',
                        html: '<div style="background: #4F46E5; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.3);"></div>',
                        iconSize: [16, 16],
                        iconAnchor: [8, 8]
                      })}
                    >
                      <Popup>
                        <div>
                          <strong>{t('routeCard.yourLocation')}</strong>
                        </div>
                      </Popup>
                    </Marker>
                  )}
                </MapContainer>
              </div>
            )}
          </div>

          {/* Close Button */}
          <button
            onClick={() => setShowMobileBottomSheet(false)}
            className="absolute top-4 left-4 z-50 bg-white rounded-full shadow-lg p-3 hover:bg-gray-100 transition-colors"
            aria-label="Close mobile view"
          >
            <span className="text-xl">×</span>
          </button>

          {/* Bottom Sheet with Route Details */}
          {!useMobileNavigation && (
            <RouteBottomSheet
              isOpen={true}
              onClose={() => setShowMobileBottomSheet(false)}
              snapPoints={[0.25, 0.5, 0.85]}
              initialSnapPoint={0.5}
            >
              {/* Route Header */}
              <div className="mb-4">
                <h2 className="text-xl font-bold text-gray-900 mb-2">
                  {origin} → {destination}
                </h2>
                
                {/* Route Stats */}
                <div className="flex items-center flex-wrap gap-2 text-sm text-gray-600">
                  <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                    <span className="mr-1">⏱️</span>
                    <span className="font-semibold">{duration} min</span>
                  </div>
                  <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                    <span className="mr-1">📏</span>
                    <span className="font-semibold">{distance} km</span>
                  </div>
                  {transfers > 0 && (
                    <div className="flex items-center bg-indigo-50 px-3 py-1.5 rounded-lg">
                      <span className="mr-1">🔄</span>
                      <span className="font-semibold">{transfers}</span>
                    </div>
                  )}
                </div>

                {/* Transit Lines */}
                {lines.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {lines.map((line, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-sm font-bold"
                      >
                        {line}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Start Navigation Button */}
              {steps.length > 0 && (
                <button
                  onClick={() => setUseMobileNavigation(true)}
                  className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-4 px-6 rounded-xl transition-colors shadow-lg flex items-center justify-center space-x-2 mb-4"
                >
                  <span className="text-2xl">🧭</span>
                  <span className="text-lg">{t('routeCard.startNavigation')}</span>
                </button>
              )}

              {/* Compact Steps List */}
              {steps.length > 0 && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-gray-700 mb-3 flex items-center">
                    <span className="mr-2">📋</span>
                    {t('routeCard.allSteps')}
                  </h3>
                  {steps.map((step, idx) => (
                    <div 
                      key={idx} 
                      className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex-shrink-0 w-7 h-7 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center font-bold text-sm">
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-lg">{getStepIcon(step.mode)}</span>
                          <span className="font-medium text-gray-800 text-sm">
                            {step.instruction || step.description}
                          </span>
                        </div>
                        {step.duration && (
                          <span className="text-xs text-gray-500">
                            ~{Math.round(step.duration / 60)} min
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </RouteBottomSheet>
          )}

          {/* Mobile Navigation Mode - Swipeable */}
          {useMobileNavigation && steps.length > 0 && (
            <div className="absolute inset-0 bg-white z-50 flex flex-col">
              {/* Header */}
              <div className="bg-indigo-600 text-white p-4 flex items-center justify-between">
                <h2 className="text-lg font-bold">Navigation</h2>
                <button
                  onClick={() => setUseMobileNavigation(false)}
                  className="bg-white bg-opacity-20 hover:bg-opacity-30 rounded-full p-2 transition-colors"
                  aria-label="Exit navigation"
                >
                  <span className="text-xl">×</span>
                </button>
              </div>

              {/* Swipeable Navigation */}
              <SwipeableStepNavigation
                steps={steps}
                onStepChange={() => {}}
              />
            </div>
          )}
        </div>
      </>
    );
  }

  // Desktop View - Original card layout
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

      {/* Alternative Routes Section */}
      {alternativeRoutes.length > 1 && !showAlternativeRoutes && (
        <div className={`p-4 border-b ${
          isDarkMode 
            ? 'bg-gradient-to-r from-indigo-900 to-purple-900 border-indigo-800' 
            : 'bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-100'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🔍</span>
              <div>
                <div className={`font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                  {alternativeRoutes.length} Route Options Available
                </div>
                <div className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  Compare routes and choose the best option for you
                </div>
              </div>
            </div>
            <button
              onClick={() => setShowAlternativeRoutes(true)}
              className={`px-6 py-2 rounded-lg font-semibold transition-colors shadow-md flex items-center space-x-2 ${
                isDarkMode
                  ? 'bg-indigo-600 hover:bg-indigo-500 text-white'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              <span>{t('routeCard.compareRoutes')}</span>
              <span>→</span>
            </button>
          </div>
        </div>
      )}

      {/* Multi-Route Comparison View */}
      {showAlternativeRoutes && alternativeRoutes.length > 1 && (
        <div className={`p-6 border-b ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
          <div className="flex items-center justify-between mb-4">
            <h3 className={`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              Route Comparison
            </h3>
            <button
              onClick={() => setShowAlternativeRoutes(false)}
              className={`transition-colors ${
                isDarkMode 
                  ? 'text-gray-400 hover:text-white' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              aria-label="Close comparison"
            >
              <span className="text-xl">×</span>
            </button>
          </div>
          <MultiRouteComparison
            routes={alternativeRoutes}
            routeComparison={generateRouteComparison(alternativeRoutes)}
            onRouteSelect={(route, index) => {
              setSelectedAlternativeIndex(index);
            }}
            darkMode={isDarkMode}
          />
        </div>
      )}

      {/* Map Visualization */}
      {hasMapData && (
        <div className="map-container relative" style={{ height: '300px' }}>
          {isMapLoading && <MapLoadingSkeleton t={t} />}
          <MapContainer 
            center={center} 
            zoom={13} 
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom={false}
            whenReady={() => {
              setTimeout(() => setIsMapLoading(false), 300);
            }}
          >
            <TileLayer
              attribution='© CARTO © OpenStreetMap contributors'
              url='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png'
              maxZoom={19}
              keepBuffer={4}
              updateWhenIdle={true}
              updateWhenZooming={false}
            />
            
            {/* Interactive Map Controls */}
            <MapControls center={center} onGeolocationFound={handleGeolocationFound} />
            
            {/* Route Lines */}
            {routes.map((route, idx) => {
              // Handle both 'lon' and 'lng' field names
              const positions = route.coordinates?.map(coord => [
                coord.lat, 
                coord.lon || coord.lng
              ]).filter(pos => pos[0] !== undefined && pos[1] !== undefined) || [];
              
              if (positions.length === 0) return null;
              
              // Determine line style based on mode
              const getLineStyle = (mode) => {
                const styles = {
                  walk: { weight: 3, color: '#9CA3AF', opacity: 0.6 },
                  metro: { weight: 5, color: '#DC2626', opacity: 0.8 },
                  bus: { weight: 4, color: '#2563EB', opacity: 0.8 },
                  tram: { weight: 4, color: '#16A34A', opacity: 0.8 },
                  ferry: { weight: 4, color: '#0891B2', opacity: 0.8 },
                  default: { weight: 4, color: '#4F46E5', opacity: 0.7 }
                };
                return styles[mode?.toLowerCase()] || styles.default;
              };
              
              const lineStyle = getLineStyle(route.mode);
              
              return (
                <AnimatedPolyline
                  key={`route-${idx}`}
                  positions={positions}
                  color={route.color || lineStyle.color}
                  weight={lineStyle.weight}
                  opacity={lineStyle.opacity}
                  speed={enableAnimations ? 30 : 0}
                />
              );
            })}
            
            {/* Transfer Point Markers - Pulsing */}
            {steps.filter(step => step.mode === 'transfer').map((step, idx) => {
              if (!step.lat || !step.lon) return null;
              return (
                <PulsingMarker
                  key={`transfer-${idx}`}
                  position={[step.lat, step.lon]}
                  label={`Transfer: ${step.instruction || step.description}`}
                  isPulse={true}
                />
              );
            })}
            
            {/* Markers */}
            {markers.map((marker, idx) => {
              // Handle both 'lon' and 'lng' field names
              const markerLat = marker.lat || marker.latitude;
              const markerLon = marker.lon || marker.lng || marker.longitude;
              
              if (!markerLat || !markerLon) {
                console.warn(`⚠️ Marker ${idx} missing coordinates:`, marker);
                return null;
              }
              
              return (
                <Marker 
                  key={`marker-${idx}`} 
                  position={[markerLat, markerLon]}
                >
                  <Popup>
                    <div>
                      <strong>{marker.label || marker.name}</strong>
                      {marker.type && <div className="text-sm text-gray-600">{marker.type}</div>}
                    </div>
                  </Popup>
                </Marker>
              );
            })}
            
            {/* User Location Marker */}
            {userLocation && (
              <Marker 
                position={userLocation}
                icon={L.divIcon({
                  className: 'user-location-marker',
                  html: '<div style="background: #4F46E5; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(0,0,0,0.3);"></div>',
                  iconSize: [16, 16],
                  iconAnchor: [8, 8]
                })}
              >
                <Popup>
                  <div>
                    <strong>{t('routeCard.yourLocation')}</strong>
                  </div>
                </Popup>
              </Marker>
            )}
            
            {/* Map Controls - Zoom, Recenter, Fullscreen, Geolocation */}
            <MapControls center={center} onGeolocationFound={() => {}} />
          </MapContainer>
        </div>
      )}

      {/* Step-by-Step Directions */}
      {steps.length > 0 && (
        <div className="p-4 border-t">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-semibold text-gray-800 flex items-center">
              <span className="mr-2">📋</span>
              Step-by-Step Directions:
            </h4>
            <button
              onClick={() => setIsNavigationMode(true)}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded-lg transition-colors shadow-sm flex items-center space-x-1 text-sm"
              title="Start step-by-step navigation"
            >
              <span>🎯</span>
              <span>{t('routeCard.startNavigation')}</span>
            </button>
          </div>
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
            <span>{t('routeCard.startNavigation')}</span>
          </button>
          
          {/* Secondary Action - Save Route */}
          <button 
            onClick={handleSaveRoute}
            className={`px-4 py-3 border-2 font-medium rounded-lg transition-all flex items-center space-x-1 ${
              isSaved 
                ? 'bg-indigo-100 border-indigo-400 text-indigo-700' 
                : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700'
            }`}
            title={isSaved ? "Unsave route" : "Save route for later"}
          >
            <span>{isSaved ? '⭐' : '☆'}</span>
            <span className="hidden sm:inline">{isSaved ? 'Saved' : 'Save'}</span>
          </button>
          
          {/* Secondary Actions - Copy */}
          <button 
            onClick={handleCopyRoute}
            className="copy-route-btn px-4 py-3 border-2 border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700 font-medium rounded-lg transition-colors flex items-center space-x-1"
            title="Copy route details"
          >
            <span>📋</span>
            <span className="hidden sm:inline">{t('routeCard.copy')}</span>
          </button>
          
          {/* Secondary Actions - Share with React state */}
          <button 
            onClick={handleShareRoute}
            className={`share-route-btn px-4 py-3 border-2 font-medium rounded-lg transition-all flex items-center space-x-1 ${
              shareStatus.state === 'success' 
                ? 'bg-green-100 border-green-400 text-green-700' 
                : shareStatus.state === 'error'
                ? 'bg-red-100 border-red-400 text-red-700'
                : 'border-gray-300 hover:border-indigo-400 hover:bg-indigo-50 text-gray-700'
            }`}
            title="Share route"
            disabled={shareStatus.state !== 'idle'}
          >
            <span>{shareStatus.state === 'success' ? '✓' : shareStatus.state === 'error' ? '❌' : '🔗'}</span>
            <span className="hidden sm:inline">{shareStatus.message || 'Share'}</span>
          </button>
        </div>
      </div>
      
      {/* Step-by-Step Navigation Modal */}
      {isNavigationMode && steps.length > 0 && (
        <StepByStepNavigation
          steps={steps}
          onClose={() => setIsNavigationMode(false)}
        />
      )}
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
