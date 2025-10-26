/**
 * Transportation API Service
 * ===========================
 * Production-grade frontend client for Istanbul AI Transportation System
 * 
 * Features:
 * - Live İBB open data integration
 * - GPS-based route planning
 * - Multi-modal transportation (metro, tram, bus, ferry, walking)
 * - Step-by-step directions with map rendering
 * - Real-time route alternatives
 * - Accessibility information
 * - Error handling and retry logic
 * 
 * Backend Integration:
 * - /ai/chat endpoint with transportation queries
 * - Enhanced GPS route planner
 * - Transportation directions service
 * - OSRM routing for walking segments
 */

import { fetchWithRetry, handleApiError } from '../utils/errorHandler';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const CHAT_API_URL = `${BASE_URL}/ai/chat`;

/**
 * Get transportation directions between two points
 * @param {Object} params - Route parameters
 * @param {Object} params.from - Origin location {lat, lng, name}
 * @param {Object} params.to - Destination location {lat, lng, name}
 * @param {string} params.mode - Transportation mode: 'transit', 'walking', 'driving'
 * @param {boolean} params.accessible - Prefer accessible routes
 * @param {boolean} params.alternatives - Return alternative routes
 * @param {string} params.departureTime - ISO datetime string for departure
 * @returns {Promise<Object>} Route data with steps, map geometry, and alternatives
 */
export const getTransportationDirections = async ({
  from,
  to,
  mode = 'transit',
  accessible = false,
  alternatives = true,
  departureTime = null
}) => {
  try {
    // Build natural language query for the AI chat system
    const accessibleText = accessible ? ' with accessibility features' : '';
    const timeText = departureTime ? ` departing at ${new Date(departureTime).toLocaleTimeString()}` : '';
    
    const query = `Get directions from ${from.name || `${from.lat}, ${from.lng}`} to ${to.name || `${to.lat}, ${to.lng}`} using ${mode}${accessibleText}${timeText}. Include step-by-step instructions, transfer points, and map data.`;

    console.log('🗺️ Transportation query:', query);

    const response = await fetchWithRetry(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: query,
        context: {
          from_location: from,
          to_location: to,
          mode,
          accessible,
          alternatives,
          departure_time: departureTime
        }
      }),
      timeout: 30000
    }, {
      maxAttempts: 2,
      baseDelay: 1000
    });

    const data = await response.json();
    
    // Extract transportation data from AI response
    const route = parseTransportationResponse(data);
    
    console.log('✅ Transportation directions received:', route);
    return route;

  } catch (error) {
    console.error('❌ Transportation directions failed:', error);
    throw handleApiError(error, null, 'Transportation Directions');
  }
};

/**
 * Get route from GPS coordinates (user location)
 * @param {Object} params - GPS route parameters
 * @param {Object} params.userLocation - Current GPS location {latitude, longitude, accuracy}
 * @param {Object} params.destination - Destination {lat, lng, name}
 * @param {string} params.mode - Transportation mode
 * @param {boolean} params.accessible - Prefer accessible routes
 * @returns {Promise<Object>} GPS-based route with live tracking support
 */
export const getGPSRoute = async ({
  userLocation,
  destination,
  mode = 'transit',
  accessible = false
}) => {
  try {
    const query = `Navigate me from my current location (${userLocation.latitude}, ${userLocation.longitude}) to ${destination.name || destination.lat + ',' + destination.lng} using ${mode}. Use live GPS tracking.`;

    console.log('📍 GPS route query:', query);

    const response = await fetchWithRetry(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: query,
        context: {
          gps_location: userLocation,
          destination,
          mode,
          accessible,
          enable_tracking: true
        }
      }),
      timeout: 30000
    });

    const data = await response.json();
    const route = parseTransportationResponse(data);
    
    console.log('✅ GPS route received:', route);
    return route;

  } catch (error) {
    console.error('❌ GPS route failed:', error);
    throw handleApiError(error, null, 'GPS Route');
  }
};

/**
 * Get nearby transit stations and stops
 * @param {Object} location - Location {lat, lng}
 * @param {number} radius - Search radius in meters (default: 500m)
 * @param {Array<string>} types - Transit types to search: ['metro', 'tram', 'bus', 'ferry']
 * @returns {Promise<Array>} Nearby transit stations
 */
export const getNearbyTransit = async (location, radius = 500, types = null) => {
  try {
    const typesText = types ? types.join(', ') : 'all transit';
    const query = `Show me ${typesText} stations within ${radius} meters of ${location.lat}, ${location.lng}`;

    console.log('🚉 Nearby transit query:', query);

    const response = await fetchWithRetry(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: query,
        context: {
          location,
          radius,
          types,
          query_type: 'nearby_transit'
        }
      }),
      timeout: 15000
    });

    const data = await response.json();
    const stations = parseNearbyTransitResponse(data);
    
    console.log('✅ Nearby transit stations:', stations.length);
    return stations;

  } catch (error) {
    console.error('❌ Nearby transit search failed:', error);
    throw handleApiError(error, null, 'Nearby Transit');
  }
};

/**
 * Get live schedule for a specific transit line/station
 * @param {string} stationName - Station name
 * @param {string} lineId - Transit line ID (e.g., 'M2', 'T1')
 * @returns {Promise<Object>} Live schedule data
 */
export const getTransitSchedule = async (stationName, lineId = null) => {
  try {
    const lineText = lineId ? ` on line ${lineId}` : '';
    const query = `Show me the schedule for ${stationName}${lineText}. Include next arrivals and real-time updates.`;

    console.log('📅 Transit schedule query:', query);

    const response = await fetchWithRetry(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: query,
        context: {
          station_name: stationName,
          line_id: lineId,
          query_type: 'transit_schedule'
        }
      }),
      timeout: 15000
    });

    const data = await response.json();
    const schedule = parseScheduleResponse(data);
    
    console.log('✅ Transit schedule received');
    return schedule;

  } catch (error) {
    console.error('❌ Transit schedule failed:', error);
    throw handleApiError(error, null, 'Transit Schedule');
  }
};

/**
 * Search for routes between common Istanbul locations
 * @param {string} originName - Origin location name
 * @param {string} destinationName - Destination location name
 * @param {string} mode - Transportation mode
 * @returns {Promise<Object>} Route data
 */
export const searchRouteByName = async (originName, destinationName, mode = 'transit') => {
  try {
    const query = `How do I get from ${originName} to ${destinationName} using ${mode}?`;

    console.log('🔍 Route search query:', query);

    const response = await fetchWithRetry(CHAT_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        message: query,
        context: {
          origin_name: originName,
          destination_name: destinationName,
          mode,
          query_type: 'route_search'
        }
      }),
      timeout: 30000
    });

    const data = await response.json();
    const route = parseTransportationResponse(data);
    
    console.log('✅ Route search completed');
    return route;

  } catch (error) {
    console.error('❌ Route search failed:', error);
    throw handleApiError(error, null, 'Route Search');
  }
};

/**
 * Parse transportation response from AI chat
 * Extracts route data, steps, map geometry, and metadata
 */
const parseTransportationResponse = (data) => {
  // Check if response has transportation data
  const hasTransportData = data.metadata?.category === 'transportation' ||
                          data.metadata?.has_transportation_data ||
                          data.response?.includes('directions') ||
                          data.response?.includes('route');

  if (!hasTransportData && !data.map_data) {
    console.warn('⚠️ No transportation data in response');
    return {
      error: 'No transportation data found',
      message: data.response || 'Unable to find route information',
      hasRoute: false
    };
  }

  // Extract route steps from response
  const steps = extractRouteSteps(data);
  
  // Extract map geometry
  const geometry = extractMapGeometry(data);
  
  // Extract metadata
  const metadata = {
    totalDistance: extractDistance(data),
    totalDuration: extractDuration(data),
    modes: extractModes(data),
    transfers: extractTransfers(data),
    accessibility: extractAccessibility(data),
    alternatives: extractAlternatives(data)
  };

  return {
    hasRoute: true,
    steps,
    geometry,
    metadata,
    summary: data.response || '',
    raw: data
  };
};

/**
 * Extract route steps from AI response
 */
const extractRouteSteps = (data) => {
  const steps = [];
  
  // Check map_data for route polyline and steps
  if (data.map_data?.route_polyline) {
    // Route polyline exists - create steps from it
    const polyline = data.map_data.route_polyline;
    steps.push({
      type: 'route',
      instruction: 'Follow the highlighted route',
      geometry: polyline,
      mode: data.metadata?.transport_mode || 'transit'
    });
  }

  // Check metadata for transportation steps
  if (data.metadata?.transportation_steps) {
    return data.metadata.transportation_steps.map(step => ({
      type: step.type || 'transit',
      instruction: step.instruction || step.description,
      distance: step.distance,
      duration: step.duration,
      mode: step.mode || step.type,
      line: step.line_name,
      stops: step.stops_count,
      start: step.start_location,
      end: step.end_location,
      geometry: step.waypoints
    }));
  }

  // Parse from text response if structured data not available
  const text = data.response || '';
  const lines = text.split('\n');
  
  lines.forEach(line => {
    // Look for step indicators
    if (line.match(/^\d+\./) || line.includes('→') || line.includes('Take') || line.includes('Walk')) {
      steps.push({
        type: 'instruction',
        instruction: line.trim(),
        mode: detectMode(line)
      });
    }
  });

  return steps;
};

/**
 * Extract map geometry (polyline/route path)
 */
const extractMapGeometry = (data) => {
  if (data.map_data?.route_polyline) {
    return {
      type: 'LineString',
      coordinates: data.map_data.route_polyline.map(p => [p.lng, p.lat])
    };
  }

  if (data.metadata?.route_geometry) {
    return data.metadata.route_geometry;
  }

  // Create simple geometry from start/end if available
  if (data.map_data?.locations && data.map_data.locations.length >= 2) {
    const locs = data.map_data.locations;
    return {
      type: 'LineString',
      coordinates: locs.map(loc => [loc.lon, loc.lat])
    };
  }

  return null;
};

/**
 * Helper functions to extract metadata
 */
const extractDistance = (data) => {
  if (data.metadata?.total_distance) return data.metadata.total_distance;
  const match = (data.response || '').match(/(\d+\.?\d*)\s*(km|kilometers|m|meters)/i);
  return match ? parseFloat(match[1]) : null;
};

const extractDuration = (data) => {
  if (data.metadata?.total_duration) return data.metadata.total_duration;
  const match = (data.response || '').match(/(\d+)\s*(min|minutes|hour|hours)/i);
  if (match) {
    const value = parseInt(match[1]);
    const unit = match[2].toLowerCase();
    return unit.startsWith('hour') ? value * 60 : value;
  }
  return null;
};

const extractModes = (data) => {
  if (data.metadata?.transport_modes) return data.metadata.transport_modes;
  const text = (data.response || '').toLowerCase();
  const modes = [];
  if (text.includes('metro') || text.includes('m1') || text.includes('m2')) modes.push('metro');
  if (text.includes('tram') || text.includes('t1')) modes.push('tram');
  if (text.includes('bus')) modes.push('bus');
  if (text.includes('ferry') || text.includes('vapur')) modes.push('ferry');
  if (text.includes('walk')) modes.push('walking');
  return modes.length > 0 ? modes : ['transit'];
};

const extractTransfers = (data) => {
  if (data.metadata?.transfers) return data.metadata.transfers;
  const text = data.response || '';
  const transferMatch = text.match(/(\d+)\s*transfer/i);
  return transferMatch ? parseInt(transferMatch[1]) : 0;
};

const extractAccessibility = (data) => {
  if (data.metadata?.accessibility) return data.metadata.accessibility;
  const text = (data.response || '').toLowerCase();
  return {
    wheelchairAccessible: text.includes('wheelchair') || text.includes('accessible'),
    elevatorAvailable: text.includes('elevator') || text.includes('asansör'),
    rampAvailable: text.includes('ramp')
  };
};

const extractAlternatives = (data) => {
  if (data.metadata?.alternative_routes) {
    return data.metadata.alternative_routes.map(alt => ({
      summary: alt.summary,
      duration: alt.duration,
      distance: alt.distance,
      modes: alt.modes
    }));
  }
  return [];
};

const detectMode = (text) => {
  const lower = text.toLowerCase();
  if (lower.includes('metro') || lower.includes('m1') || lower.includes('m2')) return 'metro';
  if (lower.includes('tram') || lower.includes('t1')) return 'tram';
  if (lower.includes('bus')) return 'bus';
  if (lower.includes('ferry') || lower.includes('vapur')) return 'ferry';
  if (lower.includes('walk')) return 'walking';
  return 'transit';
};

/**
 * Parse nearby transit response
 */
const parseNearbyTransitResponse = (data) => {
  if (data.map_data?.locations) {
    return data.map_data.locations.filter(loc => 
      loc.type === 'station' || loc.type === 'transit'
    );
  }

  // Parse from text if no structured data
  const stations = [];
  const text = data.response || '';
  const lines = text.split('\n');
  
  lines.forEach(line => {
    // Look for station names with distances
    const match = line.match(/([A-Za-zığüşöçİĞÜŞÖÇ\s]+)\s*[:-]\s*(\d+)\s*m/i);
    if (match) {
      stations.push({
        name: match[1].trim(),
        distance: parseInt(match[2]),
        type: 'station'
      });
    }
  });

  return stations;
};

/**
 * Parse transit schedule response
 */
const parseScheduleResponse = (data) => {
  const schedule = {
    station: null,
    line: null,
    nextArrivals: [],
    lastUpdate: new Date()
  };

  // Extract from metadata if available
  if (data.metadata?.schedule) {
    return data.metadata.schedule;
  }

  // Parse from text response
  const text = data.response || '';
  const lines = text.split('\n');
  
  lines.forEach(line => {
    // Look for time patterns
    const timeMatch = line.match(/(\d{1,2}):(\d{2})/g);
    if (timeMatch) {
      schedule.nextArrivals.push(...timeMatch.map(t => ({
        time: t,
        estimated: true
      })));
    }
  });

  return schedule;
};

/**
 * Health check for transportation service
 */
export const checkTransportationServiceHealth = async () => {
  try {
    const response = await fetchWithRetry(`${BASE_URL}/health`, {
      method: 'GET',
      timeout: 5000
    });
    
    const health = await response.json();
    return health;
    
  } catch (error) {
    console.error('Transportation service health check failed:', error);
    return { status: 'offline', error: error.message };
  }
};

export default {
  getTransportationDirections,
  getGPSRoute,
  getNearbyTransit,
  getTransitSchedule,
  searchRouteByName,
  checkTransportationServiceHealth
};
