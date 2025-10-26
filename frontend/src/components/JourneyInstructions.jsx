import React, { useState } from 'react';
import ChatMapView from './ChatMapView';
import './JourneyInstructions.css';

/**
 * JourneyInstructions Component
 * Displays complete GPS-to-destination journey with walking and transit stages
 * Google Maps-quality turn-by-turn navigation UI with integrated map visualization
 */
const JourneyInstructions = ({ journey, darkMode = false }) => {
  const [currentStage, setCurrentStage] = useState(0);
  const [expanded, setExpanded] = useState(true);
  const [showMap, setShowMap] = useState(false);

  if (!journey || !journey.summary) {
    return null;
  }

  const { summary, walking_to_transit, transit_journey, walking_to_destination } = journey;

  // Build stages array
  const stages = [];
  
  // Stage 1: Walking to transit
  if (walking_to_transit) {
    stages.push({
      type: 'walking',
      icon: '🚶',
      title: 'Walk to Transit',
      subtitle: walking_to_transit.end_stop?.stop_name || 'Transit Stop',
      duration: walking_to_transit.directions.total_duration_min,
      distance: walking_to_transit.directions.total_distance_m,
      details: walking_to_transit.directions,
      transportType: walking_to_transit.end_stop?.transport_type
    });
  }

  // Stage 2+: Transit segments
  if (transit_journey && transit_journey.segments) {
    transit_journey.segments.forEach((segment, idx) => {
      const icon = getTransportIcon(segment.transport_type);
      stages.push({
        type: 'transit',
        icon: icon,
        title: segment.line_name,
        subtitle: `${segment.from_stop_name} → ${segment.to_stop_name}`,
        duration: segment.duration_minutes,
        stops: segment.stops_count,
        transportType: segment.transport_type,
        details: segment
      });

      // Add transfer if not last segment
      if (idx < transit_journey.segments.length - 1) {
        const nextSegment = transit_journey.segments[idx + 1];
        if (segment.to_stop !== nextSegment.from_stop) {
          stages.push({
            type: 'transfer',
            icon: '🔄',
            title: 'Transfer',
            subtitle: `Change to ${nextSegment.line_name}`,
            duration: 3,
            details: {
              from: segment.to_stop_name,
              to: nextSegment.from_stop_name
            }
          });
        }
      }
    });
  }

  // Final stage: Walking to destination
  if (walking_to_destination) {
    stages.push({
      type: 'walking',
      icon: '🚶',
      title: 'Walk to Destination',
      subtitle: journey.destination?.name || 'Final Destination',
      duration: walking_to_destination.total_duration_min,
      distance: walking_to_destination.total_distance_m,
      details: walking_to_destination
    });
  }

  const getTransportIcon = (type) => {
    const icons = {
      metro: '🚇',
      tram: '🚊',
      rail: '🚆',
      marmaray: '🚆',
      funicular: '🚡',
      bus: '🚌',
      ferry: '⛴️'
    };
    return icons[type?.toLowerCase()] || '🚇';
  };

  const formatDistance = (meters) => {
    if (meters < 1000) {
      return `${meters}m`;
    }
    return `${(meters / 1000).toFixed(1)}km`;
  };

  const formatDuration = (minutes) => {
    if (minutes < 1) return '< 1 min';
    if (minutes === 1) return '1 min';
    return `${minutes} min`;
  };

  return (
    <div className={`journey-instructions ${darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <div className="journey-header">
        <div className="journey-title">
          <span className="title-icon">🎯</span>
          <h3>Your Journey</h3>
          <button 
            className="expand-btn"
            onClick={() => setExpanded(!expanded)}
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '▼' : '▶'}
          </button>
        </div>
        
        <div className="journey-summary">
          <div className="summary-item">
            <span className="summary-icon">⏱️</span>
            <span className="summary-value">{summary.total_duration_min}</span>
            <span className="summary-label">min</span>
          </div>
          <div className="summary-item">
            <span className="summary-icon">📍</span>
            <span className="summary-value">{summary.total_distance_km}</span>
            <span className="summary-label">km</span>
          </div>
          {summary.total_transfers > 0 && (
            <div className="summary-item">
              <span className="summary-icon">🔄</span>
              <span className="summary-value">{summary.total_transfers}</span>
              <span className="summary-label">transfers</span>
            </div>
          )}
          <div className="summary-item">
            <span className="summary-icon">💰</span>
            <span className="summary-value">₺{summary.estimated_cost_tl}</span>
          </div>
        </div>
      </div>

      {/* Stages */}
      {expanded && (
        <div className="journey-stages">
          {stages.map((stage, index) => (
            <div
              key={index}
              className={`journey-stage ${stage.type} ${currentStage === index ? 'active' : ''} ${currentStage > index ? 'completed' : ''}`}
              onClick={() => setCurrentStage(index)}
            >
              <div className="stage-timeline">
                <div className="stage-marker">
                  <span className="stage-icon">{stage.icon}</span>
                </div>
                {index < stages.length - 1 && <div className="stage-line"></div>}
              </div>

              <div className="stage-content">
                <div className="stage-header">
                  <div className="stage-info">
                    <h4 className="stage-title">{stage.title}</h4>
                    <p className="stage-subtitle">{stage.subtitle}</p>
                  </div>
                  <div className="stage-meta">
                    {stage.duration && (
                      <span className="stage-duration">
                        {formatDuration(stage.duration)}
                      </span>
                    )}
                    {stage.distance && (
                      <span className="stage-distance">
                        {formatDistance(stage.distance)}
                      </span>
                    )}
                    {stage.stops && (
                      <span className="stage-stops">
                        {stage.stops} stops
                      </span>
                    )}
                  </div>
                </div>

                {/* Expanded details for active stage */}
                {currentStage === index && stage.details && (
                  <div className="stage-details">
                    {stage.type === 'walking' && stage.details.steps && (
                      <div className="walking-steps">
                        {stage.details.steps.map((step, stepIdx) => (
                          <div key={stepIdx} className="walking-step">
                            <span className="step-number">{stepIdx + 1}</span>
                            <div className="step-content">
                              <p className="step-instruction">{step.instruction}</p>
                              <span className="step-meta">
                                {formatDistance(step.distance_m)} · {step.duration_text}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {stage.type === 'transit' && (
                      <div className="transit-details">
                        <div className="transit-info">
                          <span className="transit-label">Direction:</span>
                          <span className="transit-value">{stage.details.to_stop_name}</span>
                        </div>
                        <div className="transit-info">
                          <span className="transit-label">Platform:</span>
                          <span className="transit-value">Follow signs for {stage.title}</span>
                        </div>
                      </div>
                    )}

                    {stage.type === 'transfer' && (
                      <div className="transfer-details">
                        <p className="transfer-instruction">
                          Follow signs for {stage.subtitle}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="journey-actions">
        <button className="action-btn primary" onClick={() => setCurrentStage(0)}>
          <span className="btn-icon">🚀</span>
          Start Journey
        </button>
        <button 
          className="action-btn secondary"
          onClick={() => setShowMap(!showMap)}
        >
          <span className="btn-icon">📍</span>
          {showMap ? 'Hide Map' : 'Show on Map'}
        </button>
        <button className="action-btn secondary">
          <span className="btn-icon">📤</span>
          Share Route
        </button>
      </div>

      {/* Map Visualization */}
      {showMap && journey.gps_start && (
        <div className="journey-map-container">
          <ChatMapView
            mapData={{
              locations: [
                {
                  name: 'Your Location',
                  lat: journey.gps_start.latitude,
                  lon: journey.gps_start.longitude,
                  type: 'origin'
                },
                ...(walking_to_transit?.end_stop ? [{
                  name: walking_to_transit.end_stop.stop_name,
                  lat: walking_to_transit.end_stop.latitude,
                  lon: walking_to_transit.end_stop.longitude,
                  type: 'transit'
                }] : []),
                ...(transit_journey?.segments?.map(seg => ({
                  name: seg.to_stop_name,
                  lat: seg.to_stop?.latitude || 41.0082,
                  lon: seg.to_stop?.longitude || 28.9784,
                  type: 'transit'
                })) || [])
              ],
              user_location: {
                lat: journey.gps_start.latitude,
                lng: journey.gps_start.longitude
              }
            }}
            darkMode={darkMode}
          />
        </div>
      )}

      {/* Progress Bar */}
      <div className="journey-progress">
        <div 
          className="progress-bar"
          style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
        />
      </div>
    </div>
  );
};

export default JourneyInstructions;
