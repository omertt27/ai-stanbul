/**
 * TripPlanCard Component
 * ======================
 * Displays multi-day Istanbul trip itineraries with:
 * - Day-by-day breakdown
 * - Color-coded attractions by day
 * - Duration and walking distance info
 * - Transit lines used
 * - Integration with MapVisualization
 * 
 * Props:
 * - tripPlan: Object containing trip plan data from backend
 * - onDaySelect: Callback when a day is selected (optional)
 */

import React, { useState } from 'react';
import MapVisualization from './MapVisualization';

// Category emoji mapping
const categoryEmoji = {
  mosque: '🕌',
  museum: '🏛️',
  market: '🛍️',
  landmark: '🏰',
  palace: '👑',
  viewpoint: '🌅',
  neighborhood: '🏘️',
  street: '🚶',
  square: '⭐',
  tour: '🚢',
  food: '🍽️',
  experience: '💆',
  park: '🌳',
  default: '📍'
};

// Day color palette
const dayColors = [
  '#4285F4',  // Blue - Day 1
  '#EA4335',  // Red - Day 2
  '#34A853',  // Green - Day 3
  '#FBBC05',  // Yellow - Day 4
  '#9C27B0',  // Purple - Day 5
];

const TripPlanCard = ({ tripPlan, onDaySelect }) => {
  const [selectedDay, setSelectedDay] = useState(null); // null = show all days
  const [showMap, setShowMap] = useState(true);

  if (!tripPlan) {
    return null;
  }

  const {
    name,
    name_tr,
    duration_days,
    days = [],
    metadata = {},
    markers = [],
    routes = [],
    coordinates = [],
    center,
    zoom
  } = tripPlan;

  // Filter map data by selected day
  const filteredMapData = selectedDay ? {
    ...tripPlan,
    markers: markers.filter(m => m.day === selectedDay),
    routes: routes.filter(r => r.day === selectedDay),
    coordinates: coordinates, // Keep all for context
  } : tripPlan;

  const handleDayClick = (dayNumber) => {
    const newDay = selectedDay === dayNumber ? null : dayNumber;
    setSelectedDay(newDay);
    if (onDaySelect) {
      onDaySelect(newDay);
    }
  };

  return (
    <div className="trip-plan-card">
      {/* Header */}
      <div className="trip-header">
        <div className="trip-title-section">
          <span className="trip-icon">🗓️</span>
          <div>
            <h3 className="trip-title">{name}</h3>
            <p className="trip-subtitle">{name_tr}</p>
          </div>
        </div>
        <div className="trip-meta">
          <span className="meta-badge">{duration_days} Day{duration_days > 1 ? 's' : ''}</span>
          <span className="meta-badge">{metadata.total_attractions || days.reduce((acc, d) => acc + (d.attractions?.length || 0), 0)} Attractions</span>
        </div>
      </div>

      {/* Day Selector Tabs */}
      <div className="day-tabs">
        <button
          className={`day-tab ${selectedDay === null ? 'active' : ''}`}
          onClick={() => handleDayClick(null)}
          style={{ borderColor: selectedDay === null ? '#1a73e8' : 'transparent' }}
        >
          🗺️ All Days
        </button>
        {days.map((day) => (
          <button
            key={day.day_number}
            className={`day-tab ${selectedDay === day.day_number ? 'active' : ''}`}
            onClick={() => handleDayClick(day.day_number)}
            style={{ 
              borderColor: selectedDay === day.day_number ? day.color || dayColors[day.day_number - 1] : 'transparent',
              color: selectedDay === day.day_number ? day.color || dayColors[day.day_number - 1] : '#555'
            }}
          >
            <span className="day-dot" style={{ backgroundColor: day.color || dayColors[day.day_number - 1] }}></span>
            Day {day.day_number}
          </button>
        ))}
      </div>

      {/* Map Toggle */}
      <div className="map-toggle">
        <button onClick={() => setShowMap(!showMap)}>
          {showMap ? '📍 Hide Map' : '🗺️ Show Map'}
        </button>
      </div>

      {/* Map Visualization */}
      {showMap && (
        <div className="trip-map">
          <MapVisualization 
            mapData={filteredMapData}
            height="350px"
          />
        </div>
      )}

      {/* Days Itinerary */}
      <div className="days-container">
        {days.filter(day => selectedDay === null || day.day_number === selectedDay).map((day) => (
          <div 
            key={day.day_number} 
            className="day-card"
            style={{ borderLeftColor: day.color || dayColors[day.day_number - 1] }}
          >
            <div className="day-header">
              <div className="day-title">
                <span className="day-number" style={{ backgroundColor: day.color || dayColors[day.day_number - 1] }}>
                  Day {day.day_number}
                </span>
                <h4>{day.title}</h4>
              </div>
              <div className="day-meta">
                <span>⏱️ {Math.round(day.total_duration / 60)}h</span>
                <span>🚶 {day.walking_distance?.toFixed(1) || '?'} km</span>
              </div>
            </div>
            
            <p className="day-theme">{day.theme}</p>
            
            {/* Transit Lines */}
            {day.transit_lines && day.transit_lines.length > 0 && (
              <div className="transit-lines">
                {day.transit_lines.map((line, idx) => (
                  <span key={idx} className="transit-badge">
                    {line.startsWith('M') ? '🚇' : line.startsWith('T') ? '🚊' : line === 'FERRY' ? '⛴️' : '🚆'} {line}
                  </span>
                ))}
              </div>
            )}

            {/* Attractions List */}
            <div className="attractions-list">
              {(day.attractions || []).map((attraction, idx) => (
                <div key={idx} className="attraction-item">
                  <span className="attraction-number">{idx + 1}</span>
                  <div className="attraction-info">
                    <span className="attraction-name">
                      {categoryEmoji[attraction.category] || categoryEmoji.default} {attraction.name}
                    </span>
                    <span className="attraction-duration">{attraction.duration} min</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Highlights */}
      {metadata.highlights && metadata.highlights.length > 0 && (
        <div className="highlights-section">
          <h4>✨ Trip Highlights</h4>
          <div className="highlights-list">
            {metadata.highlights.map((highlight, idx) => (
              <span key={idx} className="highlight-tag">{highlight}</span>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .trip-plan-card {
          background: white;
          border-radius: 12px;
          box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
          margin: 16px 0;
          overflow: hidden;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .trip-header {
          background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
          color: white;
          padding: 16px 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex-wrap: wrap;
          gap: 12px;
        }

        .trip-title-section {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .trip-icon {
          font-size: 32px;
        }

        .trip-title {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .trip-subtitle {
          margin: 2px 0 0;
          font-size: 13px;
          opacity: 0.9;
        }

        .trip-meta {
          display: flex;
          gap: 10px;
        }

        .meta-badge {
          background: rgba(255, 255, 255, 0.2);
          padding: 6px 12px;
          border-radius: 16px;
          font-size: 13px;
          font-weight: 500;
        }

        .day-tabs {
          display: flex;
          gap: 4px;
          padding: 12px 16px;
          background: #f8f9fa;
          overflow-x: auto;
          -webkit-overflow-scrolling: touch;
        }

        .day-tab {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          border: none;
          border-bottom: 3px solid transparent;
          background: transparent;
          border-radius: 6px 6px 0 0;
          cursor: pointer;
          font-size: 13px;
          font-weight: 500;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .day-tab:hover {
          background: rgba(0, 0, 0, 0.05);
        }

        .day-tab.active {
          background: white;
          font-weight: 600;
        }

        .day-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .map-toggle {
          padding: 8px 16px;
          text-align: right;
        }

        .map-toggle button {
          background: #f0f0f0;
          border: none;
          padding: 6px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 12px;
        }

        .trip-map {
          border-top: 1px solid #e0e0e0;
          border-bottom: 1px solid #e0e0e0;
        }

        .days-container {
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .day-card {
          background: #fafafa;
          border-radius: 10px;
          padding: 16px;
          border-left: 4px solid #4285F4;
        }

        .day-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 8px;
          flex-wrap: wrap;
          gap: 8px;
        }

        .day-title {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .day-number {
          color: white;
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .day-title h4 {
          margin: 0;
          font-size: 15px;
        }

        .day-meta {
          display: flex;
          gap: 12px;
          font-size: 12px;
          color: #666;
        }

        .day-theme {
          color: #555;
          font-size: 13px;
          margin: 6px 0 12px;
          font-style: italic;
        }

        .transit-lines {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 12px;
        }

        .transit-badge {
          background: #e8f0fe;
          color: #1a73e8;
          padding: 4px 10px;
          border-radius: 12px;
          font-size: 11px;
          font-weight: 600;
        }

        .attractions-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .attraction-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }

        .attraction-number {
          width: 24px;
          height: 24px;
          background: #e8f0fe;
          color: #1a73e8;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 600;
          flex-shrink: 0;
        }

        .attraction-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          flex: 1;
          gap: 8px;
        }

        .attraction-name {
          font-size: 13px;
          font-weight: 500;
        }

        .attraction-duration {
          font-size: 11px;
          color: #888;
          white-space: nowrap;
        }

        .highlights-section {
          padding: 16px 20px;
          background: #f8f9fa;
          border-top: 1px solid #e0e0e0;
        }

        .highlights-section h4 {
          margin: 0 0 10px;
          font-size: 14px;
        }

        .highlights-list {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .highlight-tag {
          background: #fff;
          border: 1px solid #ddd;
          padding: 5px 12px;
          border-radius: 16px;
          font-size: 12px;
          color: #333;
        }

        @media (max-width: 768px) {
          .trip-header {
            padding: 12px 16px;
          }

          .trip-title {
            font-size: 16px;
          }

          .day-tabs {
            padding: 10px 12px;
          }

          .days-container {
            padding: 12px;
          }

          .attraction-info {
            flex-direction: column;
            align-items: flex-start;
            gap: 2px;
          }
        }
      `}</style>
    </div>
  );
};

export default TripPlanCard;
