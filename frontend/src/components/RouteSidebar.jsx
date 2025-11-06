/**
 * Route Sidebar Component
 * =======================
 * Displays itinerary with drag-and-drop reordering
 * 
 * Features:
 * - Drag-and-drop waypoint reordering
 * - Turn-by-turn directions
 * - Visit duration tracking
 * - Location details
 * - Add/remove waypoints
 */

import React, { useState, useCallback } from 'react';
import './RouteSidebar.css';

const RouteSidebar = ({
  waypoints = [],
  route = null,
  onWaypointReorder,
  onWaypointRemove,
  onWaypointClick,
  transportMode = 'walk',
  showDirections = true,
  className = ''
}) => {
  const [draggedIndex, setDraggedIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);

  // Handle drag and drop
  const handleDragStart = useCallback((e, index) => {
    setDraggedIndex(index);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.target);
  }, []);

  const handleDragOver = useCallback((e, index) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverIndex(index);
  }, []);

  const handleDrop = useCallback((e, dropIndex) => {
    e.preventDefault();
    
    if (draggedIndex === null || draggedIndex === dropIndex) {
      setDraggedIndex(null);
      setDragOverIndex(null);
      return;
    }

    const newWaypoints = [...waypoints];
    const draggedWaypoint = newWaypoints[draggedIndex];
    
    // Remove from old position
    newWaypoints.splice(draggedIndex, 1);
    // Insert at new position
    newWaypoints.splice(dropIndex, 0, draggedWaypoint);
    
    if (onWaypointReorder) {
      onWaypointReorder(newWaypoints);
    }
    
    setDraggedIndex(null);
    setDragOverIndex(null);
  }, [draggedIndex, waypoints, onWaypointReorder]);

  const handleDragEnd = useCallback(() => {
    setDraggedIndex(null);
    setDragOverIndex(null);
  }, []);

  // Get marker emoji based on type
  const getMarkerEmoji = (waypoint, index) => {
    if (index === 0) return '🏁';
    if (index === waypoints.length - 1) return '🎯';
    
    const emojiMap = {
      museum: '🏛️',
      restaurant: '🍽️',
      cafe: '☕',
      attraction: '⭐',
      default: '📍'
    };
    
    return emojiMap[waypoint.category] || emojiMap.default;
  };

  // Calculate route statistics
  const routeStats = route ? {
    totalDistance: (route.total_distance / 1000).toFixed(2),
    totalDuration: Math.round(route.total_duration / 60),
    totalLocations: waypoints.length
  } : null;

  return (
    <div className={`route-sidebar ${className}`}>
      {/* Header */}
      <div className="sidebar-header">
        <h2>Your Itinerary</h2>
        {routeStats && (
          <div className="route-summary">
            <div className="summary-item">
              <span className="summary-icon">📏</span>
              <span className="summary-text">{routeStats.totalDistance} km</span>
            </div>
            <div className="summary-item">
              <span className="summary-icon">⏱️</span>
              <span className="summary-text">{routeStats.totalDuration} min</span>
            </div>
            <div className="summary-item">
              <span className="summary-icon">📍</span>
              <span className="summary-text">{routeStats.totalLocations} stops</span>
            </div>
          </div>
        )}
      </div>

      {/* Waypoints list */}
      <div className="waypoints-list">
        {waypoints.map((waypoint, index) => (
          <div
            key={`waypoint-${index}-${waypoint.name}`}
            className={`waypoint-item ${
              draggedIndex === index ? 'dragging' : ''
            } ${dragOverIndex === index ? 'drag-over' : ''}`}
            draggable={waypoints.length > 2}
            onDragStart={(e) => handleDragStart(e, index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDrop={(e) => handleDrop(e, index)}
            onDragEnd={handleDragEnd}
            onClick={() => onWaypointClick && onWaypointClick(waypoint, index)}
          >
            <div className="waypoint-header">
              <div className="waypoint-marker">
                <span className="marker-emoji">{getMarkerEmoji(waypoint, index)}</span>
                <span className="marker-number">{index + 1}</span>
              </div>
              <div className="waypoint-info">
                <h3 className="waypoint-name">{waypoint.name}</h3>
                {waypoint.category && (
                  <p className="waypoint-category">{waypoint.category}</p>
                )}
              </div>
              {waypoints.length > 2 && onWaypointRemove && (
                <button
                  className="remove-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onWaypointRemove(index);
                  }}
                  title="Remove waypoint"
                >
                  ❌
                </button>
              )}
            </div>

            {waypoint.address && (
              <p className="waypoint-address">📍 {waypoint.address}</p>
            )}

            {waypoint.visit_duration && (
              <p className="waypoint-duration">⏱️ Visit: {waypoint.visit_duration} min</p>
            )}

            {/* Show directions to next waypoint */}
            {showDirections && route && route.segments && route.segments[index] && (
              <div className="directions-section">
                <div className="direction-summary">
                  <span className="direction-icon">
                    {transportMode === 'walk' ? '🚶' : 
                     transportMode === 'drive' ? '🚗' : 
                     transportMode === 'bike' ? '🚴' : '🚇'}
                  </span>
                  <span className="direction-text">
                    {(route.segments[index].distance / 1000).toFixed(2)} km • {' '}
                    {Math.round(route.segments[index].duration / 60)} min to next stop
                  </span>
                </div>
                
                {route.segments[index].instructions && route.segments[index].instructions.length > 0 && (
                  <div className="turn-by-turn">
                    <details className="directions-details">
                      <summary>Turn-by-turn directions</summary>
                      <ol className="instructions-list">
                        {route.segments[index].instructions.map((instruction, idx) => (
                          <li key={`instruction-${index}-${idx}`} className="instruction-item">
                            <span className="instruction-icon">
                              {instruction.type === 'turn right' ? '➡️' :
                               instruction.type === 'turn left' ? '⬅️' :
                               instruction.type === 'straight' ? '⬆️' :
                               instruction.type === 'arrive' ? '🎯' : '📍'}
                            </span>
                            <span className="instruction-text">
                              {instruction.instruction}
                            </span>
                            {instruction.distance && (
                              <span className="instruction-distance">
                                ({instruction.distance.toFixed(0)}m)
                              </span>
                            )}
                          </li>
                        ))}
                      </ol>
                    </details>
                  </div>
                )}
              </div>
            )}

            {/* Connector line to next waypoint */}
            {index < waypoints.length - 1 && (
              <div className="waypoint-connector"></div>
            )}
          </div>
        ))}
      </div>

      {/* Footer with actions */}
      <div className="sidebar-footer">
        <p className="drag-hint">
          💡 Drag waypoints to reorder your itinerary
        </p>
      </div>
    </div>
  );
};

export default RouteSidebar;
