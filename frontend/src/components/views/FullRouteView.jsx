/**
 * FullRouteView Component
 * =======================
 * 
 * Complete route display with embedded map and all advanced features.
 * This is a wrapper around the existing RouteCard component.
 * 
 * Features:
 * - All Priority 1-5.1 features
 * - Embedded map with MapCore
 * - Multi-route comparison
 * - Step-by-step navigation
 * - Mobile optimizations
 * - Dark mode support
 * - Save/Share/Copy actions
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React from 'react';
import RouteCard from '../RouteCard';

const FullRouteView = ({ data, darkMode, onCopy, onShare, onStartNavigation, ...props }) => {
  // RouteCard expects routeData with route_info and map_data
  const routeData = {
    route_info: data.route_info || data.routeInfo,
    map_data: data.map_data || data.mapData,
    alternatives: data.alternatives || data.alternative_routes || [],
    ...data
  };

  return (
    <RouteCard
      routeData={routeData}
      darkMode={darkMode}
      {...props}
    />
  );
};

export default FullRouteView;
