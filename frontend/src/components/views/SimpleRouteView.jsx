/**
 * SimpleRouteView Component
 * ==========================
 * 
 * Simple route display without embedded map.
 * Wrapper around TransportationRouteCard for legacy compatibility.
 * 
 * Features:
 * - Step-by-step directions
 * - Transit line colors
 * - Confidence indicators
 * - Transfer highlights
 * - No embedded map (lighter weight)
 * 
 * Author: AI Istanbul Team
 * Date: January 12, 2026
 */

import React from 'react';
import TransportationRouteCard from '../TransportationRouteCard';

const SimpleRouteView = ({ data, darkMode, ...props }) => {
  // TransportationRouteCard expects routeData
  const routeData = data.route_info || data.routeInfo || data.routeData || data;

  return (
    <TransportationRouteCard
      routeData={routeData}
      darkMode={darkMode}
      {...props}
    />
  );
};

export default SimpleRouteView;
