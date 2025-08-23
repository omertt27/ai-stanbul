import React from 'react';
import { GoogleMap, Marker, useJsApiLoader } from '@react-google-maps/api';

const containerStyle = {
  width: '100%',
  height: '400px',
};

function MapView({ locations }) {
  const { isLoaded } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY
  });

  const center = locations && locations.length > 0
    ? { lat: locations[0].lat, lng: locations[0].lng }
    : { lat: 51.505, lng: -0.09 };

  return isLoaded ? (
    <GoogleMap
      mapContainerStyle={containerStyle}
      center={center}
      zoom={13}
    >
      {locations.map((loc, idx) => (
        <Marker key={idx} position={{ lat: loc.lat, lng: loc.lng }} label={loc.label} />
      ))}
    </GoogleMap>
  ) : <div className="bg-gray-800 text-white flex items-center justify-center h-[400px]">Loading Map...</div>;
}

export default React.memo(MapView);
