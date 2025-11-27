import React from 'react';

/**
 * Restaurant Card Component
 * Displays restaurant information with photo in chat responses
 */
const RestaurantCard = ({ restaurant, index }) => {
  // Get photo URL from Google Places API
  const getPhotoUrl = (photoReference) => {
    if (!photoReference) return null;
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    return `https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference=${photoReference}&key=${apiKey}`;
  };

  const photoUrl = restaurant.photos && restaurant.photos.length > 0 
    ? getPhotoUrl(restaurant.photos[0].photo_reference || restaurant.photos[0])
    : null;

  const priceLevel = restaurant.price_level;
  const priceSymbols = ['💰', '💰💰', '💰💰💰', '💰💰💰💰'];
  const priceDisplay = priceLevel ? priceSymbols[priceLevel - 1] : null;

  return (
    <div className="restaurant-card mb-4 rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow duration-200 bg-white dark:bg-gray-800">
      {/* Restaurant Photo */}
      {photoUrl && (
        <div className="restaurant-photo relative h-48 overflow-hidden">
          <img 
            src={photoUrl} 
            alt={restaurant.name}
            className="w-full h-full object-cover"
            loading="lazy"
            onError={(e) => {
              e.target.style.display = 'none';
            }}
          />
          {/* Restaurant Index Badge */}
          <div className="absolute top-2 left-2 bg-blue-500 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
            {index + 1}
          </div>
        </div>
      )}

      {/* Restaurant Info */}
      <div className="p-4">
        {/* Name */}
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
          {!photoUrl && `${index + 1}. `}{restaurant.name}
        </h3>

        {/* Rating */}
        {restaurant.rating && (
          <div className="flex items-center gap-2 mb-2">
            <div className="flex items-center">
              <span className="text-yellow-500">⭐</span>
              <span className="ml-1 font-semibold text-gray-900 dark:text-white">
                {restaurant.rating}
              </span>
              <span className="text-gray-500 dark:text-gray-400">/5</span>
            </div>
            {restaurant.user_ratings_total && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                ({restaurant.user_ratings_total} reviews)
              </span>
            )}
          </div>
        )}

        {/* Price Level */}
        {priceDisplay && (
          <div className="mb-2">
            <span className="text-sm">💸 {priceDisplay}</span>
          </div>
        )}

        {/* Location */}
        {(restaurant.vicinity || restaurant.formatted_address) && (
          <div className="flex items-start gap-1 mb-2 text-sm">
            <span>📍</span>
            <span className="text-gray-700 dark:text-gray-300 line-clamp-2">
              {restaurant.vicinity || restaurant.formatted_address}
            </span>
          </div>
        )}

        {/* Cuisine Types */}
        {restaurant.types && restaurant.types.length > 0 && (
          <div className="flex items-start gap-1 mb-2 text-sm">
            <span>🍽️</span>
            <span className="text-gray-700 dark:text-gray-300">
              {restaurant.types
                .filter(type => !type.includes('_') && type !== 'establishment' && type !== 'point_of_interest')
                .slice(0, 3)
                .map(type => type.charAt(0).toUpperCase() + type.slice(1))
                .join(', ')}
            </span>
          </div>
        )}

        {/* Opening Hours */}
        {restaurant.opening_hours && restaurant.opening_hours.open_now !== undefined && (
          <div className="mt-2">
            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              restaurant.opening_hours.open_now 
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
            }`}>
              🕒 {restaurant.opening_hours.open_now ? 'Open now' : 'Closed now'}
            </span>
          </div>
        )}

        {/* View on Maps Button */}
        {restaurant.place_id && (
          <a
            href={`https://www.google.com/maps/place/?q=place_id:${restaurant.place_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 inline-flex items-center text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
          >
            View on Maps →
          </a>
        )}
      </div>
    </div>
  );
};

export default RestaurantCard;
