/**
 * Quick Integration: Add Route Maps to Main Chat
 * =============================================
 * 
 * Add these changes to integrate LeafletNavigationMap into Chatbot.jsx
 */

// ============================================
// STEP 1: Add Import (near top of Chatbot.jsx)
// ============================================

import LeafletNavigationMap from './components/LeafletNavigationMap';


// ============================================
// STEP 2: Add Route Rendering to Messages
// ============================================

// Find the message rendering section (around line 1600-1700)
// Add this AFTER the message content, BEFORE ChatRouteIntegration:

{/* Inline Route Map - Show when backend returns route data */}
{msg.metadata?.route_data && (
  <div style={{
    marginTop: '1rem',
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
  }}>
    <LeafletNavigationMap
      routeData={msg.metadata.route_data}
      pois={msg.metadata.pois || []}
      userLocation={msg.metadata.user_location || null}
      height="400px"
    />
    
    {/* Route Summary */}
    {msg.metadata.route_data.distance && (
      <div style={{
        padding: '1rem',
        background: darkMode ? '#374151' : '#f9fafb',
        display: 'flex',
        justifyContent: 'space-around',
        fontSize: '0.9rem',
        borderTop: '1px solid #e5e7eb'
      }}>
        <div>
          <strong>📍 Distance:</strong> {msg.metadata.route_data.distance}
        </div>
        <div>
          <strong>⏱️ Duration:</strong> {msg.metadata.route_data.duration}
        </div>
        {msg.metadata.route_data.mode && (
          <div>
            <strong>🚶 Mode:</strong> {msg.metadata.route_data.mode}
          </div>
        )}
      </div>
    )}
  </div>
)}

{/* POI/Museum Cards - Rich information display */}
{msg.metadata?.pois && msg.metadata.pois.length > 0 && (
  <div style={{
    marginTop: '1rem',
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1rem'
  }}>
    {msg.metadata.pois.map((poi, poiIndex) => (
      <div
        key={poiIndex}
        style={{
          background: darkMode ? '#374151' : 'white',
          borderRadius: '8px',
          padding: '1rem',
          borderLeft: '3px solid #2196F3',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}
      >
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem' }}>
          {poi.name}
        </h4>
        
        {poi.category && (
          <div style={{
            fontSize: '0.75rem',
            textTransform: 'uppercase',
            color: '#666',
            marginBottom: '0.5rem'
          }}>
            {poi.category}
          </div>
        )}
        
        {/* Details */}
        {(poi.visiting_duration || poi.entrance_fee || poi.distance) && (
          <div style={{
            display: 'flex',
            gap: '0.75rem',
            marginBottom: '0.75rem',
            fontSize: '0.875rem',
            flexWrap: 'wrap'
          }}>
            {poi.visiting_duration && (
              <span>⏱️ {poi.visiting_duration}</span>
            )}
            {poi.entrance_fee && (
              <span>💰 {poi.entrance_fee}</span>
            )}
            {poi.distance && (
              <span>📍 {typeof poi.distance === 'number' 
                ? `${poi.distance.toFixed(1)} km` 
                : poi.distance}
              </span>
            )}
          </div>
        )}
        
        {/* Highlights */}
        {poi.highlights && poi.highlights.length > 0 && (
          <div style={{ marginBottom: '0.75rem' }}>
            <strong style={{ fontSize: '0.875rem' }}>✨ Must-See:</strong>
            <ul style={{
              margin: '0.25rem 0 0 0',
              paddingLeft: '1.25rem',
              fontSize: '0.875rem'
            }}>
              {poi.highlights.slice(0, 3).map((highlight, hIndex) => (
                <li key={hIndex}>{highlight}</li>
              ))}
            </ul>
          </div>
        )}
        
        {/* Local Tips */}
        {poi.local_tips && poi.local_tips.length > 0 && (
          <div style={{
            background: 'rgba(33, 150, 243, 0.1)',
            borderRadius: '6px',
            padding: '0.75rem',
            marginTop: '0.75rem'
          }}>
            <strong style={{ fontSize: '0.875rem', color: '#2196F3' }}>
              💡 Local Tips:
            </strong>
            <ul style={{
              margin: '0.5rem 0 0 0',
              paddingLeft: '1.25rem',
              fontSize: '0.875rem'
            }}>
              {poi.local_tips.map((tip, tIndex) => (
                <li key={tIndex}>{tip}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    ))}
  </div>
)}

{/* District Information */}
{msg.metadata?.district_info && (
  <div style={{
    marginTop: '1rem',
    background: darkMode ? '#374151' : 'white',
    borderRadius: '8px',
    padding: '1rem',
    borderLeft: '3px solid #FF9800',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
  }}>
    <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem' }}>
      📍 {msg.metadata.district_info.name}
    </h4>
    
    {msg.metadata.district_info.description && (
      <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.9rem' }}>
        {msg.metadata.district_info.description}
      </p>
    )}
    
    {msg.metadata.district_info.best_time && (
      <div style={{ marginBottom: '0.75rem', fontSize: '0.875rem' }}>
        <strong>⏰ Best Time to Visit:</strong> {msg.metadata.district_info.best_time}
      </div>
    )}
    
    {msg.metadata.district_info.local_tips && msg.metadata.district_info.local_tips.length > 0 && (
      <div style={{
        background: 'rgba(255, 152, 0, 0.1)',
        borderRadius: '6px',
        padding: '0.75rem',
        marginTop: '0.75rem'
      }}>
        <strong style={{ fontSize: '0.875rem', color: '#FF9800' }}>
          💡 Insider Tips:
        </strong>
        <ul style={{
          margin: '0.5rem 0 0 0',
          paddingLeft: '1.25rem',
          fontSize: '0.875rem'
        }}>
          {msg.metadata.district_info.local_tips.map((tip, tIndex) => (
            <li key={tIndex}>{tip}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
)}

{/* Itinerary Timeline */}
{msg.metadata?.total_itinerary && (
  <div style={{
    marginTop: '1rem',
    background: darkMode ? '#374151' : 'white',
    borderRadius: '8px',
    padding: '1rem',
    borderLeft: '3px solid #4CAF50',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
  }}>
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '1rem'
    }}>
      <h4 style={{ margin: 0, fontSize: '1rem' }}>
        🗺️ Your Optimized Itinerary
      </h4>
      <div style={{ display: 'flex', gap: '1rem', fontSize: '0.875rem' }}>
        {msg.metadata.total_itinerary.total_distance && (
          <span>🚶 {msg.metadata.total_itinerary.total_distance}</span>
        )}
        {msg.metadata.total_itinerary.total_time && (
          <span>⏱️ {msg.metadata.total_itinerary.total_time}</span>
        )}
      </div>
    </div>
    
    {msg.metadata.total_itinerary.recommended_breaks && msg.metadata.total_itinerary.recommended_breaks.length > 0 && (
      <div>
        <strong style={{ fontSize: '0.875rem', color: '#4CAF50' }}>
          🧃 Recommended Breaks:
        </strong>
        <div style={{ marginTop: '0.5rem' }}>
          {msg.metadata.total_itinerary.recommended_breaks.map((brk, bIndex) => (
            <div
              key={bIndex}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '0.5rem',
                background: 'rgba(76, 175, 80, 0.1)',
                borderRadius: '4px',
                marginTop: bIndex > 0 ? '0.5rem' : 0,
                fontSize: '0.875rem'
              }}
            >
              <span style={{ fontWeight: '500' }}>{brk.location}</span>
              <span style={{ color: '#666' }}>{brk.activity}</span>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
)}


// ============================================
// STEP 3: Example Backend Response Format
// ============================================

/**
 * Your backend should return metadata like this:
 * 
 * {
 *   "response": "Here's a perfect museum tour route in Sultanahmet...",
 *   "intent": "route_planning_museums",
 *   "metadata": {
 *     "route_data": {
 *       "route": [[41.0086, 28.9802], [41.0115, 28.9833], ...],
 *       "startLocation": [41.0086, 28.9802],
 *       "endLocation": [41.0115, 28.9833],
 *       "distance": "3.2 km",
 *       "duration": "4-5 hours",
 *       "mode": "walking"
 *     },
 *     "pois": [
 *       {
 *         "name": "Hagia Sophia",
 *         "category": "Museum",
 *         "coordinates": [41.0086, 28.9802],
 *         "visiting_duration": "45-60 minutes",
 *         "entrance_fee": "Free (mosque)",
 *         "highlights": [
 *           "Byzantine mosaics",
 *           "Massive dome (31m diameter)",
 *           "Deesis Mosaic"
 *         ],
 *         "local_tips": [
 *           "Visit early morning to avoid crowds",
 *           "Dress modestly (shoulders and knees covered)",
 *           "Remove shoes before entering"
 *         ]
 *       },
 *       {
 *         "name": "Topkapi Palace",
 *         "category": "Museum",
 *         "coordinates": [41.0115, 28.9833],
 *         "visiting_duration": "2-3 hours",
 *         "entrance_fee": "₺200",
 *         "distance": 0.5,
 *         "highlights": [
 *           "Imperial Treasury",
 *           "Harem quarters",
 *           "Bosphorus views"
 *         ],
 *         "local_tips": [
 *           "Buy tickets online to skip queues",
 *           "Closed on Tuesdays",
 *           "Harem requires separate ticket"
 *         ]
 *       }
 *     ],
 *     "district_info": {
 *       "name": "Sultanahmet",
 *       "description": "Historic peninsula, heart of Old Istanbul",
 *       "best_time": "Early morning (8-10 AM) or late afternoon",
 *       "local_tips": [
 *         "Many museums closed on Mondays",
 *         "Tram stop: Sultanahmet (T1 line)",
 *         "Wear comfortable walking shoes"
 *       ]
 *     },
 *     "total_itinerary": {
 *       "total_time": "4-5 hours",
 *       "total_distance": "3.2 km",
 *       "recommended_breaks": [
 *         {
 *           "location": "Sultanahmet Square",
 *           "activity": "Turkish coffee break"
 *         },
 *         {
 *           "location": "Gulhane Park",
 *           "activity": "Scenic rest with Bosphorus view"
 *         }
 *       ]
 *     }
 *   }
 * }
 */


// ============================================
// STEP 4: Test Queries
// ============================================

/**
 * Try these queries in your chatbot:
 * 
 * 1. "Show me museums in Sultanahmet with a route"
 * 2. "Plan a museum tour in the old city"
 * 3. "I want to explore Beyoğlu district"
 * 4. "Best route for visiting 3 museums"
 * 5. "Museums near Taksim Square"
 * 6. "Walking tour of historical sites"
 * 
 * These should trigger the navigation backend and return
 * metadata with route_data, pois, district_info, etc.
 */


// ============================================
// STEP 5: Styling (Optional)
// ============================================

/**
 * Add to App.css for better dark mode support:
 * 
 * .rich-navigation-message {
 *   --card-bg: #f9fafb;
 *   --text-primary: #1f2937;
 * }
 * 
 * .dark .rich-navigation-message {
 *   --card-bg: #374151;
 *   --text-primary: #e5e7eb;
 * }
 */
