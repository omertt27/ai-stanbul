import React from 'react';

function Tips({ darkMode }) {
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <div className="max-w-4xl mx-auto">
        <h1>Istanbul Travel Tips & Guides</h1>
        
        <p>
          Make the most of your Istanbul experience with these insider tips, local customs, 
          and practical advice from seasoned travelers and locals.
        </p>

        <h2>🚇 Getting Around Istanbul</h2>
      <h3>Public Transportation</h3>
      <ul>
        <li><strong>Istanbul Card (Istanbulkart):</strong> Essential for all public transport. Buy at metro stations, valid for metro, bus, tram, ferry, and funicular.</li>
        <li><strong>Metro & Tram:</strong> Clean, efficient, and connects major tourist areas. Metro runs until midnight, trams until around 1 AM.</li>
        <li><strong>Ferries:</strong> Don't miss the Bosphorus ferry rides - they're both transportation and a scenic tour!</li>
        <li><strong>Dolmuş:</strong> Shared minibuses that run on fixed routes. Cheaper than taxis but can be confusing for tourists.</li>
      </ul>

      <h3>Taxis & Ride-sharing</h3>
      <ul>
        <li>Always insist the taxi meter is turned on ("Taksimetre açar mısınız?")</li>
        <li>BiTaksi and Uber operate in Istanbul</li>
        <li>Airport taxi: Use official taxi stands, expect 45-60 TL to city center</li>
      </ul>

      <h2>💰 Money & Payments</h2>
      <ul>
        <li><strong>Currency:</strong> Turkish Lira (TL). Euros and USD widely accepted but you'll get better rates paying in lira.</li>
        <li><strong>ATMs:</strong> Widely available, use bank ATMs for better rates</li>
        <li><strong>Credit Cards:</strong> Accepted in most restaurants and shops, but carry cash for small vendors</li>
        <li><strong>Tipping:</strong> 10-15% at restaurants if service charge not included. Round up for taxis.</li>
      </ul>

      <h2>🍽️ Food & Dining Tips</h2>
      <h3>Local Etiquette</h3>
      <ul>
        <li>Breakfast is important - try a traditional Turkish breakfast (kahvaltı)</li>
        <li>Lunch: 12-3 PM, Dinner: 7-10 PM (later than European times)</li>
        <li>Street food is generally safe and delicious - try döner, simit, and midye dolma</li>
        <li>Tea (çay) is offered everywhere - accepting is polite</li>
      </ul>

      <h3>Must-Try Foods</h3>
      <ul>
        <li><strong>Kebabs:</strong> Try beyti, adana, or İskender kebab</li>
        <li><strong>Seafood:</strong> Especially good along the Bosphorus</li>
        <li><strong>Sweets:</strong> Turkish delight, baklava, künefe</li>
        <li><strong>Drinks:</strong> Turkish coffee, raki (anise spirit), fresh pomegranate juice</li>
      </ul>

      <h2>🕌 Cultural Tips & Etiquette</h2>
      <h3>Mosque Visits</h3>
      <ul>
        <li>Dress modestly: cover shoulders, knees. Women should cover hair</li>
        <li>Remove shoes before entering</li>
        <li>Don't visit during prayer times (5 times daily)</li>
        <li>Free to enter, but donations appreciated</li>
      </ul>

      <h3>Social Customs</h3>
      <ul>
        <li>Handshakes are common for greetings</li>
        <li>Remove shoes when entering homes</li>
        <li>Don't point the sole of your foot at someone</li>
        <li>Turks are very hospitable - don't be surprised by generous invitations</li>
      </ul>

      <h2>🛍️ Shopping & Bargaining</h2>
      <ul>
        <li><strong>Grand Bazaar & Spice Bazaar:</strong> Bargaining expected. Start at 30-50% of asking price</li>
        <li><strong>Fixed Prices:</strong> Stores with price tags usually don't negotiate</li>
        <li><strong>Best Buys:</strong> Turkish carpets, ceramics, spices, Turkish delight, evil eye charms</li>
        <li><strong>Tax Refund:</strong> Available for purchases over 100 TL at participating stores</li>
      </ul>

      <h2>📱 Practical Information</h2>
      <h3>Communication</h3>
      <ul>
        <li><strong>WiFi:</strong> Available in most cafes, restaurants, and hotels</li>
        <li><strong>SIM Cards:</strong> Available at airport and phone shops, need passport</li>
        <li><strong>Language:</strong> Turkish is main language, but English widely spoken in tourist areas</li>
      </ul>

      <h3>Safety</h3>
      <ul>
        <li>Istanbul is generally very safe for tourists</li>
        <li>Watch for pickpockets in crowded areas like Sultanahmet and Galata Bridge</li>
        <li>Beware of aggressive carpet sellers and fake police scams</li>
        <li>Keep copies of important documents</li>
      </ul>

      <h2>⏰ Best Times to Visit</h2>
      <h3>Seasonal Guide</h3>
      <ul>
        <li><strong>Spring (April-May):</strong> Perfect weather, fewer crowds, blooming tulips</li>
        <li><strong>Summer (June-August):</strong> Hot and crowded, but great for Bosphorus activities</li>
        <li><strong>Fall (September-November):</strong> Excellent weather, fewer tourists</li>
        <li><strong>Winter (December-March):</strong> Mild, rainy, fewer crowds, cozy indoor experiences</li>
      </ul>

      <h3>Daily Planning</h3>
      <ul>
        <li>Start early to avoid crowds at major attractions</li>
        <li>Plan indoor activities for midday heat in summer</li>
        <li>Friday afternoons are busy due to prayer times</li>
        <li>Many museums closed on Mondays</li>
      </ul>

      <h2>🎯 Neighborhood Quick Guide</h2>
      <ul>
        <li><strong>Sultanahmet:</strong> Historic sites (Blue Mosque, Hagia Sophia, Topkapi)</li>
        <li><strong>Beyoğlu/Galata:</strong> Nightlife, Galata Tower, modern dining</li>
        <li><strong>Kadıköy:</strong> Local vibe, great food scene, less touristy</li>
        <li><strong>Beşiktaş:</strong> Dolmabahçe Palace, upscale shopping</li>
        <li><strong>Üsküdar:</strong> Asian side, authentic local experience</li>
      </ul>

      <div style={{ 
        marginTop: '3rem', 
        padding: '1.5rem', 
        borderRadius: '0.75rem', 
        background: darkMode ? '#1f2937' : '#f0f9ff',
        border: darkMode ? '1px solid #374151' : '1px solid #bae6fd'
      }}>
        <h2>💡 Pro Tips from Locals</h2>
        <ul>
          <li>Download offline maps before exploring</li>
          <li>Learn basic Turkish phrases - locals appreciate the effort</li>
          <li>Always carry tissue paper and hand sanitizer</li>
          <li>Try local specialties in each neighborhood</li>
          <li>Take photos at sunset from Galata Bridge or Pierre Loti Hill</li>
          <li>Book popular restaurants in advance, especially on weekends</li>
        </ul>
      </div>

      <p style={{ marginTop: '2rem', fontStyle: 'italic' }}>
        Remember, our AI assistant knows all this and more! Ask specific questions about any 
        topic covered here for personalized recommendations based on your interests and travel style. 
        Hoş geldiniz - Welcome to Istanbul! 🇹🇷
      </p>
      </div>
    </div>
  );
}

export default Tips;
