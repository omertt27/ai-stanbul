import React from 'react';

function About({ darkMode }) {
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <div className="max-w-4xl mx-auto">
        <h1>About AI Istanbul Guide</h1>
        
        <p>
          Welcome to AI Istanbul Guide, your intelligent companion for exploring the magical city of Istanbul. 
          Our AI-powered assistant combines the best of technology with local expertise to provide you with 
          personalized recommendations and insights about Turkey's cultural capital.
        </p>

        <h2>What We Offer</h2>
        <ul>
          <li><strong>Restaurant Recommendations</strong> - Real-time data from Google Maps to find the best dining experiences</li>
          <li><strong>Museum & Cultural Sites</strong> - Curated information about Istanbul's rich historical and cultural attractions</li>
          <li><strong>Neighborhood Guides</strong> - Detailed insights about districts like Beyoğlu, Fatih, Beşiktaş, and more</li>
          <li><strong>Historical Attractions</strong> - Mosques, palaces, monuments, and architectural wonders</li>
          <li><strong>Local Insights</strong> - Hidden gems and authentic experiences beyond tourist hotspots</li>
        </ul>

        <h2>Our Mission</h2>
        <p>
          We believe that every visitor to Istanbul deserves to experience the city like a local. Our AI assistant 
          is trained to understand your preferences and provide personalized recommendations that match your interests, 
          whether you're interested in Ottoman history, Turkish cuisine, Byzantine architecture, or modern art.
        </p>

        <h2>Why Choose AI Istanbul Guide?</h2>
        <ul>
          <li><strong>Personalized Experience</strong> - Tailored recommendations based on your interests</li>
          <li><strong>Real-Time Information</strong> - Up-to-date restaurant data and opening hours</li>
          <li><strong>Local Expertise</strong> - Curated content from Istanbul locals and cultural experts</li>
          <li><strong>24/7 Availability</strong> - Get help planning your Istanbul adventure anytime</li>
          <li><strong>No Hidden Costs</strong> - Completely free to use with no registration required</li>
        </ul>

        <h2>The Technology</h2>
        <p>
          Our platform combines artificial intelligence with carefully curated local data. Restaurant recommendations 
          come from Google Maps API for real-time accuracy, while our cultural and historical content is manually 
          curated by local experts to ensure authenticity and depth.
        </p>

        <h2>Start Your Journey</h2>
        <p>
          Ready to explore Istanbul? Head back to our chat interface and ask about anything that interests you - 
          from the best Turkish breakfast spots to Byzantine churches, from rooftop bars with Bosphorus views to 
          traditional hammams. Your Istanbul adventure begins with a simple question!
        </p>
      </div>
    </div>
  );
}

export default About;
