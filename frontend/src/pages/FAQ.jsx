import React from 'react';

function FAQ({ darkMode }) {
  const faqs = [
    {
      question: "How accurate are the restaurant recommendations?",
      answer: "Our restaurant recommendations come directly from Google Maps API, providing real-time ratings, reviews, and location data. We combine this with AI analysis to give you the most relevant suggestions based on your preferences."
    },
    {
      question: "Do you have information about all Istanbul districts?",
      answer: "Yes! We have comprehensive data about all major Istanbul districts including Sultanahmet, Beyoğlu, Kadıköy, Beşiktaş, Üsküdar, and many more. Each district entry includes historical context, main attractions, and cultural significance."
    },
    {
      question: "Can I get recommendations in different languages?",
      answer: "Currently, our AI responds in English, but it understands queries in multiple languages. We're working on expanding multilingual support to serve travelers from around the world better."
    },
    {
      question: "How often is the information updated?",
      answer: "Restaurant data is updated in real-time through Google Maps API. Our museum and attraction information is manually curated and updated regularly to ensure accuracy. We review and update our database monthly."
    },
    {
      question: "Is this service completely free?",
      answer: "Yes! AI Istanbul Guide is completely free to use. We're funded by donations and maintain this as an open-source project to help visitors discover the best of Istanbul."
    },
    {
      question: "Can I save my favorite recommendations?",
      answer: "Currently, conversations are not saved between sessions for privacy reasons. However, we're considering adding optional account features that would allow you to save favorites while maintaining your privacy."
    },
    {
      question: "What types of questions can I ask?",
      answer: "You can ask about restaurants, museums, historical sites, neighborhoods, cultural events, transportation, shopping areas, local customs, and much more. Our AI is trained specifically for Istanbul tourism and local knowledge."
    },
    {
      question: "Do you provide real-time information like opening hours?",
      answer: "For restaurants, we provide real-time information through Google Maps API including current opening hours. For museums and attractions, we provide general opening hours, but we recommend checking official websites for the most current information."
    },
    {
      question: "Can I use this on my mobile phone?",
      answer: "Absolutely! Our website is fully responsive and optimized for mobile devices. You can access it through any web browser on your phone or tablet."
    },
    {
      question: "How can I report incorrect information?",
      answer: "If you find any incorrect information, please report it through our GitHub issues page or contact us directly. We take accuracy seriously and investigate all reports promptly."
    }
  ];

  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <div className="max-w-4xl mx-auto">
        <h1>Frequently Asked Questions</h1>
        
        <p>
          Find answers to common questions about AI Istanbul Guide. If you don't find what you're 
          looking for, feel free to ask our AI assistant or contact us directly.
        </p>

      <div style={{ marginTop: '2rem' }}>
        {faqs.map((faq, index) => (
          <div key={index} style={{ marginBottom: '2rem', borderBottom: darkMode ? '1px solid #374151' : '1px solid #e5e7eb', paddingBottom: '1.5rem' }}>
            <h3 style={{ 
              fontSize: '1.125rem', 
              fontWeight: '600', 
              marginBottom: '0.75rem',
              color: darkMode ? '#f9fafb' : '#1f2937'
            }}>
              {faq.question}
            </h3>
            <p style={{ 
              color: darkMode ? '#d1d5db' : '#4b5563',
              lineHeight: '1.6'
            }}>
              {faq.answer}
            </p>
          </div>
        ))}
      </div>

      <div style={{ 
        marginTop: '3rem', 
        padding: '1.5rem', 
        borderRadius: '0.75rem', 
        background: darkMode ? '#1f2937' : '#f9fafb',
        border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb'
      }}>
        <h2>Still Have Questions?</h2>
        <p style={{ marginBottom: '1rem' }}>
          Can't find the answer you're looking for? Here are some ways to get help:
        </p>
        <ul>
          <li>Ask our AI assistant directly in the chat</li>
          <li>Check our <a href="#" onClick={() => window.location.reload()}>Tips & Guides</a> page</li>
          <li>Contact us through GitHub Issues</li>
          <li>Email us at help@aiistanbul.guide</li>
        </ul>
      </div>
    </div>
    </div>
  );
}

export default FAQ;
