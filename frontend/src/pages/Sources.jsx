import React from 'react';
import { Link } from 'react-router-dom';
import '../App.css';

const Sources = () => (
  <div className="static-page">
    {/* AI Istanbul Logo - Top Left */}
    <Link to="/" style={{textDecoration: 'none'}} className="fixed z-50">
      <div className="logo-istanbul logo-move-top-left">
        <span className="logo-text">
          A/<span style={{fontWeight: 400}}>STANBUL</span>
        </span>
      </div>
    </Link>
    
    <h1>Sources & Technology</h1>
    <h2>Restaurant Data</h2>
    <ul>
      <li>Google Maps Places API for live restaurant and place data</li>
      <li>OpenAI for conversational AI</li>
      <li>Official Istanbul tourism and municipality resources</li>
      <li>Community recommendations and user feedback</li>
    </ul>
    <h2>Cultural & Historical Data</h2>
    <ul>
      <li>Istanbul Metropolitan Municipality — Official tourism data</li>
      <li>Turkish Ministry of Culture and Tourism — Cultural heritage information</li>
      <li>Local Museums — Direct partnerships for accurate details</li>
      <li>UNESCO World Heritage Sites — Historical significance data</li>
      <li>Local Cultural Experts — Verified insights and recommendations</li>
    </ul>
    <h2>Technology Stack</h2>
    <ul>
      <li>React, Vite, Tailwind CSS for the frontend</li>
      <li>FastAPI, SQLAlchemy, OpenAI, Google Maps API for the backend</li>
    </ul>
  </div>
);

export default Sources;
