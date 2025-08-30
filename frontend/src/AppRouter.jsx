import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import App from './App';
import About from './pages/About';
import Sources from './pages/Sources';
import Donate from './pages/Donate';
import NavBar from './components/NavBar';

const AppRouter = () => (
  <Router>
    {/* Chatbot outline for all pages */}
    <div className="chatbot-outline"></div>
    <NavBar />
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/about" element={<About />} />
      <Route path="/sources" element={<Sources />} />
      <Route path="/donate" element={<Donate />} />
    </Routes>
  </Router>
);

export default AppRouter;
