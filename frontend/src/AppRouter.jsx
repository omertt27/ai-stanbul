import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import App from "./App";
import Chatbot from './Chatbot';
import About from "./pages/About";
import Contact from "./pages/Contact";
import FAQ from "./pages/FAQ";
import Donate from "./pages/Donate";
import OfflineSettings from "./pages/OfflineSettings";
import BlogList from "./pages/BlogList";
import BlogPost from "./pages/BlogPost";
import NewBlogPost from "./pages/NewBlogPost";
import IntelligentRoutePlanner from "./pages/IntelligentRoutePlanner";
import LLMAnalyticsDashboard from "./components/LLMAnalyticsDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import GDPR from "./pages/GDPR";
import Sources from "./pages/Sources";
import TermsOfService from "./pages/TermsOfService";
import Privacy from "./pages/Privacy";
import './App.css';

function AppRouterContent() {
  const location = useLocation();
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  
  // Update window width on resize
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  const isMobile = windowWidth < 768;
  
  // Hide NavBar on mobile for chat pages to maximize screen space
  const hideNavBar = isMobile && (
    location.pathname === '/chat' || 
    location.pathname === '/chatbot'
  );
  
  return (
    <div className="min-h-screen flex flex-col">
      {!hideNavBar && <NavBar />}
      <main className="flex-1">
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/chat" element={<Chatbot />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route path="/route-planner" element={<IntelligentRoutePlanner />} />
            <Route path="/analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/llm-analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/admin" element={<AdminDashboard />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/donate" element={<Donate />} />
            <Route path="/blog" element={<BlogList />} />
            <Route path="/blog/new" element={<NewBlogPost />} />
            <Route path="/blog/:id" element={<BlogPost />} />
            <Route path="/offline-settings" element={<OfflineSettings />} />
            <Route path="/gdpr" element={<GDPR />} />
            <Route path="/sources" element={<Sources />} />
            <Route path="/terms" element={<TermsOfService />} />
            <Route path="/privacy" element={<Privacy />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </div>
  );
}

function AppRouter() {
  return (
    <Router>
      <AppRouterContent />
    </Router>
  );
}

export default AppRouter;
