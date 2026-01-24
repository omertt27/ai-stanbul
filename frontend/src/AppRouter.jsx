import React, { useState, useEffect, lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import CookieConsent from './components/CookieConsent';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

// Critical pages - loaded immediately
import App from "./App";
import Chatbot from './Chatbot';

// Lazy-loaded pages - loaded on demand
const About = lazy(() => import("./pages/About"));
const Contact = lazy(() => import("./pages/Contact"));
const FAQ = lazy(() => import("./pages/FAQ"));
const Donate = lazy(() => import("./pages/Donate"));
const OfflineSettings = lazy(() => import("./pages/OfflineSettings"));
const BlogList = lazy(() => import("./pages/BlogList"));
const BlogPost = lazy(() => import("./pages/BlogPost"));
const BlogAdmin = lazy(() => import("./pages/BlogAdmin"));
const NewBlogPostPro = lazy(() => import("./pages/NewBlogPostPro"));
const IntelligentRoutePlanner = lazy(() => import("./pages/IntelligentRoutePlanner"));
const LLMAnalyticsDashboard = lazy(() => import("./components/LLMAnalyticsDashboard"));
const AdminDashboard = lazy(() => import("./pages/AdminDashboard"));
const GDPR = lazy(() => import("./pages/GDPR"));
const Sources = lazy(() => import("./pages/Sources"));
const TermsOfService = lazy(() => import("./pages/TermsOfService"));
const Privacy = lazy(() => import("./pages/Privacy"));

// Loading fallback component
const PageLoader = () => (
  <div style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '60vh',
    flexDirection: 'column',
    gap: '1rem'
  }}>
    <div style={{
      width: '50px',
      height: '50px',
      border: '4px solid #f3f3f3',
      borderTop: '4px solid #3498db',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite'
    }}></div>
    <p style={{ color: '#666', fontSize: '0.9rem' }}>Loading page...</p>
    <style>{`
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `}</style>
  </div>
);

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
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/chat" element={
              <ErrorBoundary>
                <Chatbot />
              </ErrorBoundary>
            } />
            <Route path="/chatbot" element={
              <ErrorBoundary>
                <Chatbot />
              </ErrorBoundary>
            } />
            <Route path="/route-planner" element={<IntelligentRoutePlanner />} />
            <Route path="/analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/llm-analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/admin" element={<AdminDashboard />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/donate" element={<Donate />} />
            <Route path="/blog" element={<BlogList />} />
            <Route path="/blog/admin" element={<BlogAdmin />} />
            <Route path="/blog/new" element={<NewBlogPostPro />} />
            <Route path="/blog/edit/:id" element={<NewBlogPostPro />} />
            <Route path="/blog/:id" element={<BlogPost />} />
            <Route path="/offline-settings" element={<OfflineSettings />} />
            <Route path="/gdpr" element={<GDPR />} />
            <Route path="/sources" element={<Sources />} />
            <Route path="/terms" element={<TermsOfService />} />
            <Route path="/privacy" element={<Privacy />} />
          </Routes>
        </Suspense>
        </main>
        <Footer />
        <CookieConsent />
      </div>
  );
}

function AppRouter() {
  return (
    <Router
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <AppRouterContent />
    </Router>
  );
}

export default AppRouter;
