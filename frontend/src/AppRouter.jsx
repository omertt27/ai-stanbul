import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
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
import './App.css';

function AppRouter() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <NavBar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<App />} />
            <Route path="/chat" element={<Chatbot />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route path="/route-planner" element={<IntelligentRoutePlanner />} />
            <Route path="/analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/llm-analytics" element={<LLMAnalyticsDashboard />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/faq" element={<FAQ />} />
            <Route path="/donate" element={<Donate />} />
            <Route path="/blog" element={<BlogList />} />
            <Route path="/blog/new" element={<NewBlogPost />} />
            <Route path="/blog/:id" element={<BlogPost />} />
            <Route path="/offline-settings" element={<OfflineSettings />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default AppRouter;
