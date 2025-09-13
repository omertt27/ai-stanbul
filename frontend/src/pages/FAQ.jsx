import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import '../App.css';

function FAQ() {
  const { darkMode } = useTheme();
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  const faqs = [
    {
      question: "How accurate are the restaurant recommendations?",
      answer: "Our restaurant recommendations come directly from Google Maps API, providing real-time ratings, reviews, and location data. We combine this with AI analysis to give you the most relevant suggestions based on your preferences.",
      category: "accuracy"
    },
    {
      question: "Do you have information about all Istanbul districts?",
      answer: "Yes! We have comprehensive data about all major Istanbul districts including Sultanahmet, Beyoğlu, Kadıköy, Beşiktaş, Üsküdar, and many more. Each district entry includes historical context, main attractions, and cultural significance.",
      category: "coverage"
    },
    {
      question: "Can I get recommendations in different languages?",
      answer: "Currently, our AI responds in English, but it understands queries in multiple languages. We're working on expanding multilingual support to serve travelers from around the world better.",
      category: "features"
    },
    {
      question: "How often is the information updated?",
      answer: "Restaurant data is updated in real-time through Google Maps API. Our museum and attraction information is manually curated and updated regularly to ensure accuracy. We review and update our database monthly.",
      category: "accuracy"
    },
    {
      question: "Is this service completely free?",
      answer: "Yes! AI Istanbul Guide is completely free to use. We're funded by donations and maintain this as an open-source project to help visitors discover the best of Istanbul.",
      category: "general"
    },
    {
      question: "Can I save my favorite recommendations?",
      answer: "Currently, conversations are not saved between sessions for privacy reasons. However, we're considering adding optional account features that would allow you to save favorites while maintaining your privacy.",
      category: "features"
    },
    {
      question: "What types of questions can I ask?",
      answer: "You can ask about restaurants, museums, historical sites, neighborhoods, cultural events, transportation, shopping areas, local customs, and much more. Our AI is trained specifically for Istanbul tourism and local knowledge.",
      category: "usage"
    },
    {
      question: "Do you provide real-time information like opening hours?",
      answer: "For restaurants, we provide real-time information through Google Maps API including current opening hours. For museums and attractions, we provide general opening hours, but we recommend checking official websites for the most current information.",
      category: "accuracy"
    },
    {
      question: "Can I use this on my mobile phone?",
      answer: "Absolutely! Our website is fully responsive and optimized for mobile devices. You can access it through any web browser on your phone or tablet.",
      category: "general"
    },
    {
      question: "How can I report incorrect information?",
      answer: "If you find any incorrect information, please report it through our GitHub issues page or contact us directly. We take accuracy seriously and investigate all reports promptly.",
      category: "general"
    }
  ];

  const getCategoryColor = (category) => {
    const colors = {
      accuracy: 'from-green-500 to-emerald-600',
      coverage: 'from-blue-500 to-indigo-600',
      features: 'from-purple-500 to-pink-600',
      usage: 'from-orange-500 to-red-600',
      general: 'from-gray-500 to-gray-600'
    };
    return colors[category] || colors.general;
  };

  const getCategoryIcon = (category) => {
    const icons = {
      accuracy: '',
      coverage: '',
      features: '',
      usage: '',
      general: ''
    };
    return icons[category] || icons.general;
  };

  return (
    <div className={`min-h-screen w-full pt-16 px-4 pb-8 transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50'
    }`}>
      <div className="max-w-6xl mx-auto">
        {/* Hero Section */}
        <div className="pb-12">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 mb-6">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h1 className={`text-5xl font-bold mb-6 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Frequently Asked <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Questions</span>
            </h1>
            <p className={`text-xl leading-relaxed transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Find answers to common questions about AI Istanbul Guide
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-6 pb-20">
        {/* FAQ Items */}
        <div className="space-y-4 mb-12">
          {faqs.map((faq, index) => (
            <div key={index} className={`rounded-xl transition-all duration-300 ${
              darkMode 
                ? 'bg-gray-800 border border-gray-700 hover:border-gray-600' 
                : 'bg-white shadow-lg border border-gray-100 hover:shadow-xl'
            }`}>
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full p-6 text-left focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 rounded-xl"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-start space-x-4 flex-1">
                    <div className={`inline-flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-r ${getCategoryColor(faq.category)} flex-shrink-0`}>
                      <span className="text-white text-lg">{getCategoryIcon(faq.category)}</span>
                    </div>
                    <h3 className={`text-lg font-semibold transition-colors duration-300 ${
                      darkMode ? 'text-white' : 'text-gray-800'
                    }`}>
                      {faq.question}
                    </h3>
                  </div>
                  <div className="flex-shrink-0 ml-4">
                    <svg 
                      className={`w-6 h-6 transition-transform duration-300 ${
                        openIndex === index ? 'rotate-180' : ''
                      } ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </button>
              
              <div className={`overflow-hidden transition-all duration-300 ${
                openIndex === index ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
              }`}>
                <div className="px-6 pb-6">
                  <div className="ml-14">
                    <p className={`leading-relaxed transition-colors duration-300 ${
                      darkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      {faq.answer}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Stats */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className={`text-center p-6 rounded-xl transition-colors duration-300 ${
            darkMode 
              ? 'bg-gray-800 border border-gray-700' 
              : 'bg-white shadow-lg border border-gray-100'
          }`}>
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r from-green-500 to-emerald-600 mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className={`text-xl font-semibold mb-2 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Real-time Data
            </h3>
            <p className={`text-sm transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Live restaurant & place information via Google Maps API
            </p>
          </div>

          <div className={`text-center p-6 rounded-xl transition-colors duration-300 ${
            darkMode 
              ? 'bg-gray-800 border border-gray-700' 
              : 'bg-white shadow-lg border border-gray-100'
          }`}>
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r from-blue-500 to-indigo-600 mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0-9v9" />
              </svg>
            </div>
            <h3 className={`text-xl font-semibold mb-2 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              100% Free
            </h3>
            <p className={`text-sm transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Completely free to use, powered by community support
            </p>
          </div>

          <div className={`text-center p-6 rounded-xl transition-colors duration-300 ${
            darkMode 
              ? 'bg-gray-800 border border-gray-700' 
              : 'bg-white shadow-lg border border-gray-100'
          }`}>
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r from-purple-500 to-pink-600 mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className={`text-xl font-semibold mb-2 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Mobile Optimized
            </h3>
            <p className={`text-sm transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Perfect experience on any device, anywhere in Istanbul
            </p>
          </div>
        </div>

        {/* Still Have Questions */}
        <div className={`rounded-xl p-8 text-center transition-colors duration-300 ${
          darkMode 
            ? 'bg-gradient-to-r from-gray-800 to-gray-700' 
            : 'bg-gradient-to-r from-indigo-50 to-purple-50'
        }`}>
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 mb-6">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Still Have Questions?
          </h2>
          <p className={`text-lg mb-6 transition-colors duration-300 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Can't find the answer you're looking for? Here are some ways to get help:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <Link
              to="/"
              className={`p-4 rounded-lg transition-all duration-300 hover:scale-105 ${
                darkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 border border-gray-700' 
                  : 'bg-white hover:bg-gray-50 shadow-lg border border-gray-100'
              }`}
            >
              <div className="text-2xl mb-2">�</div>
              <h3 className={`font-semibold mb-1 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Ask Our AI
              </h3>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Get instant answers from our AI assistant
              </p>
            </Link>

            <a
              href="mailto:help@aiistanbul.guide"
              className={`p-4 rounded-lg transition-all duration-300 hover:scale-105 ${
                darkMode 
                  ? 'bg-gray-800 hover:bg-gray-700 border border-gray-700' 
                  : 'bg-white hover:bg-gray-50 shadow-lg border border-gray-100'
              }`}
            >
              <h3 className={`font-semibold mb-1 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Email Us
              </h3>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Direct support from our team
              </p>
            </a>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

export default FAQ;
