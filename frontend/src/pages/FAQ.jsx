import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import '../App.css';

function FAQ() {
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




  return (
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900" style={{ marginTop: '0px', paddingLeft: '1.5rem', paddingRight: '1.5rem', paddingBottom: '3rem' }}>
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-24">
        {/* Hero Section */}
        <div className="pb-16">
        <div className="max-w-4xl mx-auto px-8 text-center">
          <div className="mb-8">
            <h1 className="text-5xl font-bold mb-6 pt-28 transition-colors duration-300 text-white">
              Frequently Asked <span className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Questions</span>
            </h1>
            <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
              Find answers to common questions about AI Istanbul Guide
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-8 pb-24">
        {/* FAQ Items */}
        <div className="space-y-6 mb-16">
          {faqs.map((faq, index) => (
            <div key={index} className="rounded-xl transition-all duration-300 bg-gray-800 border border-gray-700 hover:border-gray-600">
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full p-8 text-left focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 rounded-xl"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-start flex-1">
                    <h3 className="text-lg font-semibold transition-colors duration-300 text-white">
                      {faq.question}
                    </h3>
                  </div>
                  <div className="flex-shrink-0 ml-4">
                    <svg 
                      className={`w-6 h-6 transition-transform duration-300 ${
                        openIndex === index ? 'rotate-180' : ''
                      } text-gray-400`} 
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
                <div className="px-8 pb-8">
                  <div className="">
                    <p className="leading-relaxed transition-colors duration-300 text-gray-300">
                      {faq.answer}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Stats */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="text-center p-8 rounded-xl transition-colors duration-300 bg-gray-800 border border-gray-700">
            <h3 className="text-xl font-semibold mb-2 transition-colors duration-300 text-white">
              Real-time Data
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              Live restaurant & place information via Google Maps API
            </p>
          </div>

          <div className="text-center p-8 rounded-xl transition-colors duration-300 bg-gray-800 border border-gray-700">
            <h3 className="text-xl font-semibold mb-2 transition-colors duration-300 text-white">
              100% Free
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              Completely free to use, powered by community support
            </p>
          </div>

          <div className="text-center p-8 rounded-xl transition-colors duration-300 bg-gray-800 border border-gray-700">
            <h3 className="text-xl font-semibold mb-2 transition-colors duration-300 text-white">
              Mobile Optimized
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              Perfect experience on any device, anywhere in Istanbul
            </p>
          </div>
        </div>

        {/* Still Have Questions */}
        <div className="rounded-xl p-12 text-center transition-colors duration-300 bg-gradient-to-r from-gray-800 to-gray-700">
          <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-white">
            Still Have Questions?
          </h2>
          <p className="text-lg mb-6 transition-colors duration-300 text-gray-300">
            Can't find the answer you're looking for? Here are some ways to get help:
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <Link
              to="/"
              className="p-4 rounded-lg transition-all duration-300 hover:scale-105 bg-gray-800 hover:bg-gray-700 border border-gray-700"
            >
              <h3 className="font-semibold mb-1 transition-colors duration-300 text-white">
                Ask Our AI
              </h3>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                Get instant answers from our AI assistant
              </p>
            </Link>

            <a
              href="mailto:help@aiistanbul.guide"
              className="p-4 rounded-lg transition-all duration-300 hover:scale-105 bg-gray-800 hover:bg-gray-700 border border-gray-700"
            >
              <h3 className="font-semibold mb-1 transition-colors duration-300 text-white">
                Email Us
              </h3>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                Direct support from our team
              </p>
            </a>
          </div>
        </div>
      </div>
      </div>
      </div>
    </div>
  );
}

export default FAQ;
