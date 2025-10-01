import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import '../App.css';

function FAQ() {
  const { t } = useTranslation();
  const [openIndex, setOpenIndex] = useState(null);

  const toggleFAQ = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  const faqs = [
    {
      question: t('faq.questions.whatIsKam.question'),
      answer: t('faq.questions.whatIsKam.answer'),
      category: "general"
    },
    {
      question: t('faq.questions.accuracy.question'),
      answer: t('faq.questions.accuracy.answer'),
      category: "accuracy"
    },
    {
      question: t('faq.questions.coverage.question'),
      answer: t('faq.questions.coverage.answer'),
      category: "coverage"
    },
    {
      question: t('faq.questions.updates.question'),
      answer: t('faq.questions.updates.answer'),
      category: "accuracy"
    },
    {
      question: t('faq.questions.free.question'),
      answer: t('faq.questions.free.answer'),
      category: "general"
    }
  ];

  return (
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900 mobile-scroll-optimized" style={{ marginTop: '0px', paddingLeft: '1.5rem', paddingRight: '1.5rem', paddingBottom: '3rem' }}>
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-24">
        {/* Hero Section */}
        <div className="pb-16">
        <div className="max-w-4xl mx-auto px-8 text-center">
          <div className="mb-8">
            <h1 className="text-5xl font-bold mb-6 pt-28 transition-colors duration-300 text-white">
              {t('faq.title')}
            </h1>
            <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
              {t('faq.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-8 pb-24">
        {/* FAQ Items */}
        <div className="space-y-6 mb-16">
          {faqs.map((faq, index) => (
            <div key={index} className="rounded-xl transition-all duration-300 bg-gray-800 border border-gray-700 hover:border-gray-600 mobile-touch-optimized">
              <button
                onClick={() => toggleFAQ(index)}
                className="w-full p-8 text-left focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 rounded-xl mobile-touch-optimized"
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
              {t('faq.stats.realTimeData.title')}
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              {t('faq.stats.realTimeData.description')}
            </p>
          </div>

          <div className="text-center p-8 rounded-xl transition-colors duration-300 bg-gray-800 border border-gray-700">
            <h3 className="text-xl font-semibold mb-2 transition-colors duration-300 text-white">
              {t('faq.stats.free.title')}
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              {t('faq.stats.free.description')}
            </p>
          </div>

          <div className="text-center p-8 rounded-xl transition-colors duration-300 bg-gray-800 border border-gray-700">
            <h3 className="text-xl font-semibold mb-2 transition-colors duration-300 text-white">
              {t('faq.stats.mobileOptimized.title')}
            </h3>
            <p className="text-sm transition-colors duration-300 text-gray-300">
              {t('faq.stats.mobileOptimized.description')}
            </p>
          </div>
        </div>

        {/* Still Have Questions */}
        <div className="rounded-xl p-12 text-center transition-colors duration-300 bg-gradient-to-r from-gray-800 to-gray-700">
          <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-white">
            {t('faq.stillHaveQuestions.title')}
          </h2>
          <p className="text-lg mb-6 transition-colors duration-300 text-gray-300">
            {t('faq.stillHaveQuestions.subtitle')}
          </p>
          
          <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
            <Link
              to="/chat"
              className="p-4 rounded-lg transition-all duration-300 hover:scale-105 bg-gray-800 hover:bg-gray-700 border border-gray-700"
            >
              <h3 className="font-semibold mb-1 transition-colors duration-300 text-white">
                {t('faq.stillHaveQuestions.askAI.title')}
              </h3>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                {t('faq.stillHaveQuestions.askAI.description')}
              </p>
            </Link>

            <a
              href="mailto:omertahtaci@aistanbul.net"
              className="p-4 rounded-lg transition-all duration-300 hover:scale-105 bg-gray-800 hover:bg-gray-700 border border-gray-700"
            >
              <h3 className="font-semibold mb-1 transition-colors duration-300 text-white">
                {t('faq.stillHaveQuestions.emailUs.title')}
              </h3>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                {t('faq.stillHaveQuestions.emailUs.description')}
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
