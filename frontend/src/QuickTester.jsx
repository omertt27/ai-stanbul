import React, { useState } from 'react';
import { testInputs } from './test-inputs';

const QuickTester = ({ onTestInput }) => {
  const [selectedCategory, setSelectedCategory] = useState('restaurants');
  const [isVisible, setIsVisible] = useState(false);

  const categorizedTests = {
    restaurants: testInputs.filter(t => t.category === "Restaurants").slice(0, 5),
    attractions: testInputs.filter(t => t.category === "Attractions").slice(0, 5),
    districts: testInputs.filter(t => t.category === "Districts").slice(0, 5),
    transportation: testInputs.filter(t => t.category === "Transportation").slice(0, 3),
    culture: testInputs.filter(t => t.category === "Culture").slice(0, 3),
    practical: testInputs.filter(t => t.category === "Practical").slice(0, 3)
  };

  if (!isVisible) {
    return (
      <button
        onClick={() => setIsVisible(true)}
        className="fixed bottom-4 right-4 z-40 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg shadow-lg transition-all duration-200 text-sm"
      >
        🧪 Quick Test
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 w-80 bg-gray-800 border border-gray-600 rounded-lg shadow-xl">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Quick Test Chatbot</h3>
          <button
            onClick={() => setIsVisible(false)}
            className="text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>

        <div className="mb-3">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full p-2 bg-gray-700 text-white rounded border border-gray-600 text-sm"
          >
            <option value="restaurants">Restaurants</option>
            <option value="attractions">Attractions</option>
            <option value="districts">Districts</option>
            <option value="transportation">Transportation</option>
            <option value="culture">Culture</option>
            <option value="practical">Practical</option>
          </select>
        </div>

        <div className="space-y-2 max-h-60 overflow-y-auto">
          {categorizedTests[selectedCategory]?.map((test) => (
            <button
              key={test.id}
              onClick={() => {
                onTestInput(test.input);
                setIsVisible(false);
              }}
              className="w-full text-left p-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            >
              <div className="font-medium text-xs text-gray-300 mb-1">
                {test.category}
              </div>
              <div className="text-white text-sm leading-tight">
                {test.input}
              </div>
            </button>
          ))}
        </div>

        <div className="mt-4 pt-3 border-t border-gray-600">
          <a
            href="/test-chatbot"
            className="block w-full text-center px-3 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded text-sm transition-colors"
          >
            Full Test Suite (50 tests)
          </a>
        </div>
      </div>
    </div>
  );
};

export default QuickTester;
