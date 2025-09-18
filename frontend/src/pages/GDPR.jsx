import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
import GDPRDataManager from '../components/GDPRDataManager';

const GDPRPage = () => {
  const { darkMode } = useTheme();

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-blue-50 to-indigo-100'
    }`}>
      <div className="container mx-auto px-4 py-8">
        <GDPRDataManager />
      </div>
    </div>
  );
};

export default GDPRPage;
