import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
import GDPRDataManager from '../components/GDPRDataManager';

const GDPRPage = () => {
  const { darkMode } = useTheme();

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      darkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'
    }`} style={{ paddingTop: '6rem', paddingBottom: '2rem' }}>
      <div className="container mx-auto px-4 py-8">
        <GDPRDataManager />
      </div>
    </div>
  );
};

export default GDPRPage;