import React from 'react';
import { useTheme } from '../contexts/ThemeContext';
import GDPRDataManager from '../components/GDPRDataManager';

const GDPRPage = () => {
  const { darkMode } = useTheme();

  return (
    <div className="chatbot-background min-h-screen transition-colors duration-300" style={{ paddingTop: '6rem' }}>
      <div className="container mx-auto px-4 py-8">
        <GDPRDataManager />
      </div>
    </div>
  );
};

export default GDPRPage;
