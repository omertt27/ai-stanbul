import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';

const AppRouter = () => {
  return (
    <Router>
      <div>
        <h1>Test - React App Loading</h1>
        <Routes>
          <Route path="/" element={<App />} />
        </Routes>
      </div>
    </Router>
  );
};

export default AppRouter;
