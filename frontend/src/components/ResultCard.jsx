import React from 'react';

const ResultCard = ({ title, description, onClick }) => (
  <div className="result-card border rounded p-4 mb-2 shadow hover:shadow-lg transition cursor-pointer" onClick={onClick}>
    <h2 className="text-lg font-bold mb-1">{title}</h2>
    <p className="text-gray-700">{description}</p>
  </div>
);

export default ResultCard;
