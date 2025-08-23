import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import Chat from './components/Chat';
import ResultCard from './components/ResultCard';
import MapView from './components/MapView';
import { fetchResults } from './api/api';
import './App.css';

const App = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [messages, setMessages] = useState([
    { user: 'KAM', text: 'hoşgeldin Başkan' }
  ]);

  useEffect(() => {
    const audio = new Audio('/welcome.mp3');
    audio.volume = 0
    audio.play();
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    setMessages([...messages, { user: 'You', text: query }]);
    try {
      const data = await fetchResults(query);
      setResults(data.results || []);
    } catch (err) {
      setResults([]);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-6">
      <h1 className="text-3xl font-bold text-center">AI-Stanbul</h1>
      <SearchBar value={query} onChange={e => setQuery(e.target.value)} onSubmit={handleSearch} />
      <Chat messages={messages} />
      <div>
        {results.map((res, idx) => (
          <ResultCard key={idx} title={res.title} description={res.description} />
        ))}
      </div>
      {/* Pass a default locations prop to MapView for testing */}
      <MapView locations={[{ lat: 51.505, lng: -0.09, label: 'Default Location' }]} />
    </div>
  );
};

export default App;
