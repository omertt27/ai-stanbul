// API utility that works for both local and deployed environments
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/ai';

export const fetchResults = async (query) => {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) throw new Error('API error');
  return response.json();
};
