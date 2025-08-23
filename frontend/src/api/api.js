// Example API utility
export const fetchResults = async (query) => {
  const response = await fetch('http://localhost:8000/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_input: query }),
  });
  if (!response.ok) throw new Error('API error');
  return response.json();
};
