// Example API utility
export const fetchResults = async (query) => {
  // Replace with your actual API endpoint
  const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error('API error');
  return response.json();
};
