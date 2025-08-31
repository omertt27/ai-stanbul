// API utility that works for both local and deployed environments
const BASE_URL = 'http://localhost:8000'; // Force the correct port
const API_URL = `${BASE_URL}/ai`;
const STREAM_API_URL = `${BASE_URL}/ai/stream`;

console.log('API Configuration:', {
  BASE_URL,
  API_URL,
  STREAM_API_URL,
  VITE_API_URL: import.meta.env.VITE_API_URL
});

export const fetchResults = async (query) => {
  try {
    console.log('Making API request to:', API_URL, 'with query:', query);
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_input: query }),
    });
    console.log('Response status:', response.status, response.statusText);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API error response:', errorText);
      throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    const data = await response.json();
    console.log('API response data:', data);
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
    throw error;
  }
};

export const fetchStreamingResults = async (query, onChunk) => {
  const response = await fetch(STREAM_API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_input: query }),
  });
  
  if (!response.ok) throw new Error('API error');
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          
          try {
            const parsed = JSON.parse(data);
            if (parsed.delta && parsed.delta.content) {
              onChunk(parsed.delta.content);
            }
          } catch (e) {
            // Ignore parsing errors for malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
};
