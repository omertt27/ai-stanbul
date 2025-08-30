// API utility that works for both local and deployed environments
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';
const API_URL = `${BASE_URL}/ai`;
const STREAM_API_URL = `${BASE_URL}/ai/stream`;

export const fetchResults = async (query) => {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_input: query }),
  });
  if (!response.ok) throw new Error('API error');
  return response.json();
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
