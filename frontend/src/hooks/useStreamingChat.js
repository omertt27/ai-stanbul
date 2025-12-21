/**
 * useStreamingChat Hook
 * 
 * React hook for consuming streaming chat responses.
 * Provides real-time word-by-word display like ChatGPT.
 * 
 * Usage:
 * ```jsx
 * const { 
 *   streamingText, 
 *   isStreaming, 
 *   sendMessage,
 *   error 
 * } = useStreamingChat();
 * 
 * // In your component:
 * <button onClick={() => sendMessage("Hello!")}>Send</button>
 * <p>{streamingText}</p>
 * ```
 */

import { useState, useCallback, useRef } from 'react';
import { fetchStreamingChat, fetchUnifiedChatV2, getSessionId } from '../api/api';

/**
 * Hook for streaming chat responses
 * @param {Object} options - Configuration options
 * @param {boolean} options.enableStreaming - Enable streaming (default: true)
 * @param {string} options.language - Language code (default: 'en')
 * @param {Function} options.onMessageComplete - Callback when message completes
 * @param {boolean} options.fallbackToRegular - Fall back to regular API if streaming fails
 */
export const useStreamingChat = (options = {}) => {
  const {
    enableStreaming = true,
    language = 'en',
    onMessageComplete,
    fallbackToRegular = true
  } = options;

  const [streamingText, setStreamingText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [intent, setIntent] = useState(null);
  
  const abortControllerRef = useRef(null);
  const sessionIdRef = useRef(getSessionId());

  /**
   * Send a message and receive streaming response
   */
  const sendMessage = useCallback(async (message, messageOptions = {}) => {
    const {
      gpsLocation = null,
      sessionId = sessionIdRef.current,
      lang = language
    } = messageOptions;

    // Reset state
    setStreamingText('');
    setError(null);
    setMetadata(null);
    setIntent(null);
    setIsStreaming(true);

    try {
      if (enableStreaming) {
        // Use streaming API
        await fetchStreamingChat(message, {
          sessionId,
          language: lang,
          gpsLocation,
          
          onStart: (data) => {
            console.log('🚀 Streaming started:', data);
            if (data.intent) {
              setIntent(data.intent);
            }
          },
          
          onToken: (token, fullText) => {
            setStreamingText(fullText);
          },
          
          onComplete: (finalText, meta) => {
            console.log('✅ Streaming complete');
            setStreamingText(finalText);
            setMetadata(meta);
            setIsStreaming(false);
            
            if (onMessageComplete) {
              onMessageComplete({
                text: finalText,
                metadata: meta,
                intent: meta?.intent || intent
              });
            }
          },
          
          onError: (err) => {
            console.error('❌ Streaming error:', err);
            
            if (fallbackToRegular) {
              // Fall back to regular API
              console.log('⚡ Falling back to regular chat API');
              sendRegularMessage(message, messageOptions);
            } else {
              setError(err);
              setIsStreaming(false);
            }
          }
        });
      } else {
        // Use regular API directly
        await sendRegularMessage(message, messageOptions);
      }
    } catch (err) {
      console.error('Send message error:', err);
      
      if (fallbackToRegular && enableStreaming) {
        await sendRegularMessage(message, messageOptions);
      } else {
        setError(err);
        setIsStreaming(false);
      }
    }
  }, [enableStreaming, language, fallbackToRegular, onMessageComplete, intent]);

  /**
   * Send message using regular (non-streaming) API
   */
  const sendRegularMessage = useCallback(async (message, messageOptions = {}) => {
    const {
      gpsLocation = null,
      sessionId = sessionIdRef.current,
      lang = language
    } = messageOptions;

    try {
      const response = await fetchUnifiedChatV2(message, {
        sessionId,
        language: lang,
        gpsLocation,
        usePureLLM: false
      });

      const responseText = response.response || response.message || '';
      
      // Simulate streaming effect for better UX
      await simulateStreaming(responseText);
      
      setMetadata({
        intent: response.intent,
        confidence: response.confidence,
        method: response.method
      });
      
      if (onMessageComplete) {
        onMessageComplete({
          text: responseText,
          metadata: response,
          intent: response.intent
        });
      }
    } catch (err) {
      setError(err);
    } finally {
      setIsStreaming(false);
    }
  }, [language, onMessageComplete]);

  /**
   * Simulate streaming effect for regular API responses
   */
  const simulateStreaming = useCallback(async (text) => {
    const words = text.split(' ');
    let currentText = '';
    
    for (let i = 0; i < words.length; i++) {
      currentText += (i === 0 ? '' : ' ') + words[i];
      setStreamingText(currentText);
      
      // Variable delay for more natural feel
      const delay = Math.random() * 30 + 20; // 20-50ms per word
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }, []);

  /**
   * Cancel ongoing streaming
   */
  const cancelStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  /**
   * Clear current response
   */
  const clearResponse = useCallback(() => {
    setStreamingText('');
    setError(null);
    setMetadata(null);
    setIntent(null);
  }, []);

  return {
    // State
    streamingText,
    isStreaming,
    error,
    metadata,
    intent,
    
    // Actions
    sendMessage,
    cancelStreaming,
    clearResponse,
    
    // Session
    sessionId: sessionIdRef.current
  };
};

export default useStreamingChat;
