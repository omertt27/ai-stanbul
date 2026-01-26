/**
 * useVoiceInput Hook
 * ===================
 * A robust, ChatGPT-style voice input hook for React
 * 
 * Features:
 * - Multi-language support (English, Turkish, Russian, French, Arabic, German)
 * - Cross-browser support (Chrome, Safari, Firefox, Edge)
 * - iOS Safari special handling with native input fallback
 * - Continuous listening mode (like ChatGPT mobile)
 * - Silence detection with auto-stop
 * - Interim results with visual feedback
 * - Error handling with user-friendly messages
 * - Haptic feedback for mobile
 * - Noise level detection
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Supported languages for Istanbul tourism app
export const SUPPORTED_VOICE_LANGUAGES = {
  'en-US': { code: 'en-US', name: 'English', flag: '🇺🇸', shortName: 'EN' },
  'tr-TR': { code: 'tr-TR', name: 'Türkçe', flag: '🇹🇷', shortName: 'TR' },
  'ru-RU': { code: 'ru-RU', name: 'Русский', flag: '🇷🇺', shortName: 'RU' },
  'fr-FR': { code: 'fr-FR', name: 'Français', flag: '🇫🇷', shortName: 'FR' },
  'ar-SA': { code: 'ar-SA', name: 'العربية', flag: '🇸🇦', shortName: 'AR' },
  'de-DE': { code: 'de-DE', name: 'Deutsch', flag: '🇩🇪', shortName: 'DE' },
};

// Default language order for the picker
export const LANGUAGE_ORDER = ['en-US', 'tr-TR', 'ru-RU', 'fr-FR', 'ar-SA', 'de-DE'];

// Detect best language from browser settings
export const detectBrowserLanguage = () => {
  const browserLang = navigator.language || navigator.userLanguage || 'en-US';
  const langPrefix = browserLang.split('-')[0].toLowerCase();
  
  // Map browser language prefixes to our supported languages
  const langMap = {
    'en': 'en-US',
    'tr': 'tr-TR',
    'ru': 'ru-RU',
    'fr': 'fr-FR',
    'ar': 'ar-SA',
    'de': 'de-DE',
  };
  
  return langMap[langPrefix] || 'en-US';
};

// Browser detection utilities
const getBrowserInfo = () => {
  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua) && !window.MSStream;
  const isAndroid = /Android/.test(ua);
  const isSafari = /Safari/.test(ua) && !/Chrome|CriOS|FxiOS/.test(ua);
  const isChrome = /Chrome|CriOS/.test(ua) && !/Edge|Edg/.test(ua);
  const isFirefox = /Firefox|FxiOS/.test(ua);
  const isEdge = /Edge|Edg/.test(ua);
  const isMobile = isIOS || isAndroid || /Mobile/.test(ua);
  
  return { isIOS, isAndroid, isSafari, isChrome, isFirefox, isEdge, isMobile };
};

// Check if Web Speech API is supported
const checkSpeechSupport = () => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  return {
    supported: !!SpeechRecognition,
    SpeechRecognition
  };
};

// Haptic feedback helper
const triggerHaptic = (pattern = 'light') => {
  if (!('vibrate' in navigator)) return;
  
  const patterns = {
    light: 10,
    medium: 25,
    heavy: 50,
    success: [10, 50, 10],
    error: [50, 100, 50]
  };
  
  try {
    navigator.vibrate(patterns[pattern] || patterns.light);
  } catch (e) {
    // Vibrate failed, ignore
  }
};

/**
 * Main voice input hook
 * 
 * @param {Object} options Configuration options
 * @param {Function} options.onResult Callback when final transcript is ready
 * @param {Function} options.onInterim Callback for interim results (visual feedback)
 * @param {Function} options.onError Callback for errors
 * @param {Function} options.onStateChange Callback when listening state changes
 * @param {string} options.language Preferred language ('en-US', 'tr-TR', or 'auto')
 * @param {boolean} options.continuous Whether to keep listening after speech ends
 * @param {number} options.silenceTimeout Ms to wait in silence before stopping (default: 2000)
 * @param {number} options.maxDuration Maximum recording duration in ms (default: 60000)
 */
const useVoiceInput = (options = {}) => {
  const {
    onResult,
    onInterim,
    onError,
    onStateChange,
    language = 'en-US', // Default to English for Istanbul tourism app
    continuous = false,
    silenceTimeout = 2500,
    maxDuration = 60000
  } = options;

  // State
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [interimTranscript, setInterimTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState('');
  const [error, setError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [browserInfo, setBrowserInfo] = useState(null);

  // Refs for cleanup and state management
  const recognitionRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const maxDurationTimerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const isListeningRef = useRef(false);
  const accumulatedTranscriptRef = useRef('');

  // Initialize browser info
  useEffect(() => {
    setBrowserInfo(getBrowserInfo());
  }, []);

  // Check support on mount
  useEffect(() => {
    const { supported } = checkSpeechSupport();
    setIsSupported(supported);
    
    if (!supported) {
      console.warn('Web Speech API not supported in this browser');
    }
  }, []);

  // Update listening ref when state changes
  useEffect(() => {
    isListeningRef.current = isListening;
    onStateChange?.(isListening);
  }, [isListening, onStateChange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
      cleanupAudioAnalysis();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup audio analysis resources
  const cleanupAudioAnalysis = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        audioContextRef.current.close();
      } catch (e) {
        // Ignore close errors
      }
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
  }, []);

  // Setup audio level analysis for visual feedback
  const setupAudioAnalysis = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateAudioLevel = () => {
        if (!isListeningRef.current || !analyserRef.current) {
          return;
        }

        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedLevel = Math.min(average / 128, 1); // Normalize to 0-1
        setAudioLevel(normalizedLevel);

        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };

      updateAudioLevel();
      return stream;
    } catch (err) {
      console.warn('Audio analysis setup failed:', err);
      // Continue without audio level - not critical
      return null;
    }
  }, []);

  // Clear silence timer
  const clearSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  // Reset silence timer (called when speech is detected)
  const resetSilenceTimer = useCallback(() => {
    clearSilenceTimer();
    
    if (!continuous && isListeningRef.current) {
      silenceTimerRef.current = setTimeout(() => {
        console.log('Silence timeout - stopping listening');
        stopListening();
      }, silenceTimeout);
    }
  }, [continuous, silenceTimeout, clearSilenceTimer]);

  // Get the appropriate language code (supports 6 languages)
  const getLanguageCode = useCallback(() => {
    if (language === 'auto') {
      return detectBrowserLanguage();
    }
    // Validate that it's a supported language
    if (SUPPORTED_VOICE_LANGUAGES[language]) {
      return language;
    }
    // Fallback to English
    return 'en-US';
  }, [language]);

  // Stop listening
  const stopListening = useCallback(() => {
    console.log('Stopping voice recognition...');
    
    // Clear timers
    clearSilenceTimer();
    if (maxDurationTimerRef.current) {
      clearTimeout(maxDurationTimerRef.current);
      maxDurationTimerRef.current = null;
    }

    // Stop recognition
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.warn('Error stopping recognition:', e);
      }
    }

    // Cleanup audio
    cleanupAudioAnalysis();

    // Update state
    setIsListening(false);
    setInterimTranscript('');
    
    // Return accumulated transcript
    const finalResult = accumulatedTranscriptRef.current.trim();
    if (finalResult) {
      setFinalTranscript(finalResult);
      onResult?.(finalResult);
      accumulatedTranscriptRef.current = '';
    }

    triggerHaptic('medium');
  }, [clearSilenceTimer, cleanupAudioAnalysis, onResult]);

  // Start listening
  const startListening = useCallback(async () => {
    const { supported, SpeechRecognition } = checkSpeechSupport();
    
    if (!supported) {
      const errorMsg = browserInfo?.isIOS 
        ? 'Voice input requires Safari 14.1+ on iOS. Please update your browser or type your message.'
        : 'Voice input is not supported in this browser. Please try Chrome, Edge, or Safari.';
      setError(errorMsg);
      onError?.(errorMsg);
      return false;
    }

    // If already listening, stop first
    if (isListeningRef.current) {
      stopListening();
      return false;
    }

    setError(null);
    accumulatedTranscriptRef.current = '';

    try {
      // Request microphone permission first
      // This helps ensure permission dialog shows before recognition starts
      if (navigator.mediaDevices?.getUserMedia) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          // Don't stop the stream yet if we'll use it for audio analysis
          if (!browserInfo?.isIOS) {
            // On non-iOS, we can use the stream for audio level analysis
            mediaStreamRef.current = stream;
          } else {
            // On iOS, just release it - audio analysis doesn't work well
            stream.getTracks().forEach(track => track.stop());
          }
        } catch (permError) {
          if (permError.name === 'NotAllowedError' || permError.name === 'PermissionDeniedError') {
            const errorMsg = 'Microphone access denied. Please allow microphone permissions in your browser settings.';
            setError(errorMsg);
            onError?.(errorMsg);
            return false;
          }
          // For other errors (like NotFoundError), continue - recognition might still work
          console.warn('Microphone permission request issue:', permError);
        }
      }

      // Create recognition instance
      const recognition = new SpeechRecognition();
      recognitionRef.current = recognition;

      // Configure recognition
      const langCode = getLanguageCode();
      recognition.lang = langCode;
      recognition.continuous = continuous || browserInfo?.isMobile; // Mobile: keep listening
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      // iOS Safari specific adjustments
      if (browserInfo?.isIOS && browserInfo?.isSafari) {
        // iOS Safari has issues with continuous mode
        recognition.continuous = false;
        // Shorter timeout for iOS
        recognition.maxAlternatives = 1;
      }

      // Event handlers
      recognition.onstart = () => {
        console.log('Speech recognition started');
        setIsListening(true);
        triggerHaptic('light');
        
        // Setup audio level analysis (except on iOS where it doesn't work well)
        if (!browserInfo?.isIOS) {
          setupAudioAnalysis();
        }

        // Start max duration timer
        maxDurationTimerRef.current = setTimeout(() => {
          console.log('Max duration reached - stopping');
          stopListening();
        }, maxDuration);

        // Start initial silence timer
        resetSilenceTimer();
      };

      recognition.onresult = (event) => {
        // Reset silence timer - user is speaking
        resetSilenceTimer();

        let interim = '';
        let final = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const transcript = result[0].transcript;

          if (result.isFinal) {
            final += transcript;
          } else {
            interim += transcript;
          }
        }

        // Update interim display
        if (interim) {
          setInterimTranscript(interim);
          onInterim?.(interim);
        }

        // Accumulate final results
        if (final) {
          const trimmedFinal = final.trim();
          if (accumulatedTranscriptRef.current) {
            accumulatedTranscriptRef.current += ' ' + trimmedFinal;
          } else {
            accumulatedTranscriptRef.current = trimmedFinal;
          }
          
          setInterimTranscript('');
          setFinalTranscript(accumulatedTranscriptRef.current);

          // In non-continuous mode, stop after getting final result
          if (!continuous && !browserInfo?.isMobile) {
            setTimeout(() => {
              stopListening();
            }, 100);
          }
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);

        let errorMsg = '';
        let shouldStop = true;

        switch (event.error) {
          case 'not-allowed':
          case 'permission-denied':
            errorMsg = 'Microphone access denied. Please allow microphone permissions.';
            break;
          case 'no-speech':
            // No speech detected - this is common, not a real error
            errorMsg = 'No speech detected. Tap to try again.';
            break;
          case 'audio-capture':
            errorMsg = 'No microphone found. Please check your device.';
            break;
          case 'network':
            errorMsg = 'Network error. Please check your connection.';
            break;
          case 'aborted':
            // User cancelled - no error message needed
            shouldStop = true;
            break;
          case 'service-not-allowed':
            // Browser policy prevents speech recognition
            errorMsg = 'Voice input is not available. Please type your message.';
            break;
          default:
            errorMsg = 'Voice input failed. Please try again or type your message.';
        }

        if (errorMsg) {
          setError(errorMsg);
          onError?.(errorMsg);
          triggerHaptic('error');
        }

        if (shouldStop) {
          stopListening();
        }
      };

      recognition.onend = () => {
        console.log('Speech recognition ended');
        
        // If we're still supposed to be listening (continuous mode), restart
        if (isListeningRef.current && continuous) {
          try {
            recognition.start();
          } catch (e) {
            console.warn('Failed to restart recognition:', e);
            stopListening();
          }
        } else {
          // Finalize and stop
          stopListening();
        }
      };

      recognition.onspeechstart = () => {
        console.log('Speech started');
        resetSilenceTimer();
      };

      recognition.onspeechend = () => {
        console.log('Speech ended');
        // Don't immediately stop - wait for final results
      };

      // Start recognition
      recognition.start();
      return true;

    } catch (e) {
      console.error('Failed to start speech recognition:', e);
      
      const errorMsg = e.message?.includes('already started')
        ? 'Voice input is already active.'
        : 'Failed to start voice input. Please try again.';
      
      setError(errorMsg);
      onError?.(errorMsg);
      setIsListening(false);
      triggerHaptic('error');
      return false;
    }
  }, [
    browserInfo,
    continuous,
    getLanguageCode,
    maxDuration,
    onError,
    onInterim,
    resetSilenceTimer,
    setupAudioAnalysis,
    stopListening
  ]);

  // Toggle listening state
  const toggleListening = useCallback(async () => {
    if (isListening) {
      stopListening();
    } else {
      await startListening();
    }
  }, [isListening, startListening, stopListening]);

  // Change language
  const setLanguage = useCallback((newLanguage) => {
    if (recognitionRef.current) {
      recognitionRef.current.lang = newLanguage === 'auto' 
        ? (navigator.language?.startsWith('tr') ? 'tr-TR' : 'en-US')
        : newLanguage;
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    isListening,
    isSupported,
    interimTranscript,
    finalTranscript,
    error,
    audioLevel,
    browserInfo,
    
    // Actions
    startListening,
    stopListening,
    toggleListening,
    setLanguage,
    clearError
  };
};

export default useVoiceInput;
export { getBrowserInfo, checkSpeechSupport, triggerHaptic };
