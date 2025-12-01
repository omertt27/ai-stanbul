import React from 'react';
import safeStorage from '../utils/safeStorage';

/**
 * Global Error Boundary Component
 * Catches React errors and localStorage corruption issues
 * Provides recovery UI with options to clear corrupted data
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      isStorageError: false
    };
  }

  static getDerivedStateFromError(error) {
    // Check if this is a storage-related error
    const isStorageError = 
      error?.message?.includes('localStorage') ||
      error?.message?.includes('JSON') ||
      error?.message?.includes('parse') ||
      error?.message?.includes('storage');

    return {
      hasError: true,
      isStorageError
    };
  }

  componentDidCatch(error, errorInfo) {
    console.error('❌ ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo
    });

    // Log to external error tracking service if available
    if (window.gtag) {
      window.gtag('event', 'exception', {
        description: error.toString(),
        fatal: true
      });
    }
  }

  handleClearStorage = () => {
    try {
      // Clear all corrupted data
      safeStorage.clearAll();
      
      // Also clear sessionStorage
      sessionStorage.clear();
      
      console.log('✅ Storage cleared successfully');
      
      // Reload the page to start fresh
      window.location.reload();
    } catch (error) {
      console.error('Failed to clear storage:', error);
      alert('Failed to clear storage. Please try manually clearing your browser data.');
    }
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      const { error, errorInfo, isStorageError } = this.state;
      
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }}>
          <div style={{
            maxWidth: '600px',
            background: 'white',
            borderRadius: '16px',
            padding: '40px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
            textAlign: 'center'
          }}>
            <div style={{
              fontSize: '64px',
              marginBottom: '20px'
            }}>
              {isStorageError ? '🔧' : '😕'}
            </div>
            
            <h1 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              marginBottom: '16px',
              color: '#1a202c'
            }}>
              {isStorageError 
                ? 'Data Recovery Needed' 
                : 'Something Went Wrong'}
            </h1>
            
            <p style={{
              fontSize: '16px',
              color: '#4a5568',
              marginBottom: '24px',
              lineHeight: '1.6'
            }}>
              {isStorageError 
                ? 'Your browser storage contains corrupted data. This usually happens after clearing cookies or cache. Click below to fix this automatically.'
                : 'We encountered an unexpected error while loading the application.'}
            </p>

            {isStorageError ? (
              <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                <button
                  onClick={this.handleClearStorage}
                  style={{
                    padding: '12px 24px',
                    fontSize: '16px',
                    fontWeight: '600',
                    color: 'white',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'transform 0.2s',
                  }}
                  onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                  onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                >
                  🔧 Fix & Reload
                </button>
                
                <button
                  onClick={this.handleReload}
                  style={{
                    padding: '12px 24px',
                    fontSize: '16px',
                    fontWeight: '600',
                    color: '#667eea',
                    background: 'white',
                    border: '2px solid #667eea',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'transform 0.2s',
                  }}
                  onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                  onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
                >
                  🔄 Try Again
                </button>
              </div>
            ) : (
              <button
                onClick={this.handleReload}
                style={{
                  padding: '12px 24px',
                  fontSize: '16px',
                  fontWeight: '600',
                  color: 'white',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                }}
                onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
                onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
              >
                🔄 Reload Page
              </button>
            )}

            {/* Show technical details in development */}
            {import.meta.env.DEV && error && (
              <details style={{
                marginTop: '24px',
                textAlign: 'left',
                background: '#f7fafc',
                padding: '16px',
                borderRadius: '8px',
                fontSize: '12px'
              }}>
                <summary style={{
                  cursor: 'pointer',
                  fontWeight: '600',
                  marginBottom: '8px'
                }}>
                  Technical Details (Dev Mode)
                </summary>
                <pre style={{
                  overflow: 'auto',
                  color: '#e53e3e',
                  marginBottom: '8px'
                }}>
                  {error.toString()}
                </pre>
                {errorInfo && (
                  <pre style={{
                    overflow: 'auto',
                    color: '#4a5568',
                    fontSize: '11px'
                  }}>
                    {errorInfo.componentStack}
                  </pre>
                )}
              </details>
            )}

            <p style={{
              marginTop: '24px',
              fontSize: '14px',
              color: '#a0aec0'
            }}>
              If the problem persists, please contact support at{' '}
              <a href="mailto:omertahtaci@aistanbul.net" style={{color: '#667eea'}}>
                omertahtaci@aistanbul.net
              </a>
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
