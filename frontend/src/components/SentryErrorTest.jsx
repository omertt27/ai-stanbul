import React from 'react';
import * as Sentry from '@sentry/react';

/**
 * Test component to demonstrate Sentry error tracking
 * This component can be temporarily added to test Sentry integration
 */
export function SentryErrorTest() {
  const [showError, setShowError] = React.useState(false);

  const triggerError = () => {
    // This will be caught by Sentry
    throw new Error('Test error for Sentry integration - This is intentional!');
  };

  const triggerCaptureException = () => {
    try {
      throw new Error('Manually captured test error');
    } catch (error) {
      Sentry.captureException(error);
      console.log('Error captured and sent to Sentry');
    }
  };

  const triggerCaptureMessage = () => {
    Sentry.captureMessage('Test message from Istanbul AI frontend', 'info');
    console.log('Test message sent to Sentry');
  };

  if (showError) {
    triggerError();
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      padding: '20px',
      backgroundColor: '#fff',
      border: '2px solid #e74c3c',
      borderRadius: '8px',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      zIndex: 9999,
    }}>
      <h3 style={{ marginTop: 0, color: '#e74c3c' }}>Sentry Test Panel</h3>
      <p style={{ fontSize: '12px', color: '#666' }}>
        Test Sentry error tracking integration
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <button
          onClick={() => setShowError(true)}
          style={{
            padding: '10px',
            backgroundColor: '#e74c3c',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          Trigger React Error
        </button>
        <button
          onClick={triggerCaptureException}
          style={{
            padding: '10px',
            backgroundColor: '#3498db',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Capture Exception
        </button>
        <button
          onClick={triggerCaptureMessage}
          style={{
            padding: '10px',
            backgroundColor: '#2ecc71',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Send Test Message
        </button>
      </div>
      <p style={{ fontSize: '10px', color: '#999', marginBottom: 0 }}>
        Remove this component in production
      </p>
    </div>
  );
}

// Error Boundary component using Sentry
export const SentryErrorBoundary = Sentry.withErrorBoundary(
  ({ children }) => children,
  {
    fallback: ({ error, resetError }) => (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        maxWidth: '600px',
        margin: '100px auto',
      }}>
        <h2 style={{ color: '#e74c3c' }}>Oops! Something went wrong</h2>
        <p style={{ color: '#666', marginBottom: '20px' }}>
          An error occurred and has been reported to our team.
        </p>
        <details style={{ 
          textAlign: 'left', 
          padding: '15px', 
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          marginBottom: '20px'
        }}>
          <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
            Error Details
          </summary>
          <pre style={{ 
            marginTop: '10px', 
            fontSize: '12px',
            overflow: 'auto',
            color: '#e74c3c'
          }}>
            {error?.toString()}
          </pre>
        </details>
        <button
          onClick={resetError}
          style={{
            padding: '12px 24px',
            backgroundColor: '#3498db',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          Try Again
        </button>
      </div>
    ),
    showDialog: false, // Set to true to show Sentry's user feedback dialog
  }
);

export default SentryErrorTest;
