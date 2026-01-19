/**
 * Production-Safe Logger
 * Only logs in development mode, silent in production
 * 
 * Usage:
 *   import logger from './utils/logger.js';
 *   logger.info('User logged in', { userId: 123 });
 *   logger.error('API failed', error);
 *   logger.debug('Debugging data', data);
 */

const isDevelopment = import.meta.env.DEV || import.meta.env.MODE === 'development';
const isProduction = import.meta.env.PROD || import.meta.env.MODE === 'production';

// Force enable/disable logging (can be toggled via localStorage)
const FORCE_ENABLE = localStorage.getItem('ENABLE_LOGS') === 'true';
const FORCE_DISABLE = localStorage.getItem('DISABLE_LOGS') === 'true';

// Determine if logging should be enabled
const LOGGING_ENABLED = FORCE_ENABLE || (isDevelopment && !FORCE_DISABLE);

// Log levels
const LogLevel = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
  CRITICAL: 4
};

// Current log level (can be configured via environment)
const currentLogLevel = import.meta.env.VITE_LOG_LEVEL 
  ? LogLevel[import.meta.env.VITE_LOG_LEVEL.toUpperCase()] 
  : LogLevel.DEBUG;

class Logger {
  constructor(namespace = 'App') {
    // Store namespace on a private field to avoid shadowing the prototype method `namespace()`
    this._namespace = namespace;
  }

  /**
   * Create a namespaced logger
   * @param {string} namespace - Logger namespace (e.g., 'ChatService', 'GPS')
   * @returns {Logger}
   */
  namespace(namespace) {
    return new Logger(namespace);
  }

  /**
   * Format log message with namespace and timestamp
   */
  _format(level, message, ...args) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    const prefix = `[${timestamp}] [${this._namespace}] [${level}]`;
    return [prefix, message, ...args];
  }

  /**
   * Debug - Detailed diagnostic information
   */
  debug(message, ...args) {
    if (!LOGGING_ENABLED || currentLogLevel > LogLevel.DEBUG) return;
    console.log(...this._format('DEBUG', message, ...args));
  }

  /**
   * Info - General informational messages
   */
  info(message, ...args) {
    if (!LOGGING_ENABLED || currentLogLevel > LogLevel.INFO) return;
    console.log(...this._format('INFO', message, ...args));
  }

  /**
   * Warn - Warning messages
   */
  warn(message, ...args) {
    if (!LOGGING_ENABLED || currentLogLevel > LogLevel.WARN) return;
    console.warn(...this._format('WARN', message, ...args));
  }

  /**
   * Error - Error messages (ALWAYS logged, even in production)
   */
  error(message, ...args) {
    // Errors are always logged, even in production (but sanitized)
    if (isProduction) {
      // In production, only log error message without sensitive data
      console.error(`[${this._namespace}] Error:`, message);
    } else {
      console.error(...this._format('ERROR', message, ...args));
    }
  }

  /**
   * Critical - Critical errors that need immediate attention
   */
  critical(message, ...args) {
    // Critical errors are always logged
    console.error(...this._format('CRITICAL', message, ...args));
    
    // In production, you might want to send to error tracking service
    if (isProduction) {
      // TODO: Send to Sentry, LogRocket, or your error tracking service
      this._sendToErrorTracking(message, args);
    }
  }

  /**
   * Group - Group related logs together
   */
  group(label, collapsed = false) {
    if (!LOGGING_ENABLED) return;
    collapsed ? console.groupCollapsed(label) : console.group(label);
  }

  /**
   * End group
   */
  groupEnd() {
    if (!LOGGING_ENABLED) return;
    console.groupEnd();
  }

  /**
   * Table - Display data as a table
   */
  table(data) {
    if (!LOGGING_ENABLED) return;
    console.table(data);
  }

  /**
   * Time - Start a timer
   */
  time(label) {
    if (!LOGGING_ENABLED) return;
    console.time(`[${this._namespace}] ${label}`);
  }

  /**
   * Time End - End a timer
   */
  timeEnd(label) {
    if (!LOGGING_ENABLED) return;
    console.timeEnd(`[${this._namespace}] ${label}`);
  }

  /**
   * Send critical errors to error tracking service
   */
  _sendToErrorTracking(message, args) {
    // Sentry integration for production error tracking
    if (typeof window !== 'undefined' && window.Sentry) {
      try {
        window.Sentry.captureException(new Error(message), {
          level: 'error',
          extra: {
            namespace: this._namespace,
            args: args,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href
          }
        });
      } catch (err) {
        // Fallback if Sentry fails
        console.error('[Logger] Failed to send to Sentry:', err);
      }
    }
  }
}

// Create default logger instance
const logger = new Logger('App');

// Export both the class and default instance
export { Logger, LogLevel };
export default logger;

// Development helper: Enable/disable logs via browser console
if (typeof window !== 'undefined') {
  window.enableLogs = () => {
    localStorage.setItem('ENABLE_LOGS', 'true');
    localStorage.removeItem('DISABLE_LOGS');
    console.log('✅ Logging enabled. Reload page to apply.');
  };
  
  window.disableLogs = () => {
    localStorage.setItem('DISABLE_LOGS', 'true');
    localStorage.removeItem('ENABLE_LOGS');
    console.log('🔇 Logging disabled. Reload page to apply.');
  };

  // Show logging status on load
  if (isDevelopment) {
    console.log(`🔍 Logging: ${LOGGING_ENABLED ? 'ENABLED' : 'DISABLED'} (${isDevelopment ? 'development' : 'production'} mode)`);
    console.log('💡 Use enableLogs() or disableLogs() to toggle');
  }
}
