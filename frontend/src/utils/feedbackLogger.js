// Feedback logging utility
class FeedbackLogger {
  constructor() {
    this.feedbacks = this.loadFeedbacks();
  }

  // Load existing feedbacks from localStorage
  loadFeedbacks() {
    try {
      const stored = localStorage.getItem('ai-stanbul-feedbacks');
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error loading feedbacks:', error);
      return [];
    }
  }

  // Save feedbacks to localStorage
  saveFeedbacks() {
    try {
      localStorage.setItem('ai-stanbul-feedbacks', JSON.stringify(this.feedbacks));
    } catch (error) {
      console.error('Error saving feedbacks:', error);
    }
  }

  // Log a feedback event
  logFeedback(messageText, feedbackType, userQuery = '') {
    const feedback = {
      id: Date.now() + Math.random(), // Simple unique ID
      timestamp: new Date().toISOString(),
      userQuery,
      messageText: messageText.substring(0, 500), // Limit text length
      feedbackType, // 'good' or 'bad'
      sessionId: this.getSessionId()
    };

    this.feedbacks.push(feedback);
    this.saveFeedbacks();

    // Also log to console for immediate observation
    console.log('📊 Feedback Logged:', feedback);

    // Send to backend if available (optional)
    this.sendToBackend(feedback);

    return feedback;
  }

  // Get or create session ID
  getSessionId() {
    let sessionId = sessionStorage.getItem('ai-stanbul-session');
    if (!sessionId) {
      sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      sessionStorage.setItem('ai-stanbul-session', sessionId);
    }
    return sessionId;
  }

  // Get all feedbacks
  getAllFeedbacks() {
    return [...this.feedbacks];
  }

  // Get feedback statistics
  getStatistics() {
    const total = this.feedbacks.length;
    const good = this.feedbacks.filter(f => f.feedbackType === 'good').length;
    const bad = this.feedbacks.filter(f => f.feedbackType === 'bad').length;
    
    return {
      total,
      good,
      bad,
      goodPercentage: total > 0 ? ((good / total) * 100).toFixed(1) : 0,
      badPercentage: total > 0 ? ((bad / total) * 100).toFixed(1) : 0
    };
  }

  // Export feedbacks as JSON
  exportFeedbacks() {
    const data = {
      exportedAt: new Date().toISOString(),
      statistics: this.getStatistics(),
      feedbacks: this.getAllFeedbacks()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ai-stanbul-feedbacks-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Clear all feedbacks
  clearFeedbacks() {
    this.feedbacks = [];
    this.saveFeedbacks();
    console.log('🗑️ All feedbacks cleared');
  }

  // Optional: Send feedback to backend
  async sendToBackend(feedback) {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
      const response = await fetch(`${cleanApiUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedback)
      });
      
      if (response.ok) {
        console.log('✅ Feedback sent to backend');
      }
    } catch (error) {
      // Silently fail if backend doesn't have feedback endpoint
      console.log('ℹ️ Backend feedback endpoint not available');
    }
  }
}

// Create singleton instance
export const feedbackLogger = new FeedbackLogger();

// Make it globally available for console debugging
if (typeof window !== 'undefined') {
  window.feedbackLogger = feedbackLogger;
}
