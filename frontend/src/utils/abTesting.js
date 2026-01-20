/**
 * A/B Testing Framework
 * =====================
 * Simple client-side A/B testing for mobile components
 * 
 * Features:
 * - User bucketing (consistent across sessions)
 * - Variant assignment
 * - Conversion tracking
 * - Results export
 * 
 * Tests:
 * 1. Mobile Components vs Standard UI (50/50 split)
 * 2. Quick Reply Suggestions (control vs smart suggestions)
 * 3. Voice Input Prominence (always visible vs contextual)
 */

const AB_TEST_STORAGE_KEY = 'kam_ab_tests';
const USER_BUCKET_KEY = 'kam_ab_bucket';

/**
 * Available A/B tests
 */
export const AB_TESTS = {
  MOBILE_COMPONENTS: 'mobile_components_vs_standard',
  SMART_QUICK_REPLIES: 'smart_quick_replies',
  VOICE_INPUT_PROMINENCE: 'voice_input_prominence'
};

/**
 * Test variants
 */
export const VARIANTS = {
  CONTROL: 'control',
  TREATMENT: 'treatment'
};

class ABTestingManager {
  constructor() {
    // Lazy initialization - don't access localStorage until actually needed
    this._userBucket = null;
    this._assignments = null;
    this._conversions = null;
    this._initialized = false;
  }

  /**
   * Initialize the manager (called lazily on first use)
   */
  _ensureInitialized() {
    if (this._initialized) return;
    this._userBucket = this.getUserBucket();
    this._assignments = this.loadAssignments();
    this._conversions = this.loadConversions();
    this._initialized = true;
  }

  /**
   * Get or create a persistent user bucket (0-99)
   */
  getUserBucket() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return Math.floor(Math.random() * 100);
      }
      const stored = localStorage.getItem(USER_BUCKET_KEY);
      if (stored !== null) {
        return parseInt(stored, 10);
      }
    } catch (e) {
      console.warn('Failed to load user bucket:', e);
    }

    // Generate new bucket (0-99)
    const bucket = Math.floor(Math.random() * 100);
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        localStorage.setItem(USER_BUCKET_KEY, bucket.toString());
      }
    } catch (e) {
      console.warn('Failed to save user bucket:', e);
    }
    return bucket;
  }

  /**
   * Load test assignments from storage
   */
  loadAssignments() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return {};
      }
      const stored = localStorage.getItem(AB_TEST_STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        return data.assignments || {};
      }
    } catch (e) {
      console.warn('Failed to load A/B test assignments:', e);
    }
    return {};
  }

  /**
   * Load conversions from storage
   */
  loadConversions() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return {};
      }
      const stored = localStorage.getItem(AB_TEST_STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        return data.conversions || {};
      }
    } catch (e) {
      console.warn('Failed to load A/B test conversions:', e);
    }
    return {};
  }

  /**
   * Save data to storage
   */
  save() {
    try {
      if (typeof window === 'undefined' || !window.localStorage) {
        return;
      }
      const data = {
        assignments: this._assignments,
        conversions: this._conversions,
        userBucket: this._userBucket,
        lastUpdated: new Date().toISOString()
      };
      localStorage.setItem(AB_TEST_STORAGE_KEY, JSON.stringify(data));
    } catch (e) {
      console.warn('Failed to save A/B test data:', e);
    }
  }

  /**
   * Get variant for a test (consistent across sessions)
   */
  getVariant(testName, splitPercentage = 50) {
    this._ensureInitialized();
    
    // Return existing assignment if available
    if (this._assignments[testName]) {
      return this._assignments[testName];
    }

    // Assign based on user bucket
    const variant = this._userBucket < splitPercentage 
      ? VARIANTS.TREATMENT 
      : VARIANTS.CONTROL;

    // Save assignment
    this._assignments[testName] = variant;
    this.save();

    console.log(`[A/B Test] ${testName}: Assigned to ${variant} (bucket: ${this._userBucket})`);
    return variant;
  }

  /**
   * Check if user is in treatment group
   */
  isTreatment(testName, splitPercentage = 50) {
    this._ensureInitialized();
    return this.getVariant(testName, splitPercentage) === VARIANTS.TREATMENT;
  }

  /**
   * Check if user is in control group
   */
  isControl(testName, splitPercentage = 50) {
    this._ensureInitialized();
    return this.getVariant(testName, splitPercentage) === VARIANTS.CONTROL;
  }

  /**
   * Track a conversion event
   */
  trackConversion(testName, conversionType = 'primary', metadata = {}) {
    this._ensureInitialized();
    
    if (!this._conversions[testName]) {
      this._conversions[testName] = [];
    }

    this._conversions[testName].push({
      variant: this.getVariant(testName),
      conversionType,
      metadata,
      timestamp: new Date().toISOString()
    });

    this.save();
    console.log(`[A/B Test] ${testName}: Conversion tracked (${conversionType})`);
  }

  /**
   * Get test results
   */
  getResults(testName) {
    this._ensureInitialized();
    
    const conversions = this._conversions[testName] || [];
    
    const controlConversions = conversions.filter(c => c.variant === VARIANTS.CONTROL);
    const treatmentConversions = conversions.filter(c => c.variant === VARIANTS.TREATMENT);

    return {
      testName,
      variant: this.getVariant(testName),
      conversions: {
        control: {
          count: controlConversions.length,
          types: this.groupBy(controlConversions, 'conversionType')
        },
        treatment: {
          count: treatmentConversions.length,
          types: this.groupBy(treatmentConversions, 'conversionType')
        }
      }
    };
  }

  /**
   * Get all test results
   */
  getAllResults() {
    this._ensureInitialized();
    
    const results = {};
    for (const testName in this._assignments) {
      results[testName] = this.getResults(testName);
    }
    return results;
  }

  /**
   * Helper: Group array by key
   */
  groupBy(array, key) {
    return array.reduce((acc, item) => {
      const group = item[key];
      acc[group] = (acc[group] || 0) + 1;
      return acc;
    }, {});
  }

  /**
   * Reset all tests (for debugging)
   */
  reset() {
    this.assignments = {};
    this.conversions = {};
    this.save();
    console.log('[A/B Test] All tests reset');
  }
}

// Export singleton instance
export const abTesting = new ABTestingManager();

// Export convenience functions
export const getVariant = (testName, splitPercentage) => 
  abTesting.getVariant(testName, splitPercentage);

export const isTreatment = (testName, splitPercentage) => 
  abTesting.isTreatment(testName, splitPercentage);

export const isControl = (testName, splitPercentage) => 
  abTesting.isControl(testName, splitPercentage);

export const trackConversion = (testName, conversionType, metadata) => 
  abTesting.trackConversion(testName, conversionType, metadata);

export const getResults = (testName) => 
  abTesting.getResults(testName);

export const getAllResults = () => 
  abTesting.getAllResults();

export default abTesting;
