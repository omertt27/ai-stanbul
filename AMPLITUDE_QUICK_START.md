# Amplitude Implementation - Quick Start 🚀

**Status:** ✅ Ready to Deploy  
**Date:** December 7, 2025

## What Was Done

### 1. ✅ Frontend Integration
**File:** `/frontend/index.html`
- Added Amplitude SDK snippet in `<head>`
- Enabled Session Replay (100% sample rate)
- Enabled Autocapture for automatic event tracking

### 2. ✅ Admin Dashboard - Helper Functions
**File:** `/admin/amplitude-helpers.js`
- Created comprehensive tracking helper functions
- 40+ pre-built event tracking functions
- Safe wrapper to prevent errors

### 3. ✅ Documentation
**File:** `/AMPLITUDE_ANALYTICS_INTEGRATION.md`
- Complete integration guide
- Custom event examples
- Best practices
- Testing instructions

---

## How to Use

### Method 1: Add Snippet Manually to Admin Dashboard

Open `/admin/dashboard.html` and add this **before** the `</head>` tag:

```html
<!-- Amplitude Analytics with Session Replay & Autocapture -->
<script src="https://cdn.amplitude.com/libs/analytics-browser-2.11.1-min.js.gz"></script>
<script src="https://cdn.amplitude.com/libs/plugin-session-replay-browser-1.23.2-min.js.gz"></script>
<script>
  window.amplitude.add(window.sessionReplay.plugin({sampleRate: 1}));
  window.amplitude.init('d1288055adec91b66d5bce71829fc4d', {
    "autocapture": {
      "elementInteractions": true
    }
  });
  
  // Track admin dashboard access
  window.amplitude.track('Admin Dashboard Accessed', {
    timestamp: new Date().toISOString(),
    page: 'dashboard'
  });
</script>
```

### Method 2: Use Helper Functions

In your `dashboard.js`, add this at the top:

```html
<!-- In dashboard.html, before dashboard.js -->
<script src="amplitude-helpers.js"></script>
```

Then in your JavaScript:

```javascript
// Track experiment creation
async function createExperiment() {
  // ... your existing code ...
  
  if (data.success) {
    // Use the helper!
    window.AIAnalytics.experimentCreated({
      name: experimentData.name,
      id: data.experiment.id,
      variant_count: Object.keys(experimentData.variants).length,
      metrics: experimentData.metrics
    });
  }
}
```

---

## Quick Integration Examples

### 1. Track Chat Messages

```javascript
// In your chat handler
async function sendChatMessage(message) {
  // Track message sent
  window.AIAnalytics.chatMessageSent(message, {
    language: detectLanguage(message),
    has_location: !!userLocation
  });
  
  // Call API
  const response = await fetch('/api/chat/pure-llm', ...);
  const data = await response.json();
  
  // Track response
  window.AIAnalytics.botResponse(data.response, {
    intent: data.intent,
    confidence: data.confidence,
    method: 'pure_llm'
  });
}
```

### 2. Track Experiments

```javascript
// Track create
window.AIAnalytics.experimentCreated(experiment);

// Track actions
window.AIAnalytics.experimentAction('started', experimentId);
window.AIAnalytics.experimentAction('stopped', experimentId);
window.AIAnalytics.experimentAction('deleted', experimentId);
```

### 3. Track Feature Flags

```javascript
// Track toggle
window.AIAnalytics.featureFlagAction('toggled', flagName, {
  enabled: newState,
  rollout_percentage: 50
});

// Track create
window.AIAnalytics.featureFlagAction('created', flagName, {
  enabled: true,
  rollout_percentage: 10
});
```

### 4. Track Routes

```javascript
// Track route request
window.AIAnalytics.routeRequested('Taksim', 'Sultanahmet', {
  transport_mode: 'public_transport',
  budget_friendly: true
});

// Track route selection
window.AIAnalytics.routeSelected({
  id: 'route_123',
  mode: 'tram',
  duration: 1200, // seconds
  distance: 5000, // meters
  cost: 15
});
```

### 5. Track Errors

```javascript
try {
  // Your code
} catch (error) {
  window.AIAnalytics.error('api_error', error.message, {
    endpoint: '/api/chat/pure-llm',
    status_code: response.status
  });
}
```

---

## Testing

### 1. Verify Installation

Open browser console on any page and run:

```javascript
// Test if Amplitude is loaded
console.log('Amplitude loaded:', typeof window.amplitude !== 'undefined');

// Send test event
window.amplitude.track('Test Event', {test: true});

// Or use helper
window.AIAnalytics.track('Test Event', {test: true});
```

### 2. Check Amplitude Dashboard

1. Go to https://analytics.amplitude.com/
2. Login with your credentials
3. Check Events → Live Stream
4. You should see your test events!

### 3. Test Autocapture

- Click any button on your site
- Check Amplitude for "Element Clicked" events
- Verify they're being captured automatically

---

## Integration Checklist

### Immediate (5 minutes)
- [ ] Verify Amplitude works on frontend (`/frontend/index.html` - ✅ DONE)
- [ ] Add snippet to admin dashboard (`/admin/dashboard.html`)
- [ ] Test with a simple event in console

### Phase 1 (30 minutes)
- [ ] Add helper script to admin dashboard HTML
- [ ] Track experiment CRUD operations
- [ ] Track feature flag operations
- [ ] Track learning cycle runs

### Phase 2 (1 hour)
- [ ] Track chat interactions in main app
- [ ] Track route planning flows
- [ ] Track place searches
- [ ] Track navigation events

### Phase 3 (2 hours)
- [ ] Set up user identification
- [ ] Configure user properties
- [ ] Set up error tracking globally
- [ ] Add page view tracking

### Phase 4 (Optional)
- [ ] Create Amplitude dashboards
- [ ] Set up alerts for errors
- [ ] Configure funnels
- [ ] Set up cohort analysis

---

## File Structure

```
ai-stanbul/
├── frontend/
│   └── index.html ✅ (Amplitude added)
├── admin/
│   ├── dashboard.html ⚠️ (Manual step needed)
│   ├── dashboard.js (Use helpers here)
│   └── amplitude-helpers.js ✅ (Created)
├── AMPLITUDE_ANALYTICS_INTEGRATION.md ✅ (Complete guide)
└── AMPLITUDE_QUICK_START.md ✅ (This file)
```

---

## Example: Complete Dashboard.js Integration

Here's how to integrate into your existing dashboard.js:

```javascript
// At the top of dashboard.js, after API_BASE_URL

// Amplitude helper - globally available as window.AIAnalytics

// === Track section changes ===
function initializeNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  
  navItems.forEach(item => {
    item.addEventListener('click', function() {
      const sectionId = this.getAttribute('data-section');
      
      // Track section view
      window.AIAnalytics.pageView(sectionId, {
        section_type: 'dashboard'
      });
      
      // ...existing code...
    });
  });
}

// === Track experiment operations ===
async function createExperiment() {
  // ...existing code...
  
  if (data.success) {
    window.AIAnalytics.experimentCreated({
      name: experiment.name,
      id: data.experiment.id,
      variant_count: Object.keys(experiment.variants).length
    });
    
    showSuccess('Experiment created successfully!');
    loadExperiments();
  }
}

async function deleteExperiment(id) {
  if (!confirm('Are you sure?')) return;
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/admin/experiments/experiments/${id}`, {
      method: 'DELETE'
    });
    
    if (data.success) {
      window.AIAnalytics.experimentAction('deleted', id);
      showSuccess('Experiment deleted!');
    }
  } catch (error) {
    window.AIAnalytics.error('experiment_delete_failed', error.message, {
      experiment_id: id
    });
  }
}

// === Track feature flags ===
async function toggleFlag(name, enabled) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/admin/experiments/flags/${name}`, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled})
    });
    
    if (data.success) {
      window.AIAnalytics.featureFlagAction('toggled', name, {
        enabled: enabled
      });
      showSuccess(`Flag ${enabled ? 'enabled' : 'disabled'}!`);
    }
  } catch (error) {
    window.AIAnalytics.error('flag_toggle_failed', error.message);
  }
}
```

---

## What You Get

### Automatic Tracking (Autocapture)
- ✅ Button clicks
- ✅ Form submissions
- ✅ Page navigation
- ✅ Element interactions

### Custom Events (via helpers)
- ✅ Chat messages & responses
- ✅ Route planning flow
- ✅ Experiment management
- ✅ Feature flag operations
- ✅ Learning cycle runs
- ✅ Error tracking
- ✅ User identification

### Session Replay
- ✅ Watch user sessions
- ✅ Debug issues visually
- ✅ Understand user behavior
- ✅ 100% sample rate

---

## Next Steps

1. **Now:** Add snippet to `admin/dashboard.html`
2. **Today:** Test basic tracking
3. **This Week:** Integrate helpers into dashboard.js
4. **Next Week:** Set up dashboards and alerts

---

## Support

- **Full Guide:** `AMPLITUDE_ANALYTICS_INTEGRATION.md`
- **Helpers:** `admin/amplitude-helpers.js`
- **Amplitude Docs:** https://www.docs.developers.amplitude.com/

---

**Status:** Ready to track! 🎯
