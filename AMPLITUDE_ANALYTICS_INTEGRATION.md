# Amplitude Analytics Integration Guide

**Date:** December 7, 2025  
**Project:** AI Istanbul  
**Analytics Platform:** Amplitude Browser SDK

## Overview

This guide shows how to integrate Amplitude analytics across the AI Istanbul application to track user behavior, interactions, and conversions.

---

## Setup Instructions

### 1. Browser SDK Installation

#### Method 1: CDN Snippet (Recommended) ✅

Add this code snippet **before the `</head>` tag** on every page you want to track:

```html
<!-- Amplitude Analytics with Session Replay & Autocapture -->
<script src="https://cdn.amplitude.com/libs/analytics-browser-2.11.1-min.js.gz"></script>
<script src="https://cdn.amplitude.com/libs/plugin-session-replay-browser-1.23.2-min.js.gz"></script>
<script>
  // Initialize Amplitude with Session Replay and Autocapture
  window.amplitude.add(window.sessionReplay.plugin({sampleRate: 1}));
  window.amplitude.init('d1288055adec91b66d5bce71829fc4d', {
    "autocapture": {
      "elementInteractions": true
    }
  });
</script>
```

#### Method 2: npm Package (Alternative)

```bash
npm install @amplitude/analytics-browser @amplitude/plugin-session-replay-browser
```

Then in your JavaScript:

```javascript
import * as amplitude from '@amplitude/analytics-browser';
import { sessionReplayPlugin } from '@amplitude/plugin-session-replay-browser';

amplitude.add(sessionReplayPlugin({sampleRate: 1}));
amplitude.init('d1288055adec91b66d5bce71829fc4d', {
  autocapture: {
    elementInteractions: true
  }
});
```

---

## Features Enabled

### ✅ Session Replay
- **What it does:** Records user sessions for playback
- **Sample Rate:** 100% (sampleRate: 1)
- **Use Case:** Debug user issues, understand user flows

### ✅ Autocapture
- **What it does:** Automatically tracks button clicks, form submissions, page views
- **Element Interactions:** Enabled
- **Use Case:** Zero-config event tracking

---

## Integration Points

### 1. Main Website (`index.html`)

**Location:** `/index.html`

Add the snippet in the `<head>` section:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Istanbul - Your Istanbul Travel Companion</title>
    
    <!-- Amplitude Analytics -->
    <script src="https://cdn.amplitude.com/libs/analytics-browser-2.11.1-min.js.gz"></script>
    <script src="https://cdn.amplitude.com/libs/plugin-session-replay-browser-1.23.2-min.js.gz"></script>
    <script>
      window.amplitude.add(window.sessionReplay.plugin({sampleRate: 1}));
      window.amplitude.init('d1288055adec91b66d5bce71829fc4d', {
        "autocapture": {
          "elementInteractions": true
        }
      });
    </script>
    
    <!-- Rest of your head content -->
</head>
<body>
    <!-- Your content -->
</body>
</html>
```

### 2. Admin Dashboard (`admin/dashboard.html`)

**Location:** `/admin/dashboard.html`

Same snippet in the `<head>` section to track admin activities.

### 3. Chat Interface (if separate page)

Add to any standalone chat pages.

---

## Custom Event Tracking

### Basic Event Tracking

Track custom events using:

```javascript
amplitude.track('Sign Up');
```

### Events with Properties

```javascript
amplitude.track('Route Requested', {
  origin: 'Taksim',
  destination: 'Sultanahmet',
  transport_mode: 'tram',
  estimated_time: 20
});
```

### User Identification

```javascript
// Identify user
amplitude.setUserId('user-123');

// Set user properties
amplitude.identify(
  new amplitude.Identify()
    .set('language', 'tr')
    .set('user_type', 'tourist')
    .set('location', 'Istanbul')
);
```

---

## Recommended Custom Events for AI Istanbul

### Chat & Query Events

```javascript
// User sends a chat message
amplitude.track('Chat Message Sent', {
  message_length: message.length,
  intent: 'route_planning',
  confidence: 0.95,
  language: 'tr',
  has_location: true
});

// Bot responds
amplitude.track('Bot Response Generated', {
  response_type: 'route_planning',
  response_time: 1.2,
  enhanced_with_llm: true,
  phase_used: 'pure_llm'
});
```

### Route Planning Events

```javascript
// Route requested
amplitude.track('Route Requested', {
  origin: 'Taksim',
  destination: 'Sultanahmet',
  transport_mode: 'public_transport',
  has_preferences: true,
  preference_type: 'budget_friendly'
});

// Route selected
amplitude.track('Route Selected', {
  route_id: 'route_123',
  transport_mode: 'tram',
  duration_minutes: 20,
  cost_tl: 15,
  alternative_rank: 1
});

// Navigation started
amplitude.track('Navigation Started', {
  route_id: 'route_123',
  gps_enabled: true,
  estimated_arrival: '2025-12-07T15:30:00'
});
```

### Hidden Gems & Places

```javascript
// Gem discovered
amplitude.track('Gem Discovered', {
  gem_name: 'Secret Rooftop Cafe',
  gem_category: 'cafe',
  discovery_method: 'chat',
  distance_km: 2.5
});

// Place search
amplitude.track('Place Search', {
  search_term: 'cafes near Taksim',
  results_count: 12,
  filter_applied: 'budget_friendly'
});
```

### Admin Dashboard Events

```javascript
// Experiment created
amplitude.track('Experiment Created', {
  experiment_name: 'LLM Prompt Test',
  variant_count: 2,
  duration_days: 14,
  metrics: ['accuracy', 'latency']
});

// Feature flag toggled
amplitude.track('Feature Flag Toggled', {
  flag_name: 'new_context_resolution',
  enabled: true,
  rollout_percentage: 25
});

// Learning cycle run
amplitude.track('Learning Cycle Executed', {
  patterns_learned: 5,
  feedback_analyzed: 120,
  improvements_deployed: true
});
```

### User Engagement Events

```javascript
// Page view (automatically tracked with autocapture)
// But you can add custom properties:
amplitude.track('Page Viewed', {
  page_name: 'home',
  referrer: document.referrer,
  user_language: navigator.language
});

// User feedback
amplitude.track('Feedback Submitted', {
  feedback_type: 'misclassification',
  original_intent: 'route_planning',
  correct_intent: 'hidden_gem',
  confidence: 0.75
});

// Error occurred
amplitude.track('Error Occurred', {
  error_type: 'api_error',
  error_message: 'Failed to load route',
  endpoint: '/api/route',
  status_code: 500
});
```

---

## Implementation Examples

### 1. Chat Interface Integration

```javascript
// In your chat handler
async function sendChatMessage(message) {
  const startTime = Date.now();
  
  // Track message sent
  amplitude.track('Chat Message Sent', {
    message_length: message.length,
    timestamp: new Date().toISOString()
  });
  
  try {
    // Call API
    const response = await fetch('/api/chat/pure-llm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message})
    });
    
    const data = await response.json();
    const responseTime = (Date.now() - startTime) / 1000;
    
    // Track response
    amplitude.track('Bot Response Generated', {
      intent: data.intent,
      confidence: data.confidence,
      response_time: responseTime,
      method: data.method || 'pure_llm'
    });
    
    return data;
    
  } catch (error) {
    // Track error
    amplitude.track('Chat Error', {
      error_message: error.message,
      error_type: error.name
    });
    throw error;
  }
}
```

### 2. Route Planning Integration

```javascript
// Track route request
function requestRoute(origin, destination, preferences) {
  amplitude.track('Route Requested', {
    origin: origin.name,
    destination: destination.name,
    has_preferences: !!preferences,
    preference_types: preferences ? Object.keys(preferences) : []
  });
  
  // Make API call...
}

// Track route selection
function selectRoute(route) {
  amplitude.track('Route Selected', {
    transport_mode: route.mode,
    duration_minutes: route.duration / 60,
    cost_tl: route.cost,
    distance_km: route.distance / 1000
  });
  
  // Start navigation...
}
```

### 3. Admin Dashboard Integration

```javascript
// Track experiment creation
async function createExperiment(experimentData) {
  try {
    const response = await fetch('/api/admin/experiments/experiments', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(experimentData)
    });
    
    const data = await response.json();
    
    if (data.success) {
      amplitude.track('Experiment Created', {
        experiment_name: experimentData.name,
        variant_count: Object.keys(experimentData.variants).length,
        metrics: experimentData.metrics,
        duration_days: calculateDuration(experimentData.start_date, experimentData.end_date)
      });
    }
    
    return data;
  } catch (error) {
    amplitude.track('Admin Error', {
      action: 'create_experiment',
      error: error.message
    });
  }
}
```

---

## User Properties to Track

Set these properties to segment users:

```javascript
// On user identification
amplitude.identify(
  new amplitude.Identify()
    .set('user_type', 'tourist') // tourist, local, business
    .set('language', 'tr') // tr, en
    .set('device_type', 'mobile') // mobile, desktop, tablet
    .set('signup_date', '2025-12-07')
    .set('total_queries', 42)
    .set('favorite_transport', 'tram')
    .set('preferred_budget', 'moderate')
);

// Increment counters
amplitude.identify(
  new amplitude.Identify()
    .add('total_routes_requested', 1)
    .add('total_gems_discovered', 1)
);
```

---

## Privacy & Compliance

### GDPR Compliance

```javascript
// Disable tracking for opted-out users
amplitude.setOptOut(true);

// Re-enable
amplitude.setOptOut(false);
```

### User Consent

```javascript
// Only initialize after user consent
if (userHasConsented) {
  amplitude.init('d1288055adec91b66d5bce71829fc4d', {
    autocapture: { elementInteractions: true }
  });
}
```

---

## Debugging

### Enable Debug Mode

```javascript
amplitude.init('d1288055adec91b66d5bce71829fc4d', {
  autocapture: { elementInteractions: true },
  logLevel: amplitude.Types.LogLevel.Debug
});
```

### Check Events in Console

Open browser console to see events being sent:
```
[Amplitude] Event: Chat Message Sent
[Amplitude] Properties: {message_length: 42, timestamp: "2025-12-07..."}
```

---

## Testing

### 1. Verify Installation

Open browser console and type:
```javascript
amplitude.track('Test Event', {test: true});
```

Check Amplitude dashboard for the event.

### 2. Test Session Replay

- Navigate through your app
- Click buttons, fill forms
- Check Amplitude dashboard → Session Replay

### 3. Test Autocapture

- Click any button
- Check Amplitude dashboard → Element Clicked events

---

## Dashboard Setup

### Recommended Charts

1. **Chat Activity**
   - Event: Chat Message Sent
   - Segment by: intent, language
   - Chart type: Line over time

2. **Route Planning Funnel**
   - Steps: Route Requested → Route Selected → Navigation Started
   - Chart type: Funnel

3. **User Engagement**
   - Events: Page Viewed, Chat Message Sent, Route Requested
   - Chart type: Stacked area

4. **Error Tracking**
   - Event: Error Occurred
   - Group by: error_type
   - Chart type: Bar

5. **A/B Test Performance**
   - Event: Experiment Created
   - Metrics: variant_count, completion_rate
   - Chart type: Bar comparison

---

## Best Practices

### ✅ Do's
- Track key user actions (chat, routes, gems)
- Use descriptive event names ("Route Requested" not "route_req")
- Include relevant properties
- Set user properties for segmentation
- Test in development first

### ❌ Don'ts
- Don't track PII (names, emails) without consent
- Don't send sensitive data (passwords, tokens)
- Don't track every tiny interaction (reduces signal)
- Don't forget to handle errors
- Don't mix tracking logic with business logic

---

## Implementation Checklist

- [ ] Add Amplitude snippet to `index.html`
- [ ] Add Amplitude snippet to `admin/dashboard.html`
- [ ] Add snippet to any other HTML pages
- [ ] Implement chat tracking in JavaScript
- [ ] Implement route planning tracking
- [ ] Implement admin dashboard tracking
- [ ] Set up user identification
- [ ] Test events in development
- [ ] Verify events in Amplitude dashboard
- [ ] Set up key charts and funnels
- [ ] Configure alerts for errors
- [ ] Add privacy controls (opt-out)

---

## Next Steps

1. **Immediate:** Add the snippet to all HTML pages
2. **Week 1:** Implement custom events for core features
3. **Week 2:** Set up dashboard and charts
4. **Week 3:** Analyze data and optimize
5. **Ongoing:** Monitor and iterate

---

## Support Resources

- **Amplitude Docs:** https://www.docs.developers.amplitude.com/
- **Browser SDK:** https://www.docs.developers.amplitude.com/data/sdks/browser-2/
- **Event Taxonomy:** https://help.amplitude.com/hc/en-us/articles/360047138392

---

**Status:** Ready to implement! 🚀
