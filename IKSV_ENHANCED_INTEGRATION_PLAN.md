# İKSV Events Integration - Enhanced URL Configuration
# This file documents the enhanced URL structure for comprehensive İKSV event scraping

## Enhanced İKSV URL Coverage

### Main İKSV Portal
- https://www.iksv.org/en - Main English site
- https://www.iksv.org/tr - Main Turkish site
- https://www.iksv.org/en#event-calendar-section - Calendar section (EN)
- https://www.iksv.org/tr#event-calendar-section - Calendar section (TR)

### Festival-Specific Sites
1. **Istanbul Music Festival** (June)
   - https://muzik.iksv.org/en
   - https://muzik.iksv.org/tr

2. **Istanbul Film Festival** (April)
   - https://film.iksv.org/en
   - https://film.iksv.org/tr

3. **Istanbul Jazz Festival** (July)
   - https://caz.iksv.org/en
   - https://caz.iksv.org/tr

4. **Istanbul Theatre Festival** (May-June)
   - https://tiyatro.iksv.org/en
   - https://tiyatro.iksv.org/tr

5. **Istanbul Biennial** (September, odd years)
   - https://bienal.iksv.org/en
   - https://bienal.iksv.org/tr

### Venue-Specific Sites
1. **Salon İKSV** (Regular concerts and performances)
   - https://www.iksv.org/en/salon
   - https://www.iksv.org/tr/salon

2. **General Events Pages**
   - https://www.iksv.org/en/events
   - https://www.iksv.org/tr/etkinlikler

## Enhanced Data Fields

### Current Fields
- title: Event title
- venue: Venue name
- date_str: Date string
- source: "İKSV" or "İKSV Sample"
- fetched_at: Timestamp
- event_number: Sequential ID

### Proposed Additional Fields
- **artist**: Performer/artist name(s) for concerts/shows
- **ticket_url**: Direct link to ticket purchase
- **price_range**: Price information (e.g., "50-100 TL", "Free")
- **category**: Event category (Music, Film, Theatre, Art, etc.)
- **festival**: Parent festival name if applicable
- **language**: Performance language(s)
- **duration**: Event duration in minutes
- **accessibility**: Accessibility information
- **age_restriction**: Age requirements if any
- **description**: Detailed event description
- **image_url**: Event poster/image URL

## Implementation Priorities

### Phase 1: URL Expansion ✅ COMPLETE
- ✅ Add all festival-specific domains
- ✅ Add Salon İKSV regular events
- ✅ Add bilingual support (EN/TR scraping)
- ✅ Implement domain-specific parsing logic
- ✅ **MAJOR: Selenium integration working!** (7 real events extracted)

### Phase 2: Data Quality Enhancement
- 📋 Extract ticket URLs and pricing
- 📋 Extract artist/performer information
- 📋 Improve category detection
- 📋 Add festival parent linking

### Phase 3: Advanced Features
- 📋 Implement pagination for large event lists
- 📋 Add image/poster scraping
- 📋 Venue coordinate mapping
- 📋 Integration with official İKSV API (if available)

### Phase 4: Real-time Updates
- 📋 Weekly refresh schedule
- 📋 Event change detection
- 📋 Notification system for new events
- 📋 Better cache management

## Venue Mapping Enhancement

### Current İKSV Venues
- Zorlu PSM (Zorlu Performing Arts Center)
- Harbiye Muhsin Ertuğrul Stage
- Salon İKSV
- İKSV Cultural Center
- Multiple venues (for Biennial)

### Proposed Venue Database
```python
IKSV_VENUES = {
    'salon_iksv': {
        'name': {'tr': 'Salon İKSV', 'en': 'Salon İKSV'},
        'address': 'Sadi Konuralp Caddesi No:5 Şişhane, Beyoğlu',
        'capacity': 300,
        'coordinates': {'lat': 41.0272, 'lng': 28.9744},
        'accessibility': True,
    },
    'zorlu_psm': {
        'name': {'tr': 'Zorlu PSM', 'en': 'Zorlu PSM'},
        'address': 'Levazım, Koru Sokağı No:2, 34340 Beşiktaş',
        'capacity': 2000,
        'coordinates': {'lat': 41.0688, 'lng': 29.0094},
        'accessibility': True,
    },
    # Add more venues...
}
```

## Error Handling Improvements

### Current Issues
- SSL certificate verification disabled (security risk)
- Limited retry logic
- No rate limiting
- Generic error messages

### Proposed Enhancements
- Implement proper SSL certificate handling
- Add exponential backoff retry
- Rate limiting between requests
- Detailed error logging with context
- Fallback to cached data on failures

## Testing Strategy

### Unit Tests
- Test each URL domain scraping independently
- Validate data field extraction
- Test duplicate detection
- Verify bilingual support

### Integration Tests
- End-to-end event fetching and caching
- Database integration validation
- Search and filter functionality
- API endpoint testing

### Performance Tests
- Measure scraping time per domain
- Cache hit/miss rates
- Memory usage monitoring
- Concurrent request handling

## Next Steps

1. **Implement Enhanced URL List** (Priority: High)
   - Update `fetch_iksv_events()` to include all domains
   - Add domain-specific parsing hints
   - Test each domain individually

2. **Add Enhanced Data Fields** (Priority: Medium)
   - Extend event data structure
   - Update parser to extract additional fields
   - Update database schema if needed

3. **Improve Error Handling** (Priority: Medium)
   - Add retry logic with backoff
   - Implement rate limiting
   - Better logging and monitoring

4. **Venue Database Integration** (Priority: Low)
   - Create comprehensive venue mapping
   - Add coordinate data for map integration
   - Link events to venue details

5. **API Research** (Priority: Low)
   - Check if İKSV has official API
   - Document API endpoints if available
   - Plan migration from scraping to API
