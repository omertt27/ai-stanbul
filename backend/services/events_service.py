"""
Events Service
Handle event queries with temporal parsing and IKSV integration
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import calendar
import logging
import sys
import os

# Add parent directory to path for events database import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import events database functions
try:
    from data.events_database import (
        load_live_iksv_events,
        get_all_events,
        get_events_by_month,
        get_current_and_upcoming_events,
        search_events,
        get_iksv_events_only,
        get_live_events_metadata
    )
    EVENTS_DATABASE_AVAILABLE = True
except ImportError:
    try:
        # Fallback for different import context
        from backend.data.events_database import (
            load_live_iksv_events,
            get_all_events,
            get_events_by_month,
            get_current_and_upcoming_events,
            search_events,
            get_iksv_events_only,
            get_live_events_metadata
        )
        EVENTS_DATABASE_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ Events database not available: {e}")
        EVENTS_DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EventsService:
    """Real-time and curated events information with temporal parsing"""
    
    def __init__(self):
        self.events_cache = {}
        self.recurring_events = self._load_recurring_events()
        self.seasonal_events = self._load_seasonal_events()
        self.venues = self._load_venues()
        
        # Load İKSV events from database
        self.iksv_events = []
        if EVENTS_DATABASE_AVAILABLE:
            try:
                self.iksv_events = load_live_iksv_events()
                logger.info(f"✅ Loaded {len(self.iksv_events)} İKSV events from database")
            except Exception as e:
                logger.warning(f"Could not load İKSV events: {e}")
                self.iksv_events = []
        
    def _load_recurring_events(self) -> Dict[str, List[Dict]]:
        """Load weekly recurring events"""
        return {
            'monday': [
                {
                    'name': 'Kadıköy Organic Market',
                    'time': '09:00-18:00',
                    'type': 'market',
                    'location': 'Kadıköy Moda',
                    'description': 'Fresh organic produce from local farmers'
                }
            ],
            'tuesday': [
                {
                    'name': 'Beşiktaş Fish Market',
                    'time': '06:00-14:00',
                    'type': 'market',
                    'location': 'Beşiktaş waterfront',
                    'description': 'Fresh fish from Bosphorus and Black Sea'
                }
            ],
            'wednesday': [
                {
                    'name': 'Karaköy Food Market',
                    'time': '09:00-18:00',
                    'type': 'market',
                    'location': 'Karaköy Square',
                    'description': 'Street food and local produce'
                }
            ],
            'thursday': [
                {
                    'name': 'Cihangir Antique Market',
                    'time': '10:00-19:00',
                    'type': 'market',
                    'location': 'Cihangir neighborhood',
                    'description': 'Vintage items and antiques'
                }
            ],
            'friday': [
                {
                    'name': 'Friday Prayers at Blue Mosque',
                    'time': '13:00-14:00',
                    'type': 'religious',
                    'location': 'Sultanahmet',
                    'visitor_note': 'Non-Muslims welcome but dress modestly, observe from designated areas'
                },
                {
                    'name': 'Balat Street Market',
                    'time': '09:00-18:00',
                    'type': 'market',
                    'location': 'Balat neighborhood',
                    'description': 'Colorful streets with local vendors'
                },
                {
                    'name': 'Weekend Nightlife Start',
                    'time': '22:00-04:00',
                    'type': 'nightlife',
                    'location': 'Beyoğlu, Karaköy, Bebek',
                    'description': 'Clubs and bars come alive on Friday nights'
                }
            ],
            'saturday': [
                {
                    'name': 'Beşiktaş Farmers Market',
                    'time': '08:00-18:00',
                    'type': 'market',
                    'location': 'Beşiktaş Square',
                    'description': 'Large weekly market with fresh produce, cheese, olives'
                },
                {
                    'name': 'Live Music at Babylon',
                    'time': '21:00',
                    'type': 'music',
                    'location': 'Beyoğlu',
                    'note': 'Check schedule for specific artists',
                    'website': 'babylon.com.tr'
                },
                {
                    'name': 'Nardis Jazz Club',
                    'time': '21:30',
                    'type': 'music',
                    'location': 'Galata',
                    'description': 'Live jazz performances',
                    'website': 'nardisjazz.com'
                },
                {
                    'name': 'Ortaköy Street Performances',
                    'time': '14:00-22:00',
                    'type': 'entertainment',
                    'location': 'Ortaköy Square',
                    'description': 'Street musicians and artists'
                }
            ],
            'sunday': [
                {
                    'name': 'Flea Market at Kadıköy',
                    'time': '10:00-19:00',
                    'type': 'market',
                    'location': 'Moda coast, Kadıköy',
                    'description': 'Vintage items, books, antiques, handicrafts'
                },
                {
                    'name': 'Bomonti Sunday Brunch Scene',
                    'time': '10:00-16:00',
                    'type': 'food',
                    'location': 'Bomonti neighborhood',
                    'description': 'Popular brunch spots with live music'
                },
                {
                    'name': 'Istiklal Street Performers',
                    'time': '12:00-22:00',
                    'type': 'entertainment',
                    'location': 'Istiklal Street',
                    'description': 'Street musicians, dancers, and artists'
                }
            ]
        }
    
    def _load_seasonal_events(self) -> Dict[str, List[Dict]]:
        """Load seasonal and annual events"""
        return {
            'spring': [
                {
                    'name': 'Istanbul Tulip Festival',
                    'month': 'April',
                    'type': 'festival',
                    'description': 'Millions of tulips bloom across the city',
                    'locations': ['Emirgan Park', 'Gülhane Park', 'Sultanahmet Square']
                },
                {
                    'name': 'Istanbul Film Festival',
                    'month': 'April',
                    'type': 'cultural',
                    'description': 'International and Turkish cinema',
                    'organizer': 'İKSV'
                },
                {
                    'name': 'Istanbul Coffee Festival',
                    'month': 'May',
                    'type': 'food',
                    'description': 'Coffee culture and tasting events'
                }
            ],
            'summer': [
                {
                    'name': 'Istanbul Music Festival',
                    'month': 'June',
                    'type': 'music',
                    'description': 'Classical music and opera',
                    'organizer': 'İKSV',
                    'venues': ['Multiple venues across Istanbul']
                },
                {
                    'name': 'Istanbul Jazz Festival',
                    'month': 'July',
                    'type': 'music',
                    'description': 'International and local jazz artists',
                    'organizer': 'İKSV'
                },
                {
                    'name': 'Bosphorus Night Cruises',
                    'months': 'June-September',
                    'type': 'activity',
                    'description': 'Evening boat tours with dinner and entertainment'
                },
                {
                    'name': 'Outdoor Cinema Events',
                    'months': 'June-August',
                    'type': 'cultural',
                    'description': 'Open-air movie screenings',
                    'locations': ['Various parks and rooftops']
                }
            ],
            'fall': [
                {
                    'name': 'Istanbul Biennial',
                    'month': 'September-November (odd years)',
                    'type': 'art',
                    'description': 'Contemporary art exhibition',
                    'organizer': 'İKSV'
                },
                {
                    'name': 'Contemporary Istanbul Art Fair',
                    'month': 'November',
                    'type': 'art',
                    'description': 'Major contemporary art fair'
                },
                {
                    'name': 'Akbank Jazz Festival',
                    'month': 'October',
                    'type': 'music',
                    'description': 'Jazz, blues, and world music'
                }
            ],
            'winter': [
                {
                    'name': 'New Year Celebrations',
                    'month': 'December 31',
                    'type': 'celebration',
                    'description': 'Fireworks, parties, special dinners',
                    'locations': ['Taksim Square', 'Ortaköy', 'Bosphorus boats']
                },
                {
                    'name': 'Christmas Markets',
                    'month': 'December',
                    'type': 'market',
                    'description': 'Limited but growing tradition',
                    'locations': ['Nişantaşı', 'Beyoğlu']
                },
                {
                    'name': 'Istanbul Shopping Fest',
                    'month': 'January-February',
                    'type': 'shopping',
                    'description': 'City-wide sales and discounts'
                }
            ]
        }
    
    def _load_venues(self) -> Dict[str, Dict]:
        """Load regular event venues"""
        return {
            'babylon': {
                'name': 'Babylon',
                'type': 'music_venue',
                'location': 'Beyoğlu',
                'description': 'Live music, rock, alternative, electronic',
                'website': 'babylon.com.tr',
                'typical_events': 'Concerts 2-3 times per week'
            },
            'salon_iksv': {
                'name': 'Salon İKSV',
                'type': 'cultural_center',
                'location': 'Beyoğlu',
                'description': 'Concerts, theater, talks',
                'website': 'saloniks.iksv.org',
                'typical_events': 'Multiple events weekly'
            },
            'nardis_jazz': {
                'name': 'Nardis Jazz Club',
                'type': 'jazz_club',
                'location': 'Galata',
                'description': 'Live jazz every night',
                'website': 'nardisjazz.com',
                'typical_events': 'Daily jazz performances at 21:30'
            },
            'zorlu_psm': {
                'name': 'Zorlu PSM',
                'type': 'performing_arts',
                'location': 'Zincirlikuyu',
                'description': 'Theater, concerts, musicals',
                'typical_events': 'Major performances and shows'
            }
        }
    
    def parse_temporal_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse temporal expressions from query"""
        query_lower = query.lower()
        now = datetime.now()
        
        # Today
        if 'today' in query_lower:
            return {
                'type': 'specific_day',
                'date': now,
                'label': 'today'
            }
        
        # Tonight
        if 'tonight' in query_lower:
            return {
                'type': 'tonight',
                'date': now,
                'label': 'tonight'
            }
        
        # Tomorrow
        if 'tomorrow' in query_lower:
            tomorrow = now + timedelta(days=1)
            return {
                'type': 'specific_day',
                'date': tomorrow,
                'label': 'tomorrow'
            }
        
        # This weekend
        if 'this weekend' in query_lower or 'weekend' in query_lower:
            # Find next Saturday
            days_until_saturday = (5 - now.weekday()) % 7
            if days_until_saturday == 0 and now.weekday() == 5:  # Already Saturday
                saturday = now
            else:
                saturday = now + timedelta(days=days_until_saturday if days_until_saturday > 0 else 7)
            sunday = saturday + timedelta(days=1)
            
            return {
                'type': 'weekend',
                'dates': [saturday, sunday],
                'label': 'this weekend'
            }
        
        # This week
        if 'this week' in query_lower:
            days_remaining = 7 - now.weekday()
            end_of_week = now + timedelta(days=days_remaining)
            return {
                'type': 'week',
                'start_date': now,
                'end_date': end_of_week,
                'label': 'this week'
            }
        
        # This month
        if 'this month' in query_lower:
            last_day = calendar.monthrange(now.year, now.month)[1]
            end_of_month = datetime(now.year, now.month, last_day)
            return {
                'type': 'month',
                'start_date': now,
                'end_date': end_of_month,
                'label': 'this month'
            }
        
        # Specific day names
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(days):
            if day in query_lower:
                current_day = now.weekday()
                days_until = (i - current_day) % 7
                if days_until == 0:
                    days_until = 7  # Next week if today
                target_date = now + timedelta(days=days_until)
                return {
                    'type': 'specific_day',
                    'date': target_date,
                    'label': day
                }
        
        return None
    
    def get_events_by_timeframe(self, timeframe_data: Dict[str, Any]) -> List[Dict]:
        """Get events for parsed timeframe, including İKSV events"""
        if not timeframe_data:
            return []
        
        events = []
        timeframe_type = timeframe_data['type']
        
        # Helper function to check if İKSV event matches timeframe
        def matches_iksv_event(event: Dict, start_date: datetime, end_date: datetime) -> bool:
            """Check if İKSV event falls within timeframe"""
            event_date_str = event.get('date', '')
            if not event_date_str:
                return False
            
            try:
                # Parse various date formats
                from dateutil import parser
                event_date = parser.parse(event_date_str)
                return start_date.date() <= event_date.date() <= end_date.date()
            except:
                # Fallback: check if date string contains month/year
                try:
                    if '-' in event_date_str:  # Format: YYYY-MM-DD
                        event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
                        return start_date.date() <= event_date.date() <= end_date.date()
                except:
                    pass
                return False
        
        if timeframe_type == 'specific_day':
            date = timeframe_data['date']
            day_name = date.strftime('%A').lower()
            
            # Get recurring events for this day
            day_events = self.recurring_events.get(day_name, [])
            for event in day_events:
                events.append({
                    **event,
                    'date': date.strftime('%B %d, %Y'),
                    'day': day_name.capitalize(),
                    'source': 'recurring'
                })
            
            # Add İKSV events for this specific day
            for iksv_event in self.iksv_events:
                if matches_iksv_event(iksv_event, date, date):
                    events.append({
                        **iksv_event,
                        'source': 'iksv',
                        'day': day_name.capitalize()
                    })
        
        elif timeframe_type == 'tonight':
            date = timeframe_data['date']
            day_name = date.strftime('%A').lower()
            
            # Filter for evening events
            day_events = self.recurring_events.get(day_name, [])
            for event in day_events:
                time_str = event.get('time', '')
                # Check if it's an evening event (after 18:00)
                if ':' in time_str:
                    try:
                        hour = int(time_str.split(':')[0].split('-')[0])
                        if hour >= 18 or hour <= 4:  # Evening or late night
                            events.append({
                                **event,
                                'date': date.strftime('%B %d, %Y'),
                                'day': day_name.capitalize(),
                                'timing': 'tonight',
                                'source': 'recurring'
                            })
                    except:
                        pass
            
            # Add İKSV events that are suitable for tonight
            for iksv_event in self.iksv_events:
                if matches_iksv_event(iksv_event, date, date):
                    # Check if it's an evening event
                    event_type = iksv_event.get('type', '').lower()
                    if any(keyword in event_type for keyword in ['concert', 'theater', 'performance', 'show']):
                        events.append({
                            **iksv_event,
                            'source': 'iksv',
                            'timing': 'tonight',
                            'day': day_name.capitalize()
                        })
        
        elif timeframe_type == 'weekend':
            for date in timeframe_data['dates']:
                day_name = date.strftime('%A').lower()
                day_events = self.recurring_events.get(day_name, [])
                for event in day_events:
                    events.append({
                        **event,
                        'date': date.strftime('%B %d, %Y'),
                        'day': day_name.capitalize(),
                        'source': 'recurring'
                    })
                
                # Add İKSV events for weekend days
                for iksv_event in self.iksv_events:
                    if matches_iksv_event(iksv_event, date, date):
                        # Avoid duplicates
                        if not any(e.get('id') == iksv_event.get('id') for e in events):
                            events.append({
                                **iksv_event,
                                'source': 'iksv',
                                'day': day_name.capitalize()
                            })
        
        elif timeframe_type in ['week', 'month']:
            # Get all recurring events for the period
            start = timeframe_data['start_date']
            end = timeframe_data['end_date']
            current = start
            
            while current <= end:
                day_name = current.strftime('%A').lower()
                day_events = self.recurring_events.get(day_name, [])
                for event in day_events:
                    events.append({
                        **event,
                        'date': current.strftime('%B %d, %Y'),
                        'day': day_name.capitalize(),
                        'source': 'recurring'
                    })
                current += timedelta(days=1)
            
            # Add all İKSV events in this timeframe
            for iksv_event in self.iksv_events:
                if matches_iksv_event(iksv_event, start, end):
                    # Avoid duplicates
                    if not any(e.get('id') == iksv_event.get('id') for e in events):
                        events.append({
                            **iksv_event,
                            'source': 'iksv'
                        })
        
        return events
    
    def get_seasonal_events(self) -> List[Dict]:
        """Get current season's events"""
        now = datetime.now()
        month = now.month
        
        if month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        elif month in [9, 10, 11]:
            season = 'fall'
        else:
            season = 'winter'
        
        return self.seasonal_events.get(season, [])
    
    def format_events_response(self, events: List[Dict], timeframe_label: str = None, 
                               include_iksv: bool = True) -> str:
        """Format events into readable response with İKSV integration"""
        if not events and not include_iksv:
            return self._get_no_events_response(timeframe_label)
        
        response = f"🎭 **Events in Istanbul"
        if timeframe_label:
            response += f" {timeframe_label.title()}"
        response += "**\n\n"
        
        if events:
            # Separate İKSV events from recurring events
            iksv_events = [e for e in events if e.get('source') == 'iksv']
            recurring_events = [e for e in events if e.get('source') == 'recurring']
            
            # Show İKSV events first (they're the real featured events!)
            if iksv_events:
                response += "🎪 **Featured İKSV Events:**\n\n"
                for event in iksv_events[:5]:  # Show top 5 İKSV events
                    # Get bilingual name
                    name = event.get('name', {})
                    if isinstance(name, dict):
                        event_name = name.get('en', name.get('tr', 'Event'))
                    else:
                        event_name = name
                    
                    response += f"• **{event_name}**\n"
                    
                    # Date
                    if event.get('date'):
                        response += f"  📅 {event['date']}\n"
                    
                    # Venue
                    venue = event.get('venue', {})
                    if isinstance(venue, dict):
                        venue_name = venue.get('en', venue.get('tr', ''))
                    else:
                        venue_name = venue
                    if venue_name:
                        response += f"  📍 {venue_name}\n"
                    
                    # Description (use shorter English version if available)
                    description = event.get('description', {})
                    if isinstance(description, dict):
                        desc_text = description.get('en', description.get('tr', ''))
                    else:
                        desc_text = description
                    if desc_text and len(desc_text) < 150:
                        response += f"  ℹ️ {desc_text}\n"
                    
                    # Tags
                    tags = event.get('tags', [])
                    if tags:
                        response += f"  �️ {', '.join(tags[:3])}\n"
                    
                    # Booking URL
                    if event.get('booking_url'):
                        response += f"  🎟️ Book at: {event['booking_url']}\n"
                    
                    response += "\n"
                
                response += "💡 More events at **iksv.org** and **Biletix.com**\n\n"
            
            # Then show recurring/regular events
            if recurring_events:
                response += "📅 **Regular Events & Activities:**\n\n"
                
                # Group recurring events by type
                grouped = {}
                for event in recurring_events:
                    event_type = event.get('type', 'other')
                    if event_type not in grouped:
                        grouped[event_type] = []
                    grouped[event_type].append(event)
                
                # Format by type
                type_emojis = {
                    'market': '🛍️',
                    'music': '🎵',
                    'cultural': '🎭',
                    'religious': '🕌',
                    'nightlife': '🌙',
                    'food': '🍽️',
                    'entertainment': '🎪',
                    'art': '🎨',
                    'festival': '🎉'
                }
                
                for event_type, type_events in grouped.items():
                    emoji = type_emojis.get(event_type, '📅')
                    response += f"**{emoji} {event_type.replace('_', ' ').title()}:**\n"
                    
                    for event in type_events[:3]:  # Limit to 3 per type
                        response += f"• **{event['name']}**"
                        if event.get('day'):
                            response += f" ({event['day']}"
                            if event.get('time'):
                                response += f", {event['time']}"
                            response += ")"
                        response += "\n"
                        
                        if event.get('location'):
                            response += f"  📍 {event['location']}\n"
                        
                        if event.get('description'):
                            response += f"  ℹ️ {event['description']}\n"
                        
                        response += "\n"
        
        else:
            # No events found, show İKSV info
            if include_iksv:
                response += "💡 **İKSV** (Istanbul Foundation for Culture and Arts):\n"
                response += "   🌐 iksv.org | concerts, theater, exhibitions\n"
                response += "   📍 Venues: Salon İKSV, Babylon, and more\n\n"
        
        # Add seasonal events if available
        seasonal = self.get_seasonal_events()
        if seasonal:
            response += "\n**🌸 This Season:**\n"
            for event in seasonal[:3]:  # Show top 3
                response += f"• **{event['name']}** ({event.get('month', 'TBA')})\n"
                if event.get('description'):
                    response += f"  {event['description']}\n"
        
        response += "\n💡 **Pro Tip:** Book tickets in advance via Biletix or venue websites!\n"
        
        return response
    
    def _get_no_events_response(self, timeframe_label: str = None) -> str:
        """Fallback response when no specific events found"""
        response = f"🎭 **Events in Istanbul"
        if timeframe_label:
            response += f" {timeframe_label.title()}"
        response += "**\n\n"
        
        response += "While I don't have specific events for this timeframe, here's how to find what's happening:\n\n"
        response += "**🎵 Live Music & Concerts:**\n"
        response += "• **İKSV** - iksv.org (Salon, festivals, major events)\n"
        response += "• **Babylon** - babylon.com.tr (Rock, electronic, alternative)\n"
        response += "• **Nardis Jazz Club** - nardisjazz.com (Live jazz every night)\n\n"
        
        response += "**🎭 Cultural Events:**\n"
        response += "• Check İKSV calendar for exhibitions and performances\n"
        response += "• Zorlu PSM for theater and musicals\n"
        response += "• Istanbul Modern for art exhibitions\n\n"
        
        response += "**📅 Regular Events:**\n"
        seasonal = self.get_seasonal_events()
        for event in seasonal[:2]:
            response += f"• {event['name']} ({event.get('month', 'TBA')})\n"
        
        return response


# Singleton instance
_events_service = None

def get_events_service() -> EventsService:
    """Get or create events service instance"""
    global _events_service
    if _events_service is None:
        _events_service = EventsService()
    return _events_service
