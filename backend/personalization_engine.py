#!/usr/bin/env python3
"""
Istanbul Specialized AI Assistant
Provides context-aware, personalized recommendations for Istanbul tourists
"""

import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from specialized_models import UserProfile, TransportRoute, TurkishPhrases, LocalTips
from typing import Dict, List, Optional

class IstanbulPersonalizationEngine:
    """Personalized recommendations engine for Istanbul tourists"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_user_profile(self, session_id: str) -> UserProfile:
        """Get existing user profile or create a new one"""
        profile = self.db.query(UserProfile).filter(
            UserProfile.session_id == session_id
        ).first()
        
        if not profile:
            profile = UserProfile(session_id=session_id)
            self.db.add(profile)
            self.db.commit()
            self.db.refresh(profile)
        
        return profile
    
    def update_user_context(self, session_id: str, **kwargs) -> UserProfile:
        """Update user context from conversation"""
        profile = self.get_or_create_user_profile(session_id)
        
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.last_updated = datetime.utcnow()
        self.db.commit()
        return profile
    
    def extract_user_context(self, user_input: str, session_id: str) -> Dict:
        """Extract user context clues from conversation"""
        context_updates = {}
        user_input_lower = user_input.lower()
        
        # Dietary restrictions
        if any(word in user_input_lower for word in ['vegetarian', 'vegan', 'no meat']):
            context_updates['dietary_restrictions'] = 'vegetarian'
        elif any(word in user_input_lower for word in ['halal', 'muslim food']):
            context_updates['dietary_restrictions'] = 'halal'
        
        # Location context
        districts = ['kadikoy', 'sultanahmet', 'beyoglu', 'besiktas', 'uskudar', 'fatih']
        for district in districts:
            if f'staying in {district}' in user_input_lower or f'hotel in {district}' in user_input_lower:
                context_updates['accommodation_district'] = district.title()
        
        # Time context
        if 'days left' in user_input_lower or 'leaving in' in user_input_lower:
            # Extract number of days (simple regex could be added here)
            for i in range(1, 15):
                if f'{i} day' in user_input_lower:
                    context_updates['days_remaining'] = i
                    break
        
        # Budget context
        if any(word in user_input_lower for word in ['cheap', 'budget', 'affordable']):
            context_updates['budget_level'] = 'budget'
        elif any(word in user_input_lower for word in ['luxury', 'expensive', 'high-end']):
            context_updates['budget_level'] = 'luxury'
        elif any(word in user_input_lower for word in ['mid-range', 'moderate']):
            context_updates['budget_level'] = 'mid-range'
        
        # Update profile if we found context
        if context_updates:
            self.update_user_context(session_id, **context_updates)
        
        return context_updates
    
    def get_personalized_recommendations(self, query: str, session_id: str) -> Dict:
        """Generate personalized recommendations based on user profile"""
        profile = self.get_or_create_user_profile(session_id)
        recommendations = {
            'personalized': True,
            'user_context': {},
            'recommendations': [],
            'transportation': [],
            'cultural_tips': []
        }
        
        # Add user context to response
        if profile.dietary_restrictions:
            recommendations['user_context']['dietary'] = profile.dietary_restrictions
        if profile.accommodation_district:
            recommendations['user_context']['staying_in'] = profile.accommodation_district
        if profile.days_remaining:
            recommendations['user_context']['days_left'] = profile.days_remaining
        if profile.budget_level:
            recommendations['user_context']['budget'] = profile.budget_level
        
        return recommendations
    
    def get_ferry_schedule(self, from_location: str, to_location: str) -> List[Dict]:
        """Get ferry schedules between locations"""
        routes = self.db.query(TransportRoute).filter(
            TransportRoute.transport_type == "ferry",
            TransportRoute.from_location.ilike(f"%{from_location}%"),
            TransportRoute.to_location.ilike(f"%{to_location}%"),
            TransportRoute.is_active == True
        ).all()
        
        ferry_info = []
        for route in routes:
            current_time = datetime.now().time()
            next_departure = self._calculate_next_departure(
                route.first_departure, 
                route.last_departure, 
                route.frequency_minutes, 
                current_time
            )
            
            ferry_info.append({
                'route_name': route.route_name,
                'from': route.from_location,
                'to': route.to_location,
                'duration': f"{route.duration_minutes} minutes",
                'price': f"{route.price_try} TL",
                'next_departure': next_departure,
                'frequency': f"Every {route.frequency_minutes} minutes",
                'notes': route.notes
            })
        
        return ferry_info
    
    def _calculate_next_departure(self, first_dep, last_dep, frequency, current_time):
        """Calculate next ferry departure time"""
        # Simplified calculation - in reality you'd want more sophisticated logic
        if current_time < first_dep:
            return first_dep.strftime("%H:%M")
        elif current_time > last_dep:
            return "Service ended for today"
        else:
            # Calculate approximate next departure
            minutes_since_first = (
                current_time.hour * 60 + current_time.minute -
                first_dep.hour * 60 - first_dep.minute
            )
            next_departure_minutes = ((minutes_since_first // frequency) + 1) * frequency
            next_departure_time = (
                datetime.combine(datetime.today(), first_dep) + 
                timedelta(minutes=next_departure_minutes)
            ).time()
            return next_departure_time.strftime("%H:%M")
    
    def get_cultural_context(self, query: str, location: str = None, session_id: str = None) -> List[Dict]:
        """Get relevant cultural tips and Turkish phrases with diversity and context-awareness"""
        import random
        from sqlalchemy import func
        
        # Determine context from query
        context_keywords = {
            'restaurant': ['food', 'restaurant', 'eating', 'dining', 'meal', 'breakfast', 'lunch', 'dinner'],
            'shopping': ['shopping', 'bazaar', 'market', 'buying', 'shop', 'souvenir'],
            'mosque': ['mosque', 'prayer', 'religious', 'blue mosque', 'hagia sophia'],
            'transportation': ['taxi', 'bus', 'metro', 'transport', 'ferry', 'tram'],
            'places': ['places', 'visit', 'attraction', 'sightseeing', 'landmark', 'museum'],
            'nightlife': ['nightlife', 'bar', 'club', 'drink', 'evening'],
            'culture': ['culture', 'turkish', 'traditional', 'local', 'customs']
        }
        
        relevant_categories = []
        query_lower = query.lower()
        
        for category, keywords in context_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_categories.append(category)
        
        # If no specific context, add general tips
        if not relevant_categories:
            relevant_categories = ['general', 'culture']
        
        # Get location-specific tips if location is provided
        location_specific_tips = []
        if location:
            location_keywords = {
                'sultanahmet': ['mosque', 'culture', 'tourist'],
                'beyoglu': ['nightlife', 'restaurant', 'culture'],
                'galata': ['culture', 'restaurant', 'nightlife'],
                'kadikoy': ['local', 'restaurant', 'shopping'],
                'besiktas': ['local', 'nightlife', 'restaurant'],
                'taksim': ['nightlife', 'shopping', 'transport']
            }
            
            if location.lower() in location_keywords:
                relevant_categories.extend(location_keywords[location.lower()])
        
        # Get diverse tips using randomization
        all_tips = self.db.query(LocalTips).filter(
            LocalTips.category.in_(relevant_categories + ['culture', 'general']),
            LocalTips.importance_level.in_(['essential', 'helpful', 'interesting'])
        ).all()
        
        # Create contextual tips based on query intent - do this first
        contextual_tips = self._generate_contextual_tips(query_lower, location)
        
        # Randomize tip selection to provide variety
        selected_tips = []
        if len(all_tips) > 3:
            # Group tips by category for better variety
            tips_by_category = {}
            for tip in all_tips:
                if tip.category not in tips_by_category:
                    tips_by_category[tip.category] = []
                tips_by_category[tip.category].append(tip)
            
            # Prioritize contextual tips first
            selected_tips.extend(contextual_tips[:2])  # Take up to 2 contextual tips
            
            # Fill remaining slots with database tips (avoiding duplicates)
            remaining_slots = 3 - len(selected_tips)
            if remaining_slots > 0:
                # Select at most 1 tip per category for diversity
                for category_tips in tips_by_category.values():
                    if category_tips and remaining_slots > 0:
                        selected_tips.append(random.choice(category_tips))
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
                
                # If we still need more tips, randomly select from remaining
                if remaining_slots > 0:
                    remaining_tips = [tip for tip in all_tips if tip not in selected_tips]
                    additional_needed = min(remaining_slots, len(remaining_tips))
                    if additional_needed > 0:
                        selected_tips.extend(random.sample(remaining_tips, additional_needed))
        else:
            # If we have few database tips, use contextual tips to fill
            selected_tips.extend(contextual_tips)
            selected_tips.extend(all_tips)
        
        # Combine all tips, ensuring we have unique tips
        seen_titles = set()
        final_tips = []
        for tip in selected_tips:
            title = tip.tip_title if hasattr(tip, 'tip_title') else tip['title']
            if title not in seen_titles:
                final_tips.append(tip)
                seen_titles.add(title)
                if len(final_tips) >= 3:
                    break
        
        # Get contextual Turkish phrases
        phrases = self.db.query(TurkishPhrases).filter(
            TurkishPhrases.category.in_(relevant_categories + ['essential'])
        ).order_by(func.random()).limit(4).all()
        
        cultural_info = {
            'phrases': [
                {
                    'english': phrase.english_phrase,
                    'turkish': phrase.turkish_phrase,
                    'pronunciation': phrase.pronunciation,
                    'context': phrase.context
                } for phrase in phrases
            ],
            'tips': [
                {
                    'title': tip.tip_title if hasattr(tip, 'tip_title') else tip['title'],
                    'content': tip.tip_content if hasattr(tip, 'tip_content') else tip['content'],
                    'importance': getattr(tip, 'importance_level', 'helpful') if hasattr(tip, 'importance_level') else 'helpful'
                } for tip in final_tips  # Use final_tips instead of all_tip_objects
            ]
        }
        
        return cultural_info
    
    def _generate_contextual_tips(self, query_lower: str, location: str = None) -> List[Dict]:
        """Generate dynamic contextual tips based on the query and location"""
        contextual_tips = []
        
        # Location-specific contextual tips
        if location:
            location_lower = location.lower()
            if 'sultanahmet' in location_lower or 'fatih' in location_lower:
                contextual_tips.append({
                    'title': 'Sultanahmet Timing',
                    'content': 'Visit Blue Mosque and Hagia Sophia early morning (8-9 AM) or late afternoon to avoid crowds. Both are within walking distance.',
                    'source': 'contextual'
                })
                contextual_tips.append({
                    'title': 'Historic Peninsula Tips',
                    'content': 'Wear comfortable walking shoes in Sultanahmet - the area has cobblestone streets. Many attractions are clustered together.',
                    'source': 'contextual'
                })
            elif 'beyoglu' in location_lower or 'galata' in location_lower:
                contextual_tips.append({
                    'title': 'Beyoğlu Exploration',
                    'content': 'Walk down Istiklal Street in the evening when it\'s most vibrant. Don\'t miss the historic Galata Tower for panoramic views.',
                    'source': 'contextual'
                })
                contextual_tips.append({
                    'title': 'Galata Bridge Walk',
                    'content': 'Walk across Galata Bridge at sunset for stunning Golden Horn views. Fishermen line the bridge all day.',
                    'source': 'contextual'
                })
            elif 'kadikoy' in location_lower:
                contextual_tips.append({
                    'title': 'Kadıköy Local Experience',
                    'content': 'Kadıköy Tuesday Market is perfect for authentic local shopping. Try balık ekmek (fish sandwich) at the waterfront.',
                    'source': 'contextual'
                })
                contextual_tips.append({
                    'title': 'Asian Side Culture',
                    'content': 'Kadıköy has a more local, less touristy vibe. Great for experiencing authentic Istanbul culture and cuisine.',
                    'source': 'contextual'
                })
            elif 'taksim' in location_lower:
                contextual_tips.append({
                    'title': 'Taksim Square Area',
                    'content': 'Taksim is the main transport hub. From here you can easily reach Beyoğlu, Galata, and other European side neighborhoods.',
                    'source': 'contextual'
                })
            elif 'besiktas' in location_lower:
                contextual_tips.append({
                    'title': 'Beşiktaş Waterfront',
                    'content': 'Don\'t miss the Dolmabahçe Palace and the vibrant waterfront area. Great ferry connections to Asian side.',
                    'source': 'contextual'
                })
        
        # Query-specific contextual tips
        if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
            contextual_tips.append({
                'title': 'Turkish Dining Hours',
                'content': 'Turks eat dinner late (8-10 PM). For lunch, try a "lokanta" (local restaurant) for authentic home-style cooking.',
                'source': 'contextual'
            })
        
        if any(word in query_lower for word in ['shopping', 'bazaar', 'market']):
            contextual_tips.append({
                'title': 'Market Timing',
                'content': 'Visit markets in the morning for the freshest items. Neighborhood pazars (weekly markets) offer the most authentic experience.',
                'source': 'contextual'
            })
        
        if any(word in query_lower for word in ['nightlife', 'bar', 'club']):
            contextual_tips.append({
                'title': 'Istanbul Nightlife',
                'content': 'Nightlife starts late in Istanbul (10 PM+). Rooftop bars offer amazing city views, especially in Beyoğlu and Karaköy.',
                'source': 'contextual'
            })
        
        if any(word in query_lower for word in ['transport', 'metro', 'ferry']):
            contextual_tips.append({
                'title': 'Ferry Experience',
                'content': 'Take a ferry ride across the Bosphorus for scenic views. It\'s often faster than traffic and costs the same as metro.',
                'source': 'contextual'
            })
        
        # Add some general contextual tips based on time/season (these can be random)
        general_tips = [
            {
                'title': 'Best Photo Times',
                'content': 'Golden hour (sunset) from Galata Tower or Pierre Loti offers the most stunning Istanbul photos. Arrive 30 minutes early.',
                'source': 'contextual'
            },
            {
                'title': 'Local Etiquette',
                'content': 'Turks are very hospitable. If invited for tea, it\'s polite to accept. Small gifts from your country are appreciated.',
                'source': 'contextual'
            },
            {
                'title': 'Currency Tips',
                'content': 'Turkish Lira is the currency. Many places accept cards, but have some cash for street vendors and small shops.',
                'source': 'contextual'
            },
            {
                'title': 'Language Helper',
                'content': 'Download Google Translate with Turkish offline. Many young people speak English, especially in tourist areas.',
                'source': 'contextual'
            }
        ]
        
        # Add 1-2 random general tips if we don't have enough contextual ones
        import random
        if len(contextual_tips) < 2:
            remaining_needed = 2 - len(contextual_tips)
            contextual_tips.extend(random.sample(general_tips, min(remaining_needed, len(general_tips))))
        
        print(f"DEBUG: Generated {len(contextual_tips)} contextual tips for location '{location}' and query '{query_lower[:50]}'")
        return contextual_tips

def format_personalized_response(
    base_response: str, 
    user_context: Dict, 
    transportation: List[Dict], 
    cultural_info: Dict
) -> str:
    """Format a personalized response with context"""
    
    response_parts = [base_response]
    
    # Add personalized context
    if user_context:
        context_notes = []
        if user_context.get('dietary'):
            context_notes.append(f"🌱 Noted: You prefer {user_context['dietary']} options")
        if user_context.get('staying_in'):
            context_notes.append(f"📍 You're staying in {user_context['staying_in']}")
        if user_context.get('days_left'):
            context_notes.append(f"⏰ {user_context['days_left']} days left to explore")
        
        if context_notes:
            response_parts.append("\n**Personal Context:**\n" + "\n".join(context_notes))
    
    # Add transportation info
    if transportation:
        response_parts.append("\n🚢 **Getting There:**")
        for transport in transportation[:2]:  # Limit to 2 options
            response_parts.append(
                f"• **{transport['route_name']}**: {transport['duration']}, "
                f"next departure {transport['next_departure']}, {transport['price']}"
            )
    
    # Add cultural context
    if cultural_info.get('phrases'):
        response_parts.append("\n🇹🇷 **Useful Turkish Phrases:**")
        for phrase in cultural_info['phrases'][:3]:
            response_parts.append(
                f"• *{phrase['english']}* → **{phrase['turkish']}** "
                f"({phrase['pronunciation']})"
            )
    
    if cultural_info.get('tips'):
        response_parts.append("\n💡 **Local Tips:**")
        for tip in cultural_info['tips'][:2]:
            response_parts.append(f"• **{tip['title']}**: {tip['content']}")
    
    return "\n".join(response_parts)
