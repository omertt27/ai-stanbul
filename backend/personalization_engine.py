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
    
    def get_cultural_context(self, query: str) -> List[Dict]:
        """Get relevant cultural tips and Turkish phrases"""
        # Determine context from query
        context_keywords = {
            'restaurant': ['food', 'restaurant', 'eating', 'dining'],
            'shopping': ['shopping', 'bazaar', 'market', 'buying'],
            'mosque': ['mosque', 'prayer', 'religious'],
            'transportation': ['taxi', 'bus', 'metro', 'transport']
        }
        
        relevant_categories = []
        query_lower = query.lower()
        
        for category, keywords in context_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_categories.append(category)
        
        # If no specific context, add general tips
        if not relevant_categories:
            relevant_categories = ['general']
        
        # Get Turkish phrases
        phrases = self.db.query(TurkishPhrases).filter(
            TurkishPhrases.category.in_(relevant_categories + ['essential'])
        ).limit(5).all()
        
        # Get local tips
        tips = self.db.query(LocalTips).filter(
            LocalTips.category.in_(relevant_categories + ['culture']),
            LocalTips.importance_level.in_(['essential', 'helpful'])
        ).limit(3).all()
        
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
                    'title': tip.tip_title,
                    'content': tip.tip_content,
                    'importance': tip.importance_level
                } for tip in tips
            ]
        }
        
        return cultural_info

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
