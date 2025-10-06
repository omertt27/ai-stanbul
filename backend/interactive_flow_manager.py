#!/usr/bin/env python3
"""
Interactive Flow Manager - UX & Interaction Design
================================================

This system provides guided flows instead of open-ended chat:
1. "Plan my day" → pick attractions → get itinerary
2. Interactive suggestions to guide users
3. Step-by-step flows for common tasks
4. Context-aware follow-up suggestions
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

class FlowType(Enum):
    DAY_PLANNING = "day_planning"
    RESTAURANT_DISCOVERY = "restaurant_discovery"
    MUSEUM_TOUR = "museum_tour"
    TRANSPORTATION_HELP = "transportation_help"
    DISTRICT_EXPLORATION = "district_exploration"
    CULTURAL_EXPERIENCE = "cultural_experience"
    SHOPPING_GUIDE = "shopping_guide"

class FlowStep(Enum):
    INITIAL = "initial"
    PREFERENCE_GATHERING = "preference_gathering"
    OPTIONS_PRESENTATION = "options_presentation"
    SELECTION_CONFIRMATION = "selection_confirmation"
    ITINERARY_GENERATION = "itinerary_generation"
    COMPLETION = "completion"

@dataclass
class FlowState:
    """Current state of an interactive flow"""
    flow_type: FlowType
    current_step: FlowStep
    user_preferences: Dict[str, Any]
    selected_options: List[Dict[str, Any]]
    session_data: Dict[str, Any]
    next_suggestions: List[str]

@dataclass
class InteractiveOption:
    """An interactive option presented to the user"""
    option_id: str
    title: str
    description: str
    category: str
    metadata: Dict[str, Any]
    quick_actions: List[str]

class InteractiveFlowManager:
    """Manages guided flows and interactive suggestions"""
    
    def __init__(self):
        self.active_flows = {}  # session_id -> FlowState
        self.flow_templates = self._initialize_flow_templates()
        
    def _initialize_flow_templates(self) -> Dict[FlowType, Dict[str, Any]]:
        """Initialize guided flow templates"""
        
        return {
            FlowType.DAY_PLANNING: {
                'name': 'Day Planning Assistant',
                'description': 'Let me help you plan the perfect day in Istanbul',
                'steps': {
                    FlowStep.INITIAL: {
                        'message': '🗓️ **Plan Your Perfect Istanbul Day**\n\nI\'ll help you create a personalized itinerary! First, tell me:',
                        'options': [
                            {'id': 'duration_half', 'title': 'Half Day (4-5 hours)', 'icon': '⏱️'},
                            {'id': 'duration_full', 'title': 'Full Day (8-10 hours)', 'icon': '📅'},
                            {'id': 'duration_evening', 'title': 'Evening Only (3-4 hours)', 'icon': '🌆'}
                        ],
                        'quick_actions': ['Skip to recommendations', 'I need transport help first']
                    },
                    FlowStep.PREFERENCE_GATHERING: {
                        'message': '🎯 **What interests you most?** (Select all that apply)',
                        'options': [
                            {'id': 'history', 'title': 'Historical Sites', 'icon': '🏛️', 'desc': 'Mosques, palaces, ancient sites'},
                            {'id': 'food', 'title': 'Food & Dining', 'icon': '🍽️', 'desc': 'Local cuisine, restaurants, food tours'},
                            {'id': 'culture', 'title': 'Arts & Culture', 'icon': '🎭', 'desc': 'Museums, galleries, cultural experiences'},
                            {'id': 'views', 'title': 'Scenic Views', 'icon': '📸', 'desc': 'Bosphorus, rooftops, panoramic spots'},
                            {'id': 'local', 'title': 'Local Life', 'icon': '🏘️', 'desc': 'Markets, neighborhoods, authentic experiences'},
                            {'id': 'shopping', 'title': 'Shopping', 'icon': '🛍️', 'desc': 'Bazaars, boutiques, souvenirs'}
                        ]
                    },
                    FlowStep.OPTIONS_PRESENTATION: {
                        'message': '✨ **Here are your personalized recommendations:**',
                        'format': 'attraction_list_with_actions'
                    }
                }
            },
            
            FlowType.RESTAURANT_DISCOVERY: {
                'name': 'Restaurant Discovery Guide',
                'description': 'Find the perfect dining experience',
                'steps': {
                    FlowStep.INITIAL: {
                        'message': '🍽️ **Find Your Perfect Istanbul Restaurant**\n\nWhat kind of dining experience are you looking for?',
                        'options': [
                            {'id': 'romantic', 'title': 'Romantic Dinner', 'icon': '💕', 'desc': 'Perfect for couples'},
                            {'id': 'family', 'title': 'Family Friendly', 'icon': '👨‍👩‍👧‍👦', 'desc': 'Great for kids too'},
                            {'id': 'local', 'title': 'Authentic Local', 'icon': '🏘️', 'desc': 'Where locals eat'},
                            {'id': 'upscale', 'title': 'Fine Dining', 'icon': '✨', 'desc': 'Special occasion'},
                            {'id': 'quick', 'title': 'Quick & Casual', 'icon': '🥙', 'desc': 'Fast, delicious food'},
                            {'id': 'view', 'title': 'With a View', 'icon': '🌆', 'desc': 'Scenic dining'}
                        ]
                    },
                    FlowStep.PREFERENCE_GATHERING: {
                        'message': '🏘️ **Which area would you prefer?**',
                        'options': [
                            {'id': 'sultanahmet', 'title': 'Sultanahmet', 'desc': 'Historic area, tourist-friendly'},
                            {'id': 'beyoglu', 'title': 'Beyoğlu/Taksim', 'desc': 'Modern, lively nightlife'},
                            {'id': 'galata', 'title': 'Galata/Karaköy', 'desc': 'Trendy, artistic neighborhood'},
                            {'id': 'kadikoy', 'title': 'Kadıköy', 'desc': 'Asian side, authentic local'},
                            {'id': 'besiktas', 'title': 'Beşiktaş', 'desc': 'Bosphorus views, upscale'},
                            {'id': 'anywhere', 'title': 'Anywhere Good', 'desc': 'Show me the best options'}
                        ]
                    }
                }
            },
            
            FlowType.TRANSPORTATION_HELP: {
                'name': 'Transportation Assistant',
                'description': 'Navigate Istanbul like a local',
                'steps': {
                    FlowStep.INITIAL: {
                        'message': '🚇 **Istanbul Transportation Guide**\n\nWhat do you need help with?',
                        'options': [
                            {'id': 'airport', 'title': 'From/To Airport', 'icon': '✈️'},
                            {'id': 'metro', 'title': 'Metro System', 'icon': '🚇'},
                            {'id': 'ferry', 'title': 'Ferry Routes', 'icon': '⛴️'},
                            {'id': 'taxi', 'title': 'Taxis & Apps', 'icon': '🚖'},
                            {'id': 'istanbulkart', 'title': 'Istanbulkart Info', 'icon': '💳'},
                            {'id': 'route', 'title': 'Specific Route', 'icon': '🗺️'}
                        ]
                    }
                }
            },
            
            FlowType.MUSEUM_TOUR: {
                'name': 'Museum Tour Planner',
                'description': 'Explore Istanbul\'s rich history',
                'steps': {
                    FlowStep.INITIAL: {
                        'message': '🏛️ **Plan Your Museum Experience**\n\nWhat type of history/culture interests you most?',
                        'options': [
                            {'id': 'ottoman', 'title': 'Ottoman Empire', 'icon': '🏰', 'desc': 'Palaces, imperial history'},
                            {'id': 'byzantine', 'title': 'Byzantine Era', 'icon': '⛪', 'desc': 'Churches, mosaics, Christian history'},
                            {'id': 'modern', 'title': 'Modern Art', 'icon': '🎨', 'desc': 'Contemporary Turkish art'},
                            {'id': 'archaeology', 'title': 'Ancient History', 'icon': '🏺', 'desc': 'Artifacts from all eras'},
                            {'id': 'religious', 'title': 'Religious Sites', 'icon': '🕌', 'desc': 'Mosques, churches, spiritual sites'},
                            {'id': 'overview', 'title': 'Best Overview', 'icon': '📚', 'desc': 'Mix of everything important'}
                        ]
                    }
                }
            }
        }
    
    def start_flow(self, flow_type: FlowType, session_id: str, 
                   initial_query: str = "") -> Dict[str, Any]:
        """Start a new guided flow"""
        
        flow_template = self.flow_templates.get(flow_type)
        if not flow_template:
            return {'error': 'Flow type not supported'}
        
        # Initialize flow state
        flow_state = FlowState(
            flow_type=flow_type,
            current_step=FlowStep.INITIAL,
            user_preferences={},
            selected_options=[],
            session_data={'initial_query': initial_query},
            next_suggestions=[]
        )
        
        self.active_flows[session_id] = flow_state
        
        # Get initial step content
        initial_step = flow_template['steps'][FlowStep.INITIAL]
        
        return {
            'flow_type': flow_type.value,
            'flow_name': flow_template['name'],
            'step': FlowStep.INITIAL.value,
            'message': initial_step['message'],
            'options': initial_step['options'],
            'quick_actions': initial_step.get('quick_actions', []),
            'progress': 0.2  # 20% complete after starting
        }
    
    def process_flow_input(self, session_id: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input in an active flow"""
        
        if session_id not in self.active_flows:
            return {'error': 'No active flow found'}
        
        flow_state = self.active_flows[session_id]
        
        # Process input based on current step
        if flow_state.current_step == FlowStep.INITIAL:
            return self._process_initial_step(flow_state, user_input)
        elif flow_state.current_step == FlowStep.PREFERENCE_GATHERING:
            return self._process_preferences(flow_state, user_input)
        elif flow_state.current_step == FlowStep.OPTIONS_PRESENTATION:
            return self._process_selection(flow_state, user_input)
        else:
            return self._generate_final_response(flow_state, user_input)
    
    def _process_initial_step(self, flow_state: FlowState, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process initial step input"""
        
        selected_option = user_input.get('selected_option')
        if not selected_option:
            return {'error': 'Please select an option'}
        
        # Store user preference
        flow_state.user_preferences['initial_choice'] = selected_option
        
        # Move to next step
        flow_template = self.flow_templates[flow_state.flow_type]
        
        if FlowStep.PREFERENCE_GATHERING in flow_template['steps']:
            flow_state.current_step = FlowStep.PREFERENCE_GATHERING
            next_step = flow_template['steps'][FlowStep.PREFERENCE_GATHERING]
            
            return {
                'flow_type': flow_state.flow_type.value,
                'step': FlowStep.PREFERENCE_GATHERING.value,
                'message': next_step['message'],
                'options': next_step['options'],
                'progress': 0.4,
                'can_skip': True
            }
        else:
            # Skip to results
            return self._generate_recommendations(flow_state)
    
    def _process_preferences(self, flow_state: FlowState, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process preference gathering step"""
        
        selected_options = user_input.get('selected_options', [])
        flow_state.user_preferences['interests'] = selected_options
        
        # Generate recommendations based on preferences
        return self._generate_recommendations(flow_state)
    
    def _generate_recommendations(self, flow_state: FlowState) -> Dict[str, Any]:
        """Generate personalized recommendations based on flow state"""
        
        if flow_state.flow_type == FlowType.DAY_PLANNING:
            return self._generate_day_plan(flow_state)
        elif flow_state.flow_type == FlowType.RESTAURANT_DISCOVERY:
            return self._generate_restaurant_recommendations(flow_state)
        elif flow_state.flow_type == FlowType.MUSEUM_TOUR:
            return self._generate_museum_tour(flow_state)
        elif flow_state.flow_type == FlowType.TRANSPORTATION_HELP:
            return self._generate_transport_help(flow_state)
        else:
            return {'error': 'Flow type not implemented'}
    
    def _generate_day_plan(self, flow_state: FlowState) -> Dict[str, Any]:
        """Generate a day plan itinerary"""
        
        preferences = flow_state.user_preferences
        duration = preferences.get('initial_choice', 'duration_full')
        interests = preferences.get('interests', [])
        
        # Sample itinerary generation (would integrate with retrieval system)
        itinerary_items = []
        
        if 'history' in interests:
            itinerary_items.extend([
                {
                    'time': '9:00 AM',
                    'activity': 'Hagia Sophia',
                    'duration': '1.5 hours',
                    'type': 'historical',
                    'description': 'Start with Istanbul\'s most iconic monument',
                    'next_actions': ['Get directions', 'See opening hours', 'Book skip-the-line']
                },
                {
                    'time': '11:00 AM', 
                    'activity': 'Topkapi Palace',
                    'duration': '2 hours',
                    'type': 'historical',
                    'description': 'Ottoman imperial palace with stunning views',
                    'next_actions': ['Get directions', 'See highlights', 'Plan lunch nearby']
                }
            ])
        
        if 'food' in interests:
            itinerary_items.append({
                'time': '1:00 PM',
                'activity': 'Lunch at Hamdi Restaurant',
                'duration': '1.5 hours',
                'type': 'dining',
                'description': 'Famous for kebabs with Golden Horn views',
                'next_actions': ['Make reservation', 'See menu', 'Get directions']
            })
        
        if 'views' in interests:
            itinerary_items.append({
                'time': '3:30 PM',
                'activity': 'Galata Tower',
                'duration': '1 hour',
                'type': 'scenic',
                'description': '360° panoramic views of Istanbul',
                'next_actions': ['Skip the line tips', 'Best photo spots', 'Nearby cafes']
            })
        
        return {
            'flow_type': flow_state.flow_type.value,
            'step': 'itinerary_complete',
            'message': f'🎉 **Your Perfect Istanbul Day Plan**\n\nBased on your interests in {", ".join(interests)}, here\'s your personalized itinerary:',
            'itinerary': itinerary_items,
            'summary': {
                'total_duration': '8 hours',
                'walking_distance': '3.2 km',
                'estimated_cost': 'Moderate (tickets + meals)',
                'difficulty': 'Easy'
            },
            'quick_actions': [
                'Adjust timing',
                'Add more activities', 
                'Find restaurants nearby',
                'Get transport directions',
                'Export to calendar'
            ],
            'progress': 1.0
        }
    
    def _generate_restaurant_recommendations(self, flow_state: FlowState) -> Dict[str, Any]:
        """Generate restaurant recommendations"""
        
        preferences = flow_state.user_preferences
        dining_type = preferences.get('initial_choice')
        area = preferences.get('area', 'anywhere')
        
        # Sample restaurants (would integrate with retrieval system)
        restaurants = [
            {
                'id': 'hamdi',
                'name': 'Hamdi Restaurant',
                'type': 'Traditional Turkish',
                'area': 'Eminönü',
                'rating': 4.6,
                'price_range': '$$',
                'highlight': 'Famous lamb dishes with Golden Horn views',
                'quick_actions': ['See menu', 'Make reservation', 'Get directions', 'Read reviews']
            },
            {
                'id': 'mikla',
                'name': 'Mikla',
                'type': 'Modern Turkish',
                'area': 'Beyoğlu',
                'rating': 4.8,
                'price_range': '$$$',
                'highlight': 'Rooftop fine dining with Bosphorus panorama',
                'quick_actions': ['Check availability', 'See tasting menu', 'Dress code info']
            },
            {
                'id': 'ciya',
                'name': 'Çiya Sofrası',
                'type': 'Regional Turkish',
                'area': 'Kadıköy',
                'rating': 4.7,
                'price_range': '$',
                'highlight': 'Authentic Anatolian dishes, locals\' favorite',
                'quick_actions': ['Popular dishes', 'How to get there', 'Peak hours']
            }
        ]
        
        return {
            'flow_type': flow_state.flow_type.value,
            'step': 'recommendations_ready',
            'message': f'🍽️ **Perfect Restaurants for {dining_type.replace("_", " ").title()}**\n\nHere are my top recommendations:',
            'restaurants': restaurants,
            'quick_actions': [
                'Filter by price',
                'Show map view',
                'Book now',
                'Find similar',
                'Plan full evening'
            ],
            'progress': 1.0
        }
    
    def _generate_museum_tour(self, flow_state: FlowState) -> Dict[str, Any]:
        """Generate museum tour recommendations"""
        
        preferences = flow_state.user_preferences
        interest_type = preferences.get('initial_choice')
        
        museum_recommendations = []
        
        if interest_type == 'byzantine':
            museum_recommendations = [
                {
                    'name': 'Hagia Sophia',
                    'priority': 'Must-see',
                    'duration': '1.5-2 hours',
                    'highlights': ['Byzantine mosaics', 'Imperial architecture', 'Christian-Islamic history'],
                    'insider_tip': 'Visit early morning or late afternoon for fewer crowds'
                },
                {
                    'name': 'Chora Church Museum',
                    'priority': 'Highly recommended',
                    'duration': '1 hour',
                    'highlights': ['Best preserved Byzantine mosaics', 'Peaceful atmosphere'],
                    'insider_tip': 'Photography allowed, but no flash'
                }
            ]
        elif interest_type == 'ottoman':
            museum_recommendations = [
                {
                    'name': 'Topkapi Palace',
                    'priority': 'Must-see',
                    'duration': '2-3 hours',
                    'highlights': ['Imperial chambers', 'Treasury', 'Harem quarters'],
                    'insider_tip': 'Buy harem ticket separately for complete experience'
                }
            ]
        
        return {
            'flow_type': flow_state.flow_type.value,
            'step': 'tour_ready',
            'message': f'🏛️ **Your {interest_type.title()} Museum Tour**',
            'museums': museum_recommendations,
            'tour_plan': {
                'suggested_order': 'Historical chronology',
                'total_time': '4-5 hours',
                'walking_between': '15 minutes average'
            },
            'quick_actions': [
                'Optimize route',
                'Check opening hours',
                'Buy combined tickets',
                'Audio guide info',
                'Add lunch break'
            ],
            'progress': 1.0
        }
    
    def _generate_transport_help(self, flow_state: FlowState) -> Dict[str, Any]:
        """Generate transportation help"""
        
        help_type = flow_state.user_preferences.get('initial_choice')
        
        if help_type == 'metro':
            return {
                'flow_type': flow_state.flow_type.value,
                'step': 'help_ready',
                'message': '🚇 **Istanbul Metro System Guide**',
                'info': {
                    'overview': 'Istanbul has 8 metro lines (M1-M8) connecting major districts',
                    'payment': 'Use Istanbulkart - available at stations and kiosks',
                    'key_lines': [
                        {'line': 'M1', 'route': 'Airport → Yenikapi', 'tourist_spots': 'Sultanahmet connection'},
                        {'line': 'M2', 'route': 'Hacıosman → Vezneciler', 'tourist_spots': 'Taksim, Şişhane (Galata)'},
                        {'line': 'M4', 'route': 'Kadıköy → Sabiha Gökçen', 'tourist_spots': 'Asian side exploration'}
                    ]
                },
                'quick_actions': [
                    'Get Istanbulkart',
                    'Plan specific route',
                    'Download offline map',
                    'Check service hours',
                    'See all transport options'
                ],
                'progress': 1.0
            }
        
        return {'error': 'Transport help type not implemented'}
    
    def get_suggested_flows(self, query: str) -> List[Dict[str, Any]]:
        """Get suggested flows based on user query"""
        
        query_lower = query.lower()
        suggestions = []
        
        # Analyze query and suggest appropriate flows
        if any(word in query_lower for word in ['plan', 'day', 'itinerary', 'visit', 'see']):
            suggestions.append({
                'flow_type': FlowType.DAY_PLANNING.value,
                'title': '📅 Plan Your Day',
                'description': 'Create a personalized Istanbul itinerary',
                'estimated_time': '2-3 minutes'
            })
        
        if any(word in query_lower for word in ['restaurant', 'eat', 'food', 'dinner', 'lunch']):
            suggestions.append({
                'flow_type': FlowType.RESTAURANT_DISCOVERY.value,
                'title': '🍽️ Find Great Restaurants',
                'description': 'Discover the perfect dining experience',
                'estimated_time': '1-2 minutes'
            })
        
        if any(word in query_lower for word in ['museum', 'history', 'culture', 'art']):
            suggestions.append({
                'flow_type': FlowType.MUSEUM_TOUR.value,
                'title': '🏛️ Plan Museum Tour',
                'description': 'Explore Istanbul\'s rich cultural heritage',
                'estimated_time': '2 minutes'
            })
        
        if any(word in query_lower for word in ['transport', 'metro', 'ferry', 'taxi', 'get to']):
            suggestions.append({
                'flow_type': FlowType.TRANSPORTATION_HELP.value,
                'title': '🚇 Transportation Help',
                'description': 'Navigate Istanbul like a local',
                'estimated_time': '1 minute'
            })
        
        # Always show day planning as fallback
        if not suggestions:
            suggestions.append({
                'flow_type': FlowType.DAY_PLANNING.value,
                'title': '📅 Plan Your Day',
                'description': 'Not sure where to start? Let me help plan your day!',
                'estimated_time': '2-3 minutes'
            })
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def get_flow_status(self, session_id: str) -> Dict[str, Any]:
        """Get current flow status"""
        
        if session_id not in self.active_flows:
            return {'active': False}
        
        flow_state = self.active_flows[session_id]
        
        return {
            'active': True,
            'flow_type': flow_state.flow_type.value,
            'current_step': flow_state.current_step.value,
            'progress': self._calculate_progress(flow_state),
            'can_restart': True,
            'can_skip_to_results': flow_state.current_step in [FlowStep.PREFERENCE_GATHERING]
        }
    
    def _calculate_progress(self, flow_state: FlowState) -> float:
        """Calculate flow completion progress"""
        
        step_progress = {
            FlowStep.INITIAL: 0.2,
            FlowStep.PREFERENCE_GATHERING: 0.5,
            FlowStep.OPTIONS_PRESENTATION: 0.8,
            FlowStep.COMPLETION: 1.0
        }
        
        return step_progress.get(flow_state.current_step, 0.0)

# Global instance
interactive_flow_manager = InteractiveFlowManager()

def start_guided_flow(flow_type_str: str, session_id: str, initial_query: str = "") -> Dict[str, Any]:
    """Start a guided flow"""
    try:
        flow_type = FlowType(flow_type_str)
        return interactive_flow_manager.start_flow(flow_type, session_id, initial_query)
    except ValueError:
        return {'error': f'Unknown flow type: {flow_type_str}'}

def process_flow_interaction(session_id: str, user_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process user interaction in active flow"""
    return interactive_flow_manager.process_flow_input(session_id, user_input)

def get_flow_suggestions(query: str) -> List[Dict[str, Any]]:
    """Get suggested flows for a query"""
    return interactive_flow_manager.get_suggested_flows(query)
