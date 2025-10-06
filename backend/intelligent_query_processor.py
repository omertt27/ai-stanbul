#!/usr/bin/env python3
"""
Intelligent Query Processor - No LLMs Required
==============================================

This system processes NLP results and generates intelligent responses by:
1. Querying the appropriate databases based on intent
2. Applying business logic and filtering
3. Generating human-like responses using templates
4. Handling context and follow-up queries
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime

from lightweight_nlp_system import (
    lightweight_nlp, QueryIntent, ExtractedEntities, 
    QueryContext, process_natural_query
)

# Import lightweight retrieval system for retrieval-first design
try:
    from lightweight_retrieval_system import lightweight_retrieval_system, search_content_lightweight, SearchResult
    RETRIEVAL_SYSTEM_AVAILABLE = True
    print("✅ Lightweight Retrieval System integrated for retrieval-first design")
except ImportError as e:
    print(f"⚠️ Lightweight Retrieval System not available: {e}")
    RETRIEVAL_SYSTEM_AVAILABLE = False

# Import interactive flow manager for guided UX
try:
    from interactive_flow_manager import interactive_flow_manager, get_flow_suggestions
    INTERACTIVE_FLOWS_AVAILABLE = True
    print("✅ Interactive Flow Manager integrated for guided UX")
except ImportError as e:
    print(f"⚠️ Interactive Flow Manager not available: {e}")
    INTERACTIVE_FLOWS_AVAILABLE = False

# Import seasonal events for up-to-date info
try:
    from seasonal_events_manager import get_current_seasonal_events, get_daily_istanbul_updates
    SEASONAL_EVENTS_AVAILABLE = True
    print("✅ Seasonal Events Manager integrated for daily updates")
except ImportError as e:
    print(f"⚠️ Seasonal Events Manager not available: {e}")
    SEASONAL_EVENTS_AVAILABLE = False

# Import query analytics system for tracking
try:
    from query_analytics_system import (
        log_query_analytics, QueryStatus, track_user_satisfaction,
        get_analytics_dashboard
    )
    ANALYTICS_AVAILABLE = True
    print("✅ Query Analytics System integrated")
except ImportError as e:
    print(f"⚠️ Query Analytics System not available: {e}")
    ANALYTICS_AVAILABLE = False

# Import automated pipeline for data management
try:
    from automated_data_pipeline import automated_pipeline, get_pipeline_dashboard
    PIPELINE_AVAILABLE = True
    print("✅ Automated Data Pipeline integrated")
except ImportError as e:
    print(f"⚠️ Automated Data Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

class IntelligentQueryProcessor:
    """Main processor that handles NLP results and generates responses"""
    
    def __init__(self, database_manager=None):
        self.database_manager = database_manager
        self.nlp_system = lightweight_nlp
        
    def process_user_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a user query and return intelligent response"""
        start_time = datetime.now()
        query_status = QueryStatus.SUCCESS
        error_details = None
        
        try:
            # Step 1: Process query with NLP
            nlp_result = process_natural_query(query)
            
            # Step 2: Generate primary response
            primary_response = self._generate_primary_response(nlp_result)
            
            # Step 3: Add seasonal context if available
            if SEASONAL_EVENTS_AVAILABLE:
                seasonal_context = self._add_seasonal_context(nlp_result, primary_response)
                primary_response['seasonal_info'] = seasonal_context
            
            # Step 4: Add retrieval results if available
            if RETRIEVAL_SYSTEM_AVAILABLE:
                retrieval_results = search_content_lightweight(query, max_results=5)
                primary_response['related_content'] = [
                    {
                        'title': result.title,
                        'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        'score': result.score,
                        'type': result.metadata.get('type', 'general')
                    }
                    for result in retrieval_results
                ]
            
            # Step 5: Suggest interactive flows if available
            if INTERACTIVE_FLOWS_AVAILABLE:
                flow_suggestions = get_flow_suggestions(nlp_result.intent.value, nlp_result.entities)
                primary_response['suggested_flows'] = [
                    {
                        'flow_name': flow.flow_name,
                        'description': flow.description,
                        'next_step': flow.steps[0] if flow.steps else None
                    }
                    for flow in flow_suggestions[:3]  # Top 3 suggestions
                ]
            
            # Calculate response metrics
            end_time = datetime.now()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            results_count = len(primary_response.get('results', []))
            confidence_score = primary_response.get('confidence', 0.8)
            
            # Track analytics if available
            if ANALYTICS_AVAILABLE:
                query_id = log_query_analytics(
                    raw_query=query,
                    processed_query=nlp_result.processed_query,
                    intent=nlp_result.intent.value,
                    extracted_entities=nlp_result.entities.__dict__,
                    query_type=primary_response.get('response_type', 'general'),
                    response_time_ms=response_time_ms,
                    results_count=results_count,
                    status=query_status,
                    confidence_score=confidence_score,
                    session_id=session_id,
                    used_fallback=primary_response.get('used_fallback', False)
                )
                primary_response['query_id'] = query_id
            
            # Add metadata
            primary_response.update({
                'query_processing': {
                    'intent': nlp_result.intent.value,
                    'entities': nlp_result.entities.__dict__,
                    'context_level': nlp_result.context.context_level,
                    'processing_time_ms': response_time_ms,
                    'has_seasonal_context': SEASONAL_EVENTS_AVAILABLE,
                    'has_retrieval': RETRIEVAL_SYSTEM_AVAILABLE,
                    'has_interactive_flows': INTERACTIVE_FLOWS_AVAILABLE
                },
                'system_info': {
                    'version': '2.0_analytics_enabled',
                    'capabilities': {
                        'nlp_processing': True,
                        'seasonal_events': SEASONAL_EVENTS_AVAILABLE,
                        'content_retrieval': RETRIEVAL_SYSTEM_AVAILABLE,
                        'interactive_flows': INTERACTIVE_FLOWS_AVAILABLE,
                        'analytics_tracking': ANALYTICS_AVAILABLE,
                        'automated_pipeline': PIPELINE_AVAILABLE
                    }
                }
            })
            
            return primary_response
            
        except Exception as e:
            query_status = QueryStatus.ERROR
            error_details = str(e)
            
            # Track failed query if analytics available
            if ANALYTICS_AVAILABLE:
                end_time = datetime.now()
                response_time_ms = (end_time - start_time).total_seconds() * 1000
                
                log_query_analytics(
                    raw_query=query,
                    processed_query=query,  # fallback
                    intent="error",
                    extracted_entities={},
                    query_type="error",
                    response_time_ms=response_time_ms,
                    results_count=0,
                    status=query_status,
                    confidence_score=0.0,
                    session_id=session_id,
                    error_details=error_details,
                    used_fallback=True
                )
            
            # Return error response
            return {
                'message': "I encountered an error processing your query. Please try again.",
                'success': False,
                'error': str(e),
                'response_type': 'error',
                'system_info': {'error_tracked': ANALYTICS_AVAILABLE}
            }
    
    def _query_data_sources(self, intent: QueryIntent, entities: ExtractedEntities, 
                           context: QueryContext) -> Dict[str, Any]:
        """Query appropriate data sources based on intent and entities"""
        
        if intent == QueryIntent.RESTAURANT_SEARCH:
            return self._query_restaurants(entities)
        
        elif intent == QueryIntent.MUSEUM_INFO:
            return self._query_museums(entities)
        
        elif intent == QueryIntent.DIRECTIONS:
            return self._query_directions(entities)
        
        elif intent == QueryIntent.DISTRICT_INFO:
            return self._query_district_info(entities)
        
        elif intent == QueryIntent.TRANSPORTATION:
            return self._query_transportation(entities)
        
        elif intent == QueryIntent.CULTURAL_ETIQUETTE:
            return self._query_cultural_info(entities)
        
        elif intent == QueryIntent.EVENTS:
            return self._query_events(entities)
        
        elif intent == QueryIntent.SHOPPING:
            return self._query_shopping(entities)
        
        elif intent == QueryIntent.NIGHTLIFE:
            return self._query_nightlife(entities)
        
        else:
            return self._query_general_info(entities)
    
    def _query_restaurants(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Query restaurant database with RETRIEVAL-FIRST design (TF-IDF + Keyword Search)"""
        
        # **RETRIEVAL-FIRST APPROACH**: Try lightweight retrieval first
        if RETRIEVAL_SYSTEM_AVAILABLE and lightweight_retrieval_system.initialized:
            
            # Build semantic query from entities
            query_parts = []
            if entities.cuisine:
                query_parts.append(entities.cuisine)
            if entities.district:
                query_parts.append(f"in {entities.district}")
            if entities.atmosphere:
                query_parts.append(entities.atmosphere)
            if entities.budget:
                query_parts.append(entities.budget)
            
            search_query = " ".join(query_parts) if query_parts else "restaurant"
            
            print(f"🔍 Retrieval search query: '{search_query}'")
            
            # Perform lightweight retrieval search
            search_results = search_content_lightweight(search_query, content_types=['restaurant'], top_k=8)
            
            if search_results:
                restaurants = []
                for result in search_results:
                    restaurant_data = result.metadata.copy()
                    restaurant_data['match_score'] = result.score
                    restaurant_data['search_method'] = 'vector_search'
                    restaurants.append(restaurant_data)
                
                print(f"✅ Retrieval search found {len(restaurants)} restaurants")
                
                return {
                    'restaurants': restaurants[:5],  # Top 5 matches
                    'total_found': len(restaurants),
                    'search_method': 'retrieval_first',
                    'filters_applied': self._get_applied_filters(entities)
                }
        
        # **FALLBACK**: Traditional database filtering approach
        print("🔄 Falling back to traditional database search")
        
        # Use database manager if available
        if self.database_manager and hasattr(self.database_manager, 'restaurants'):
            restaurants = self.database_manager.restaurants
        else:
            # Fallback to sample data
            restaurants = self._get_sample_restaurants()
        
        # Apply intelligent filtering
        filtered_restaurants = []
        
        for restaurant in restaurants:
            match_score = 0
            
            # District matching
            if entities.district:
                if entities.district.lower() in restaurant.get('district', '').lower():
                    match_score += 3
                elif entities.district.lower() in restaurant.get('location', '').lower():
                    match_score += 2
            
            # Cuisine matching
            if entities.cuisine:
                if entities.cuisine.lower() in restaurant.get('cuisine', '').lower():
                    match_score += 3
                elif entities.cuisine.lower() in restaurant.get('tags', []):
                    match_score += 2
            
            # Budget matching
            if entities.budget:
                restaurant_budget = restaurant.get('price_range', 'moderate')
                if entities.budget == restaurant_budget:
                    match_score += 2
                elif self._budget_compatible(entities.budget, restaurant_budget):
                    match_score += 1
            
            # Atmosphere matching
            if entities.atmosphere:
                restaurant_atmosphere = restaurant.get('atmosphere', [])
                if isinstance(restaurant_atmosphere, list):
                    if entities.atmosphere in restaurant_atmosphere:
                        match_score += 2
                elif entities.atmosphere in restaurant_atmosphere:
                    match_score += 2
            
            # Group size compatibility
            if entities.group_size:
                capacity = restaurant.get('capacity', 'any')
                if capacity == 'any' or self._group_size_compatible(entities.group_size, capacity):
                    match_score += 1
            
            # Only include restaurants with decent match scores
            if match_score >= 2 or not any([entities.district, entities.cuisine, entities.budget]):
                restaurant['match_score'] = match_score
                restaurant['search_method'] = 'database_filter'
                filtered_restaurants.append(restaurant)
        
        # Sort by match score and take top results
        filtered_restaurants.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return {
            'restaurants': filtered_restaurants[:5],  # Top 5 matches
            'total_found': len(filtered_restaurants),
            'search_method': 'database_filter',
            'filters_applied': self._get_applied_filters(entities)
        }
    
    def _query_museums(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Query museum database with RETRIEVAL-FIRST design (Vector + Keyword Search)"""
        
        # **RETRIEVAL-FIRST APPROACH**: Try lightweight retrieval first
        if RETRIEVAL_SYSTEM_AVAILABLE and lightweight_retrieval_system.initialized:
            
            # Build semantic query from entities
            query_parts = []
            if entities.museum_name:
                query_parts.append(entities.museum_name)
            if entities.district:
                query_parts.append(f"in {entities.district}")
            
            # Add general museum query if no specific name
            if not entities.museum_name:
                query_parts.append("museum")
            
            search_query = " ".join(query_parts) if query_parts else "museum istanbul"
            
            print(f"🔍 Retrieval search query: '{search_query}'")
            
            # Perform lightweight retrieval search
            search_results = search_content_lightweight(search_query, content_types=['museum'], top_k=6)
            
            if search_results:
                museums = []
                for result in search_results:
                    museum_data = {
                        'name': result.title,
                        'district': result.metadata.get('location', 'Istanbul'),
                        'historical_period': result.metadata.get('historical_period', 'Various periods'),
                        'description': result.description,
                        'practical_info': f"Open: {result.metadata.get('opening_hours', {}).get('daily', 'Check schedule')}",
                        'match_score': result.score,
                        'search_method': 'vector_search'
                    }
                    museums.append(museum_data)
                
                print(f"✅ Retrieval search found {len(museums)} museums")
                
                return {
                    'museums': museums[:3],  # Top 3 museums
                    'total_found': len(museums),
                    'search_method': 'retrieval_first'
                }
        
        # **FALLBACK**: Traditional database approach
        print("🔄 Falling back to traditional museum database search")
        
        # Use our expanded museum database
        try:
            from accurate_museum_database import istanbul_museums
            museums = istanbul_museums.museums
        except ImportError:
            museums = {}
        
        filtered_museums = []
        
        # If specific museum requested
        if entities.museum_name:
            # Try exact match first
            for key, museum in museums.items():
                if entities.museum_name.lower() in museum.name.lower():
                    filtered_museums.append({
                        'name': museum.name,
                        'district': museum.location,
                        'historical_period': museum.historical_period,
                        'significance': museum.historical_significance,
                        'practical_info': f"Open: {museum.opening_hours.get('daily', 'Check schedule')}",
                        'description': f"{museum.architectural_style}. {', '.join(museum.key_features[:2])}.",
                        'match_score': 5,
                        'search_method': 'database_match'
                    })
                    break
        
        # District-based filtering
        elif entities.district:
            for key, museum in museums.items():
                if entities.district.lower() in museum.location.lower():
                    filtered_museums.append({
                        'name': museum.name,
                        'district': museum.location,
                        'description': f"{museum.architectural_style}. {', '.join(museum.key_features[:2])}.",
                        'match_score': 3,
                        'search_method': 'database_filter'
                    })
        
        # General museum recommendations
        else:
            # Get top popular museums
            popular_museums = ['hagia_sophia', 'topkapi_palace', 'blue_mosque', 'galata_tower', 'pera_museum']
            for key in popular_museums:
                if key in museums:
                    museum = museums[key]
                    filtered_museums.append({
                        'name': museum.name,
                        'district': museum.location,
                        'description': f"{museum.architectural_style}. {', '.join(museum.key_features[:2])}.",
                        'match_score': 4,
                        'search_method': 'database_popular'
                    })
        
        return {
            'museums': filtered_museums[:3],  # Top 3 museums
            'total_found': len(filtered_museums),
            'search_method': 'database_fallback'
        }
    
    def _query_directions(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Generate directions based on entities"""
        
        routes = []
        
        # Simple route generation based on district
        if entities.district:
            district = entities.district.lower()
            
            # Metro routes to popular districts
            metro_routes = {
                'sultanahmet': {
                    'line': 'M1 Vezneciler-Üniversite',
                    'station': 'Vezneciler',
                    'walking_info': 'Walk 5 minutes towards Sultanahmet Square'
                },
                'beyoğlu': {
                    'line': 'M2 Taksim-Şişhane',
                    'station': 'Şişhane or Taksim',
                    'walking_info': 'Walk down Istiklal Street'
                },
                'kadıköy': {
                    'line': 'M4 Kadıköy-Kartal',
                    'station': 'Kadıköy',
                    'walking_info': 'Exit towards Bahariye Street'
                }
            }
            
            if district in metro_routes:
                routes.append({
                    'type': 'metro',
                    'line': metro_routes[district]['line'],
                    'station': metro_routes[district]['station'],
                    'walking_info': metro_routes[district]['walking_info'],
                    'duration': '20-30 minutes'
                })
        
        # Transportation-specific routes
        if entities.transportation_type:
            if entities.transportation_type == 'ferry':
                routes.append({
                    'type': 'ferry',
                    'from': 'Eminönü or Karaköy',
                    'to': 'Asian side ports',
                    'info': 'Ferries run every 15-20 minutes with beautiful Bosphorus views',
                    'duration': '15-25 minutes'
                })
        
        return {'routes': routes}
    
    def _query_district_info(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get district information"""
        
        district_info = {
            'sultanahmet': {
                'characteristics': 'its historic significance and major tourist attractions',
                'highlights': 'Home to Hagia Sophia, Blue Mosque, and Topkapi Palace.'
            },
            'beyoğlu': {
                'characteristics': 'its vibrant nightlife and modern culture',
                'highlights': 'Istiklal Street, Galata Tower, and numerous art galleries.'
            },
            'kadıköy': {
                'characteristics': 'its authentic local atmosphere and great food scene',
                'highlights': 'Bahariye Street, local markets, and trendy cafes.'
            },
            'galata': {
                'characteristics': 'its artistic vibe and historic architecture',
                'highlights': 'Galata Tower, art galleries, and cozy cafes with Bosphorus views.'
            }
        }
        
        district = entities.district.lower() if entities.district else 'istanbul'
        info = district_info.get(district, {
            'characteristics': 'its unique Istanbul charm',
            'highlights': 'Many interesting places to discover.'
        })
        
        return {'district_info': info}
    
    def _query_transportation(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get transportation information"""
        
        transport_info = {
            'metro': {
                'description': 'Istanbul has multiple metro lines connecting major districts',
                'tips': 'Get an Istanbulkart for easy travel. Lines M1, M2, M3, and M4 cover most tourist areas.',
                'cost': 'Very affordable public transport option'
            },
            'ferry': {
                'description': 'Ferries connect European and Asian sides with scenic Bosphorus views',
                'tips': 'Great for sightseeing while traveling. Main routes from Eminönü, Karaköy, and Beşiktaş.',
                'cost': 'Moderate pricing, great value for the experience'
            },
            'taxi': {
                'description': 'Taxis and ride-sharing apps like BiTaksi are widely available',
                'tips': 'Use BiTaksi or Uber for transparent pricing. Traditional taxis should use the meter.',
                'cost': 'Higher cost but convenient for door-to-door service'
            }
        }
        
        transport_type = entities.transportation_type or 'general'
        info = transport_info.get(transport_type, transport_info['metro'])
        
        return {'transport_info': info}
    
    def _query_cultural_info(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get cultural etiquette information"""
        
        guidelines = [
            "Remove shoes when entering mosques",
            "Dress modestly when visiting religious sites",
            "It's polite to learn a few Turkish phrases like 'Merhaba' (Hello) and 'Teşekkürler' (Thank you)",
            "Tipping 10-15% is appreciated in restaurants",
            "Friday prayers are important - avoid visiting mosques during prayer times",
            "Turkish hospitality is legendary - don't be surprised by warm welcomes!"
        ]
        
        return {'guidelines': ' • '.join(guidelines)}
    
    def _query_events(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get event information (placeholder for future implementation)"""
        return {
            'events': [],
            'message': 'Event information feature coming soon! For now, check local event websites or ask at your hotel.'
        }
    
    def _query_shopping(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get shopping information"""
        
        shopping_spots = [
            {
                'name': 'Grand Bazaar',
                'type': 'Traditional Market',
                'district': 'Fatih',
                'description': 'Historic covered market with 4000 shops selling carpets, jewelry, and souvenirs'
            },
            {
                'name': 'Spice Bazaar',
                'type': 'Food Market', 
                'district': 'Eminönü',
                'description': 'Aromatic market for Turkish delights, spices, and traditional foods'
            },
            {
                'name': 'Istiklal Street',
                'type': 'Shopping Street',
                'district': 'Beyoğlu',
                'description': 'Modern shopping street with international brands and local boutiques'
            }
        ]
        
        return {'shopping_spots': shopping_spots}
    
    def _query_nightlife(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Get nightlife information"""
        
        nightlife_spots = [
            {
                'name': 'Istiklal Street Area',
                'type': 'Bar District',
                'district': 'Beyoğlu',
                'description': 'Numerous bars, pubs, and clubs along the famous pedestrian street'
            },
            {
                'name': 'Ortaköy',
                'type': 'Waterfront Bars',
                'district': 'Beşiktaş',
                'description': 'Trendy bars with Bosphorus views, popular with locals and tourists'
            },
            {
                'name': 'Kadıköy',
                'type': 'Local Scene',
                'district': 'Kadıköy',
                'description': 'Authentic local bars and venues, less touristy atmosphere'
            }
        ]
        
        return {'nightlife_spots': nightlife_spots}
    
    def _query_general_info(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Handle general information requests"""
        
        return {
            'response': "I can help you with restaurants, museums, directions, cultural tips, and general information about Istanbul. What specifically would you like to know?"
        }
    
    def _generate_intelligent_response(self, intent: QueryIntent, entities: ExtractedEntities, 
                                     data: Dict[str, Any], context: QueryContext) -> str:
        """Generate intelligent, human-like responses"""
        
        # Use our NLP system's response generation
        response = self.nlp_system.generate_response(intent, entities, data, context)
        
        # Add contextual enhancements
        response = self._enhance_response_with_context(response, entities, data)
        
        return response
    
    def _enhance_response_with_context(self, response: str, entities: ExtractedEntities, 
                                     data: Dict[str, Any]) -> str:
        """Add contextual information to enhance the response"""
        
        enhancements = []
        
        # Add practical tips based on time
        if entities.time:
            if entities.time in ['night', 'evening', 'dinner']:
                enhancements.append("💡 **Evening Tip:** Many places get busy after 7 PM, so consider making a reservation.")
            elif entities.time in ['morning', 'breakfast']:
                enhancements.append("🌅 **Morning Tip:** Turkish breakfast is a must-try experience - it's more like a feast!")
        
        # Add group size considerations
        if entities.group_size and entities.group_size > 4:
            enhancements.append(f"👥 **Large Group Tip:** For {entities.group_size} people, I'd recommend calling ahead to ensure seating.")
        
        # Add budget considerations
        if entities.budget == 'budget':
            enhancements.append("💰 **Budget Tip:** Look for 'lokanta' style restaurants for authentic, affordable meals.")
        elif entities.budget == 'upscale':
            enhancements.append("✨ **Upscale Tip:** Many fine dining restaurants offer lunch menus at better value.")
        
        # Add district-specific tips
        if entities.district:
            district_tips = {
                'sultanahmet': "🏛️ **Sultanahmet Tip:** Buy a Museum Pass Istanbul to save money and skip lines at major attractions.",
                'beyoğlu': "🎭 **Beyoğlu Tip:** The area comes alive at night - perfect for evening exploration.",
                'kadıköy': "🌍 **Kadıköy Tip:** This is where locals go - great for authentic experiences away from tourist crowds."
            }
            
            district_key = entities.district.lower()
            if district_key in district_tips:
                enhancements.append(district_tips[district_key])
        
        # Add enhancements to response
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response
    
    def _get_sample_restaurants(self) -> List[Dict[str, Any]]:
        """Fallback sample restaurant data"""
        return [
            {
                'name': 'Hamdi Restaurant',
                'cuisine': 'turkish',
                'district': 'Eminönü',
                'price_range': 'moderate',
                'atmosphere': ['traditional', 'view'],
                'description': 'Famous for lamb dishes with Golden Horn views',
                'tags': ['kebab', 'meat', 'traditional']
            },
            {
                'name': 'Karaköy Lokantası',
                'cuisine': 'turkish',
                'district': 'Karaköy',
                'price_range': 'upscale',
                'atmosphere': ['modern', 'romantic'],
                'description': 'Contemporary Turkish cuisine in elegant setting',
                'tags': ['fine dining', 'modern', 'seafood']
            },
            {
                'name': 'Çiya Sofrası',
                'cuisine': 'turkish',
                'district': 'Kadıköy',
                'price_range': 'budget',
                'atmosphere': ['local', 'casual'],
                'description': 'Authentic Anatolian dishes, locals\' favorite',
                'tags': ['authentic', 'local', 'traditional']
            }
        ]
    
    def _serialize_entities(self, entities: ExtractedEntities) -> Dict[str, Any]:
        """Convert entities to serializable dictionary"""
        return {
            'district': entities.district,
            'cuisine': entities.cuisine,
            'budget': entities.budget,
            'atmosphere': entities.atmosphere,
            'time': entities.time,
            'group_size': entities.group_size,
            'museum_name': entities.museum_name,
            'transportation_type': entities.transportation_type,
            'activity_type': entities.activity_type,
            'specific_location': entities.specific_location
        }
    
    def _get_applied_filters(self, entities: ExtractedEntities) -> List[str]:
        """Get list of filters that were applied"""
        filters = []
        
        if entities.district:
            filters.append(f"district: {entities.district}")
        if entities.cuisine:
            filters.append(f"cuisine: {entities.cuisine}")
        if entities.budget:
            filters.append(f"budget: {entities.budget}")
        if entities.atmosphere:
            filters.append(f"atmosphere: {entities.atmosphere}")
        
        return filters
    
    def _budget_compatible(self, requested_budget: str, restaurant_budget: str) -> bool:
        """Check if budgets are compatible"""
        budget_hierarchy = {'budget': 1, 'moderate': 2, 'upscale': 3}
        
        requested_level = budget_hierarchy.get(requested_budget, 2)
        restaurant_level = budget_hierarchy.get(restaurant_budget, 2)
        
        # Allow one level up or down
        return abs(requested_level - restaurant_level) <= 1
    
    def _group_size_compatible(self, group_size: int, capacity: str) -> bool:
        """Check if group size is compatible with restaurant capacity"""
        if capacity == 'any':
            return True
        
        # Simple compatibility check
        if group_size <= 2:
            return True
        elif group_size <= 6:
            return capacity in ['medium', 'large', 'any']
        else:
            return capacity in ['large', 'any']
    
    def _should_suggest_interactive_flow(self, intent: QueryIntent, entities: ExtractedEntities, 
                                       query: str) -> Optional[Dict[str, Any]]:
        """Determine if user should be guided to an interactive flow"""
        
        if not INTERACTIVE_FLOWS_AVAILABLE:
            return None
        
        query_lower = query.lower()
        
        # Detect planning-type queries
        planning_keywords = ['plan', 'itinerary', 'schedule', 'day', 'visit', 'see', 'do', 'recommendations']
        if any(keyword in query_lower for keyword in planning_keywords):
            if not entities.specific_location and not entities.museum_name:
                # General planning query - suggest interactive flow
                return {
                    'suggestion_type': 'interactive_flow',
                    'flow_suggestions': get_flow_suggestions(query),
                    'message': '🎯 **Let me help you plan the perfect experience!**\n\nI can guide you through some quick questions to create personalized recommendations:',
                    'benefits': [
                        'Get personalized recommendations',
                        'Save time with guided questions', 
                        'Discover hidden gems based on your interests',
                        'Create a complete itinerary'
                    ]
                }
        
        # Detect restaurant discovery queries
        if intent == QueryIntent.RESTAURANT_SEARCH and not entities.cuisine and not entities.district:
            return {
                'suggestion_type': 'restaurant_flow',
                'flow_suggestions': [
                    {
                        'flow_type': 'restaurant_discovery',
                        'title': '🍽️ Restaurant Discovery Guide',
                        'description': 'Find the perfect dining experience with guided questions',
                        'estimated_time': '1-2 minutes'
                    }
                ],
                'message': '🍽️ **Find Your Perfect Restaurant**\n\nLet me guide you to the ideal dining experience with a few quick questions:',
                'quick_preview': 'I\'ll ask about atmosphere, location preferences, and cuisine type'
            }
        
        # Detect museum/cultural queries
        if intent == QueryIntent.MUSEUM_INFO and not entities.museum_name:
            return {
                'suggestion_type': 'museum_flow',
                'flow_suggestions': [
                    {
                        'flow_type': 'museum_tour',
                        'title': '🏛️ Museum Tour Planner', 
                        'description': 'Create a personalized cultural experience',
                        'estimated_time': '2 minutes'
                    }
                ],
                'message': '🏛️ **Plan Your Cultural Journey**\n\nI can help you discover the perfect museums and cultural sites:',
                'quick_preview': 'Tell me your interests and I\'ll create a personalized tour'
            }
        
        return None

    def _add_seasonal_context(self, intent: QueryIntent, entities: ExtractedEntities) -> Optional[str]:
        """Add relevant seasonal context to responses"""
        
        if not SEASONAL_EVENTS_AVAILABLE:
            return None
        
        try:
            # Get current seasonal events
            seasonal_events = get_current_seasonal_events()
            daily_updates = get_daily_istanbul_updates()
            
            if not seasonal_events and not daily_updates.get('daily_insights'):
                return None
            
            seasonal_context = "🌟 **Current Season Tips:**\n"
            
            # Add relevant seasonal events
            relevant_events = []
            for event in seasonal_events[:2]:  # Max 2 events
                if event['impact_level'] >= 3:  # Only high-impact events
                    # Check if event is relevant to current intent
                    if self._is_event_relevant(event, intent, entities):
                        relevant_events.append(event)
            
            for event in relevant_events:
                seasonal_context += f"• **{event['title']}**: {event['description']}\n"
                if event['recommendations']:
                    seasonal_context += f"  💡 {event['recommendations'][0]}\n"
            
            # Add daily insights
            insights = daily_updates.get('daily_insights', [])
            if insights and not relevant_events:
                seasonal_context += f"• {insights[0]}\n"
            
            return seasonal_context if len(seasonal_context) > 50 else None
            
        except Exception as e:
            print(f"⚠️ Error adding seasonal context: {e}")
            return None
    
    def _is_event_relevant(self, event: Dict[str, Any], intent: QueryIntent, 
                          entities: ExtractedEntities) -> bool:
        """Check if a seasonal event is relevant to the current query"""
        
        event_type = event.get('type', '')
        
        # Restaurant queries - relevant if affects dining
        if intent == QueryIntent.RESTAURANT_SEARCH:
            return event_type in ['weather', 'seasonal', 'religious']
        
        # Museum queries - relevant if affects cultural activities
        if intent == QueryIntent.MUSEUM_INFO:
            return event_type in ['cultural', 'festival', 'weather', 'seasonal']
        
        # Transportation queries - always relevant for transport events
        if intent == QueryIntent.TRANSPORTATION or intent == QueryIntent.DIRECTIONS:
            return event_type in ['transport', 'weather', 'seasonal']
        
        # General planning - most events relevant
        if intent == QueryIntent.GENERAL_INFO:
            return event['impact_level'] >= 4
        
        # District info - relevant if location matches
        if intent == QueryIntent.DISTRICT_INFO and entities.district:
            return entities.district.lower() in event.get('location', '').lower()
        
        return False

# Global instance
intelligent_processor = IntelligentQueryProcessor()

def process_intelligent_query(query: str, session_id: str = "default", 
                            database_manager=None) -> Dict[str, Any]:
    """Main function to process queries intelligently without LLMs"""
    
    if database_manager:
        intelligent_processor.database_manager = database_manager
    
    return intelligent_processor.process_user_query(query, session_id)
