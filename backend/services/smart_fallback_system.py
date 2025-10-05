"""
Smart Fallback Architecture for AI Istanbul
Multi-layered fallback system ensuring robust responses when primary AI fails
Implements 4-tier fallback: Semantic Cache -> Template+KG -> Rule-based -> Generic
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import random

logger = logging.getLogger(__name__)

class FallbackLevel(Enum):
    """Fallback priority levels"""
    SEMANTIC_CACHE = "semantic_cache"      # 70-80%
    TEMPLATE_KNOWLEDGE = "template_kg"     # 15-20%
    RULE_BASED = "rule_based"             # 3-5%
    GENERIC_HELPFUL = "generic"           # 1-2%

@dataclass
class FallbackResponse:
    """Structured fallback response"""
    content: str
    confidence: float
    source: FallbackLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    related_info: List[Dict] = field(default_factory=list)
    processing_time_ms: int = 0

@dataclass
class FallbackContext:
    """Context for fallback decision making"""
    original_query: str
    intent_type: str
    confidence_score: float
    extracted_entities: Dict[str, List[str]]
    user_profile: Optional[Dict] = None
    session_context: Optional[Dict] = None
    failure_reason: str = ""

class SmartFallbackSystem:
    """
    Multi-layered smart fallback system
    Provides intelligent responses when primary AI systems fail or have low confidence
    """
    
    def __init__(self, knowledge_graph=None, semantic_cache=None):
        self.knowledge_graph = knowledge_graph
        self.semantic_cache = semantic_cache
        
        # Fallback templates organized by intent and context
        self.templates = self._initialize_templates()
        
        # Rule-based response patterns
        self.rule_patterns = self._initialize_rule_patterns()
        
        # Generic helpful responses
        self.generic_responses = self._initialize_generic_responses()
        
        # Fallback statistics
        self.stats = {
            'total_fallbacks': 0,
            'semantic_cache_hits': 0,
            'template_kg_used': 0,
            'rule_based_used': 0,
            'generic_used': 0,
            'success_rate_by_level': {level.value: 0 for level in FallbackLevel}
        }
        
        # Response quality tracking
        self.quality_scores = defaultdict(list)
        
        logger.info("Smart Fallback System initialized")
    
    def get_fallback_response(self, context: FallbackContext) -> FallbackResponse:
        """
        Main fallback method - tries each layer in order
        """
        start_time = datetime.now()
        self.stats['total_fallbacks'] += 1
        
        # Layer 1: Semantic Cache (70-80% success target)
        cache_response = self._try_semantic_cache_fallback(context)
        if cache_response and cache_response.confidence >= 0.7:
            self.stats['semantic_cache_hits'] += 1
            self._update_quality_tracking(FallbackLevel.SEMANTIC_CACHE, cache_response.confidence)
            cache_response.processing_time_ms = self._get_processing_time(start_time)
            return cache_response
        
        # Layer 2: Template + Knowledge Graph (15-20% success target)
        template_response = self._try_template_knowledge_fallback(context)
        if template_response and template_response.confidence >= 0.6:
            self.stats['template_kg_used'] += 1
            self._update_quality_tracking(FallbackLevel.TEMPLATE_KNOWLEDGE, template_response.confidence)
            template_response.processing_time_ms = self._get_processing_time(start_time)
            return template_response
        
        # Layer 3: Rule-based Generation (3-5% success target)
        rule_response = self._try_rule_based_fallback(context)
        if rule_response and rule_response.confidence >= 0.4:
            self.stats['rule_based_used'] += 1
            self._update_quality_tracking(FallbackLevel.RULE_BASED, rule_response.confidence)
            rule_response.processing_time_ms = self._get_processing_time(start_time)
            return rule_response
        
        # Layer 4: Generic Helpful Response (1-2% usage)
        generic_response = self._get_generic_helpful_response(context)
        self.stats['generic_used'] += 1
        self._update_quality_tracking(FallbackLevel.GENERIC_HELPFUL, generic_response.confidence)
        generic_response.processing_time_ms = self._get_processing_time(start_time)
        
        return generic_response
    
    def _try_semantic_cache_fallback(self, context: FallbackContext) -> Optional[FallbackResponse]:
        """Layer 1: Try semantic cache for similar queries"""
        if not self.semantic_cache:
            return None
        
        try:
            # Try semantic similarity search in cache
            similar_responses = self._find_similar_cached_responses(
                context.original_query, 
                context.intent_type,
                threshold=0.75
            )
            
            if similar_responses:
                best_match = similar_responses[0]
                
                # Adapt cached response to current context
                adapted_content = self._adapt_cached_response(
                    best_match['content'], 
                    context
                )
                
                return FallbackResponse(
                    content=adapted_content,
                    confidence=best_match['similarity'] * 0.9,  # Slight penalty for adaptation
                    source=FallbackLevel.SEMANTIC_CACHE,
                    metadata={
                        'original_similarity': best_match['similarity'],
                        'cache_source': best_match.get('query', 'unknown'),
                        'adaptation_applied': True
                    },
                    suggestions=self._generate_cache_suggestions(context)
                )
        
        except Exception as e:
            logger.warning(f"Semantic cache fallback failed: {e}")
        
        return None
    
    def _try_template_knowledge_fallback(self, context: FallbackContext) -> Optional[FallbackResponse]:
        """Layer 2: Template-based response with knowledge graph enrichment"""
        
        # Find appropriate template
        template = self._select_template(context.intent_type, context.extracted_entities)
        if not template:
            return None
        
        try:
            # Get knowledge graph data to enrich template
            kg_data = self._get_knowledge_graph_enrichment(context)
            
            # Fill template with available data
            filled_content = self._fill_template(template, context, kg_data)
            
            # Calculate confidence based on data availability
            confidence = self._calculate_template_confidence(template, kg_data, context)
            
            return FallbackResponse(
                content=filled_content,
                confidence=confidence,
                source=FallbackLevel.TEMPLATE_KNOWLEDGE,
                metadata={
                    'template_id': template.get('id', 'unknown'),
                    'kg_data_points': len(kg_data),
                    'entities_matched': len(context.extracted_entities)
                },
                suggestions=self._generate_template_suggestions(context, kg_data),
                related_info=self._get_related_info_from_kg(context, kg_data)
            )
        
        except Exception as e:
            logger.warning(f"Template+KG fallback failed: {e}")
        
        return None
    
    def _try_rule_based_fallback(self, context: FallbackContext) -> Optional[FallbackResponse]:
        """Layer 3: Rule-based response generation"""
        
        try:
            # Match query patterns against rule patterns
            matched_rules = self._match_rule_patterns(context.original_query, context.intent_type)
            
            if matched_rules:
                best_rule = matched_rules[0]
                
                # Generate response using rule
                generated_content = self._generate_rule_response(best_rule, context)
                
                confidence = best_rule['confidence'] * 0.8  # Rule-based penalty
                
                return FallbackResponse(
                    content=generated_content,
                    confidence=confidence,
                    source=FallbackLevel.RULE_BASED,
                    metadata={
                        'rule_pattern': best_rule['pattern'],
                        'rule_type': best_rule['type']
                    },
                    suggestions=self._generate_rule_suggestions(context)
                )
        
        except Exception as e:
            logger.warning(f"Rule-based fallback failed: {e}")
        
        return None
    
    def _get_generic_helpful_response(self, context: FallbackContext) -> FallbackResponse:
        """Layer 4: Generic helpful response - always succeeds"""
        
        # Select appropriate generic response based on context
        generic_response = self._select_generic_response(context)
        
        # Add context-specific suggestions
        suggestions = self._generate_generic_suggestions(context)
        
        return FallbackResponse(
            content=generic_response['content'],
            confidence=0.3,  # Low but acceptable confidence
            source=FallbackLevel.GENERIC_HELPFUL,
            metadata={
                'response_type': generic_response['type'],
                'fallback_reason': context.failure_reason
            },
            suggestions=suggestions
        )
    
    def _initialize_templates(self) -> Dict[str, List[Dict]]:
        """Initialize response templates by intent type"""
        return {
            "attractions": [
                {
                    "id": "attraction_basic_info",
                    "pattern": r"(tell me about|what is|information about) (.+)",
                    "template": """About {attraction_name}:

{basic_info}

🕒 **Opening Hours:** {opening_hours}
💰 **Entrance Fee:** {entrance_fee}
📍 **Location:** {location}

{nearby_info}

{visit_tips}

Would you like specific information about visiting hours, transportation, or nearby attractions?""",
                    "required_fields": ["attraction_name"],
                    "optional_fields": ["basic_info", "opening_hours", "entrance_fee", "location", "nearby_info", "visit_tips"]
                },
                {
                    "id": "attraction_visiting_info",
                    "pattern": r"(how to|best time|when to visit) (.+)",
                    "template": """Best Time to Visit {attraction_name}:

⏰ **Optimal Time:** {best_time}
🚶 **Expected Duration:** {visit_duration}
📸 **Photography:** {photography_info}

**Getting There:**
{transport_options}

**Nearby Attractions:**
{nearby_attractions}

Need help planning your route or want restaurant recommendations nearby?""",
                    "required_fields": ["attraction_name"],
                    "optional_fields": ["best_time", "visit_duration", "photography_info", "transport_options", "nearby_attractions"]
                }
            ],
            
            "transportation": [
                {
                    "id": "route_basic",
                    "pattern": r"(how to get|route|directions) (.+) to (.+)",
                    "template": """Route from {from_location} to {to_location}:

🚇 **Recommended Route:**
{route_steps}

⏱️ **Journey Time:** {duration}
💰 **Cost:** {cost}

**Alternative Options:**
{alternatives}

{transport_tips}

Need more specific timing or want to add stops along the way?""",
                    "required_fields": ["from_location", "to_location"],
                    "optional_fields": ["route_steps", "duration", "cost", "alternatives", "transport_tips"]
                }
            ],
            
            "food_dining": [
                {
                    "id": "restaurant_recommendations",
                    "pattern": r"(restaurant|food|eat|dining) (.+)",
                    "template": """Food & Dining in {area}:

🍽️ **Top Recommendations:**
{restaurant_list}

**Cuisine Types Available:**
{cuisine_types}

**Budget Options:**
{budget_options}

{dietary_info}

Would you like specific restaurant details or directions to any of these places?""",
                    "required_fields": ["area"],
                    "optional_fields": ["restaurant_list", "cuisine_types", "budget_options", "dietary_info"]
                }
            ],
            
            "general": [
                {
                    "id": "general_help",
                    "pattern": r".*",
                    "template": """I'd be happy to help you with Istanbul tourism information!

Based on your query about "{query}", here are some ways I can assist:

{relevant_topics}

**Popular Topics I Can Help With:**
• 🏛️ Historical attractions and museums
• 🚇 Transportation and getting around
• 🍽️ Restaurants and local cuisine
• 🛍️ Shopping and markets
• 🗺️ Area exploration and neighborhoods

{personalized_suggestions}

What specific aspect would you like to know more about?""",
                    "required_fields": ["query"],
                    "optional_fields": ["relevant_topics", "personalized_suggestions"]
                }
            ]
        }
    
    def _initialize_rule_patterns(self) -> List[Dict]:
        """Initialize rule-based response patterns"""
        return [
            {
                "pattern": r"\b(opening hours?|open|close|schedule)\b.*\b(hagia sophia|ayasofya)\b",
                "type": "practical_info",
                "confidence": 0.8,
                "response_template": "Hagia Sophia is open daily from 9:00 AM to 7:00 PM (April-October) and 9:00 AM to 5:00 PM (November-March). Last entry is 1 hour before closing. Entrance fee is 200 TL for adults."
            },
            {
                "pattern": r"\b(opening hours?|open|close|schedule)\b.*\b(blue mosque|sultanahmet mosque)\b",
                "type": "practical_info", 
                "confidence": 0.8,
                "response_template": "Blue Mosque is open daily except during prayer times. Best visiting hours are: 8:30-11:30 AM, 1:00-2:30 PM, 4:00-6:00 PM, and 7:30-9:00 PM. Entry is free, but dress modestly."
            },
            {
                "pattern": r"\b(metro|tram|bus|transport)\b.*\b(sultanahmet|taksim|galata)\b",
                "type": "transportation",
                "confidence": 0.7,
                "response_template": "For public transport in Istanbul, use the Istanbulkart. Metro M2 connects Taksim to other areas. T1 tram serves Sultanahmet. The system operates 6:00 AM to midnight, with reduced service on Sundays."
            },
            {
                "pattern": r"\b(restaurant|food|eat|dining)\b.*\b(turkish|traditional|local)\b",
                "type": "food_dining",
                "confidence": 0.6,
                "response_template": "For authentic Turkish cuisine, try traditional dishes like kebab, meze, dolma, and baklava. Popular areas for dining include Sultanahmet, Beyoğlu, and Kadıköy. Don't miss Turkish breakfast and tea culture!"
            },
            {
                "pattern": r"\b(shop|shopping|buy|souvenir)\b.*\b(grand bazaar|market|bazaar)\b",
                "type": "shopping",
                "confidence": 0.7,
                "response_template": "Grand Bazaar is perfect for shopping Turkish carpets, jewelry, ceramics, and souvenirs. Open Monday-Saturday 9:00 AM-7:00 PM. Remember to bargain! Also visit Spice Bazaar for Turkish delight and spices."
            }
        ]
    
    def _initialize_generic_responses(self) -> List[Dict]:
        """Initialize generic helpful responses"""
        return [
            {
                "type": "general_help",
                "content": """I'm here to help you explore Istanbul! While I couldn't provide specific details for your request, I can assist you with:

🏛️ **Attractions**: Hagia Sophia, Blue Mosque, Topkapi Palace, Galata Tower
🚇 **Transportation**: Metro, tram, bus routes and schedules  
🍽️ **Dining**: Turkish cuisine recommendations and restaurant suggestions
🛍️ **Shopping**: Markets, bazaars, and local shopping areas
🗺️ **Areas**: Sultanahmet, Beyoğlu, Galata, and other neighborhoods

Could you try rephrasing your question or let me know what specific aspect of Istanbul interests you most?"""
            },
            {
                "type": "encouragement",
                "content": """I'd love to help you discover Istanbul! This amazing city has so much to offer, from ancient Byzantine and Ottoman sites to vibrant modern neighborhoods.

While I need a bit more context to give you the perfect answer, I'm equipped to help with:
• Historical sites and cultural attractions
• Getting around the city efficiently  
• Food recommendations and dining experiences
• Shopping and local markets
• Neighborhood exploration

What would you like to explore first? Feel free to ask about any specific place or activity!"""
            },
            {
                "type": "clarification",
                "content": """I want to make sure I give you the most helpful information about Istanbul! Could you help me understand what you're looking for?

For example, you might ask:
• "How do I get from Taksim to Hagia Sophia?"
• "What are the best Turkish restaurants in Sultanahmet?"  
• "What time does the Grand Bazaar open?"
• "What should I see in the Galata area?"

Istanbul has incredible history, amazing food, and fascinating neighborhoods - I'm here to help you make the most of your visit!"""
            }
        ]
    
    def _find_similar_cached_responses(self, query: str, intent: str, threshold: float = 0.75) -> List[Dict]:
        """Find semantically similar cached responses"""
        if not self.semantic_cache:
            return []
        
        try:
            # This would use the semantic cache's similarity search
            # For now, return mock data - replace with actual semantic search
            mock_similar = [
                {
                    'content': 'Previous similar response about Istanbul attractions...',
                    'similarity': 0.82,
                    'query': 'Tell me about Hagia Sophia'
                }
            ]
            return [item for item in mock_similar if item['similarity'] >= threshold]
        except Exception as e:
            logger.warning(f"Cache similarity search failed: {e}")
            return []
    
    def _adapt_cached_response(self, cached_content: str, context: FallbackContext) -> str:
        """Adapt cached response to current context"""
        adapted = cached_content
        
        # Replace entity references if found in current context
        if context.extracted_entities:
            for entity_type, entities in context.extracted_entities.items():
                if entities:
                    # Simple adaptation - would be more sophisticated in practice
                    adapted = f"Based on your interest in {', '.join(entities[:2])}, here's what I found:\n\n{adapted}"
        
        return adapted
    
    def _select_template(self, intent_type: str, entities: Dict[str, List[str]]) -> Optional[Dict]:
        """Select most appropriate template for the context"""
        templates = self.templates.get(intent_type, [])
        
        if not templates:
            templates = self.templates.get("general", [])
        
        # For now, return first available template
        # In practice, would score templates based on entity matches
        return templates[0] if templates else None
    
    def _get_knowledge_graph_enrichment(self, context: FallbackContext) -> Dict[str, Any]:
        """Get enrichment data from knowledge graph"""
        kg_data = {}
        
        if not self.knowledge_graph:
            return kg_data
        
        try:
            # Extract location entities
            locations = context.extracted_entities.get('locations', [])
            if locations:
                primary_location = locations[0]
                
                # Try to get enriched data from knowledge graph
                if hasattr(self.knowledge_graph, 'get_enriched_response'):
                    enrichment = self.knowledge_graph.get_enriched_response(
                        context.original_query, 
                        primary_location
                    )
                    
                    if enrichment.get('enriched'):
                        kg_data = enrichment.get('data', {})
        
        except Exception as e:
            logger.warning(f"Knowledge graph enrichment failed: {e}")
        
        return kg_data
    
    def _fill_template(self, template: Dict, context: FallbackContext, kg_data: Dict) -> str:
        """Fill template with available data"""
        template_str = template['template']
        
        # Create replacement dictionary
        replacements = {
            'query': context.original_query,
            'attraction_name': 'this attraction',
            'area': 'this area',
            'from_location': 'your starting point',
            'to_location': 'your destination'
        }
        
        # Add entity-based replacements
        entities = context.extracted_entities
        if entities.get('locations'):
            locations = entities['locations']
            replacements['attraction_name'] = locations[0]
            replacements['area'] = locations[0]
            if len(locations) > 1:
                replacements['from_location'] = locations[0]
                replacements['to_location'] = locations[1]
        
        # Add knowledge graph data
        if kg_data.get('primary_info'):
            primary_info = kg_data['primary_info']
            replacements.update({
                'basic_info': primary_info.get('description', 'A significant Istanbul attraction'),
                'opening_hours': primary_info.get('opening_hours', 'Please check current hours'),
                'entrance_fee': primary_info.get('entrance_fee', 'Varies'),
                'location': primary_info.get('district', 'Istanbul'),
                'best_time': primary_info.get('best_visit_time', 'Early morning or late afternoon'),
                'visit_duration': primary_info.get('visit_duration', '1-2 hours'),
                'photography_info': primary_info.get('photography', 'Photography allowed')
            })
        
        # Add nearby information
        if kg_data.get('nearby_attractions'):
            nearby = kg_data['nearby_attractions'][:3]  # Top 3
            nearby_list = '\n'.join([f"• {item.get('name', 'Nearby attraction')}" for item in nearby])
            replacements['nearby_attractions'] = nearby_list
            replacements['nearby_info'] = f"**Nearby Attractions:**\n{nearby_list}"
        
        # Add transport info
        if kg_data.get('transport_info'):
            transport = kg_data['transport_info']
            transport_list = '\n'.join([f"• {item.get('name', 'Transport option')}" for item in transport])
            replacements['transport_options'] = transport_list
        
        # Fill template with fallbacks for missing data
        for key, value in replacements.items():
            placeholder = '{' + key + '}'
            template_str = template_str.replace(placeholder, str(value))
        
        # Remove any unfilled placeholders
        template_str = re.sub(r'\{[^}]+\}', '[Information not available]', template_str)
        
        return template_str
    
    def _calculate_template_confidence(self, template: Dict, kg_data: Dict, context: FallbackContext) -> float:
        """Calculate confidence score for template-based response"""
        base_confidence = 0.6
        
        # Boost for required fields filled
        required_fields = template.get('required_fields', [])
        filled_required = sum(1 for field in required_fields if self._field_available(field, context, kg_data))
        required_boost = (filled_required / len(required_fields)) * 0.2 if required_fields else 0.1
        
        # Boost for optional fields filled  
        optional_fields = template.get('optional_fields', [])
        filled_optional = sum(1 for field in optional_fields if self._field_available(field, context, kg_data))
        optional_boost = (filled_optional / len(optional_fields)) * 0.1 if optional_fields else 0
        
        # Boost for knowledge graph enrichment
        kg_boost = 0.1 if kg_data else 0
        
        return min(0.9, base_confidence + required_boost + optional_boost + kg_boost)
    
    def _field_available(self, field: str, context: FallbackContext, kg_data: Dict) -> bool:
        """Check if a template field has available data"""
        # Check in entities
        if field in ['attraction_name', 'area', 'from_location', 'to_location']:
            return bool(context.extracted_entities.get('locations'))
        
        # Check in knowledge graph data
        if kg_data.get('primary_info'):
            field_mapping = {
                'basic_info': 'description',
                'opening_hours': 'opening_hours',
                'entrance_fee': 'entrance_fee'
            }
            kg_field = field_mapping.get(field, field)
            return kg_field in kg_data['primary_info']
        
        return False
    
    def _match_rule_patterns(self, query: str, intent_type: str) -> List[Dict]:
        """Match query against rule patterns"""
        matched_rules = []
        
        for rule in self.rule_patterns:
            if re.search(rule['pattern'], query, re.IGNORECASE):
                # Boost confidence if intent matches
                confidence = rule['confidence']
                if rule['type'].replace('_', '') in intent_type:
                    confidence *= 1.2
                
                matched_rules.append({
                    **rule,
                    'confidence': min(1.0, confidence)
                })
        
        # Sort by confidence
        return sorted(matched_rules, key=lambda x: x['confidence'], reverse=True)
    
    def _generate_rule_response(self, rule: Dict, context: FallbackContext) -> str:
        """Generate response using matched rule"""
        response = rule['response_template']
        
        # Add context-specific information if available
        if context.extracted_entities.get('locations'):
            location = context.extracted_entities['locations'][0]
            response = response.replace('this location', location)
        
        return response
    
    def _select_generic_response(self, context: FallbackContext) -> Dict:
        """Select appropriate generic response"""
        responses = self.generic_responses
        
        # Simple selection based on failure reason
        if 'low_confidence' in context.failure_reason:
            return responses[2]  # clarification
        elif 'complex' in context.failure_reason:
            return responses[1]  # encouragement  
        else:
            return responses[0]  # general help
    
    def _generate_cache_suggestions(self, context: FallbackContext) -> List[str]:
        """Generate suggestions based on cache data"""
        suggestions = [
            "Try asking about specific attractions like 'Hagia Sophia opening hours'",
            "Ask about transportation: 'How to get from Taksim to Sultanahmet'",
            "Get restaurant recommendations: 'Best Turkish restaurants in Beyoğlu'"
        ]
        return suggestions[:2]  # Return top 2
    
    def _generate_template_suggestions(self, context: FallbackContext, kg_data: Dict) -> List[str]:
        """Generate suggestions based on template and KG data"""
        suggestions = []
        
        if kg_data.get('nearby_attractions'):
            nearby = kg_data['nearby_attractions'][:2]
            for attraction in nearby:
                suggestions.append(f"Learn about nearby {attraction.get('name', 'attraction')}")
        
        if context.intent_type == 'attractions':
            suggestions.extend([
                "Ask about opening hours and entrance fees",
                "Get directions and transportation options"
            ])
        
        return suggestions[:3]
    
    def _generate_rule_suggestions(self, context: FallbackContext) -> List[str]:
        """Generate suggestions for rule-based responses"""
        return [
            "Ask for more specific details about your topic",
            "Try including location names in your question",
            "Ask about practical details like times and costs"
        ]
    
    def _generate_generic_suggestions(self, context: FallbackContext) -> List[str]:
        """Generate suggestions for generic responses"""
        return [
            "Ask about specific Istanbul attractions",
            "Get help with transportation and routes", 
            "Find restaurant and dining recommendations",
            "Explore different neighborhoods and areas"
        ]
    
    def _get_related_info_from_kg(self, context: FallbackContext, kg_data: Dict) -> List[Dict]:
        """Extract related information from knowledge graph data"""
        related_info = []
        
        if kg_data.get('nearby_attractions'):
            for attraction in kg_data['nearby_attractions'][:2]:
                related_info.append({
                    'type': 'nearby_attraction',
                    'title': attraction.get('name', 'Nearby Attraction'),
                    'description': attraction.get('description', 'Worth visiting')
                })
        
        if kg_data.get('dining_options'):
            for restaurant in kg_data['dining_options'][:1]:
                related_info.append({
                    'type': 'dining',
                    'title': restaurant.get('name', 'Local Restaurant'),
                    'description': restaurant.get('cuisine_type', 'Turkish cuisine')
                })
        
        return related_info
    
    def _update_quality_tracking(self, level: FallbackLevel, confidence: float):
        """Track quality metrics for each fallback level"""
        self.quality_scores[level.value].append(confidence)
        self.stats['success_rate_by_level'][level.value] += 1
    
    def _get_processing_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fallback system statistics"""
        total = self.stats['total_fallbacks']
        if total == 0:
            return self.stats
        
        stats = dict(self.stats)
        
        # Calculate success rates
        stats['semantic_cache_rate'] = (self.stats['semantic_cache_hits'] / total) * 100
        stats['template_kg_rate'] = (self.stats['template_kg_used'] / total) * 100
        stats['rule_based_rate'] = (self.stats['rule_based_used'] / total) * 100
        stats['generic_rate'] = (self.stats['generic_used'] / total) * 100
        
        # Quality metrics
        stats['avg_quality_by_level'] = {}
        for level, scores in self.quality_scores.items():
            if scores:
                stats['avg_quality_by_level'][level] = sum(scores) / len(scores)
        
        return stats
    
    def optimize_fallback_thresholds(self):
        """Optimize fallback thresholds based on performance data"""
        # Analyze success rates and adjust thresholds
        # This would implement adaptive threshold optimization
        pass

# Factory function
def create_smart_fallback_system(knowledge_graph=None, semantic_cache=None) -> SmartFallbackSystem:
    """Create and return smart fallback system instance"""
    return SmartFallbackSystem(knowledge_graph, semantic_cache)
