"""
Istanbul Knowledge Graph System
Rich contextual information for intelligent query responses without GPT dependency
Provides connected information about attractions, restaurants, transport, and user patterns
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""
    id: str
    name: str
    type: str  # attraction, restaurant, transport, district, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    connections: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserJourney:
    """Track user interaction patterns"""
    user_id: str
    session_id: str
    queries: List[Dict] = field(default_factory=list)
    visited_nodes: Set[str] = field(default_factory=set)
    patterns: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class IstanbulKnowledgeGraph:
    """
    Comprehensive knowledge graph for Istanbul tourism
    Enables rich, contextual responses without GPT dependency
    """
    
    def __init__(self, data_dir: str = "knowledge_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core knowledge structures
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.connections: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Behavioral pattern tracking
        self.user_journeys: Dict[str, UserJourney] = {}
        self.common_patterns: Dict[str, int] = Counter()
        self.prediction_rules: Dict[str, List[str]] = defaultdict(list)
        
        # Performance metrics
        self.usage_stats = {
            'total_queries': 0,
            'graph_hits': 0,
            'pattern_predictions': 0,
            'enrichment_success': 0
        }
        
        # Initialize the knowledge graph
        self._initialize_istanbul_knowledge()
        self._load_behavioral_patterns()
    
    def _initialize_istanbul_knowledge(self):
        """Initialize comprehensive Istanbul knowledge graph"""
        
        # Major attractions
        attractions = {
            "hagia_sophia": {
                "name": "Hagia Sophia",
                "type": "attraction",
                "district": "sultanahmet",
                "category": "historical_religious",
                "visit_duration": "1-2 hours",
                "opening_hours": "9:00-19:00 (Apr-Oct), 9:00-17:00 (Nov-Mar)",
                "entrance_fee": "200 TL",
                "best_visit_time": "early_morning",
                "photography": "allowed_no_flash",
                "accessibility": "wheelchair_accessible"
            },
            "blue_mosque": {
                "name": "Blue Mosque (Sultan Ahmed Mosque)",
                "type": "attraction", 
                "district": "sultanahmet",
                "category": "historical_religious",
                "visit_duration": "30-60 minutes",
                "opening_hours": "Daily except prayer times",
                "entrance_fee": "free",
                "best_visit_time": "early_morning_late_afternoon",
                "photography": "allowed_respectful",
                "accessibility": "partial_wheelchair_access"
            },
            "topkapi_palace": {
                "name": "Topkapi Palace",
                "type": "attraction",
                "district": "sultanahmet", 
                "category": "historical_palace",
                "visit_duration": "2-3 hours",
                "opening_hours": "9:00-18:45 (closed Tuesdays)",
                "entrance_fee": "200 TL",
                "best_visit_time": "morning",
                "photography": "allowed_some_restrictions",
                "accessibility": "limited_wheelchair_access"
            },
            "galata_tower": {
                "name": "Galata Tower",
                "type": "attraction",
                "district": "galata",
                "category": "historical_viewpoint",
                "visit_duration": "1 hour",
                "opening_hours": "9:00-20:00",
                "entrance_fee": "200 TL",
                "best_visit_time": "sunset",
                "photography": "excellent_views",
                "accessibility": "elevator_available"
            },
            "grand_bazaar": {
                "name": "Grand Bazaar",
                "type": "attraction_shopping",
                "district": "beyazit",
                "category": "shopping_historical",
                "visit_duration": "1-3 hours", 
                "opening_hours": "9:00-19:00 (closed Sundays)",
                "entrance_fee": "free",
                "best_visit_time": "morning_afternoon",
                "photography": "allowed",
                "accessibility": "crowded_difficult"
            }
        }
        
        # Restaurants and dining
        restaurants = {
            "pandeli": {
                "name": "Pandeli Restaurant",
                "type": "restaurant",
                "district": "eminonu",
                "cuisine": "ottoman_traditional",
                "price_range": "expensive",
                "specialty": "ottoman_cuisine",
                "atmosphere": "historic_elegant",
                "reservation": "recommended"
            },
            "seasons_restaurant": {
                "name": "Seasons Restaurant",
                "type": "restaurant", 
                "district": "sultanahmet",
                "cuisine": "international_turkish",
                "price_range": "expensive",
                "specialty": "fine_dining",
                "atmosphere": "luxury_hotel",
                "reservation": "required"
            },
            "hamdi_restaurant": {
                "name": "Hamdi Restaurant",
                "type": "restaurant",
                "district": "eminonu",
                "cuisine": "southeastern_turkish",
                "price_range": "moderate",
                "specialty": "kebab_meat",
                "atmosphere": "traditional_bustling",
                "reservation": "not_needed"
            }
        }
        
        # Transportation nodes
        transport = {
            "sultanahmet_tram": {
                "name": "Sultanahmet Tram Station",
                "type": "transport",
                "transport_type": "tram",
                "line": "T1",
                "connections": ["eminonu", "beyazit", "karakoy"]
            },
            "galata_bridge": {
                "name": "Galata Bridge",
                "type": "transport_landmark",
                "transport_type": "walking_ferry",
                "connections": ["eminonu", "karakoy", "golden_horn"]
            }
        }
        
        # Create nodes
        all_locations = {**attractions, **restaurants, **transport}
        
        for location_id, data in all_locations.items():
            node = KnowledgeNode(
                id=location_id,
                name=data["name"],
                type=data["type"], 
                properties=data,
                metadata={"created": datetime.now().isoformat()}
            )
            self.nodes[location_id] = node
        
        # Define rich connections
        self._create_knowledge_connections()
        
        logger.info(f"✅ Initialized Istanbul knowledge graph with {len(self.nodes)} nodes")
    
    def _create_knowledge_connections(self):
        """Create rich connections between knowledge nodes"""
        
        # Hagia Sophia connections
        self.connections["hagia_sophia"].update({
            "nearby_walking": ["blue_mosque", "topkapi_palace", "sultanahmet_tram"],
            "nearby_restaurants": ["seasons_restaurant", "pandeli"],
            "combine_visit": ["blue_mosque", "topkapi_palace", "grand_bazaar"],
            "same_district": ["blue_mosque", "topkapi_palace"],
            "transport_access": ["sultanahmet_tram"],
            "optimal_sequence": ["blue_mosque", "topkapi_palace"],
            "photography_spots": ["blue_mosque", "sultanahmet_square"],
            "cultural_theme": ["blue_mosque", "topkapi_palace"]
        })
        
        # Blue Mosque connections
        self.connections["blue_mosque"].update({
            "nearby_walking": ["hagia_sophia", "sultanahmet_tram"],
            "nearby_restaurants": ["seasons_restaurant"],
            "combine_visit": ["hagia_sophia", "topkapi_palace"],
            "same_district": ["hagia_sophia", "topkapi_palace"],
            "transport_access": ["sultanahmet_tram"],
            "optimal_sequence": ["hagia_sophia", "topkapi_palace"],
            "photography_spots": ["hagia_sophia", "sultanahmet_square"],
            "cultural_theme": ["hagia_sophia", "topkapi_palace"]
        })
        
        # Topkapi Palace connections
        self.connections["topkapi_palace"].update({
            "nearby_walking": ["hagia_sophia", "blue_mosque"],
            "nearby_restaurants": ["seasons_restaurant", "pandeli"],
            "combine_visit": ["hagia_sophia", "blue_mosque"],
            "same_district": ["hagia_sophia", "blue_mosque"],
            "transport_access": ["sultanahmet_tram"],
            "cultural_theme": ["hagia_sophia", "blue_mosque"],
            "historical_period": ["hagia_sophia"]
        })
        
        # Galata Tower connections
        self.connections["galata_tower"].update({
            "nearby_walking": ["galata_bridge"],
            "viewpoint_targets": ["hagia_sophia", "blue_mosque", "golden_horn"],
            "transport_access": ["galata_bridge"],
            "photography_best": ["sunset", "bosphorus_view"],
            "combine_visit": ["galata_bridge", "karakoy_area"]
        })
        
        # Grand Bazaar connections
        self.connections["grand_bazaar"].update({
            "nearby_walking": ["beyazit_mosque", "sultanahmet_area"],
            "shopping_theme": ["spice_bazaar", "arasta_bazaar"],
            "transport_access": ["beyazit_tram"],
            "combine_visit": ["spice_bazaar", "sultanahmet_area"],
            "cultural_experience": ["traditional_shopping", "haggling"]
        })
    
    def get_enriched_response(self, query: str, primary_node: str, 
                            user_context: Dict = None) -> Dict[str, Any]:
        """
        Get enriched response with connected information
        """
        self.usage_stats['total_queries'] += 1
        
        if primary_node not in self.nodes:
            return {'enriched': False, 'reason': 'node_not_found'}
        
        try:
            node = self.nodes[primary_node]
            enrichment = {
                'primary_info': self._get_node_info(node),
                'nearby_attractions': self._get_connected_nodes(primary_node, 'nearby_walking'),
                'dining_options': self._get_connected_nodes(primary_node, 'nearby_restaurants'),
                'transport_info': self._get_connected_nodes(primary_node, 'transport_access'),
                'combination_suggestions': self._get_connected_nodes(primary_node, 'combine_visit'),
                'optimal_sequence': self._get_optimal_visiting_sequence(primary_node),
                'contextual_tips': self._get_contextual_tips(primary_node, user_context),
                'behavioral_predictions': self._predict_next_interests(primary_node, user_context)
            }
            
            self.usage_stats['graph_hits'] += 1
            self.usage_stats['enrichment_success'] += 1
            
            return {
                'enriched': True,
                'data': enrichment,
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"Error enriching response: {e}")
            return {'enriched': False, 'reason': str(e)}
    
    def _get_node_info(self, node: KnowledgeNode) -> Dict:
        """Get comprehensive node information"""
        return {
            'name': node.name,
            'type': node.type,
            'properties': node.properties,
            'practical_info': {
                'opening_hours': node.properties.get('opening_hours'),
                'entrance_fee': node.properties.get('entrance_fee'),
                'visit_duration': node.properties.get('visit_duration'),
                'best_visit_time': node.properties.get('best_visit_time')
            }
        }
    
    def _get_connected_nodes(self, node_id: str, connection_type: str) -> List[Dict]:
        """Get connected nodes of specific type"""
        connected_ids = self.connections[node_id].get(connection_type, [])
        connected_nodes = []
        
        for connected_id in connected_ids:
            if connected_id in self.nodes:
                node = self.nodes[connected_id]
                connected_nodes.append({
                    'id': connected_id,
                    'name': node.name,
                    'type': node.type,
                    'key_info': self._get_key_info_for_type(node)
                })
        
        return connected_nodes
    
    def _get_key_info_for_type(self, node: KnowledgeNode) -> Dict:
        """Get key information based on node type"""
        if node.type == 'restaurant':
            return {
                'cuisine': node.properties.get('cuisine'),
                'price_range': node.properties.get('price_range'),
                'specialty': node.properties.get('specialty')
            }
        elif node.type == 'attraction':
            return {
                'category': node.properties.get('category'),
                'visit_duration': node.properties.get('visit_duration'),
                'entrance_fee': node.properties.get('entrance_fee')
            }
        elif node.type == 'transport':
            return {
                'transport_type': node.properties.get('transport_type'),
                'line': node.properties.get('line')
            }
        else:
            return {}
    
    def _get_optimal_visiting_sequence(self, primary_node: str) -> List[Dict]:
        """Get optimal sequence for visiting related attractions"""
        sequence_ids = self.connections[primary_node].get('optimal_sequence', [])
        sequence = []
        
        for seq_id in sequence_ids:
            if seq_id in self.nodes:
                node = self.nodes[seq_id]
                sequence.append({
                    'id': seq_id,
                    'name': node.name,
                    'visit_duration': node.properties.get('visit_duration'),
                    'walking_time': self._estimate_walking_time(primary_node, seq_id)
                })
        
        return sequence
    
    def _estimate_walking_time(self, from_node: str, to_node: str) -> str:
        """Estimate walking time between nodes"""
        # Simplified distance estimation based on districts
        same_district_pairs = [
            ['hagia_sophia', 'blue_mosque', 'topkapi_palace'],
            ['galata_tower', 'galata_bridge']
        ]
        
        for district_group in same_district_pairs:
            if from_node in district_group and to_node in district_group:
                return "5-10 minutes"
        
        return "15-25 minutes"
    
    def _get_contextual_tips(self, node_id: str, user_context: Dict = None) -> List[str]:
        """Get contextual tips based on user context and time"""
        tips = []
        node = self.nodes.get(node_id)
        
        if not node:
            return tips
        
        # Time-based tips
        current_hour = datetime.now().hour
        if current_hour < 10:
            tips.append("💡 Visit early to avoid crowds")
        elif current_hour > 16:
            tips.append("💡 Consider sunset timing for best photography")
        
        # User context tips
        if user_context:
            if user_context.get('group_type') == 'family':
                if node.properties.get('accessibility') == 'wheelchair_accessible':
                    tips.append("👨‍👩‍👧‍👦 Family-friendly with good accessibility")
            
            if user_context.get('interests', []):
                if 'photography' in user_context['interests']:
                    if node.properties.get('photography'):
                        tips.append("📸 Excellent photography opportunities")
        
        # Node-specific tips
        if node_id == 'hagia_sophia':
            tips.append("🎧 Audio guide highly recommended for historical context")
        elif node_id == 'blue_mosque':
            tips.append("👕 Dress modestly - shoulders and knees covered")
        elif node_id == 'grand_bazaar':
            tips.append("💰 Haggling is expected and part of the experience")
        
        return tips
    
    def track_user_journey(self, user_id: str, query: str, identified_nodes: List[str],
                          session_id: str = None):
        """Track user journey for behavioral pattern recognition"""
        if not session_id:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Get or create user journey
        if session_id not in self.user_journeys:
            self.user_journeys[session_id] = UserJourney(
                user_id=user_id,
                session_id=session_id
            )
        
        journey = self.user_journeys[session_id]
        
        # Add current query
        journey.queries.append({
            'query': query,
            'nodes': identified_nodes,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update visited nodes
        journey.visited_nodes.update(identified_nodes)
        
        # Detect patterns
        self._detect_journey_patterns(journey)
        
        logger.info(f"📊 Tracked journey for user {user_id}: {len(journey.queries)} queries")
    
    def _detect_journey_patterns(self, journey: UserJourney):
        """Detect patterns in user journey"""
        if len(journey.queries) < 2:
            return
        
        # Get last two queries for pattern detection
        prev_nodes = set(journey.queries[-2]['nodes'])
        curr_nodes = set(journey.queries[-1]['nodes'])
        
        # Create pattern signature
        for prev_node in prev_nodes:
            for curr_node in curr_nodes:
                pattern = f"{prev_node} -> {curr_node}"
                self.common_patterns[pattern] += 1
                
                # Update prediction rules if pattern is common
                if self.common_patterns[pattern] >= 3:
                    if curr_node not in self.prediction_rules[prev_node]:
                        self.prediction_rules[prev_node].append(curr_node)
                        logger.info(f"📈 New pattern detected: {pattern}")
    
    def _predict_next_interests(self, current_node: str, user_context: Dict = None) -> List[Dict]:
        """Predict what user might be interested in next"""
        predictions = []
        
        # Rule-based predictions from behavioral patterns
        if current_node in self.prediction_rules:
            for predicted_node in self.prediction_rules[current_node]:
                if predicted_node in self.nodes:
                    confidence = min(0.9, self.common_patterns.get(f"{current_node} -> {predicted_node}", 0) / 10)
                    predictions.append({
                        'node_id': predicted_node,
                        'name': self.nodes[predicted_node].name,
                        'confidence': confidence,
                        'reason': 'behavioral_pattern'
                    })
        
        # Knowledge graph based predictions
        graph_predictions = self.connections[current_node].get('combine_visit', [])
        for pred_node in graph_predictions:
            if pred_node in self.nodes and pred_node not in [p['node_id'] for p in predictions]:
                predictions.append({
                    'node_id': pred_node,
                    'name': self.nodes[pred_node].name,
                    'confidence': 0.7,
                    'reason': 'knowledge_graph'
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        if predictions:
            self.usage_stats['pattern_predictions'] += 1
        
        return predictions[:5]  # Return top 5 predictions
    
    def get_behavioral_insights(self) -> Dict[str, Any]:
        """Get behavioral pattern insights"""
        top_patterns = self.common_patterns.most_common(10)
        
        return {
            'total_patterns': len(self.common_patterns),
            'total_journeys': len(self.user_journeys),
            'top_patterns': [
                {'pattern': pattern, 'frequency': count}
                for pattern, count in top_patterns
            ],
            'prediction_rules': dict(self.prediction_rules),
            'usage_stats': self.usage_stats
        }
    
    def _load_behavioral_patterns(self):
        """Load saved behavioral patterns"""
        patterns_file = self.data_dir / "behavioral_patterns.json"
        
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.common_patterns.update(data.get('common_patterns', {}))
                    self.prediction_rules.update(data.get('prediction_rules', {}))
                    logger.info(f"📂 Loaded {len(self.common_patterns)} behavioral patterns")
            except Exception as e:
                logger.warning(f"⚠️ Could not load behavioral patterns: {e}")
    
    def save_behavioral_patterns(self):
        """Save behavioral patterns to disk"""
        patterns_file = self.data_dir / "behavioral_patterns.json"
        
        try:
            data = {
                'common_patterns': dict(self.common_patterns),
                'prediction_rules': dict(self.prediction_rules),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"💾 Saved {len(self.common_patterns)} behavioral patterns")
            
        except Exception as e:
            logger.error(f"❌ Error saving behavioral patterns: {e}")

    def find_related_nodes(self, query: str) -> List[str]:
        """Find nodes related to a query"""
        query_lower = query.lower()
        related_nodes = []
        
        # Search node names and properties
        for node_id, node in self.nodes.items():
            if node.name.lower() in query_lower:
                related_nodes.append(node_id)
                continue
            
            # Search in properties
            for prop_value in node.properties.values():
                if isinstance(prop_value, str) and prop_value.lower() in query_lower:
                    related_nodes.append(node_id)
                    break
        
        return related_nodes

# Integration example for existing systems
def enhance_route_planner_with_knowledge_graph():
    """Example of how to integrate knowledge graph with route planner"""
    knowledge_graph = IstanbulKnowledgeGraph()
    
    def enhanced_route_planning(origin: str, destination: str, user_context: Dict = None):
        """Enhanced route planning with knowledge graph enrichment"""
        
        # Get enriched information about destination
        enrichment = knowledge_graph.get_enriched_response(
            f"route to {destination}", destination, user_context
        )
        
        if enrichment['enriched']:
            # Add nearby attractions and dining to route suggestions
            suggestions = {
                'primary_destination': enrichment['data']['primary_info'],
                'nearby_stops': enrichment['data']['nearby_attractions'],
                'dining_options': enrichment['data']['dining_options'],
                'optimal_sequence': enrichment['data']['optimal_sequence'],
                'tips': enrichment['data']['contextual_tips']
            }
            
            return suggestions
        
        return {'route': f"Basic route from {origin} to {destination}"}

if __name__ == "__main__":
    # Test the knowledge graph
    kg = IstanbulKnowledgeGraph()
    
    # Test enriched response
    result = kg.get_enriched_response(
        "What can I visit near Hagia Sophia?", 
        "hagia_sophia",
        {'interests': ['photography'], 'group_type': 'couple'}
    )
    
    print("📊 Knowledge Graph Test Results:")
    print(f"Enriched: {result['enriched']}")
    if result['enriched']:
        print(f"Nearby attractions: {len(result['data']['nearby_attractions'])}")
        print(f"Dining options: {len(result['data']['dining_options'])}")
        print(f"Tips: {result['data']['contextual_tips']}")
    
    # Test behavioral tracking
    kg.track_user_journey("test_user", "Visit Hagia Sophia", ["hagia_sophia"])
    kg.track_user_journey("test_user", "Blue Mosque info", ["blue_mosque"])
    
    insights = kg.get_behavioral_insights()
    print(f"\n🧠 Behavioral Insights:")
    print(f"Total patterns: {insights['total_patterns']}")
    print(f"Usage stats: {insights['usage_stats']}")
