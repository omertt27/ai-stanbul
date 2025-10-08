#!/usr/bin/env python3
"""
POI Graph System with NetworkX
==============================

Point of Interest graph system for Istanbul using NetworkX.
Creates relationships between attractions, restaurants, transport, and districts.
"""

import networkx as nx
import json
import os
from typing import Dict, List, Tuple, Optional, Set
import math
from dataclasses import dataclass
from enum import Enum


class POIType(Enum):
    RESTAURANT = "restaurant"
    ATTRACTION = "attraction" 
    TRANSPORT = "transport"
    DISTRICT = "district"
    HOTEL = "hotel"
    SHOPPING = "shopping"


@dataclass
class POI:
    """Point of Interest data structure"""
    id: str
    name: str
    poi_type: POIType
    district: str
    coordinates: Tuple[float, float]  # (lat, lng)
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class POIGraphSystem:
    """
    Point of Interest graph system for city-level reasoning
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.poi_index = {}  # id -> POI object
        self.district_pois = {}  # district -> list of POI ids
        self.load_istanbul_data()
    
    def add_poi(self, poi: POI):
        """Add POI to the graph"""
        # Add node to graph
        self.graph.add_node(poi.id, 
                           name=poi.name,
                           type=poi.poi_type.value,
                           district=poi.district,
                           lat=poi.coordinates[0],
                           lng=poi.coordinates[1],
                           **poi.properties)
        
        # Index POI
        self.poi_index[poi.id] = poi
        
        # Index by district
        if poi.district not in self.district_pois:
            self.district_pois[poi.district] = []
        self.district_pois[poi.district].append(poi.id)
    
    def add_relationship(self, poi1_id: str, poi2_id: str, relationship_type: str, weight: float = 1.0):
        """Add relationship between two POIs"""
        self.graph.add_edge(poi1_id, poi2_id, 
                           relationship=relationship_type, 
                           weight=weight)
    
    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates (haversine formula)"""
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        
        # Convert to radians
        lat1_r = math.radians(lat1)
        lng1_r = math.radians(lng1)
        lat2_r = math.radians(lat2)
        lng2_r = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2_r - lat1_r
        dlng = lng2_r - lng1_r
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def load_istanbul_data(self):
        """Load Istanbul POI data and create graph"""
        # Load restaurants from existing data
        self.load_restaurants()
        
        # Load attractions
        self.load_attractions()
        
        # Load transport nodes
        self.load_transport_nodes()
        
        # Create proximity relationships
        self.create_proximity_relationships()
        
        print(f"✅ POI Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def load_restaurants(self):
        """Load restaurants from existing mock data"""
        try:
            restaurant_file = os.path.join(
                os.path.dirname(__file__), 
                '..', 'api_clients', 'google_places.py'
            )
            
            # Read restaurant data (simplified extraction)
            restaurants = [
                # Sultanahmet
                POI("rest_pandeli", "Pandeli", POIType.RESTAURANT, "Sultanahmet", 
                    (41.0122, 28.9738), {"cuisine": "Ottoman", "price": "expensive", "rating": 4.5}),
                POI("rest_deraliye", "Deraliye Ottoman Palace Cuisine", POIType.RESTAURANT, "Sultanahmet",
                    (41.0086, 28.9802), {"cuisine": "Ottoman", "price": "expensive", "rating": 4.3}),
                POI("rest_hamdi", "Hamdi Restaurant", POIType.RESTAURANT, "Sultanahmet",
                    (41.0172, 28.9738), {"cuisine": "Turkish", "price": "moderate", "rating": 4.4}),
                
                # Beyoğlu  
                POI("rest_mikla", "Mikla", POIType.RESTAURANT, "Beyoğlu",
                    (41.0369, 28.9744), {"cuisine": "Modern Turkish", "price": "expensive", "rating": 4.6}),
                POI("rest_360", "360 Istanbul", POIType.RESTAURANT, "Beyoğlu", 
                    (41.0292, 28.9744), {"cuisine": "International", "price": "expensive", "rating": 4.2}),
                POI("rest_kasibeyaz", "Kası Beyaz", POIType.RESTAURANT, "Beyoğlu",
                    (41.0311, 28.9786), {"cuisine": "Turkish", "price": "moderate", "rating": 4.3}),
                
                # Kadıköy
                POI("rest_ciya", "Çiya Sofrası", POIType.RESTAURANT, "Kadıköy",
                    (40.9939, 29.0253), {"cuisine": "Anatolian", "price": "moderate", "rating": 4.5}),
                POI("rest_kanaat", "Kanaat Lokantası", POIType.RESTAURANT, "Kadıköy",
                    (40.9886, 29.0258), {"cuisine": "Turkish", "price": "budget", "rating": 4.4}),
                
                # Beşiktaş
                POI("rest_tugra", "Tuğra Restaurant", POIType.RESTAURANT, "Beşiktaş",
                    (41.0427, 28.9833), {"cuisine": "Ottoman", "price": "expensive", "rating": 4.4}),
                POI("rest_feriye", "Feriye Palace", POIType.RESTAURANT, "Beşiktaş",
                    (41.0444, 28.9889), {"cuisine": "Turkish", "price": "expensive", "rating": 4.3}),
            ]
            
            for restaurant in restaurants:
                self.add_poi(restaurant)
                
        except Exception as e:
            print(f"⚠️ Could not load restaurant data: {e}")
    
    def load_attractions(self):
        """Load major Istanbul attractions"""
        attractions = [
            # Sultanahmet
            POI("attr_hagia_sophia", "Hagia Sophia", POIType.ATTRACTION, "Sultanahmet",
                (41.0086, 28.9802), {"category": "historic", "rating": 9.5, "visit_duration": 120}),
            POI("attr_blue_mosque", "Blue Mosque", POIType.ATTRACTION, "Sultanahmet", 
                (41.0054, 28.9768), {"category": "historic", "rating": 9.2, "visit_duration": 90}),
            POI("attr_topkapi", "Topkapi Palace", POIType.ATTRACTION, "Sultanahmet",
                (41.0115, 28.9833), {"category": "historic", "rating": 9.0, "visit_duration": 180}),
            POI("attr_grand_bazaar", "Grand Bazaar", POIType.ATTRACTION, "Sultanahmet",
                (41.0106, 28.9681), {"category": "shopping", "rating": 8.5, "visit_duration": 120}),
            
            # Beyoğlu
            POI("attr_galata_tower", "Galata Tower", POIType.ATTRACTION, "Beyoğlu",
                (41.0256, 28.9744), {"category": "historic", "rating": 8.8, "visit_duration": 60}),
            POI("attr_istiklal", "Istiklal Street", POIType.ATTRACTION, "Beyoğlu",
                (41.0369, 28.9744), {"category": "cultural", "rating": 8.2, "visit_duration": 180}),
            
            # Beşiktaş  
            POI("attr_dolmabahce", "Dolmabahçe Palace", POIType.ATTRACTION, "Beşiktaş",
                (41.0391, 28.9967), {"category": "historic", "rating": 9.1, "visit_duration": 150}),
            
            # Üsküdar
            POI("attr_maiden_tower", "Maiden's Tower", POIType.ATTRACTION, "Üsküdar", 
                (41.0211, 29.0044), {"category": "historic", "rating": 8.6, "visit_duration": 90}),
        ]
        
        for attraction in attractions:
            self.add_poi(attraction)
    
    def load_transport_nodes(self):
        """Load major transport nodes"""
        transport_nodes = [
            POI("trans_eminonu", "Eminönü Transport Hub", POIType.TRANSPORT, "Sultanahmet",
                (41.0172, 28.9738), {"types": ["ferry", "bus", "tram"], "connections": 15}),
            POI("trans_karakoy", "Karaköy Transport Hub", POIType.TRANSPORT, "Beyoğlu", 
                (41.0256, 28.9744), {"types": ["ferry", "metro", "bus"], "connections": 12}),
            POI("trans_besiktas", "Beşiktaş Ferry Terminal", POIType.TRANSPORT, "Beşiktaş",
                (41.0427, 28.9944), {"types": ["ferry", "metrobus"], "connections": 8}),
            POI("trans_kadikoy", "Kadıköy Ferry Terminal", POIType.TRANSPORT, "Kadıköy",
                (40.9939, 29.0253), {"types": ["ferry", "metro", "bus"], "connections": 10}),
        ]
        
        for transport in transport_nodes:
            self.add_poi(transport)
    
    def create_proximity_relationships(self):
        """Create relationships based on proximity and logical connections"""
        # Connect POIs within walking distance (< 0.5km)
        pois = list(self.poi_index.values())
        
        for i, poi1 in enumerate(pois):
            for poi2 in pois[i+1:]:
                distance = self.calculate_distance(poi1.coordinates, poi2.coordinates)
                
                if distance <= 0.5:  # Within 500m
                    relationship_type = "walking_distance"
                    weight = 1.0 / (distance + 0.1)  # Closer = higher weight
                    self.add_relationship(poi1.id, poi2.id, relationship_type, weight)
                
                elif distance <= 2.0:  # Within 2km
                    relationship_type = "nearby"
                    weight = 0.5 / (distance + 0.1)
                    self.add_relationship(poi1.id, poi2.id, relationship_type, weight)
        
        # Create special relationships
        self.create_thematic_relationships()
    
    def create_thematic_relationships(self):
        """Create relationships based on themes and complementary experiences"""
        # Historic sites in Sultanahmet
        historic_sultanahmet = ["attr_hagia_sophia", "attr_blue_mosque", "attr_topkapi", "attr_grand_bazaar"]
        for i, poi1 in enumerate(historic_sultanahmet):
            for poi2 in historic_sultanahmet[i+1:]:
                if self.graph.has_edge(poi1, poi2):
                    # Upgrade existing edge
                    self.graph[poi1][poi2]['relationship'] = 'historic_circuit'
                    self.graph[poi1][poi2]['weight'] *= 1.5
        
        # Fine dining with views
        view_restaurants = ["rest_mikla", "rest_360", "rest_tugra"]
        view_attractions = ["attr_galata_tower", "attr_maiden_tower"]
        
        for restaurant in view_restaurants:
            for attraction in view_attractions:
                if restaurant in self.poi_index and attraction in self.poi_index:
                    self.add_relationship(restaurant, attraction, "scenic_dining", 0.8)
    
    def find_nearby_pois(self, poi_id: str, poi_types: List[POIType] = None, max_distance: float = 1.0) -> List[Dict]:
        """Find nearby POIs of specified types"""
        if poi_id not in self.poi_index:
            return []
        
        base_poi = self.poi_index[poi_id]
        nearby = []
        
        for other_id, other_poi in self.poi_index.items():
            if other_id == poi_id:
                continue
            
            # Filter by type if specified
            if poi_types and other_poi.poi_type not in poi_types:
                continue
            
            distance = self.calculate_distance(base_poi.coordinates, other_poi.coordinates)
            
            if distance <= max_distance:
                nearby.append({
                    'poi': other_poi,
                    'distance': distance,
                    'walking_time': int(distance * 12)  # ~12 minutes per km
                })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        return nearby
    
    def find_poi_circuit(self, start_poi_id: str, poi_type: POIType, max_stops: int = 4) -> List[Dict]:
        """Find a circuit of POIs starting from a given point"""
        if start_poi_id not in self.poi_index:
            return []
        
        # Use NetworkX to find connected POIs of the same type
        circuit = []
        visited = {start_poi_id}
        current = start_poi_id
        
        for _ in range(max_stops - 1):
            # Find neighbors of specified type
            neighbors = []
            
            if current in self.graph:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        neighbor_poi = self.poi_index.get(neighbor)
                        if neighbor_poi and neighbor_poi.poi_type == poi_type:
                            edge_data = self.graph[current][neighbor]
                            neighbors.append({
                                'id': neighbor,
                                'poi': neighbor_poi,
                                'weight': edge_data.get('weight', 0.1),
                                'relationship': edge_data.get('relationship', 'nearby')
                            })
            
            if not neighbors:
                break
            
            # Choose best neighbor (highest weight)
            neighbors.sort(key=lambda x: x['weight'], reverse=True)
            best_neighbor = neighbors[0]
            
            circuit.append(best_neighbor)
            visited.add(best_neighbor['id'])
            current = best_neighbor['id']
        
        return circuit
    
    def get_poi_recommendations(self, query_location: str, preferences: Dict) -> List[Dict]:
        """Get POI recommendations based on location and preferences"""
        recommendations = []
        
        # Find POIs in specified district/location
        target_pois = []
        
        # If query_location is a district
        if query_location.lower() in self.district_pois:
            poi_ids = self.district_pois[query_location.lower()]
            target_pois = [self.poi_index[poi_id] for poi_id in poi_ids]
        
        # If query_location is a specific POI
        elif query_location in self.poi_index:
            base_poi = self.poi_index[query_location]
            nearby = self.find_nearby_pois(query_location, max_distance=2.0)
            target_pois = [item['poi'] for item in nearby]
        
        # Filter and score based on preferences
        for poi in target_pois:
            score = 0.0
            match_reasons = []
            
            # Type preference
            if preferences.get('poi_type') == poi.poi_type.value:
                score += 2.0
                match_reasons.append(f"matches {poi.poi_type.value} preference")
            
            # Category preference (for attractions)
            if poi.poi_type == POIType.ATTRACTION:
                pref_category = preferences.get('category')
                poi_category = poi.properties.get('category')
                if pref_category and poi_category == pref_category:
                    score += 1.5
                    match_reasons.append(f"matches {pref_category} category")
            
            # Rating boost
            rating = poi.properties.get('rating', 0)
            if rating >= 4.0:
                score += rating * 0.3
                match_reasons.append(f"high rating ({rating})")
            
            # Budget preference (for restaurants)
            if poi.poi_type == POIType.RESTAURANT:
                pref_budget = preferences.get('budget')
                poi_price = poi.properties.get('price')
                if pref_budget and poi_price:
                    if (pref_budget == 'budget' and poi_price in ['budget', 'moderate']) or \
                       (pref_budget == 'moderate' and poi_price in ['budget', 'moderate', 'expensive']) or \
                       (pref_budget == 'expensive'):
                        score += 1.0
                        match_reasons.append(f"fits {pref_budget} budget")
            
            if score > 0:
                recommendations.append({
                    'poi': poi,
                    'score': score,
                    'reasons': match_reasons
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]  # Top 5
    
    def generate_poi_context(self, poi_id: str) -> Dict:
        """Generate context information for a POI"""
        if poi_id not in self.poi_index:
            return {}
        
        poi = self.poi_index[poi_id]
        
        # Find nearby POIs by type
        nearby_restaurants = self.find_nearby_pois(poi_id, [POIType.RESTAURANT], 0.5)
        nearby_attractions = self.find_nearby_pois(poi_id, [POIType.ATTRACTION], 0.8)
        nearby_transport = self.find_nearby_pois(poi_id, [POIType.TRANSPORT], 1.0)
        
        context = {
            'poi': poi,
            'nearby_restaurants': nearby_restaurants[:3],
            'nearby_attractions': nearby_attractions[:3], 
            'nearby_transport': nearby_transport[:2],
            'district_info': {
                'name': poi.district,
                'poi_count': len(self.district_pois.get(poi.district.lower(), []))
            }
        }
        
        # Add circuit recommendations if it's an attraction
        if poi.poi_type == POIType.ATTRACTION:
            circuit = self.find_poi_circuit(poi_id, POIType.ATTRACTION, 3)
            context['suggested_circuit'] = circuit
        
        return context


# Global POI graph instance
poi_graph = POIGraphSystem()


def get_poi_intelligence(query_location: str, preferences: Dict = None) -> Dict:
    """
    Get comprehensive POI intelligence for AI responses
    """
    if preferences is None:
        preferences = {}
    
    recommendations = poi_graph.get_poi_recommendations(query_location, preferences)
    
    intelligence = {
        'recommendations': recommendations,
        'total_pois': poi_graph.graph.number_of_nodes(),
        'location_context': query_location
    }
    
    # Add specific POI context if querying a specific location
    if query_location in poi_graph.poi_index:
        intelligence['poi_context'] = poi_graph.generate_poi_context(query_location)
    
    return intelligence


if __name__ == "__main__":
    # Test the POI graph system
    print("🗺️ POI Graph System Test")
    print("=" * 30)
    
    # Test basic functionality
    print(f"\n1. Graph Statistics:")
    print(f"   Nodes: {poi_graph.graph.number_of_nodes()}")
    print(f"   Edges: {poi_graph.graph.number_of_edges()}")
    print(f"   Districts: {len(poi_graph.district_pois)}")
    
    print(f"\n2. Nearby POIs Test:")
    nearby = poi_graph.find_nearby_pois("attr_hagia_sophia", [POIType.RESTAURANT], 0.8)
    print(f"   Restaurants near Hagia Sophia: {len(nearby)}")
    for item in nearby[:3]:
        print(f"   • {item['poi'].name} ({item['distance']:.2f}km)")
    
    print(f"\n3. POI Recommendations Test:")
    prefs = {'poi_type': 'restaurant', 'budget': 'moderate'}
    recs = poi_graph.get_poi_recommendations("Sultanahmet", prefs)
    print(f"   Restaurant recommendations for Sultanahmet: {len(recs)}")
    for rec in recs[:3]:
        print(f"   • {rec['poi'].name} (score: {rec['score']:.1f})")
    
    print(f"\n4. Circuit Test:")
    circuit = poi_graph.find_poi_circuit("attr_hagia_sophia", POIType.ATTRACTION, 3)
    print(f"   Historic circuit from Hagia Sophia: {len(circuit)} stops")
    for stop in circuit:
        print(f"   • {stop['poi'].name} ({stop['relationship']})")
    
    print(f"\n✅ POI Graph System - OPERATIONAL")
