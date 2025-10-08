#!/usr/bin/env python3
"""
District Ontology System
========================

Formal district ontology system for Istanbul using YAML data structure.
Provides city-level reasoning and district intelligence for enhanced AI responses.
"""

import yaml
import os
from typing import Dict, List, Optional, Union
import math
from datetime import datetime


class DistrictOntologySystem:
    """
    Formal district ontology system for Istanbul intelligence
    """
    
    def __init__(self):
        self.ontology_data = None
        self.districts = {}
        self.load_ontology()
    
    def load_ontology(self):
        """Load district ontology from YAML file"""
        try:
            ontology_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 'data', 'istanbul_district_ontology.yaml'
            )
            
            with open(ontology_path, 'r', encoding='utf-8') as f:
                self.ontology_data = yaml.safe_load(f)
                self.districts = self.ontology_data.get('districts', {})
                
            print(f"✅ Loaded district ontology: {len(self.districts)} districts")
            
        except Exception as e:
            print(f"❌ Error loading district ontology: {e}")
            self.districts = {}
    
    def get_district_info(self, district_name: str) -> Optional[Dict]:
        """Get complete district information"""
        district_key = district_name.lower().replace(' ', '_').replace('ğ', 'g').replace('ş', 's')
        return self.districts.get(district_key)
    
    def get_district_characteristics(self, district_name: str) -> List[str]:
        """Get district characteristics"""
        district = self.get_district_info(district_name)
        if district:
            return district.get('characteristics', [])
        return []
    
    def get_district_category_score(self, district_name: str, category: str) -> int:
        """Get district score for specific category (1-10 scale)"""
        district = self.get_district_info(district_name)
        if district and 'categories' in district:
            return district['categories'].get(category, 0)
        return 0
    
    def find_districts_by_characteristic(self, characteristic: str) -> List[str]:
        """Find districts with specific characteristic"""
        matching_districts = []
        
        for district_key, district_data in self.districts.items():
            characteristics = district_data.get('characteristics', [])
            if characteristic.lower() in [c.lower() for c in characteristics]:
                matching_districts.append(district_data['name'])
        
        return matching_districts
    
    def find_districts_by_category_score(self, category: str, min_score: int = 7) -> List[Dict]:
        """Find districts with high scores in specific category"""
        matching_districts = []
        
        for district_key, district_data in self.districts.items():
            categories = district_data.get('categories', {})
            score = categories.get(category, 0)
            
            if score >= min_score:
                matching_districts.append({
                    'name': district_data['name'],
                    'score': score,
                    'characteristics': district_data.get('characteristics', [])
                })
        
        # Sort by score (highest first)
        matching_districts.sort(key=lambda x: x['score'], reverse=True)
        return matching_districts
    
    def get_best_districts_for_vibe(self, vibe: str) -> List[Dict]:
        """Get best districts for specific vibes/moods"""
        vibe_mappings = {
            'romantic': {'category': 'quiet', 'characteristics': ['romantic', 'scenic', 'upscale']},
            'nightlife': {'category': 'nightlife', 'characteristics': ['vibrant', 'trendy', 'nightlife']},
            'cultural': {'category': 'culture', 'characteristics': ['historic', 'cultural', 'traditional']},
            'authentic': {'category': 'authentic', 'characteristics': ['local', 'authentic', 'traditional']},
            'luxury': {'category': 'luxury', 'characteristics': ['upscale', 'luxury', 'premium']},
            'quiet': {'category': 'quiet', 'characteristics': ['quiet', 'peaceful', 'residential']},
            'sea': {'category': 'sea_access', 'characteristics': ['waterfront', 'coastal', 'bosphorus']},
            'trendy': {'category': 'nightlife', 'characteristics': ['trendy', 'modern', 'hip']}
        }
        
        mapping = vibe_mappings.get(vibe.lower())
        if not mapping:
            return []
        
        # Get districts by category score
        districts_by_score = self.find_districts_by_category_score(mapping['category'], min_score=6)
        
        # Filter by characteristics if available
        filtered_districts = []
        for district in districts_by_score:
            district_chars = [c.lower() for c in district['characteristics']]
            mapping_chars = [c.lower() for c in mapping['characteristics']]
            
            # Check if district has any of the required characteristics
            if any(char in district_chars for char in mapping_chars):
                filtered_districts.append(district)
        
        return filtered_districts[:3]  # Top 3 districts
    
    def get_transport_recommendations(self, district_name: str) -> Dict:
        """Get transport recommendations for district"""
        district = self.get_district_info(district_name)
        if district and 'transport' in district:
            return district['transport']
        return {}
    
    def get_landmarks(self, district_name: str) -> List[str]:
        """Get major landmarks in district"""
        district = self.get_district_info(district_name)
        if district and 'landmarks' in district:
            return district['landmarks']
        return []
    
    def calculate_district_similarity(self, district1: str, district2: str) -> float:
        """Calculate similarity between two districts based on characteristics"""
        d1_info = self.get_district_info(district1)
        d2_info = self.get_district_info(district2)
        
        if not d1_info or not d2_info:
            return 0.0
        
        # Compare characteristics
        chars1 = set(d1_info.get('characteristics', []))
        chars2 = set(d2_info.get('characteristics', []))
        
        if not chars1 or not chars2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_districts(self, district_name: str, top_n: int = 3) -> List[Dict]:
        """Find districts similar to given district"""
        similarities = []
        
        for district_key, district_data in self.districts.items():
            other_district = district_data['name']
            if other_district.lower() != district_name.lower():
                similarity = self.calculate_district_similarity(district_name, other_district)
                if similarity > 0:
                    similarities.append({
                        'name': other_district,
                        'similarity': similarity,
                        'characteristics': district_data.get('characteristics', [])
                    })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    
    def get_time_based_recommendations(self, district_name: str, time_context: str) -> Dict:
        """Get time-based recommendations for district"""
        district = self.get_district_info(district_name)
        if not district:
            return {}
        
        best_time = district.get('best_time', 'anytime')
        tourist_level = district.get('tourist_level', 'moderate')
        
        recommendations = {
            'best_time': best_time,
            'tourist_level': tourist_level,
            'current_advice': ''
        }
        
        # Time-based advice
        current_hour = datetime.now().hour
        
        if time_context in ['morning', 'early'] or current_hour < 10:
            if tourist_level == 'very_high':
                recommendations['current_advice'] = "Perfect timing! Morning visits are ideal to avoid crowds."
            else:
                recommendations['current_advice'] = "Great choice for a peaceful morning exploration."
                
        elif time_context in ['evening', 'night'] or current_hour > 18:
            nightlife_score = district.get('categories', {}).get('nightlife', 0)
            if nightlife_score >= 7:
                recommendations['current_advice'] = "Excellent for evening entertainment and nightlife!"
            else:
                recommendations['current_advice'] = "Consider evening dining and quiet walks."
        
        return recommendations
    
    def generate_district_context_summary(self, district_name: str) -> str:
        """Generate context summary for AI responses"""
        district = self.get_district_info(district_name)
        if not district:
            return ""
        
        characteristics = district.get('characteristics', [])
        landmarks = district.get('landmarks', [])
        
        # Build context summary
        summary_parts = []
        
        # Basic description
        char_text = ', '.join(characteristics[:3])  # Top 3 characteristics
        summary_parts.append(f"{district['name']} is known for being {char_text}")
        
        # Landmarks
        if landmarks:
            landmark_text = landmarks[0] if len(landmarks) == 1 else f"{landmarks[0]} and {len(landmarks)-1} other major attractions"
            summary_parts.append(f"featuring {landmark_text}")
        
        # Category strengths
        categories = district.get('categories', {})
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if top_categories:
            cat_descriptions = []
            for cat, score in top_categories:
                if score >= 8:
                    cat_descriptions.append(f"excellent {cat.replace('_', ' ')}")
                elif score >= 6:
                    cat_descriptions.append(f"good {cat.replace('_', ' ')}")
            
            if cat_descriptions:
                summary_parts.append(f"with {' and '.join(cat_descriptions)}")
        
        return ". ".join(summary_parts) + "."


# Global instance
district_ontology = DistrictOntologySystem()


def get_district_intelligence(district_name: str, query_context: str = '') -> Dict:
    """
    Get comprehensive district intelligence for AI responses
    """
    if not district_ontology.districts:
        return {}
    
    district_info = district_ontology.get_district_info(district_name)
    if not district_info:
        return {}
    
    intelligence = {
        'basic_info': district_info,
        'characteristics': district_ontology.get_district_characteristics(district_name),
        'landmarks': district_ontology.get_landmarks(district_name),
        'transport': district_ontology.get_transport_recommendations(district_name),
        'context_summary': district_ontology.generate_district_context_summary(district_name),
        'similar_districts': district_ontology.find_similar_districts(district_name)
    }
    
    # Add vibe-based recommendations if context suggests it
    vibe_keywords = ['romantic', 'nightlife', 'cultural', 'authentic', 'luxury', 'quiet', 'trendy']
    for vibe in vibe_keywords:
        if vibe in query_context.lower():
            intelligence['vibe_match'] = district_ontology.get_best_districts_for_vibe(vibe)
            break
    
    return intelligence


if __name__ == "__main__":
    # Test the district ontology system
    print("🏛️ District Ontology System Test")
    print("=" * 40)
    
    # Test basic functionality
    print("\n1. District Info Test:")
    info = district_ontology.get_district_info("Sultanahmet")
    if info:
        print(f"✅ Sultanahmet info loaded: {len(info)} properties")
        print(f"   Characteristics: {info.get('characteristics', [])}")
    
    print("\n2. Vibe-based District Search:")
    romantic_districts = district_ontology.get_best_districts_for_vibe('romantic')
    print(f"✅ Romantic districts: {[d['name'] for d in romantic_districts]}")
    
    nightlife_districts = district_ontology.get_best_districts_for_vibe('nightlife')
    print(f"✅ Nightlife districts: {[d['name'] for d in nightlife_districts]}")
    
    print("\n3. District Similarity Test:")
    similar = district_ontology.find_similar_districts("Sultanahmet")
    print(f"✅ Districts similar to Sultanahmet: {[d['name'] for d in similar]}")
    
    print("\n4. Intelligence Summary Test:")
    intelligence = get_district_intelligence("Beyoglu", "romantic dinner")
    print(f"✅ Beyoğlu intelligence generated: {len(intelligence)} components")
    print(f"   Context: {intelligence.get('context_summary', 'N/A')}")
    
    print("\n✅ District Ontology System - OPERATIONAL")
