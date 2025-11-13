"""
Script to integrate Istanbul district ontology into the RAG knowledge base.
Expands the knowledge base with detailed district information from the YAML ontology.
"""

import json
import yaml
from pathlib import Path

def load_ontology():
    """Load the district ontology YAML file."""
    ontology_path = Path(__file__).parent.parent / "data" / "istanbul_district_ontology.yaml"
    with open(ontology_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_knowledge_base():
    """Load the existing knowledge base."""
    kb_path = Path(__file__).parent.parent / "data" / "istanbul_knowledge_base.json"
    with open(kb_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_district_entries(ontology):
    """Convert district ontology to knowledge base format."""
    districts = []
    
    for district_id, district_data in ontology['districts'].items():
        # Create a comprehensive district entry
        district_entry = {
            "id": f"district_{district_id}",
            "name": district_data['name'],
            "name_tr": district_data['name_tr'],
            "type": district_data['type'],
            "characteristics": district_data['characteristics'],
            "description": generate_district_description(district_data),
            
            # Category scores (1-10 scale)
            "culture_score": district_data['categories']['culture'],
            "nightlife_score": district_data['categories']['nightlife'],
            "sea_access_score": district_data['categories']['sea_access'],
            "quiet_score": district_data['categories']['quiet'],
            "authentic_score": district_data['categories']['authentic'],
            "luxury_score": district_data['categories']['luxury'],
            
            # Points of interest
            "landmarks": district_data['landmarks'],
            
            # Transport information
            "transport": district_data['transport'],
            
            # Practical information
            "restaurant_density": district_data['restaurant_density'],
            "price_level": district_data['price_level'],
            "best_time": district_data['best_time'],
            "tourist_level": district_data['tourist_level'],
            
            # Searchable text for RAG
            "searchable_text": create_searchable_text(district_data)
        }
        
        districts.append(district_entry)
    
    return districts

def generate_district_description(district_data):
    """Generate a natural language description of the district."""
    name = district_data['name']
    type_desc = district_data['type'].replace('_', ' ')
    chars = ", ".join(district_data['characteristics'])
    
    # Build description
    desc = f"{name} is a {type_desc} district characterized by its {chars} nature. "
    
    # Add landmarks
    if district_data['landmarks']:
        landmarks_str = ", ".join(district_data['landmarks'][:3])
        desc += f"Notable landmarks include {landmarks_str}. "
    
    # Add practical info
    desc += f"Restaurant density is {district_data['restaurant_density']}, "
    desc += f"price level is {district_data['price_level']}, "
    desc += f"and tourist level is {district_data['tourist_level']}. "
    desc += f"Best time to visit is {district_data['best_time']}."
    
    return desc

def create_searchable_text(district_data):
    """Create comprehensive searchable text for semantic search."""
    parts = [
        district_data['name'],
        district_data['name_tr'],
        district_data['type'].replace('_', ' '),
        " ".join(district_data['characteristics']),
        " ".join(district_data['landmarks']),
        district_data['restaurant_density'],
        district_data['price_level'],
        district_data['best_time'],
        district_data['tourist_level']
    ]
    
    # Add transport info
    for transport_type, lines in district_data['transport'].items():
        parts.extend(lines)
    
    return " ".join(parts)

def create_district_relationships(ontology):
    """Create relationship entries for district connections."""
    relationships = []
    
    # Adjacent districts
    adjacent = ontology['relationships']['adjacent']
    for district, neighbors in adjacent.items():
        if neighbors:
            relationships.append({
                "id": f"rel_adjacent_{district}",
                "type": "adjacent",
                "district": district,
                "related_districts": neighbors,
                "description": f"{district} is adjacent to {', '.join(neighbors)}"
            })
    
    # Similar vibe
    similar = ontology['relationships']['similar_vibe']
    for vibe, districts in similar.items():
        relationships.append({
            "id": f"rel_vibe_{vibe}",
            "type": "similar_vibe",
            "vibe": vibe,
            "districts": districts,
            "description": f"Districts with {vibe} vibe: {', '.join(districts)}"
        })
    
    # Transport connections
    transport = ontology['relationships']['transport_connected']
    for connection_type, districts in transport.items():
        relationships.append({
            "id": f"rel_transport_{connection_type}",
            "type": "transport_connection",
            "connection": connection_type,
            "districts": districts,
            "description": f"{connection_type.replace('_', ' ').title()}: {', '.join(districts)}"
        })
    
    return relationships

def create_query_patterns(ontology):
    """Create query pattern entries for smart recommendations."""
    patterns = []
    
    for pattern_name, pattern_data in ontology['query_patterns'].items():
        patterns.append({
            "id": f"pattern_{pattern_name}",
            "query_type": pattern_name,
            "preferred_districts": pattern_data['preferred_districts'],
            "characteristics": pattern_data['characteristics'],
            "description": f"For {pattern_name} experiences, recommended districts are: {', '.join(pattern_data['preferred_districts'])}. Key characteristics: {', '.join(pattern_data['characteristics'])}."
        })
    
    return patterns

def integrate_data():
    """Main integration function."""
    print("Loading ontology and knowledge base...")
    ontology = load_ontology()
    kb = load_knowledge_base()
    
    print("Creating district entries...")
    districts = create_district_entries(ontology)
    print(f"Created {len(districts)} district entries")
    
    print("Creating relationship entries...")
    relationships = create_district_relationships(ontology)
    print(f"Created {len(relationships)} relationship entries")
    
    print("Creating query pattern entries...")
    patterns = create_query_patterns(ontology)
    print(f"Created {len(patterns)} query pattern entries")
    
    # Add to knowledge base
    kb['districts'] = districts
    kb['district_relationships'] = relationships
    kb['query_patterns'] = patterns
    
    # Save updated knowledge base
    kb_path = Path(__file__).parent.parent / "data" / "istanbul_knowledge_base.json"
    print(f"\nSaving updated knowledge base to {kb_path}...")
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Integration complete!")
    print(f"\nKnowledge base now contains:")
    for category, items in kb.items():
        if isinstance(items, list):
            print(f"  - {category}: {len(items)} items")
    
    return kb

if __name__ == "__main__":
    integrate_data()
