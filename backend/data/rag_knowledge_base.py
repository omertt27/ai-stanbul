"""
RAG Knowledge Base for AI-stanbul
Consolidates all Istanbul information for vector search

Priority: HIGH - Core Architecture Enhancement
Timeline: 1-2 days
Cost: $0 (uses existing infrastructure)
"""

from typing import List, Dict, Any
import json

# Import existing data
from backend.data.hidden_gems_database import HIDDEN_GEMS_DATABASE


class KnowledgeDocument:
    """Single document in knowledge base"""
    def __init__(
        self,
        id: str,
        title: str,
        content: str,
        category: str,
        metadata: Dict[str, Any]
    ):
        self.id = id
        self.title = title
        self.content = content
        self.category = category
        self.metadata = metadata
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'metadata': self.metadata
        }


def build_knowledge_base() -> List[KnowledgeDocument]:
    """
    Build comprehensive knowledge base from all data sources
    
    Returns:
        List of KnowledgeDocument objects
    """
    documents = []
    
    # ========== 1. HIDDEN GEMS ==========
    for district, gems in HIDDEN_GEMS_DATABASE.items():
        for gem in gems:
            doc_id = f"gem_{district}_{gem['name'].lower().replace(' ', '_')}"
            
            # Create rich content for embedding
            content = f"""
Location: {gem['name']} in {district.title()} district
Type: {gem['type']}
Description: {gem['description']}
Best time: {gem.get('best_time', 'Anytime')}
Hidden factor: {gem.get('hidden_factor', 0)}/10
Local tip: {gem.get('local_tip', '')}
Why it's special: {gem.get('why_special', 'Authentic local experience')}
""".strip()
            
            documents.append(KnowledgeDocument(
                id=doc_id,
                title=f"{gem['name']} - Hidden Gem",
                content=content,
                category='hidden_gem',
                metadata={
                    'district': district,
                    'type': gem['type'],
                    'coordinates': gem.get('coordinates', {}),
                    'hidden_factor': gem.get('hidden_factor', 5)
                }
            ))
    
    # ========== 2. TRANSPORTATION ROUTES ==========
    transport_info = [
        {
            'id': 'route_m2',
            'title': 'M2 Metro Line',
            'content': """
M2 Metro Line - Main European Side Metro
Route: Hacıosman → Yenikapı (23 stations)
Key stations: Taksim, Şişhane, Vezneciler, Yenikapı
Connects to: M1, T1, Marmaray at Yenikapı
Frequency: 3-7 minutes
Hours: 06:00-00:30
Best for: European side travel, connecting to Asian side via Marmaray
""",
            'category': 'transportation',
            'metadata': {'line': 'M2', 'type': 'metro', 'side': 'european'}
        },
        {
            'id': 'route_marmaray',
            'title': 'Marmaray Undersea Rail',
            'content': """
Marmaray - Undersea Rail Connecting Europe & Asia
Route: Gebze (Asia) → Halkalı (Europe) through Bosphorus tunnel
Key stations: Ayrılık Çeşmesi (Kadıköy side), Yenikapı (European side)
Journey time: 5 minutes underwater crossing
Connects to: M2 at Yenikapı, all Asian side metros
Frequency: 5-10 minutes
Best for: Fastest Europe↔Asia connection, avoiding traffic
Weather-proof: Always good option in rain/snow
""",
            'category': 'transportation',
            'metadata': {'line': 'Marmaray', 'type': 'train', 'side': 'both'}
        },
        {
            'id': 'route_ferry',
            'title': 'Bosphorus Ferries',
            'content': """
Bosphorus Ferries - Scenic Europe↔Asia Crossing
Main routes: 
- Eminönü ↔ Kadıköy (20 min)
- Karaköy ↔ Kadıköy (15 min)
- Beşiktaş ↔ Üsküdar (15 min)
Frequency: 15-30 minutes
Best for: Scenic route, good weather, photography
Not ideal: Rainy/cold weather, rush hour crowds
Bonus: Amazing Bosphorus views, seagull feeding
""",
            'category': 'transportation',
            'metadata': {'type': 'ferry', 'side': 'both', 'scenic': True}
        }
    ]
    
    for info in transport_info:
        documents.append(KnowledgeDocument(
            id=info['id'],
            title=info['title'],
            content=info['content'],
            category=info['category'],
            metadata=info['metadata']
        ))
    
    # ========== 3. DISTRICTS & AREAS ==========
    districts = [
        {
            'id': 'district_kadikoy',
            'title': 'Kadıköy District',
            'content': """
Kadıköy - Vibrant Asian Side Hub
Vibe: Young, artistic, bohemian atmosphere
Famous for: Street food, bars, cafes, vintage shopping, street art
Best areas: Moda (seaside), Yeldeğirmeni (art district), Çarşı (market)
Transport: Marmaray, ferries, buses, M4 metro
Weather: Coastal breeze, great for walking
Locals' favorite: Less touristy than European side
Must-visit: Tuesday market, fish market, Moda pier
""",
            'category': 'district',
            'metadata': {'side': 'asian', 'vibe': 'bohemian', 'tourist_density': 'medium'}
        },
        {
            'id': 'district_beyoglu',
            'title': 'Beyoğlu District',
            'content': """
Beyoğlu - Cultural Heart of Istanbul
Vibe: Historic, artistic, nightlife
Famous for: İstiklal Street, Galata Tower, Taksim Square
Best areas: Cihangir (hipster), Çukurcuma (antiques), Asmalımescit (bars)
Transport: M2 metro, T1 tram, funicular
Highlights: Museums, galleries, rooftop bars, vintage shops
Architecture: Ottoman-era buildings, European influence
Tourist density: Very high (but hidden spots exist)
""",
            'category': 'district',
            'metadata': {'side': 'european', 'vibe': 'cultural', 'tourist_density': 'high'}
        }
    ]
    
    for district in districts:
        documents.append(KnowledgeDocument(
            id=district['id'],
            title=district['title'],
            content=district['content'],
            category=district['category'],
            metadata=district['metadata']
        ))
    
    # ========== 4. PRACTICAL TIPS ==========
    tips = [
        {
            'id': 'tip_weather_rain',
            'title': 'Rainy Day Transportation',
            'content': """
Best Transport Options When Raining in Istanbul:
1. Marmaray - completely underground, never gets wet
2. M2 Metro - indoor stations, covered walkways
3. AVOID: Ferries (wet boarding, rough seas), long walks
4. Tip: Taksim to Kadıköy via M2→Marmaray stays 100% dry
Indoor activities: Museums, covered bazaars, shopping malls, cafes
""",
            'category': 'tip',
            'metadata': {'weather': 'rain', 'priority': 'high'}
        },
        {
            'id': 'tip_istanbulkart',
            'title': 'Istanbulkart Guide',
            'content': """
Istanbulkart - Istanbul's Transport Card
Where to buy: Airports, metro stations, kiosks, some shops
Cost: 70 TL card + add credit as needed
Usage: Tap on/off for metro, bus, tram, ferry (discounted rides)
Benefits: Cheaper than tokens, works on all transport
Transfers: Free/discounted within 2 hours
Refund: Available at main metro stations (keep receipt)
""",
            'category': 'tip',
            'metadata': {'topic': 'payment', 'priority': 'high'}
        }
    ]
    
    for tip in tips:
        documents.append(KnowledgeDocument(
            id=tip['id'],
            title=tip['title'],
            content=tip['content'],
            category=tip['category'],
            metadata=tip['metadata']
        ))
    
    print(f"✅ Built knowledge base: {len(documents)} documents")
    return documents


if __name__ == "__main__":
    # Test knowledge base building
    docs = build_knowledge_base()
    print(f"\nCategories:")
    categories = {}
    for doc in docs:
        categories[doc.category] = categories.get(doc.category, 0) + 1
    for cat, count in categories.items():
        print(f"  - {cat}: {count} documents")
