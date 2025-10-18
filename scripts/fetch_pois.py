"""
One-Time POI Fetch Script for AI Istanbul
Fetches 500+ Istanbul POIs from Google Places
Run with: python fetch_pois.py
"""

import os
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from istanbul_zones import ISTANBUL_ZONES, POI_CATEGORIES

load_dotenv()

class POIFetcher:
    def __init__(self):
        self.google_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_PLACES_API_KEY not found in .env file!")
        
        self.all_pois = []
        self.request_count = 0
        self.max_requests = 180  # Reserve 20 for errors
        
    def fetch_google_places(self, zone, category):
        """Fetch POIs from Google Places API"""
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        lat, lng = zone['center']
        params = {
            'location': f'{lat},{lng}',
            'radius': zone['radius'],
            'type': category,
            'key': self.google_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            self.request_count += 1
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                print(f"   ✓ {zone['name']}/{category}: {len(results)} places")
                return results
            else:
                print(f"   ✗ {zone['name']}/{category}: Failed ({response.status_code})")
                return []
                
        except Exception as e:
            print(f"   ✗ {zone['name']}/{category}: Error - {str(e)}")
            return []
    
    def process_poi(self, place, zone_name):
        """Process and normalize POI data"""
        location = place.get('geometry', {}).get('location', {})
        
        poi = {
            'google_place_id': place.get('place_id'),
            'name': place.get('name'),
            'type': place.get('types', [])[0] if place.get('types') else 'unknown',
            'lat': location.get('lat'),
            'lng': location.get('lng'),
            'rating': place.get('rating', 0),
            'review_count': place.get('user_ratings_total', 0),
            'address': place.get('vicinity', ''),
            'price_level': place.get('price_level', 0),
            'zone': zone_name,
            'source': 'google',
            'photos': [],
            'hours': {},
            'fetched_at': datetime.now().isoformat()
        }
        
        # Get photo reference (first photo only to save space)
        if place.get('photos'):
            photo_ref = place['photos'][0].get('photo_reference')
            if photo_ref:
                poi['photos'].append({
                    'reference': photo_ref,
                    'url': f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={self.google_api_key}"
                })
        
        return poi
    
    def deduplicate_pois(self):
        """Remove duplicate POIs based on name and location"""
        seen = set()
        unique_pois = []
        
        for poi in self.all_pois:
            # Create unique key from name + rounded coordinates
            key = f"{poi['name']}_{round(poi['lat'], 3)}_{round(poi['lng'], 3)}"
            
            if key not in seen:
                seen.add(key)
                unique_pois.append(poi)
        
        duplicates_removed = len(self.all_pois) - len(unique_pois)
        self.all_pois = unique_pois
        
        print(f"\n🔍 Deduplication: Removed {duplicates_removed} duplicates")
        print(f"   Final count: {len(self.all_pois)} unique POIs")
    
    def run_fetch(self):
        """Main fetch process"""
        print("🚀 Starting POI Fetch Process...\n")
        print(f"📍 Zones: {len(ISTANBUL_ZONES)}")
        print(f"📦 Categories: {len(POI_CATEGORIES)}")
        print(f"🎯 Max Requests: {self.max_requests}\n")
        
        start_time = time.time()
        
        # Fetch from each zone
        for zone in ISTANBUL_ZONES:
            print(f"\n📍 Fetching: {zone['name']}")
            
            # Use top 5 most important categories per zone
            important_categories = ['tourist_attraction', 'museum', 'restaurant', 'cafe', 'park']
            
            for category in important_categories:
                if self.request_count >= self.max_requests:
                    print("\n⚠️  Request limit reached!")
                    break
                
                places = self.fetch_google_places(zone, category)
                
                # Take top 10 results per category to avoid overwhelming data
                for place in places[:10]:
                    poi = self.process_poi(place, zone['name'])
                    self.all_pois.append(poi)
                
                # Rate limiting: wait between requests
                time.sleep(0.5)
            
            if self.request_count >= self.max_requests:
                break
        
        # Deduplicate
        self.deduplicate_pois()
        
        # Save to JSON
        output_file = f"pois_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_pois, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ Fetch Complete!")
        print(f"   Total POIs: {len(self.all_pois)}")
        print(f"   API Requests: {self.request_count}")
        print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   Saved to: {output_file}")
        print("="*60)
        print(f"\n📄 Next Step: python import_pois.py {output_file}")
        
        return output_file

if __name__ == "__main__":
    try:
        fetcher = POIFetcher()
        output_file = fetcher.run_fetch()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print("💡 Make sure your .env file contains GOOGLE_PLACES_API_KEY")
