#!/usr/bin/env python3
"""
Test Enhanced Blog Features
Tests weather-aware recommendations, personalization, and analytics
"""

import asyncio
import json
import requests
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlogFeaturesTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def test_weather_recommendations(self):
        """Test weather-aware blog recommendations"""
        print("\n🌤️ Testing Weather-Aware Blog Recommendations")
        print("=" * 50)
        
        try:
            # Test for different locations
            locations = ["Istanbul", "Ankara", "Izmir"]
            
            for location in locations:
                print(f"\n📍 Testing recommendations for {location}:")
                
                response = requests.get(
                    f"{self.base_url}/blog/recommendations/weather",
                    params={"location": location, "limit": 3}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        print(f"✅ Got {len(data['recommendations'])} recommendations")
                        
                        for i, rec in enumerate(data['recommendations'], 1):
                            print(f"  {i}. {rec['title']}")
                            print(f"     Relevance: {rec['relevance_score']:.1%}")
                            print(f"     Reason: {rec['reason']}")
                            if rec.get('weather_context'):
                                print(f"     Weather: {rec['weather_context']}")
                            print()
                    else:
                        print(f"❌ API returned success=false: {data}")
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"❌ Weather recommendations test failed: {e}")
    
    def test_personalized_recommendations(self):
        """Test personalized blog recommendations"""
        print("\n👤 Testing Personalized Blog Recommendations")
        print("=" * 50)
        
        try:
            # Test different user preference combinations
            test_cases = [
                {
                    "user_id": "test_user_1",
                    "categories": "food,culture",
                    "districts": "sultanahmet,galata",
                    "description": "Food lover interested in Sultanahmet & Galata"
                },
                {
                    "user_id": "test_user_2", 
                    "categories": "nightlife,art",
                    "districts": "beyoglu,karakoy",
                    "description": "Nightlife & art enthusiast in Beyoğlu & Karaköy"
                },
                {
                    "user_id": "test_user_3",
                    "categories": "shopping,hidden_gems",
                    "districts": "kadikoy,besiktas",
                    "description": "Shopping lover seeking hidden gems"
                }
            ]
            
            for test_case in test_cases:
                print(f"\n🧪 Testing: {test_case['description']}")
                
                response = requests.get(
                    f"{self.base_url}/blog/recommendations/personalized",
                    params={
                        "user_id": test_case["user_id"],
                        "categories": test_case["categories"],
                        "districts": test_case["districts"],
                        "limit": 3
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        recommendations = data['recommendations']
                        print(f"✅ Got {len(recommendations)} personalized recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            print(f"  {i}. {rec['title']}")
                            print(f"     Relevance: {rec['relevance_score']:.1%}")
                            print(f"     Reason: {rec['reason']}")
                            print()
                        
                        print(f"   User preferences: {data['user_preferences']}")
                    else:
                        print(f"❌ API returned success=false: {data}")
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"❌ Personalized recommendations test failed: {e}")
    
    def test_blog_analytics(self):
        """Test blog analytics endpoints"""
        print("\n📊 Testing Blog Analytics")
        print("=" * 50)
        
        try:
            # Test engagement tracking
            print("\n📝 Testing engagement tracking...")
            
            tracking_data = {
                "post_id": "test_post_2024",
                "user_id": "test_user_analytics",
                "event_type": "view",
                "metadata": json.dumps({"source": "test_script", "timestamp": datetime.now().isoformat()})
            }
            
            response = requests.post(
                f"{self.base_url}/blog/analytics/track",
                data=tracking_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print("✅ Engagement tracking successful")
                else:
                    print(f"❌ Tracking failed: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
            
            # Test performance analytics
            print("\n📈 Testing performance analytics...")
            
            response = requests.get(f"{self.base_url}/blog/analytics/performance")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    analytics = data['analytics']
                    print("✅ Performance analytics retrieved")
                    
                    print(f"   Top posts: {len(analytics.get('top_performing_posts', []))}")
                    print(f"   Trending categories: {len(analytics.get('trending_categories', []))}")
                    
                    if analytics.get('top_performing_posts'):
                        top_post = analytics['top_performing_posts'][0]
                        print(f"   #1 post: {top_post['title']} ({top_post['views']} views)")
                    
                else:
                    print(f"❌ Analytics failed: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
            
            # Test real-time metrics
            print("\n⚡ Testing real-time metrics...")
            
            response = requests.get(f"{self.base_url}/blog/analytics/realtime")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    metrics = data['metrics']
                    print("✅ Real-time metrics retrieved")
                    
                    print(f"   Active readers: {metrics.get('current_active_readers', 'N/A')}")
                    print(f"   Posts read today: {metrics.get('posts_read_today', 'N/A')}")
                    print(f"   New subscribers: {metrics.get('new_subscribers_today', 'N/A')}")
                    
                    trending = metrics.get('trending_now', [])
                    if trending:
                        print(f"   Trending: {', '.join(trending[:3])}")
                    
                else:
                    print(f"❌ Real-time metrics failed: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Blog analytics test failed: {e}")
    
    def test_integration_with_existing_apis(self):
        """Test integration with existing Google APIs"""
        print("\n🔗 Testing Integration with Existing APIs")
        print("=" * 50)
        
        try:
            # Test AI endpoint with weather query
            print("\n🌤️ Testing weather AI integration...")
            
            response = requests.post(
                f"{self.base_url}/ai",
                json={"message": "What's the weather like in Istanbul today?"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("response"):
                    print("✅ Weather AI integration working")
                    print(f"   Response length: {len(data['response'])} characters")
                    # Check if response contains weather-related content
                    if any(word in data['response'].lower() for word in ['weather', 'temperature', 'climate']):
                        print("   ✅ Response contains weather information")
                    else:
                        print("   ⚠️ Response may not contain weather information")
                else:
                    print(f"❌ AI API failed: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
            
            # Test restaurants integration  
            print("\n📍 Testing restaurants API integration...")
            
            response = requests.get(f"{self.base_url}/restaurants/popular")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"✅ Restaurants API integration working ({len(data)} restaurants)")
                    
                    if data:
                        restaurant = data[0]
                        print(f"   Example: {restaurant.get('name', 'N/A')}")
                        print(f"   District: {restaurant.get('district', 'N/A')}")
                else:
                    print(f"❌ Restaurants API returned unexpected data: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
            
            # Test enhanced recommendations
            print("\n🤖 Testing AI enhanced recommendations...")
            
            response = requests.get(
                f"{self.base_url}/ai/enhanced-recommendations",
                params={"query": "best restaurants in Istanbul", "limit": 3}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    recommendations = data.get('recommendations', [])
                    print(f"✅ AI recommendations working ({len(recommendations)} recommendations)")
                    
                    if recommendations:
                        rec = recommendations[0]
                        print(f"   Example: {rec.get('name', 'N/A')}")
                        print(f"   Score: {rec.get('relevance_score', 'N/A')}")
                else:
                    print(f"❌ AI recommendations failed: {data}")
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ API integration test failed: {e}")
    
    def run_all_tests(self):
        """Run all blog enhancement tests"""
        print("🚀 AI Istanbul Blog Enhancement Tests")
        print("=" * 60)
        print(f"Testing against: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Run all tests
        self.test_weather_recommendations()
        self.test_personalized_recommendations()
        self.test_blog_analytics()
        self.test_integration_with_existing_apis()
        
        print("\n" + "=" * 60)
        print("🎉 Blog enhancement tests completed!")
        print("\nNext steps:")
        print("- Integrate weather-aware recommendations in frontend")
        print("- Add personalization based on user login")
        print("- Set up analytics dashboard for content creators")
        print("- Implement real-time features with WebSockets")

def main():
    """Main test execution"""
    tester = BlogFeaturesTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
