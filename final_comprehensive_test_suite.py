#!/usr/bin/env python3
"""
Final Comprehensive Test Suite for AI Istanbul Chatbot
====================================================

This test suite contains 80 carefully crafted inputs across 5 categories:
- Daily Talks (16 inputs)
- Restaurant Advice (16 inputs) 
- District Advice (16 inputs)
- Museum Advice (16 inputs)
- Transportation Advice (16 inputs)

Each category tests different aspects of location awareness, cultural understanding,
and practical advice quality for Istanbul tourism.
"""

import json
import requests
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import statistics

@dataclass
class TestInput:
    category: str
    query: str
    expected_features: List[str]  # Features we expect to see in a good response
    location_context: str = ""    # Expected location context
    difficulty: str = "medium"    # easy, medium, hard

class FinalTestSuite:
    """Comprehensive test suite for final AI evaluation"""
    
    def __init__(self):
        self.test_inputs = self._create_comprehensive_test_inputs()
        self.api_base = "http://localhost:8001"
        
    def _create_comprehensive_test_inputs(self) -> List[TestInput]:
        """Create 80 comprehensive test inputs across all categories"""
        
        inputs = []
        
        # === DAILY TALKS (16 inputs) ===
        daily_talks = [
            TestInput(
                category="daily_talks",
                query="Merhaba! How are you today?",
                expected_features=["greeting_response", "turkish_acknowledgment", "friendly_tone"],
                difficulty="easy"
            ),
            TestInput(
                category="daily_talks", 
                query="I'm feeling overwhelmed by Istanbul's size. Any tips?",
                expected_features=["empathy", "practical_advice", "encouragement", "step_by_step_guidance"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="What's the weather like in Istanbul right now?",
                expected_features=["current_info", "seasonal_advice", "clothing_suggestions"],
                difficulty="easy"
            ),
            TestInput(
                category="daily_talks",
                query="I just arrived at Istanbul Airport. What should I do first?",
                expected_features=["arrival_guidance", "transportation_options", "first_steps", "practical_tips"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="I'm tired of tourist traps. Show me the real Istanbul!",
                expected_features=["authentic_experiences", "local_neighborhoods", "off_beaten_path", "cultural_immersion"],
                difficulty="hard"
            ),
            TestInput(
                category="daily_talks",
                query="My Turkish is terrible. Will I survive in Istanbul?",
                expected_features=["reassurance", "language_tips", "non_verbal_communication", "helpful_phrases"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="I'm traveling solo as a woman. Is Istanbul safe?",
                expected_features=["safety_advice", "solo_travel_tips", "cultural_sensitivity", "practical_precautions"],
                difficulty="hard"
            ),
            TestInput(
                category="daily_talks",
                query="What's a typical day like for locals in Istanbul?",
                expected_features=["local_lifestyle", "daily_routines", "cultural_insights", "social_customs"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="I'm on a tight budget. Can I still enjoy Istanbul?",
                expected_features=["budget_options", "free_activities", "cost_saving_tips", "value_recommendations"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="The crowds are crazy! When is the best time to visit attractions?",
                expected_features=["timing_advice", "crowd_avoidance", "alternative_times", "seasonal_insights"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="I'm here for business but have one free evening. What should I do?",
                expected_features=["time_efficient_suggestions", "evening_activities", "business_traveler_focus", "quick_experiences"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="My flight got delayed. I have 8 hours in Istanbul. Ideas?",
                expected_features=["layover_suggestions", "time_management", "airport_proximity", "luggage_considerations"],
                difficulty="hard"
            ),
            TestInput(
                category="daily_talks",
                query="I keep getting lost in Istanbul. Any navigation tips?",
                expected_features=["navigation_help", "landmark_guidance", "app_recommendations", "orientation_tips"],
                difficulty="medium"
            ),
            TestInput(
                category="daily_talks",
                query="What should I pack for a week in Istanbul?",
                expected_features=["packing_advice", "weather_considerations", "cultural_dress_codes", "practical_items"],
                difficulty="easy"
            ),
            TestInput(
                category="daily_talks",
                query="I'm homesick already. How do I connect with locals?",
                expected_features=["social_connection", "cultural_activities", "community_events", "friendship_building"],
                difficulty="hard"
            ),
            TestInput(
                category="daily_talks",
                query="What are some Turkish customs I should know about?",
                expected_features=["cultural_etiquette", "social_norms", "respectful_behavior", "cultural_sensitivity"],
                difficulty="medium"
            )
        ]
        
        # === RESTAURANT ADVICE (16 inputs) ===
        restaurant_advice = [
            TestInput(
                category="restaurant_advice",
                query="Best kebab places in Sultanahmet?",
                expected_features=["location_specific", "kebab_expertise", "sultanahmet_context", "walking_distances"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="I want authentic Turkish breakfast in Kadıköy",
                expected_features=["breakfast_specialties", "kadikoy_local_culture", "authentic_recommendations", "local_spots"],
                location_context="kadikoy",
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="Romantic dinner spots with Bosphorus view?",
                expected_features=["romantic_atmosphere", "bosphorus_views", "upscale_recommendations", "reservation_advice"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="Street food tour around Eminönü - where to start?",
                expected_features=["street_food_guide", "eminonu_specialties", "food_tour_route", "hygiene_tips"],
                location_context="eminonu",
                difficulty="hard"
            ),
            TestInput(
                category="restaurant_advice",
                query="I'm vegetarian. What are my options in Istanbul?",
                expected_features=["vegetarian_options", "turkish_vegetarian_dishes", "restaurant_recommendations", "dietary_guidance"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="Best meze restaurants in Beyoğlu?",
                expected_features=["meze_expertise", "beyoglu_dining", "sharing_culture", "wine_pairings"],
                location_context="beyoglu",
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="I have food allergies. How do I communicate this in Turkish restaurants?",
                expected_features=["allergy_communication", "turkish_phrases", "safety_precautions", "restaurant_etiquette"],
                difficulty="hard"
            ),
            TestInput(
                category="restaurant_advice",
                query="Cheap eats near Taksim Square for students?",
                expected_features=["budget_dining", "taksim_area", "student_friendly", "value_recommendations"],
                location_context="taksim",
                difficulty="easy"
            ),
            TestInput(
                category="restaurant_advice",
                query="Traditional Ottoman cuisine - where can I find it?",
                expected_features=["ottoman_cuisine", "historical_restaurants", "traditional_dishes", "cultural_context"],
                difficulty="hard"
            ),
            TestInput(
                category="restaurant_advice",
                query="Best fish restaurants along the Bosphorus?",
                expected_features=["seafood_expertise", "bosphorus_dining", "fresh_fish_guidance", "seaside_atmosphere"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="I want to try Turkish coffee and baklava - where's the best?",
                expected_features=["coffee_culture", "dessert_expertise", "traditional_preparation", "authentic_spots"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="Late night dining options in Karaköy?",
                expected_features=["late_night_dining", "karakoy_nightlife", "24_hour_options", "night_culture"],
                location_context="karakoy",
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="Family-friendly restaurants with kids' options?",
                expected_features=["family_dining", "kid_friendly", "turkish_family_culture", "practical_advice"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="How much should I tip in Istanbul restaurants?",
                expected_features=["tipping_culture", "service_expectations", "cultural_norms", "practical_amounts"],
                difficulty="easy"
            ),
            TestInput(
                category="restaurant_advice",
                query="Best rooftop restaurants with historical views?",
                expected_features=["rooftop_dining", "historical_views", "atmosphere", "photo_opportunities"],
                difficulty="medium"
            ),
            TestInput(
                category="restaurant_advice",
                query="I'm craving international food. Any recommendations?",
                expected_features=["international_cuisine", "diverse_options", "quality_recommendations", "cultural_fusion"],
                difficulty="easy"
            )
        ]
        
        # === DISTRICT ADVICE (16 inputs) ===
        district_advice = [
            TestInput(
                category="district_advice",
                query="Tell me about Sultanahmet district - what makes it special?",
                expected_features=["sultanahmet_character", "historical_significance", "main_attractions", "walking_distances"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Is Kadıköy worth visiting? What's there?",
                expected_features=["kadikoy_authenticity", "asian_side_culture", "local_experiences", "transportation_access"],
                location_context="kadikoy",
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Beyoğlu vs Sultanahmet - which should I prioritize?",
                expected_features=["district_comparison", "travel_priorities", "different_experiences", "time_allocation"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="What's the vibe like in Karaköy?",
                expected_features=["karakoy_atmosphere", "arts_culture", "dining_scene", "gentrification_context"],
                location_context="karakoy",
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="I want to experience local life. Which districts are most authentic?",
                expected_features=["authentic_neighborhoods", "local_culture", "non_touristy_areas", "cultural_immersion"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="Safe neighborhoods for accommodation?",
                expected_features=["safety_assessment", "accommodation_advice", "neighborhood_security", "practical_considerations"],
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Which area has the best nightlife?",
                expected_features=["nightlife_districts", "entertainment_options", "bar_culture", "safety_at_night"],
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Budget-friendly districts for backpackers?",
                expected_features=["budget_neighborhoods", "backpacker_culture", "affordable_options", "hostel_areas"],
                difficulty="easy"
            ),
            TestInput(
                category="district_advice",
                query="Where do young professionals live in Istanbul?",
                expected_features=["modern_districts", "professional_areas", "lifestyle_insights", "urban_development"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="Best shopping districts in Istanbul?",
                expected_features=["shopping_areas", "different_shopping_styles", "market_culture", "boutique_districts"],
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Which districts have the best street art?",
                expected_features=["street_art_areas", "cultural_expression", "artistic_neighborhoods", "creative_scenes"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="Waterfront districts with sea views?",
                expected_features=["waterfront_areas", "sea_access", "coastal_culture", "scenic_neighborhoods"],
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Which areas should I avoid as a tourist?",
                expected_features=["safety_awareness", "tourist_precautions", "area_guidance", "practical_warnings"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="Historic districts beyond the main tourist areas?",
                expected_features=["hidden_historic_areas", "lesser_known_heritage", "cultural_depth", "exploration_suggestions"],
                difficulty="hard"
            ),
            TestInput(
                category="district_advice",
                query="Which district is best for a first-time visitor?",
                expected_features=["first_visit_guidance", "accessibility", "comprehensive_experience", "practical_logistics"],
                difficulty="medium"
            ),
            TestInput(
                category="district_advice",
                query="Areas with good public transportation connections?",
                expected_features=["transportation_hubs", "connectivity", "metro_access", "mobility_convenience"],
                difficulty="medium"
            )
        ]
        
        # === MUSEUM ADVICE (16 inputs) ===
        museum_advice = [
            TestInput(
                category="museum_advice",
                query="Should I visit Hagia Sophia or Blue Mosque first?",
                expected_features=["visit_sequence", "crowd_management", "historical_context", "practical_timing"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Is the Topkapi Palace worth the entrance fee?",
                expected_features=["value_assessment", "palace_highlights", "ticket_information", "visit_duration"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Best art museums in Istanbul?",
                expected_features=["art_museum_variety", "contemporary_vs_traditional", "cultural_significance", "artistic_collections"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="I have 3 hours for museums in Sultanahmet. What should I prioritize?",
                expected_features=["time_management", "priority_ranking", "sultanahmet_museums", "efficient_routing"],
                location_context="sultanahmet",
                difficulty="hard"
            ),
            TestInput(
                category="museum_advice",
                query="Are there any free museums in Istanbul?",
                expected_features=["free_attractions", "budget_cultural_options", "accessible_culture", "hidden_gems"],
                difficulty="easy"
            ),
            TestInput(
                category="museum_advice",
                query="Istanbul Archaeological Museums - what's inside?",
                expected_features=["archaeological_collections", "historical_artifacts", "museum_highlights", "educational_value"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Photography rules in Istanbul museums?",
                expected_features=["photography_policies", "cultural_respect", "museum_etiquette", "preservation_concerns"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Modern vs traditional museums - which type should I choose?",
                expected_features=["museum_types", "personal_preference_guidance", "cultural_balance", "experience_variety"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Basilica Cistern - is it worth the hype?",
                expected_features=["cistern_experience", "underground_atmosphere", "historical_engineering", "visitor_expectations"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Best museums for understanding Ottoman history?",
                expected_features=["ottoman_history", "imperial_collections", "dynastic_artifacts", "cultural_education"],
                difficulty="hard"
            ),
            TestInput(
                category="museum_advice",
                query="Museum pass vs individual tickets - what's better?",
                expected_features=["ticket_strategy", "cost_comparison", "convenience_factors", "visit_planning"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Kid-friendly museums in Istanbul?",
                expected_features=["family_attractions", "educational_activities", "interactive_exhibits", "child_engagement"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Off-the-beaten-path museums locals recommend?",
                expected_features=["hidden_museums", "local_recommendations", "unique_collections", "cultural_authenticity"],
                difficulty="hard"
            ),
            TestInput(
                category="museum_advice",
                query="How long should I spend at each major museum?",
                expected_features=["time_allocation", "visit_duration", "museum_sizing", "tour_planning"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Religious site etiquette - what should I know?",
                expected_features=["religious_respect", "dress_codes", "behavioral_guidelines", "cultural_sensitivity"],
                difficulty="medium"
            ),
            TestInput(
                category="museum_advice",
                query="Best time of day to visit major museums?",
                expected_features=["crowd_avoidance", "lighting_conditions", "operational_hours", "visitor_strategy"],
                difficulty="medium"
            )
        ]
        
        # === TRANSPORTATION ADVICE (16 inputs) ===
        transportation_advice = [
            TestInput(
                category="transportation_advice",
                query="How do I get from Istanbul Airport to Sultanahmet?",
                expected_features=["airport_connection", "sultanahmet_access", "transport_options", "cost_comparison"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Istanbul Metro system - how does it work?",
                expected_features=["metro_guide", "system_overview", "payment_methods", "route_planning"],
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Ferry transportation - which routes are most scenic?",
                expected_features=["ferry_routes", "scenic_recommendations", "bosphorus_views", "maritime_culture"],
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Is Uber available in Istanbul? What about taxis?",
                expected_features=["ride_services", "taxi_culture", "pricing_comparison", "safety_considerations"],
                difficulty="easy"
            ),
            TestInput(
                category="transportation_advice",
                query="How do I get to the Asian side of Istanbul?",
                expected_features=["cross_continental_transport", "bridge_vs_ferry", "kadikoy_access", "route_options"],
                location_context="kadikoy",
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Public transport card (Istanbulkart) - where to buy and how to use?",
                expected_features=["istanbulkart_guide", "purchase_locations", "loading_money", "system_integration"],
                difficulty="easy"
            ),
            TestInput(
                category="transportation_advice",
                query="Walking distances between major attractions in Sultanahmet?",
                expected_features=["walking_routes", "sultanahmet_distances", "pedestrian_navigation", "time_estimates"],
                location_context="sultanahmet",
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Night transportation - what's available after midnight?",
                expected_features=["night_transport", "late_night_options", "safety_considerations", "service_hours"],
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Traffic patterns - when should I avoid driving/taking taxis?",
                expected_features=["traffic_insights", "rush_hour_patterns", "congestion_avoidance", "alternative_timing"],
                difficulty="hard"
            ),
            TestInput(
                category="transportation_advice",
                query="Getting around Beyoğlu district - best options?",
                expected_features=["beyoglu_transport", "district_mobility", "walking_vs_transport", "local_connections"],
                location_context="beyoglu",
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Day trip to Princes' Islands - how to get there?",
                expected_features=["islands_transport", "ferry_connections", "day_trip_planning", "return_schedules"],
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Accessibility for mobility-impaired visitors?",
                expected_features=["accessibility_transport", "disabled_access", "barrier_free_routes", "assistance_services"],
                difficulty="hard"
            ),
            TestInput(
                category="transportation_advice",
                query="Airport connections - Sabiha Gökçen vs Istanbul Airport?",
                expected_features=["airport_comparison", "connection_options", "travel_times", "cost_differences"],
                difficulty="medium"
            ),
            TestInput(
                category="transportation_advice",
                query="Dolmuş transportation - what is it and how to use it?",
                expected_features=["dolmus_explanation", "shared_transport", "route_system", "cultural_experience"],
                difficulty="hard"
            ),
            TestInput(
                category="transportation_advice",
                query="Best transport apps for Istanbul?",
                expected_features=["transport_apps", "digital_navigation", "real_time_info", "trip_planning"],
                difficulty="easy"
            ),
            TestInput(
                category="transportation_advice",
                query="Crossing the Bosphorus - all the different ways?",
                expected_features=["bosphorus_crossing", "bridge_options", "ferry_routes", "tunnel_connections"],
                difficulty="medium"
            )
        ]
        
        # Combine all categories
        inputs.extend(daily_talks)
        inputs.extend(restaurant_advice) 
        inputs.extend(district_advice)
        inputs.extend(museum_advice)
        inputs.extend(transportation_advice)
        
        return inputs
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all 80 test inputs and analyze results"""
        
        print("🎯 FINAL COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Running {len(self.test_inputs)} test inputs across 5 categories...")
        
        results = {
            "total_tests": len(self.test_inputs),
            "category_results": {},
            "overall_metrics": {},
            "detailed_results": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        category_scores = {}
        
        # Test each category
        for category in ["daily_talks", "restaurant_advice", "district_advice", "museum_advice", "transportation_advice"]:
            category_inputs = [t for t in self.test_inputs if t.category == category]
            print(f"\n📂 Testing {category.upper()} ({len(category_inputs)} inputs)")
            
            category_results = []
            
            for i, test_input in enumerate(category_inputs, 1):
                print(f"  {i:2d}. Testing: {test_input.query[:50]}...")
                
                # Make API call
                try:
                    response_data = self._call_api(test_input.query)
                    if response_data:
                        # Analyze response
                        analysis = self._analyze_response(test_input, response_data)
                        category_results.append(analysis)
                        
                        # Print quick result
                        relevance = analysis['scores']['relevance']
                        print(f"      Relevance: {relevance:.1f}/5 | Features: {len(analysis['detected_features'])}/{len(test_input.expected_features)}")
                    else:
                        print(f"      ❌ API Error")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)[:50]}")
                
                # Small delay between requests
                time.sleep(0.5)
            
            # Calculate category metrics
            if category_results:
                category_metrics = self._calculate_category_metrics(category_results)
                category_scores[category] = category_metrics
                results["category_results"][category] = {
                    "metrics": category_metrics,
                    "results": category_results
                }
                
                print(f"  📊 {category.upper()} Results:")
                print(f"      Avg Relevance: {category_metrics['avg_relevance']:.2f}/5")
                print(f"      Avg Completeness: {category_metrics['avg_completeness']:.2f}/5") 
                print(f"      Feature Coverage: {category_metrics['feature_coverage_rate']:.1f}%")
        
        # Calculate overall metrics
        all_results = []
        for cat_data in results["category_results"].values():
            all_results.extend(cat_data["results"])
        
        if all_results:
            overall_metrics = self._calculate_overall_metrics(all_results, category_scores)
            results["overall_metrics"] = overall_metrics
            
            print(f"\n🏆 OVERALL RESULTS:")
            print(f"    Final Score: {overall_metrics['final_score']:.2f}/5")
            print(f"    Grade: {overall_metrics['letter_grade']}")
            print(f"    Avg Relevance: {overall_metrics['avg_relevance']:.2f}/5")
            print(f"    Avg Completeness: {overall_metrics['avg_completeness']:.2f}/5")
            print(f"    Location Awareness: {overall_metrics['location_accuracy']:.1f}%")
            print(f"    Feature Coverage: {overall_metrics['overall_feature_coverage']:.1f}%")
        
        results["detailed_results"] = all_results
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _call_api(self, query: str) -> Dict[str, Any]:
        """Make API call to chatbot"""
        try:
            response = requests.post(
                f"{self.api_base}/ai/chat",
                json={"message": query},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
    
    def _analyze_response(self, test_input: TestInput, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze response quality against expected features"""
        
        response_text = response_data.get("response", "").lower()
        
        # Detect expected features
        detected_features = []
        for feature in test_input.expected_features:
            if self._feature_detected(feature, response_text, test_input):
                detected_features.append(feature)
        
        # Calculate scores
        relevance_score = self._calculate_relevance_score(test_input, response_text)
        completeness_score = len(detected_features) / len(test_input.expected_features) * 5
        cultural_awareness_score = self._calculate_cultural_awareness(response_text)
        location_accuracy = self._check_location_accuracy(test_input, response_text)
        
        return {
            "query": test_input.query,
            "category": test_input.category,
            "difficulty": test_input.difficulty,
            "response_length": len(response_data.get("response", "")),
            "expected_features": test_input.expected_features,
            "detected_features": detected_features,
            "feature_coverage": len(detected_features) / len(test_input.expected_features),
            "scores": {
                "relevance": relevance_score,
                "completeness": completeness_score,
                "cultural_awareness": cultural_awareness_score
            },
            "location_accuracy": location_accuracy,
            "response_preview": response_data.get("response", "")[:200] + "..."
        }
    
    def _feature_detected(self, feature: str, response_text: str, test_input: TestInput) -> bool:
        """Check if expected feature is present in response"""
        
        feature_indicators = {
            # Daily talks features
            "greeting_response": ["hello", "hi", "merhaba", "welcome", "good"],
            "turkish_acknowledgment": ["merhaba", "turkish", "turkey", "istanbul"],
            "empathy": ["understand", "know how", "feel", "overwhelm", "help"],
            "practical_advice": ["tip", "suggest", "recommend", "try", "should"],
            "current_info": ["weather", "temperature", "season", "climate"],
            "arrival_guidance": ["airport", "arrival", "first", "transportation"],
            "authentic_experiences": ["local", "authentic", "real", "traditional"],
            "language_tips": ["english", "speak", "language", "phrase"],
            "safety_advice": ["safe", "security", "precaution", "careful"],
            "budget_options": ["budget", "cheap", "free", "affordable"],
            "timing_advice": ["time", "early", "avoid crowd", "best time"],
            "navigation_help": ["map", "direction", "navigate", "lost"],
            "cultural_etiquette": ["custom", "etiquette", "respect", "culture"],
            
            # Restaurant features  
            "location_specific": [test_input.location_context] if test_input.location_context else ["area", "district"],
            "kebab_expertise": ["kebab", "döner", "şiş", "adana"],
            "walking_distances": ["walk", "minute", "meter", "distance"],
            "breakfast_specialties": ["breakfast", "kahvaltı", "morning", "tea"],
            "authentic_recommendations": ["authentic", "traditional", "local", "family"],
            "romantic_atmosphere": ["romantic", "view", "atmosphere", "couple"],
            "street_food_guide": ["street", "vendor", "local food", "cheap"],
            "vegetarian_options": ["vegetarian", "vegan", "vegetable", "meat-free"],
            "meze_expertise": ["meze", "appetizer", "sharing", "small plate"],
            "allergy_communication": ["allergy", "dietary", "restriction", "cannot eat"],
            "budget_dining": ["budget", "cheap", "student", "affordable"],
            "ottoman_cuisine": ["ottoman", "imperial", "historical", "traditional"],
            "seafood_expertise": ["fish", "seafood", "fresh", "catch"],
            "coffee_culture": ["coffee", "turkish coffee", "kahve", "traditional"],
            "family_dining": ["family", "child", "kid", "children"],
            "tipping_culture": ["tip", "service", "gratuity", "%"],
            
            # District features
            "sultanahmet_context": ["sultanahmet", "historic", "unesco", "imperial"],
            "historical_significance": ["history", "historical", "byzantine", "ottoman"],
            "main_attractions": ["attraction", "sight", "must see", "visit"],
            "kadikoy_authenticity": ["authentic", "local", "real", "asian side"],
            "district_comparison": ["compare", "vs", "versus", "better", "choose"],
            "karakoy_atmosphere": ["trendy", "art", "modern", "hip"],
            "arts_culture": ["art", "gallery", "culture", "creative"],
            "authentic_neighborhoods": ["authentic", "local", "real", "traditional"],
            "safety_assessment": ["safe", "security", "crime", "dangerous"],
            "nightlife_districts": ["nightlife", "bar", "club", "entertainment"],
            "budget_neighborhoods": ["budget", "cheap", "backpacker", "hostel"],
            "modern_districts": ["modern", "contemporary", "new", "development"],
            "shopping_areas": ["shopping", "shop", "market", "bazaar"],
            "waterfront_areas": ["water", "sea", "coast", "waterfront"],
            
            # Museum features
            "visit_sequence": ["first", "order", "sequence", "priority"],
            "crowd_management": ["crowd", "busy", "avoid", "time"],
            "historical_context": ["history", "historical", "significance", "context"],
            "value_assessment": ["worth", "value", "price", "cost"],
            "palace_highlights": ["palace", "room", "collection", "exhibit"],
            "art_museum_variety": ["art", "museum", "gallery", "collection"],
            "time_management": ["time", "hour", "duration", "quick"],
            "free_attractions": ["free", "no cost", "complimentary", "gratis"],
            "archaeological_collections": ["archaeological", "artifact", "ancient", "excavation"],
            "photography_policies": ["photo", "camera", "picture", "photography"],
            "museum_types": ["type", "kind", "variety", "different"],
            "cistern_experience": ["cistern", "underground", "column", "water"],
            "ottoman_history": ["ottoman", "empire", "sultan", "imperial"],
            "ticket_strategy": ["ticket", "pass", "museum pass", "entrance"],
            "family_attractions": ["family", "child", "kid", "interactive"],
            "hidden_museums": ["hidden", "secret", "unknown", "local"],
            "time_allocation": ["time", "spend", "duration", "how long"],
            "religious_respect": ["respect", "religious", "mosque", "dress"],
            
            # Transportation features
            "airport_connection": ["airport", "connection", "transport", "how to get"],
            "metro_guide": ["metro", "subway", "underground", "m1", "m2"],
            "system_overview": ["system", "network", "how it works", "guide"],
            "ferry_routes": ["ferry", "boat", "route", "bosphorus"],
            "scenic_recommendations": ["scenic", "view", "beautiful", "sightseeing"],
            "ride_services": ["uber", "taxi", "ride", "app"],
            "cross_continental_transport": ["asian side", "european side", "cross", "bridge"],
            "istanbulkart_guide": ["istanbulkart", "card", "payment", "public transport"],
            "walking_routes": ["walk", "walking", "pedestrian", "on foot"],
            "night_transport": ["night", "late", "midnight", "24 hour"],
            "traffic_insights": ["traffic", "jam", "congestion", "rush hour"],
            "islands_transport": ["island", "prince", "ferry", "day trip"],
            "accessibility_transport": ["accessible", "disabled", "wheelchair", "mobility"],
            "airport_comparison": ["airport", "sabiha", "istanbul airport", "compare"],
            "dolmus_explanation": ["dolmuş", "shared", "minibus", "collective"],
            "transport_apps": ["app", "mobile", "digital", "smartphone"],
            "bosphorus_crossing": ["bosphorus", "cross", "bridge", "tunnel"]
        }
        
        indicators = feature_indicators.get(feature, [feature.replace("_", " ")])
        return any(indicator in response_text for indicator in indicators)
    
    def _calculate_relevance_score(self, test_input: TestInput, response_text: str) -> float:
        """Calculate relevance score based on query matching and context"""
        
        query_words = test_input.query.lower().split()
        response_words = response_text.split()
        
        # Word overlap score
        overlap = len(set(query_words) & set(response_words))
        overlap_score = min(overlap / len(query_words), 1.0) * 2  # Max 2 points
        
        # Location context score
        location_score = 0
        if test_input.location_context:
            if test_input.location_context in response_text:
                location_score = 1.5  # 1.5 points for location match
        
        # Category relevance
        category_keywords = {
            "daily_talks": ["help", "tip", "advice", "suggest", "istanbul"],
            "restaurant_advice": ["restaurant", "food", "eat", "dining", "cuisine"],
            "district_advice": ["district", "area", "neighborhood", "visit", "explore"],
            "museum_advice": ["museum", "visit", "attraction", "see", "historical"],
            "transportation_advice": ["transport", "get", "travel", "metro", "bus", "ferry"]
        }
        
        category_score = 0
        if test_input.category in category_keywords:
            keywords = category_keywords[test_input.category]
            if any(keyword in response_text for keyword in keywords):
                category_score = 1.5  # 1.5 points for category relevance
        
        return min(overlap_score + location_score + category_score, 5.0)
    
    def _calculate_cultural_awareness(self, response_text: str) -> float:
        """Calculate cultural awareness score"""
        
        cultural_indicators = [
            "turkish", "ottoman", "culture", "tradition", "local", "authentic",
            "respect", "custom", "etiquette", "halal", "mosque", "prayer",
            "ramadan", "family", "hospitality", "tea", "coffee", "baklava"
        ]
        
        cultural_score = sum(1 for indicator in cultural_indicators if indicator in response_text)
        return min(cultural_score * 0.5, 5.0)  # Max 5 points
    
    def _check_location_accuracy(self, test_input: TestInput, response_text: str) -> bool:
        """Check if location context is accurately addressed"""
        
        if not test_input.location_context:
            return True  # No specific location expected
        
        return test_input.location_context in response_text
    
    def _calculate_category_metrics(self, category_results: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for a category"""
        
        if not category_results:
            return {}
        
        relevance_scores = [r["scores"]["relevance"] for r in category_results]
        completeness_scores = [r["scores"]["completeness"] for r in category_results] 
        coverage_rates = [r["feature_coverage"] * 100 for r in category_results]
        
        return {
            "avg_relevance": statistics.mean(relevance_scores),
            "avg_completeness": statistics.mean(completeness_scores),
            "feature_coverage_rate": statistics.mean(coverage_rates),
            "total_tests": len(category_results)
        }
    
    def _calculate_overall_metrics(self, all_results: List[Dict], category_scores: Dict) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        
        # Overall averages
        relevance_scores = [r["scores"]["relevance"] for r in all_results]
        completeness_scores = [r["scores"]["completeness"] for r in all_results]
        cultural_scores = [r["scores"]["cultural_awareness"] for r in all_results]
        
        avg_relevance = statistics.mean(relevance_scores)
        avg_completeness = statistics.mean(completeness_scores)
        avg_cultural = statistics.mean(cultural_scores)
        
        # Feature coverage
        feature_coverages = [r["feature_coverage"] * 100 for r in all_results]
        overall_feature_coverage = statistics.mean(feature_coverages)
        
        # Location accuracy
        location_accurate = sum(1 for r in all_results if r["location_accuracy"])
        location_accuracy = (location_accurate / len(all_results)) * 100
        
        # Final composite score
        final_score = (avg_relevance * 0.4 + avg_completeness * 0.3 + avg_cultural * 0.3)
        
        # Letter grade
        if final_score >= 4.5:
            letter_grade = "A+"
        elif final_score >= 4.0:
            letter_grade = "A"
        elif final_score >= 3.5:
            letter_grade = "B+"
        elif final_score >= 3.0:
            letter_grade = "B"
        elif final_score >= 2.5:
            letter_grade = "C+"
        elif final_score >= 2.0:
            letter_grade = "C"
        else:
            letter_grade = "F"
        
        return {
            "final_score": final_score,
            "letter_grade": letter_grade,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_cultural_awareness": avg_cultural,
            "overall_feature_coverage": overall_feature_coverage,
            "location_accuracy": location_accuracy,
            "category_breakdown": category_scores
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to files"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = f"ai_istanbul_final_test_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save readable report
        report_filename = f"ai_istanbul_final_test_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            self._write_readable_report(f, results)
        
        print(f"\n💾 Results saved:")
        print(f"    📊 JSON: {json_filename}")
        print(f"    📄 Report: {report_filename}")
    
    def _write_readable_report(self, f, results: Dict[str, Any]):
        """Write human-readable test report"""
        
        f.write("🎯 AI ISTANBUL CHATBOT - FINAL COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {results['timestamp']}\n")
        f.write(f"Total Tests: {results['total_tests']}\n\n")
        
        # Overall results
        if results.get("overall_metrics"):
            metrics = results["overall_metrics"]
            f.write("🏆 OVERALL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Final Score: {metrics['final_score']:.2f}/5.0\n")
            f.write(f"Letter Grade: {metrics['letter_grade']}\n")
            f.write(f"Average Relevance: {metrics['avg_relevance']:.2f}/5.0\n")
            f.write(f"Average Completeness: {metrics['avg_completeness']:.2f}/5.0\n")
            f.write(f"Cultural Awareness: {metrics['avg_cultural_awareness']:.2f}/5.0\n")
            f.write(f"Feature Coverage: {metrics['overall_feature_coverage']:.1f}%\n")
            f.write(f"Location Accuracy: {metrics['location_accuracy']:.1f}%\n\n")
        
        # Category breakdown
        f.write("📊 CATEGORY BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        for category, data in results.get("category_results", {}).items():
            metrics = data["metrics"]
            f.write(f"{category.upper().replace('_', ' ')}:\n")
            f.write(f"  Relevance: {metrics['avg_relevance']:.2f}/5.0\n")
            f.write(f"  Completeness: {metrics['avg_completeness']:.2f}/5.0\n") 
            f.write(f"  Feature Coverage: {metrics['feature_coverage_rate']:.1f}%\n")
            f.write(f"  Tests: {metrics['total_tests']}\n\n")
        
        # Detailed results sample
        f.write("📝 SAMPLE DETAILED RESULTS\n")
        f.write("-" * 30 + "\n")
        for result in results.get("detailed_results", [])[:10]:  # First 10 results
            f.write(f"Query: {result['query']}\n")
            f.write(f"Category: {result['category']} | Difficulty: {result['difficulty']}\n")
            f.write(f"Relevance: {result['scores']['relevance']:.1f}/5 | ")
            f.write(f"Features: {len(result['detected_features'])}/{len(result['expected_features'])}\n")
            f.write(f"Response: {result['response_preview']}\n")
            f.write("-" * 40 + "\n")

def main():
    """Run the final comprehensive test suite"""
    
    print("🚀 Starting AI Istanbul Chatbot Final Test Suite")
    print("This will test 80 diverse inputs across 5 categories...")
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not responding. Please start the backend first:")
            print("   cd backend && python app.py")
            return
    except:
        print("❌ Backend not accessible. Please start the backend first:")
        print("   cd backend && python app.py")  
        return
    
    # Run comprehensive test
    tester = FinalTestSuite()
    results = tester.run_comprehensive_test()
    
    # Print final summary
    if results.get("overall_metrics"):
        metrics = results["overall_metrics"]
        print(f"\n🎉 FINAL TEST COMPLETE!")
        print(f"   Score: {metrics['final_score']:.2f}/5.0 ({metrics['letter_grade']})")
        print(f"   Relevance: {metrics['avg_relevance']:.2f}/5.0")
        print(f"   Feature Coverage: {metrics['overall_feature_coverage']:.1f}%")
        
        if metrics['final_score'] >= 3.5:
            print("   ✅ READY FOR PRODUCTION!")
        else:
            print("   ⚠️  Needs improvement before production")

if __name__ == "__main__":
    main()
