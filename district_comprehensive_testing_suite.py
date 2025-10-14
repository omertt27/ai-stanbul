#!/usr/bin/env python3
"""
🎯 Istanbul AI District Testing & Deep Learning Enhancement System
Comprehensive test suite for 6 main districts with 50+ queries
Analyzing response quality, deep learning integration, and personalization
"""

import asyncio
import time
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

# Import our systems
try:
    from istanbul_daily_talk_system import IstanbulDailyTalkAI
    from istanbul_neighborhood_guides_system import IstanbulNeighborhoodGuidesSystem
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ System import error: {e}")
    SYSTEMS_AVAILABLE = False

class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"  # 90-100%
    VERY_GOOD = "very_good"  # 80-89%
    GOOD = "good"           # 70-79%
    FAIR = "fair"           # 60-69%
    POOR = "poor"           # <60%

class DeepLearningFeature(Enum):
    """Deep learning features to detect"""
    CULTURAL_INSIGHTS = "cultural_insights"
    SEASONAL_CONTEXT = "seasonal_context"
    PERSONALIZED_RECOMMENDATIONS = "personalized_recommendations"
    HISTORICAL_ANECDOTES = "historical_anecdotes"
    HIDDEN_GEMS_NARRATIVES = "hidden_gems_narratives"
    LOCAL_EXPERTISE = "local_expertise"
    NEIGHBORHOOD_CHARACTER = "neighborhood_character"
    VISITOR_TYPE_MATCHING = "visitor_type_matching"

@dataclass
class ResponseAnalysis:
    """Analysis results for a single response"""
    query: str
    response: str
    length: int
    quality_score: float
    quality_level: ResponseQuality
    friendliness_score: float
    readability_score: float
    correctness_score: float
    deep_learning_features: List[DeepLearningFeature]
    district_mentioned: Optional[str]
    processing_time: float
    has_personalization: bool
    cultural_depth: int
    narrative_quality: int

class IstanbulDistrictTestingSuite:
    """Comprehensive testing suite for Istanbul districts"""
    
    def __init__(self):
        if not SYSTEMS_AVAILABLE:
            raise ImportError("Required systems not available")
        
        self.ai_system = IstanbulDailyTalkAI()
        self.test_results = []
        self.districts = ["şişli", "beşiktaş", "fatih", "kadıköy", "üsküdar", "sarıyer"]
        
        # Enhanced test queries for 6 main districts (50+ total)
        self.district_queries = self._create_comprehensive_test_queries()
        
        print("🎯 Istanbul District Testing Suite Initialized")
        print(f"📊 {len(self.district_queries)} test queries prepared")
        print(f"🏘️ Testing {len(self.districts)} main districts")
    
    def _create_comprehensive_test_queries(self) -> List[Dict[str, Any]]:
        """Create 50+ comprehensive test queries for districts"""
        
        queries = []
        
        # Şişli District Queries (9 queries)
        sisli_queries = [
            {"query": "Tell me about Şişli district character and atmosphere", "expected_district": "şişli", "type": "character"},
            {"query": "What are the best shopping areas in Şişli?", "expected_district": "şişli", "type": "shopping"},
            {"query": "Show me modern neighborhoods in Şişli for young professionals", "expected_district": "şişli", "type": "demographic"},
            {"query": "I want to experience Şişli's nightlife and entertainment", "expected_district": "şişli", "type": "nightlife"},
            {"query": "What makes Şişli different from other Istanbul districts?", "expected_district": "şişli", "type": "comparison"},
            {"query": "Best cafés and coworking spaces in Şişli area", "expected_district": "şişli", "type": "lifestyle"},
            {"query": "Şişli hidden gems and local secrets", "expected_district": "şişli", "type": "hidden_gems"},
            {"query": "How is Şişli for families with children?", "expected_district": "şişli", "type": "family"},
            {"query": "Şişli district for photography and Instagram spots", "expected_district": "şişli", "type": "photography"}
        ]
        
        # Beşiktaş District Queries (9 queries)
        besiktas_queries = [
            {"query": "Describe Beşiktaş neighborhood character and vibe", "expected_district": "beşiktaş", "type": "character"},
            {"query": "Best waterfront areas and Bosphorus views in Beşiktaş", "expected_district": "beşiktaş", "type": "scenic"},
            {"query": "Beşiktaş for football fans and sports culture", "expected_district": "beşiktaş", "type": "sports"},
            {"query": "Traditional Turkish breakfast places in Beşiktaş", "expected_district": "beşiktaş", "type": "food"},
            {"query": "Beşiktaş markets and local shopping experience", "expected_district": "beşiktaş", "type": "shopping"},
            {"query": "What's special about Beşiktaş compared to Sultanahmet?", "expected_district": "beşiktaş", "type": "comparison"},
            {"query": "Beşiktaş cultural sites and museums", "expected_district": "beşiktaş", "type": "culture"},
            {"query": "Best time to visit Beşiktaş and seasonal highlights", "expected_district": "beşiktaş", "type": "seasonal"},
            {"query": "Beşiktaş local life and authentic experiences", "expected_district": "beşiktaş", "type": "authentic"}
        ]
        
        # Fatih District Queries (9 queries)
        fatih_queries = [
            {"query": "Tell me about Fatih district's historical significance", "expected_district": "fatih", "type": "historical"},
            {"query": "Best traditional neighborhoods in Fatih to explore", "expected_district": "fatih", "type": "traditional"},
            {"query": "Fatih religious sites and spiritual experiences", "expected_district": "fatih", "type": "religious"},
            {"query": "Hidden historical gems in Fatih district", "expected_district": "fatih", "type": "hidden_gems"},
            {"query": "Fatih for cultural immersion and authentic Istanbul", "expected_district": "fatih", "type": "authentic"},
            {"query": "Best areas in Fatih for first-time visitors", "expected_district": "fatih", "type": "tourism"},
            {"query": "Fatih local markets and traditional shopping", "expected_district": "fatih", "type": "shopping"},
            {"query": "What makes Fatih unique among Istanbul districts?", "expected_district": "fatih", "type": "uniqueness"},
            {"query": "Fatih photography spots for historic architecture", "expected_district": "fatih", "type": "photography"}
        ]
        
        # Kadıköy District Queries (Enhanced - 10 queries)
        kadikoy_queries = [
            {"query": "Describe Kadıköy's alternative culture and hipster scene", "expected_district": "kadıköy", "type": "culture"},
            {"query": "Best independent bookstores and cafés in Kadıköy", "expected_district": "kadıköy", "type": "lifestyle"},
            {"query": "Kadıköy street art and creative neighborhoods", "expected_district": "kadıköy", "type": "art"},
            {"query": "Why do locals prefer Kadıköy over European side?", "expected_district": "kadıköy", "type": "comparison"},
            {"query": "Kadıköy Moda area for seaside walks and views", "expected_district": "kadıköy", "type": "scenic"},
            {"query": "Best live music venues and cultural events in Kadıköy", "expected_district": "kadıköy", "type": "entertainment"},
            {"query": "Kadıköy food scene and local restaurant gems", "expected_district": "kadıköy", "type": "food"},
            {"query": "What's the character of Kadıköy compared to Beyoğlu?", "expected_district": "kadıköy", "type": "character"},
            {"query": "Kadıköy for young travelers and backpackers", "expected_district": "kadıköy", "type": "demographic"},
            {"query": "Hidden neighborhoods and local secrets in Kadıköy", "expected_district": "kadıköy", "type": "hidden_gems"}
        ]
        
        # Üsküdar District Queries (8 queries)
        uskudar_queries = [
            {"query": "Tell me about Üsküdar's traditional Asian side character", "expected_district": "üsküdar", "type": "character"},
            {"query": "Best sunset viewpoints in Üsküdar district", "expected_district": "üsküdar", "type": "scenic"},
            {"query": "Üsküdar's religious and spiritual significance", "expected_district": "üsküdar", "type": "religious"},
            {"query": "Traditional tea houses and local life in Üsküdar", "expected_district": "üsküdar", "type": "traditional"},
            {"query": "Üsküdar waterfront and Bosphorus experience", "expected_district": "üsküdar", "type": "waterfront"},
            {"query": "What makes Üsküdar authentic compared to tourist areas?", "expected_district": "üsküdar", "type": "authentic"},
            {"query": "Üsküdar for photographers and scenic views", "expected_district": "üsküdar", "type": "photography"},
            {"query": "Hidden gems and local favorites in Üsküdar", "expected_district": "üsküdar", "type": "hidden_gems"}
        ]
        
        # Sarıyer District Queries (Enhanced - 10 queries)
        sariyer_queries = [
            {"query": "Describe Sarıyer's upscale residential character", "expected_district": "sarıyer", "type": "character"},
            {"query": "Best Bosphorus villages and waterfront areas in Sarıyer", "expected_district": "sarıyer", "type": "scenic"},
            {"query": "Sarıyer's luxury shopping and high-end lifestyle", "expected_district": "sarıyer", "type": "luxury"},
            {"query": "Historical Ottoman mansions and architecture in Sarıyer", "expected_district": "sarıyer", "type": "historical"},
            {"query": "Sarıyer for nature lovers and outdoor activities", "expected_district": "sarıyer", "type": "nature"},
            {"query": "Exclusive restaurants and fine dining in Sarıyer", "expected_district": "sarıyer", "type": "dining"},
            {"query": "What makes Sarıyer special for wealthy Istanbul residents?", "expected_district": "sarıyer", "type": "exclusivity"},
            {"query": "Sarıyer's European-style neighborhoods and culture", "expected_district": "sarıyer", "type": "european"},
            {"query": "Best seasons and times to visit Sarıyer district", "expected_district": "sarıyer", "type": "seasonal"},
            {"query": "Hidden luxury spots and insider knowledge in Sarıyer", "expected_district": "sarıyer", "type": "insider"}
        ]
        
        # Add all queries to main list
        all_district_queries = [
            sisli_queries, besiktas_queries, fatih_queries, 
            kadikoy_queries, uskudar_queries, sariyer_queries
        ]
        
        for district_queries in all_district_queries:
            queries.extend(district_queries)
        
        return queries
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of all district queries"""
        
        print("\n🚀 STARTING COMPREHENSIVE DISTRICT ANALYSIS")
        print("=" * 70)
        print(f"📊 Testing {len(self.district_queries)} queries across {len(self.districts)} districts")
        print(f"🧠 Analyzing deep learning features and response quality")
        print("=" * 70)
        
        results = {
            "total_queries": len(self.district_queries),
            "districts_tested": self.districts,
            "responses": [],
            "quality_distribution": {q.value: 0 for q in ResponseQuality},
            "deep_learning_features": {f.value: 0 for f in DeepLearningFeature},
            "district_coverage": {d: {"queries": 0, "avg_quality": 0} for d in self.districts},
            "overall_metrics": {},
            "improvement_recommendations": []
        }
        
        # Process each query
        for i, query_data in enumerate(self.district_queries, 1):
            print(f"\n📝 Query {i}/{len(self.district_queries)}: {query_data['query'][:60]}...")
            
            # Create unique user for each test
            user_id = f"district_test_user_{i}"
            self.ai_system.start_conversation(user_id)
            
            # Measure response time
            start_time = time.time()
            try:
                response = self.ai_system.process_message(query_data['query'], user_id)
                processing_time = time.time() - start_time
                
                # Analyze response
                analysis = self._analyze_response(
                    query_data['query'], 
                    response, 
                    query_data['expected_district'],
                    query_data['type'],
                    processing_time
                )
                
                results["responses"].append(analysis)
                results["quality_distribution"][analysis.quality_level.value] += 1
                
                # Count deep learning features
                for feature in analysis.deep_learning_features:
                    results["deep_learning_features"][feature.value] += 1
                
                # Update district coverage
                if analysis.district_mentioned:
                    district_key = analysis.district_mentioned.lower()
                    if district_key in results["district_coverage"]:
                        results["district_coverage"][district_key]["queries"] += 1
                        if results["district_coverage"][district_key]["avg_quality"] == 0:
                            results["district_coverage"][district_key]["avg_quality"] = analysis.quality_score
                        else:
                            # Update running average
                            current_avg = results["district_coverage"][district_key]["avg_quality"]
                            query_count = results["district_coverage"][district_key]["queries"]
                            new_avg = ((current_avg * (query_count - 1)) + analysis.quality_score) / query_count
                            results["district_coverage"][district_key]["avg_quality"] = new_avg
                
                # Print quick result
                quality_emoji = self._get_quality_emoji(analysis.quality_level)
                dl_features_count = len(analysis.deep_learning_features)
                print(f"   {quality_emoji} Quality: {analysis.quality_level.value.title()} ({analysis.quality_score:.1f}%) | DL Features: {dl_features_count} | Length: {analysis.length}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Create failed analysis
                analysis = ResponseAnalysis(
                    query=query_data['query'],
                    response="ERROR",
                    length=0,
                    quality_score=0,
                    quality_level=ResponseQuality.POOR,
                    friendliness_score=0,
                    readability_score=0,
                    correctness_score=0,
                    deep_learning_features=[],
                    district_mentioned=None,
                    processing_time=0,
                    has_personalization=False,
                    cultural_depth=0,
                    narrative_quality=0
                )
                results["responses"].append(analysis)
                results["quality_distribution"][ResponseQuality.POOR.value] += 1
        
        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results)
        
        # Generate improvement recommendations
        results["improvement_recommendations"] = self._generate_improvement_recommendations(results)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _analyze_response(self, query: str, response: str, expected_district: str, 
                         query_type: str, processing_time: float) -> ResponseAnalysis:
        """Analyze a single response for quality and deep learning features"""
        
        # Basic metrics
        length = len(response)
        
        # Detect deep learning features
        dl_features = self._detect_deep_learning_features(response, query_type)
        
        # Calculate quality scores
        quality_score = self._calculate_quality_score(response, query, expected_district, query_type)
        friendliness_score = self._calculate_friendliness_score(response)
        readability_score = self._calculate_readability_score(response)
        correctness_score = self._calculate_correctness_score(response, expected_district)
        
        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        # Detect district mentioned
        district_mentioned = self._detect_district_mention(response)
        
        # Check personalization
        has_personalization = self._has_personalization(response)
        
        # Cultural and narrative depth
        cultural_depth = self._calculate_cultural_depth(response)
        narrative_quality = self._calculate_narrative_quality(response)
        
        return ResponseAnalysis(
            query=query,
            response=response,
            length=length,
            quality_score=quality_score,
            quality_level=quality_level,
            friendliness_score=friendliness_score,
            readability_score=readability_score,
            correctness_score=correctness_score,
            deep_learning_features=dl_features,
            district_mentioned=district_mentioned,
            processing_time=processing_time,
            has_personalization=has_personalization,
            cultural_depth=cultural_depth,
            narrative_quality=narrative_quality
        )
    
    def _detect_deep_learning_features(self, response: str, query_type: str) -> List[DeepLearningFeature]:
        """Detect deep learning features in response"""
        features = []
        response_lower = response.lower()
        
        # Cultural Insights
        cultural_keywords = ['culture', 'traditional', 'authentic', 'local customs', 'etiquette', 'way of life']
        if any(keyword in response_lower for keyword in cultural_keywords):
            features.append(DeepLearningFeature.CULTURAL_INSIGHTS)
        
        # Seasonal Context
        seasonal_keywords = ['season', 'spring', 'summer', 'autumn', 'winter', 'weather', 'seasonal']
        if any(keyword in response_lower for keyword in seasonal_keywords):
            features.append(DeepLearningFeature.SEASONAL_CONTEXT)
        
        # Personalized Recommendations
        personal_keywords = ['based on your', 'for you', 'personally', 'recommendation', 'perfect for']
        if any(keyword in response_lower for keyword in personal_keywords):
            features.append(DeepLearningFeature.PERSONALIZED_RECOMMENDATIONS)
        
        # Historical Anecdotes
        historical_keywords = ['history', 'historical', 'centuries', 'ottoman', 'byzantine', 'built in']
        if any(keyword in response_lower for keyword in historical_keywords):
            features.append(DeepLearningFeature.HISTORICAL_ANECDOTES)
        
        # Hidden Gems Narratives
        gems_keywords = ['hidden gem', 'secret', 'insider', 'locals know', 'off the beaten path']
        if any(keyword in response_lower for keyword in gems_keywords):
            features.append(DeepLearningFeature.HIDDEN_GEMS_NARRATIVES)
        
        # Local Expertise
        expertise_keywords = ['local favorite', 'insider tip', 'locals prefer', 'authentic experience']
        if any(keyword in response_lower for keyword in expertise_keywords):
            features.append(DeepLearningFeature.LOCAL_EXPERTISE)
        
        # Neighborhood Character
        character_keywords = ['character', 'atmosphere', 'vibe', 'personality', 'unique']
        if any(keyword in response_lower for keyword in character_keywords):
            features.append(DeepLearningFeature.NEIGHBORHOOD_CHARACTER)
        
        # Visitor Type Matching
        visitor_keywords = ['photographer', 'family', 'young traveler', 'cultural explorer', 'visitor type']
        if any(keyword in response_lower for keyword in visitor_keywords):
            features.append(DeepLearningFeature.VISITOR_TYPE_MATCHING)
        
        return features
    
    def _calculate_quality_score(self, response: str, query: str, expected_district: str, query_type: str) -> float:
        """Calculate overall quality score (0-100)"""
        score = 0
        
        # Length adequacy (0-20 points)
        if len(response) > 500:
            score += 20
        elif len(response) > 200:
            score += 15
        elif len(response) > 100:
            score += 10
        elif len(response) > 50:
            score += 5
        
        # District relevance (0-25 points)
        district_mentioned = self._detect_district_mention(response)
        if district_mentioned and district_mentioned.lower() == expected_district.lower():
            score += 25
        elif district_mentioned:
            score += 15
        elif expected_district.lower() in response.lower():
            score += 10
        
        # Content depth (0-20 points)
        depth_indicators = ['character', 'atmosphere', 'experience', 'recommendation', 'tip', 'insight']
        depth_count = sum(1 for indicator in depth_indicators if indicator in response.lower())
        score += min(depth_count * 4, 20)
        
        # Specificity (0-15 points)
        specific_indicators = ['street', 'address', 'time', 'price', 'season', 'best', 'avoid']
        specific_count = sum(1 for indicator in specific_indicators if indicator in response.lower())
        score += min(specific_count * 3, 15)
        
        # Helpfulness (0-10 points)
        helpful_indicators = ['recommend', 'suggest', 'tip', 'advice', 'should', 'consider']
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response.lower())
        score += min(helpful_count * 2, 10)
        
        # Engagement (0-10 points)
        if '!' in response or '?' in response:
            score += 5
        if len(response.split('.')) > 3:  # Multiple sentences
            score += 5
        
        return min(score, 100)
    
    def _calculate_friendliness_score(self, response: str) -> float:
        """Calculate friendliness score (0-100)"""
        score = 50  # Base score
        
        # Positive indicators
        friendly_words = ['welcome', 'love', 'enjoy', 'wonderful', 'amazing', 'great', 'perfect', 'happy']
        score += sum(5 for word in friendly_words if word in response.lower())
        
        # Exclamation marks (engagement)
        score += min(response.count('!') * 3, 15)
        
        # Personal touch
        personal_words = ['you', 'your', 'yourself']
        score += sum(2 for word in personal_words if word in response.lower())
        
        # Negative indicators
        if response.isupper():
            score -= 20
        if 'no' in response.lower() and 'not' in response.lower():
            score -= 10
        
        return min(max(score, 0), 100)
    
    def _calculate_readability_score(self, response: str) -> float:
        """Calculate readability score (0-100)"""
        if not response or len(response) < 10:
            return 0
        
        sentences = response.split('.')
        words = response.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 50
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Readability scoring
        if 10 <= avg_words_per_sentence <= 20:
            readability = 90
        elif 8 <= avg_words_per_sentence <= 25:
            readability = 80
        elif 6 <= avg_words_per_sentence <= 30:
            readability = 70
        else:
            readability = 60
        
        # Bonus for structure
        if '**' in response or '•' in response or '\n' in response:
            readability += 10
        
        return min(readability, 100)
    
    def _calculate_correctness_score(self, response: str, expected_district: str) -> float:
        """Calculate correctness score (0-100)"""
        score = 70  # Base score assuming general correctness
        
        # District accuracy
        if expected_district.lower() in response.lower():
            score += 20
        
        # Factual accuracy indicators
        accuracy_indicators = ['istanbul', 'turkey', 'turkish', 'bosphorus']
        for indicator in accuracy_indicators:
            if indicator in response.lower():
                score += 2
        
        # Penalize obvious errors
        error_indicators = ['lorem ipsum', 'error', 'not found', 'undefined']
        for error in error_indicators:
            if error in response.lower():
                score -= 30
        
        return min(max(score, 0), 100)
    
    def _determine_quality_level(self, quality_score: float) -> ResponseQuality:
        """Determine quality level from score"""
        if quality_score >= 90:
            return ResponseQuality.EXCELLENT
        elif quality_score >= 80:
            return ResponseQuality.VERY_GOOD
        elif quality_score >= 70:
            return ResponseQuality.GOOD
        elif quality_score >= 60:
            return ResponseQuality.FAIR
        else:
            return ResponseQuality.POOR
    
    def _detect_district_mention(self, response: str) -> Optional[str]:
        """Detect which district is mentioned in response"""
        response_lower = response.lower()
        
        district_variations = {
            'şişli': ['şişli', 'sisli'],
            'beşiktaş': ['beşiktaş', 'besiktas'],
            'fatih': ['fatih'],
            'kadıköy': ['kadıköy', 'kadikoy'],
            'üsküdar': ['üsküdar', 'uskudar'],
            'sarıyer': ['sarıyer', 'sariyer']
        }
        
        for district, variations in district_variations.items():
            if any(var in response_lower for var in variations):
                return district
        
        return None
    
    def _has_personalization(self, response: str) -> bool:
        """Check if response has personalization elements"""
        personal_indicators = [
            'based on your', 'for you', 'personally', 'your interests',
            'perfect for', 'recommended for you', 'tailored'
        ]
        return any(indicator in response.lower() for indicator in personal_indicators)
    
    def _calculate_cultural_depth(self, response: str) -> int:
        """Calculate cultural depth score (0-10)"""
        cultural_elements = [
            'culture', 'tradition', 'heritage', 'customs', 'authentic',
            'local way', 'etiquette', 'values', 'beliefs', 'lifestyle'
        ]
        return min(sum(1 for element in cultural_elements if element in response.lower()), 10)
    
    def _calculate_narrative_quality(self, response: str) -> int:
        """Calculate narrative quality score (0-10)"""
        narrative_elements = [
            'story', 'history', 'tale', 'legend', 'once', 'built',
            'centuries', 'founded', 'established', 'tradition'
        ]
        return min(sum(1 for element in narrative_elements if element in response.lower()), 10)
    
    def _get_quality_emoji(self, quality_level: ResponseQuality) -> str:
        """Get emoji for quality level"""
        quality_emojis = {
            ResponseQuality.EXCELLENT: "🌟",
            ResponseQuality.VERY_GOOD: "✅",
            ResponseQuality.GOOD: "👍",
            ResponseQuality.FAIR: "⚠️",
            ResponseQuality.POOR: "❌"
        }
        return quality_emojis.get(quality_level, "❓")
    
    def _calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        responses = results["responses"]
        
        if not responses:
            return {}
        
        # Quality metrics
        avg_quality = sum(r.quality_score for r in responses) / len(responses)
        avg_friendliness = sum(r.friendliness_score for r in responses) / len(responses)
        avg_readability = sum(r.readability_score for r in responses) / len(responses)
        avg_correctness = sum(r.correctness_score for r in responses) / len(responses)
        avg_processing_time = sum(r.processing_time for r in responses) / len(responses)
        
        # Deep learning metrics
        total_dl_features = sum(len(r.deep_learning_features) for r in responses)
        avg_dl_features = total_dl_features / len(responses)
        
        # Content metrics
        avg_length = sum(r.length for r in responses) / len(responses)
        avg_cultural_depth = sum(r.cultural_depth for r in responses) / len(responses)
        avg_narrative_quality = sum(r.narrative_quality for r in responses) / len(responses)
        
        # Personalization rate
        personalized_responses = sum(1 for r in responses if r.has_personalization)
        personalization_rate = (personalized_responses / len(responses)) * 100
        
        # Quality distribution percentages
        quality_percentages = {}
        for quality in ResponseQuality:
            count = results["quality_distribution"][quality.value]
            percentage = (count / len(responses)) * 100
            quality_percentages[quality.value] = percentage
        
        return {
            "avg_quality_score": avg_quality,
            "avg_friendliness_score": avg_friendliness,
            "avg_readability_score": avg_readability,
            "avg_correctness_score": avg_correctness,
            "avg_processing_time": avg_processing_time,
            "avg_dl_features_per_response": avg_dl_features,
            "total_dl_features_detected": total_dl_features,
            "avg_response_length": avg_length,
            "avg_cultural_depth": avg_cultural_depth,
            "avg_narrative_quality": avg_narrative_quality,
            "personalization_rate": personalization_rate,
            "quality_percentages": quality_percentages
        }
    
    def _generate_improvement_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        metrics = results["overall_metrics"]
        
        # Quality improvements
        if metrics["avg_quality_score"] < 75:
            recommendations.append("🎯 Increase response depth and specificity to improve overall quality scores")
        
        # Deep learning features
        if metrics["avg_dl_features_per_response"] < 3:
            recommendations.append("🧠 Enhance deep learning features: Add more cultural insights, seasonal context, and personalized recommendations")
        
        # Personalization
        if metrics["personalization_rate"] < 40:
            recommendations.append("👤 Improve personalization: Tailor suggestions based on user intent, season, and interests")
        
        # District coverage
        poor_districts = []
        for district, data in results["district_coverage"].items():
            if data["avg_quality"] < 70:
                poor_districts.append(district)
        
        if poor_districts:
            recommendations.append(f"🗺️ Enhance district coverage for: {', '.join(poor_districts)} - Add more specific local knowledge")
        
        # Content depth
        if metrics["avg_cultural_depth"] < 4:
            recommendations.append("🏛️ Add more cultural insights and historical anecdotes to responses")
        
        if metrics["avg_narrative_quality"] < 3:
            recommendations.append("📖 Expand hidden gems into mini-narratives with storytelling elements")
        
        # Quality distribution
        good_plus_percentage = (
            metrics["quality_percentages"]["good"] + 
            metrics["quality_percentages"]["very_good"] + 
            metrics["quality_percentages"]["excellent"]
        )
        
        if good_plus_percentage < 60:
            recommendations.append("📈 Target: Shift 30-40% more responses to Good or Very Good quality range")
        
        # Processing time
        if metrics["avg_processing_time"] > 2:
            recommendations.append("⚡ Optimize response time to under 2 seconds for better user experience")
        
        return recommendations
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive analysis report"""
        
        print(f"\n" + "="*80)
        print("🎯 ISTANBUL DISTRICT TESTING - COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        metrics = results["overall_metrics"]
        
        # Overall Performance Summary
        print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
        print("-" * 50)
        print(f"  📝 Total Queries Tested: {results['total_queries']}")
        print(f"  🏘️ Districts Covered: {len(results['districts_tested'])}")
        print(f"  ⭐ Average Quality Score: {metrics['avg_quality_score']:.1f}%")
        print(f"  😊 Average Friendliness: {metrics['avg_friendliness_score']:.1f}%")
        print(f"  📖 Average Readability: {metrics['avg_readability_score']:.1f}%")
        print(f"  ✅ Average Correctness: {metrics['avg_correctness_score']:.1f}%")
        print(f"  ⚡ Average Response Time: {metrics['avg_processing_time']:.2f}s")
        print(f"  📏 Average Response Length: {metrics['avg_response_length']:.0f} characters")
        
        # Quality Distribution
        print(f"\n🎯 QUALITY DISTRIBUTION:")
        print("-" * 30)
        quality_order = [ResponseQuality.EXCELLENT, ResponseQuality.VERY_GOOD, ResponseQuality.GOOD, ResponseQuality.FAIR, ResponseQuality.POOR]
        for quality in quality_order:
            percentage = metrics["quality_percentages"][quality.value]
            emoji = self._get_quality_emoji(quality)
            print(f"  {emoji} {quality.value.replace('_', ' ').title()}: {percentage:.1f}%")
        
        # Deep Learning Features Analysis
        print(f"\n🧠 DEEP LEARNING FEATURES ANALYSIS:")
        print("-" * 40)
        print(f"  🔢 Total Features Detected: {metrics['total_dl_features_detected']}")
        print(f"  📈 Average per Response: {metrics['avg_dl_features_per_response']:.1f}")
        print(f"  👤 Personalization Rate: {metrics['personalization_rate']:.1f}%")
        
        feature_counts = results["deep_learning_features"]
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  🎯 Feature Usage Breakdown:")
        for feature, count in sorted_features:
            percentage = (count / results['total_queries']) * 100
            print(f"    • {feature.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # District Coverage Analysis
        print(f"\n🗺️ DISTRICT COVERAGE ANALYSIS:")
        print("-" * 35)
        
        for district in results['districts_tested']:
            data = results['district_coverage'][district]
            if data['queries'] > 0:
                print(f"  🏘️ {district.title()}:")
                print(f"    • Queries: {data['queries']}")
                print(f"    • Avg Quality: {data['avg_quality']:.1f}%")
                
                # Quality assessment
                if data['avg_quality'] >= 80:
                    status = "✅ Excellent"
                elif data['avg_quality'] >= 70:
                    status = "👍 Good"
                elif data['avg_quality'] >= 60:
                    status = "⚠️ Needs Improvement"
                else:
                    status = "❌ Requires Enhancement"
                print(f"    • Status: {status}")
        
        # Content Quality Analysis
        print(f"\n📚 CONTENT QUALITY ANALYSIS:")
        print("-" * 35)
        print(f"  🏛️ Cultural Depth: {metrics['avg_cultural_depth']:.1f}/10")
        print(f"  📖 Narrative Quality: {metrics['avg_narrative_quality']:.1f}/10")
        
        # Improvement Recommendations
        print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS:")
        print("-" * 40)
        
        for i, recommendation in enumerate(results['improvement_recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        # Action Items
        print(f"\n🚀 PRIORITY ACTION ITEMS:")
        print("-" * 30)
        
        # Generate priority actions based on results
        priority_actions = []
        
        if metrics['avg_dl_features_per_response'] < 3:
            priority_actions.append("HIGH: Implement enhanced deep learning feature detection")
        
        if metrics['personalization_rate'] < 40:
            priority_actions.append("HIGH: Develop advanced personalization algorithms")
        
        # Check for poor districts
        poor_districts = [d for d, data in results['district_coverage'].items() if data['avg_quality'] < 70]
        if poor_districts:
            priority_actions.append(f"MEDIUM: Enhance content for {', '.join(poor_districts)}")
        
        if metrics['avg_quality_score'] < 75:
            priority_actions.append("MEDIUM: Improve overall response quality and depth")
        
        if metrics['avg_cultural_depth'] < 4:
            priority_actions.append("LOW: Add more cultural context and historical narratives")
        
        for i, action in enumerate(priority_actions, 1):
            print(f"  {i}. {action}")
        
        # Final Assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        print("-" * 25)
        
        # Calculate overall grade
        overall_score = (
            metrics['avg_quality_score'] * 0.4 +
            metrics['avg_friendliness_score'] * 0.2 +
            metrics['avg_readability_score'] * 0.2 +
            metrics['avg_correctness_score'] * 0.2
        )
        
        if overall_score >= 90:
            grade = "A+ (Excellent)"
            status = "🟢 Production Ready"
        elif overall_score >= 80:
            grade = "A (Very Good)"
            status = "🟡 Minor Improvements Needed"
        elif overall_score >= 70:
            grade = "B+ (Good)"
            status = "🟠 Moderate Improvements Needed"
        elif overall_score >= 60:
            grade = "B (Fair)"
            status = "🔴 Significant Improvements Needed"
        else:
            grade = "C (Needs Work)"
            status = "🔴 Major Overhaul Required"
        
        print(f"  🎯 Overall Score: {overall_score:.1f}%")
        print(f"  📊 System Grade: {grade}")
        print(f"  🚦 Status: {status}")
        
        print(f"\n" + "="*80)
        print("🎉 ANALYSIS COMPLETE - DETAILED RESULTS GENERATED")
        print("="*80)

async def main():
    """Main execution function"""
    if not SYSTEMS_AVAILABLE:
        print("❌ Required systems not available. Please check imports.")
        return
    
    try:
        # Initialize testing suite
        test_suite = IstanbulDistrictTestingSuite()
        
        # Run comprehensive analysis
        results = await test_suite.run_comprehensive_analysis()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"district_analysis_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            "timestamp": timestamp,
            "total_queries": results["total_queries"],
            "districts_tested": results["districts_tested"],
            "quality_distribution": results["quality_distribution"],
            "deep_learning_features": results["deep_learning_features"],
            "district_coverage": results["district_coverage"],
            "overall_metrics": results["overall_metrics"],
            "improvement_recommendations": results["improvement_recommendations"]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
