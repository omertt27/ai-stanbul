#!/usr/bin/env python3
"""
Istanbul AI Response Analysis Tool
Analyzes AI responses for readability, correctness, and completeness
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from istanbul_ai.core.user_profile import UserProfile, UserType
from istanbul_daily_talk_system import IstanbulDailyTalkAI

class ResponseAnalyzer:
    """Analyzes AI responses for quality metrics"""
    
    def __init__(self):
        print("🔄 Initializing AI system...")
        self.ai_system = IstanbulDailyTalkAI()
        print("✅ AI system ready!")
        
    def analyze_response_quality(self, response: str, query_type: str) -> Dict[str, Any]:
        """Comprehensive analysis of response quality"""
        
        # Basic metrics
        response_lower = response.lower()
        word_count = len(response.split())
        char_count = len(response)
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        # Readability metrics
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        # Content analysis
        istanbul_districts = [
            "sultanahmet", "beyoglu", "taksim", "galata", "karakoy", "besiktas", 
            "eminonu", "kadikoy", "uskudar", "fatih", "sisli", "ortakoy", "bakirkoy"
        ]
        districts_mentioned = [d for d in istanbul_districts if d in response_lower]
        
        landmarks = [
            "hagia sophia", "blue mosque", "topkapi", "galata tower", "grand bazaar", 
            "basilica cistern", "dolmabahce", "bosphorus", "maiden's tower", 
            "spice bazaar", "istiklal", "taksim square", "golden horn"
        ]
        landmarks_mentioned = [l for l in landmarks if l in response_lower]
        
        # Practical information indicators
        practical_keywords = [
            "opening hours", "open", "closes", "entrance fee", "free", "ticket", 
            "metro", "tram", "bus", "ferry", "walk", "address", "location", 
            "reservation", "book", "website", "phone"
        ]
        practical_info_count = sum(1 for keyword in practical_keywords if keyword in response_lower)
        
        # Recommendation quality
        recommendation_words = [
            "recommend", "suggest", "try", "visit", "see", "check out", 
            "must see", "don't miss", "worth", "beautiful", "amazing", "stunning"
        ]
        recommendation_count = sum(1 for word in recommendation_words if word in response_lower)
        
        # Structure analysis
        has_bullets_or_numbers = ('•' in response or 
                                 any(f"{i}." in response for i in range(1, 10)) or
                                 any(f"{i})" in response for i in range(1, 10)))
        
        has_emojis = any(ord(char) > 127 for char in response)
        
        # Correctness indicators (basic)
        has_greeting = any(word in response_lower for word in ["hello", "hi", "good", "welcome"])
        has_helpful_tone = any(word in response_lower for word in [
            "happy", "excited", "help", "pleasure", "glad", "love"
        ])
        
        # Calculate overall quality score
        quality_score = self.calculate_quality_score(
            word_count, districts_mentioned, landmarks_mentioned, 
            practical_info_count, recommendation_count, has_bullets_or_numbers,
            has_helpful_tone, query_type
        )
        
        return {
            "basic_metrics": {
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 1)
            },
            "content_analysis": {
                "districts_mentioned": len(districts_mentioned),
                "districts_list": districts_mentioned,
                "landmarks_mentioned": len(landmarks_mentioned),
                "landmarks_list": landmarks_mentioned,
                "practical_info_indicators": practical_info_count,
                "recommendation_indicators": recommendation_count
            },
            "structure_quality": {
                "has_structure": has_bullets_or_numbers,
                "has_emojis": has_emojis,
                "has_greeting": has_greeting,
                "has_helpful_tone": has_helpful_tone
            },
            "readability": {
                "appropriate_length": 50 <= word_count <= 300,
                "good_sentence_length": 10 <= avg_words_per_sentence <= 20,
                "well_structured": has_bullets_or_numbers or sentence_count >= 3
            },
            "overall_quality_score": quality_score,
            "grade": self.get_quality_grade(quality_score)
        }
    
    def calculate_quality_score(self, word_count: int, districts: List[str], 
                               landmarks: List[str], practical_info: int, 
                               recommendations: int, structured: bool,
                               helpful_tone: bool, query_type: str) -> float:
        """Calculate overall quality score (0-100)"""
        score = 0
        
        # Length appropriateness (20 points)
        if 50 <= word_count <= 300:
            score += 20
        elif 30 <= word_count <= 400:
            score += 15
        elif word_count >= 20:
            score += 10
        
        # Content relevance (30 points)
        score += min(len(districts) * 5, 15)  # Up to 15 for districts
        score += min(len(landmarks) * 3, 15)  # Up to 15 for landmarks
        
        # Practical information (20 points)
        score += min(practical_info * 3, 20)
        
        # Recommendation quality (15 points)
        score += min(recommendations * 2, 15)
        
        # Structure and tone (15 points)
        if structured:
            score += 8
        if helpful_tone:
            score += 7
        
        return min(score, 100)
    
    def get_quality_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B (Good)"
        elif score >= 60:
            return "C (Acceptable)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Needs Improvement)"
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis of different query types"""
        
        test_queries = [
            {
                "query": "I want to visit famous museums in Istanbul",
                "type": "museums",
                "user_prefs": {"interests": ["history", "art"], "user_type": UserType.CULTURAL_ENTHUSIAST}
            },
            {
                "query": "Show me beautiful mosques and religious places",
                "type": "religious_sites",
                "user_prefs": {"interests": ["architecture", "religion"], "user_type": UserType.CULTURAL_ENTHUSIAST}
            },
            {
                "query": "We have kids aged 5 and 8, what family-friendly places do you suggest?",
                "type": "family",
                "user_prefs": {"interests": ["family_fun"], "user_type": UserType.FIRST_TIME_VISITOR}
            },
            {
                "query": "I'm on a tight budget, what free attractions can I enjoy?",
                "type": "budget",
                "user_prefs": {"interests": ["free_activities"], "user_type": UserType.BUDGET_TRAVELER}
            },
            {
                "query": "What can I see in the historic Sultanahmet area?",
                "type": "district",
                "user_prefs": {"interests": ["history"], "current_location": "Sultanahmet"}
            },
            {
                "query": "It's raining, what indoor attractions do you recommend?",
                "type": "weather",
                "user_prefs": {"interests": ["culture", "shopping"], "weather_sensitive": True}
            },
            {
                "query": "Planning a romantic day with my partner, suggest beautiful spots",
                "type": "romantic",
                "user_prefs": {"interests": ["romance"], "group_type": "couple"}
            },
            {
                "query": "I love Turkish food and culture, where should I go?",
                "type": "food_culture",
                "user_prefs": {"interests": ["food", "culture"], "user_type": UserType.FOODIE}
            }
        ]
        
        print("🚀 Starting Comprehensive Response Analysis")
        print("=" * 70)
        
        results = []
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {test['type'].upper()}")
            print(f"Query: {test['query']}")
            print("-" * 50)
            
            try:
                # Create user profile
                user_id = f"analysis_user_{i}"
                user_profile = self.ai_system.get_or_create_user_profile(user_id)
                
                # Apply preferences
                prefs = test['user_prefs']
                user_profile.interests = prefs.get('interests', [])
                user_profile.user_type = prefs.get('user_type', UserType.FIRST_TIME_VISITOR)
                
                if 'current_location' in prefs:
                    user_profile.current_location = prefs['current_location']
                if 'group_type' in prefs:
                    user_profile.group_type = prefs['group_type']
                
                # Get AI response
                start_time = time.time()
                response = self.ai_system.process_message(test['query'], user_id)
                response_time = time.time() - start_time
                
                # Analyze response
                analysis = self.analyze_response_quality(response, test['type'])
                
                result = {
                    "test_id": i,
                    "query_type": test['type'],
                    "query": test['query'],
                    "response": response,
                    "response_time_seconds": round(response_time, 2),
                    "analysis": analysis
                }
                
                results.append(result)
                
                # Print analysis summary
                print(f"✅ Response generated in {response_time:.1f}s")
                print(f"📊 Quality Score: {analysis['overall_quality_score']:.1f}/100 ({analysis['grade']})")
                print(f"📝 Length: {analysis['basic_metrics']['word_count']} words")
                print(f"🗺️ Districts: {analysis['content_analysis']['districts_mentioned']}")
                print(f"🏛️ Landmarks: {analysis['content_analysis']['landmarks_mentioned']}")
                print(f"💡 Practical Info: {analysis['content_analysis']['practical_info_indicators']} indicators")
                
                # Show response preview
                if len(response) > 150:
                    print(f"📖 Preview: {response[:150]}...")
                else:
                    print(f"📖 Response: {response}")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results.append({
                    "test_id": i,
                    "query_type": test['type'],
                    "query": test['query'],
                    "error": str(e),
                    "status": "failed"
                })
            
            time.sleep(1)  # Brief pause between tests
        
        # Generate comprehensive report
        self.generate_analysis_report(results)
        
        return results
    
    def generate_analysis_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive analysis report"""
        
        successful_results = [r for r in results if 'analysis' in r]
        failed_results = [r for r in results if 'error' in r]
        
        if not successful_results:
            print("\n❌ No successful responses to analyze!")
            return
        
        # Calculate overall statistics
        scores = [r['analysis']['overall_quality_score'] for r in successful_results]
        avg_score = sum(scores) / len(scores)
        
        word_counts = [r['analysis']['basic_metrics']['word_count'] for r in successful_results]
        avg_words = sum(word_counts) / len(word_counts)
        
        total_districts = sum(r['analysis']['content_analysis']['districts_mentioned'] for r in successful_results)
        total_landmarks = sum(r['analysis']['content_analysis']['landmarks_mentioned'] for r in successful_results)
        
        structured_responses = sum(1 for r in successful_results if r['analysis']['structure_quality']['has_structure'])
        helpful_tone_responses = sum(1 for r in successful_results if r['analysis']['structure_quality']['has_helpful_tone'])
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   • Total Tests: {len(results)}")
        print(f"   • Successful: {len(successful_results)}")
        print(f"   • Failed: {len(failed_results)}")
        print(f"   • Success Rate: {(len(successful_results)/len(results)*100):.1f}%")
        
        print(f"\n⭐ QUALITY METRICS:")
        print(f"   • Average Quality Score: {avg_score:.1f}/100")
        print(f"   • Best Score: {max(scores):.1f}/100")
        print(f"   • Worst Score: {min(scores):.1f}/100")
        print(f"   • Grade Distribution:")
        
        grade_counts = {}
        for result in successful_results:
            grade = result['analysis']['grade'].split()[0]  # Get letter grade only
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        for grade, count in sorted(grade_counts.items()):
            print(f"     - {grade}: {count} responses")
        
        print(f"\n📝 CONTENT ANALYSIS:")
        print(f"   • Average Response Length: {avg_words:.1f} words")
        print(f"   • Total Districts Mentioned: {total_districts}")
        print(f"   • Total Landmarks Mentioned: {total_landmarks}")
        print(f"   • Well-Structured Responses: {structured_responses}/{len(successful_results)} ({structured_responses/len(successful_results)*100:.1f}%)")
        print(f"   • Helpful Tone: {helpful_tone_responses}/{len(successful_results)} ({helpful_tone_responses/len(successful_results)*100:.1f}%)")
        
        # Best and worst performing queries
        best_result = max(successful_results, key=lambda x: x['analysis']['overall_quality_score'])
        worst_result = min(successful_results, key=lambda x: x['analysis']['overall_quality_score'])
        
        print(f"\n🏆 BEST PERFORMING QUERY:")
        print(f"   • Type: {best_result['query_type']}")
        print(f"   • Score: {best_result['analysis']['overall_quality_score']:.1f}/100")
        print(f"   • Query: {best_result['query']}")
        
        print(f"\n⚠️ LOWEST PERFORMING QUERY:")
        print(f"   • Type: {worst_result['query_type']}")
        print(f"   • Score: {worst_result['analysis']['overall_quality_score']:.1f}/100")
        print(f"   • Query: {worst_result['query']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
        
        if avg_score < 75:
            print("   • Overall quality needs improvement")
        
        low_content_responses = [r for r in successful_results 
                               if r['analysis']['content_analysis']['districts_mentioned'] == 0]
        if low_content_responses:
            print(f"   • {len(low_content_responses)} responses lack district mentions")
        
        short_responses = [r for r in successful_results 
                          if r['analysis']['basic_metrics']['word_count'] < 50]
        if short_responses:
            print(f"   • {len(short_responses)} responses are too short")
        
        unstructured_responses = len(successful_results) - structured_responses
        if unstructured_responses > len(successful_results) * 0.3:
            print(f"   • {unstructured_responses} responses lack proper structure")
        
        print("\n✅ Analysis completed!")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"response_analysis_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "summary": {
                    "total_tests": len(results),
                    "successful_tests": len(successful_results),
                    "average_quality_score": avg_score,
                    "average_word_count": avg_words,
                    "total_districts_mentioned": total_districts,
                    "total_landmarks_mentioned": total_landmarks
                },
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Detailed results saved to: {results_file}")

def main():
    """Main function"""
    analyzer = ResponseAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n🎉 Response analysis completed!")
    print(f"Analyzed {len(results)} different query types")

if __name__ == "__main__":
    main()
