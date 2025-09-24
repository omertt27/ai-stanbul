#!/usr/bin/env python3
"""
Comprehensive Final Test Suite for AI-stanbul Chatbot
Tests 75+ diverse queries across transportation, districts, museums, restaurants, and tips
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import os

# Test queries organized by category
TEST_QUERIES = {
    "transportation": [
        "How do I get from Sultanahmet to Taksim Square?",
        "What are the metro lines in Istanbul?",
        "How much does a taxi cost from Atatürk Airport to city center?",
        "Tell me about the ferry routes to Princes' Islands",
        "How do I use the Istanbulkart for public transport?",
        "What's the best way to get from Sabiha Gökçen Airport to Sultanahmet?",
        "Are there night buses in Istanbul?",
        "How do I get to Büyükada island?",
        "What's the difference between metrobus and metro?",
        "How to get from Kadıköy to Beyoğlu?",
        "Tell me about the Marmaray train line",
        "How much is a ferry ticket to Asian side?",
        "Best way to travel from Levent to Eminönü?",
        "What time does the last metro run?",
        "How to get to Istanbul Modern museum by public transport?"
    ],
    
    "districts": [
        "What can I do in Sultanahmet district?",
        "Tell me about Beyoğlu neighborhood",
        "What's special about Kadıköy?",
        "Describe the Galata area",
        "What to see in Üsküdar?",
        "Tell me about Beşiktaş district",
        "What's in the Fatih area?",
        "Describe Şişli neighborhood",
        "What can I find in Ortaköy?",
        "Tell me about Balat district",
        "What's special about Fener area?",
        "Describe the Eyüp district",
        "What to do in Bakırköy?",
        "Tell me about Nişantaşı area",
        "What's in Bebek neighborhood?"
    ],
    
    "museums": [
        "Tell me about Topkapi Palace",
        "What can I see at Hagia Sophia?",
        "Describe the Archaeological Museums",
        "What's at the Istanbul Modern?",
        "Tell me about Dolmabahçe Palace",
        "What can I find at the Basilica Cistern?",
        "Describe the Pera Museum",
        "What's at the Rahmi M. Koç Museum?",
        "Tell me about the Turkish and Islamic Arts Museum",
        "What can I see at Galata Tower?",
        "Describe the Chora Museum",
        "What's at the Naval Museum?",
        "Tell me about the Sakıp Sabancı Museum",
        "What can I find at the Grand Bazaar?",
        "Describe the Blue Mosque interior"
    ],
    
    "restaurants": [
        "Best restaurants in Sultanahmet",
        "Where to eat traditional Turkish breakfast?",
        "Recommend kebab restaurants in Istanbul",
        "Best seafood restaurants on Bosphorus",
        "Where to find authentic Ottoman cuisine?",
        "Best rooftop restaurants with view?",
        "Recommend vegetarian restaurants in Istanbul",
        "Where to eat in Karaköy district?",
        "Best meze restaurants in Beyoğlu?",
        "Where to find the best baklava?",
        "Recommend restaurants in Kadıköy?",
        "Best Turkish coffee houses?",
        "Where to eat fresh fish sandwich?",
        "Recommend fine dining restaurants?",
        "Best street food in Istanbul?",
        "Where to find authentic döner kebab?",
        "Best restaurants with Bosphorus view?",
        "Recommend Turkish delight shops?"
    ],
    
    "tips": [
        "What should I know before visiting Istanbul?",
        "How much should I tip in restaurants?",
        "What to wear when visiting mosques?",
        "Best time to visit Istanbul?",
        "How to bargain in Grand Bazaar?",
        "Is tap water safe to drink?",
        "What are common Turkish phrases for tourists?",
        "How much cash should I carry?",
        "What's the currency in Turkey?",
        "Best areas to stay for tourists?",
        "How to avoid tourist traps?",
        "What souvenirs to buy in Istanbul?",
        "How to respect local customs?",
        "Best way to exchange money?",
        "What to do in case of emergency?",
        "How to use Turkish toilets?",
        "What foods to try in Istanbul?",
        "How to stay safe in Istanbul?"
    ]
}

class ComprehensiveTestRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def setup_session(self) -> None:
        """Setup aiohttp session"""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
    
    async def cleanup_session(self):
        """Cleanup aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def test_query(self, category: str, query: str, query_index: int) -> Dict[str, Any]:
        """Test a single query and return results"""
        start_time = time.time()
        
        # Ensure session is initialized
        if self.session is None:
            await self.setup_session()
        
        # At this point, session should definitely be initialized
        assert self.session is not None, "Session should be initialized"
        
        try:
            # Test the streaming endpoint
            async with self.session.post(
                f"{self.base_url}/ai/stream",
                json={"user_input": query}
            ) as response:
                
                if response.status != 200:
                    return {
                        "category": category,
                        "query": query,
                        "query_index": query_index,
                        "success": False,
                        "error": f"HTTP {response.status}",
                        "response_time": time.time() - start_time,
                        "answer": None,
                        "answer_length": 0,
                        "has_useful_info": False
                    }
                
                # Collect streaming response
                full_response = ""
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        chunk = line_text[6:]  # Remove 'data: ' prefix
                        if chunk and chunk != '[DONE]':
                            try:
                                data = json.loads(chunk)
                                if 'delta' in data and 'content' in data['delta']:
                                    full_response += data['delta']['content']
                            except json.JSONDecodeError:
                                continue
                
                response_time = time.time() - start_time
                
                # Analyze the response quality
                analysis = self.analyze_response(query, full_response)
                
                return {
                    "category": category,
                    "query": query,
                    "query_index": query_index,
                    "success": True,
                    "error": None,
                    "response_time": response_time,
                    "answer": full_response,
                    "answer_length": len(full_response),
                    "has_useful_info": analysis["has_useful_info"],
                    "relevance_score": analysis["relevance_score"],
                    "completeness_score": analysis["completeness_score"],
                    "accuracy_indicators": analysis["accuracy_indicators"]
                }
                
        except Exception as e:
            return {
                "category": category,
                "query": query,
                "query_index": query_index,
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "answer": None,
                "answer_length": 0,
                "has_useful_info": False
            }
    
    def analyze_response(self, query: str, response: str) -> Dict[str, Any]:
        """Analyze response quality and accuracy indicators"""
        if not response or len(response) < 10:
            return {
                "has_useful_info": False,
                "relevance_score": 0,
                "completeness_score": 0,
                "accuracy_indicators": []
            }
        
        # Convert to lowercase for analysis
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Check for useful information indicators
        useful_indicators = [
            len(response) > 50,  # Substantial response
            any(word in response_lower for word in ['istanbul', 'turkey', 'turkish']),
            '.' in response,  # Proper sentences
            not response.startswith("sorry") and not response.startswith("i don't"),
        ]
        
        has_useful_info = sum(useful_indicators) >= 2
        
        # Relevance scoring (0-100)
        relevance_keywords = self.get_relevance_keywords(query_lower)
        relevance_matches = sum(1 for keyword in relevance_keywords if keyword in response_lower)
        relevance_score = min(100, (relevance_matches / max(len(relevance_keywords), 1)) * 100)
        
        # Completeness scoring based on response length and structure
        completeness_indicators = [
            len(response) > 100,  # Detailed response
            response.count('.') > 2,  # Multiple sentences
            any(word in response_lower for word in ['located', 'address', 'time', 'cost', 'price']),
            ':' in response or '•' in response or '-' in response,  # Structured info
        ]
        completeness_score = (sum(completeness_indicators) / len(completeness_indicators)) * 100
        
        # Accuracy indicators (specific to Istanbul)
        accuracy_indicators = self.get_accuracy_indicators(query_lower, response_lower)
        
        return {
            "has_useful_info": has_useful_info,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "accuracy_indicators": accuracy_indicators
        }
    
    def get_relevance_keywords(self, query: str) -> List[str]:
        """Get expected keywords based on query type"""
        keywords = []
        
        # Transportation keywords
        if any(word in query for word in ['metro', 'bus', 'ferry', 'transport', 'get to', 'airport', 'taxi']):
            keywords.extend(['metro', 'bus', 'ferry', 'tram', 'istanbulkart', 'transport', 'station', 'line'])
        
        # District keywords
        if any(word in query for word in ['sultanahmet', 'beyoglu', 'karakoy', 'kadikoy', 'galata', 'district', 'neighborhood']):
            keywords.extend(['district', 'area', 'neighborhood', 'located', 'historic', 'modern'])
        
        # Museum/attraction keywords
        if any(word in query for word in ['museum', 'palace', 'mosque', 'hagia sophia', 'topkapi', 'blue mosque']):
            keywords.extend(['museum', 'palace', 'mosque', 'historic', 'entrance', 'hours', 'ticket'])
        
        # Restaurant keywords
        if any(word in query for word in ['restaurant', 'food', 'eat', 'kebab', 'turkish breakfast', 'cuisine']):
            keywords.extend(['restaurant', 'food', 'turkish', 'cuisine', 'traditional', 'local', 'dish'])
        
        # Tips keywords
        if any(word in query for word in ['tip', 'should i', 'what to', 'how to', 'best time', 'advice']):
            keywords.extend(['should', 'recommend', 'best', 'important', 'tip', 'advice'])
        
        return keywords
    
    def get_accuracy_indicators(self, query: str, response: str) -> List[str]:
        """Check for specific accuracy indicators"""
        indicators = []
        
        # Check for common Istanbul-specific accurate information
        if 'metro' in query and any(line in response for line in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']):
            indicators.append("Contains correct metro line references")
        
        if 'airport' in query and any(airport in response for airport in ['atatürk', 'sabiha gökçen', 'ist']):
            indicators.append("Contains correct airport names")
        
        if 'sultanahmet' in query and any(attraction in response for attraction in ['hagia sophia', 'blue mosque', 'topkapi']):
            indicators.append("Contains correct Sultanahmet attractions")
        
        if 'ferry' in query and any(term in response for term in ['bosphorus', 'golden horn', 'princes', 'kadıköy']):
            indicators.append("Contains correct ferry route information")
        
        if 'currency' in query and 'lira' in response:
            indicators.append("Contains correct currency information")
        
        if 'tipping' in query and any(tip in response for tip in ['10%', '15%', '10-15']):
            indicators.append("Contains reasonable tipping information")
        
        return indicators
    
    async def run_comprehensive_test(self):
        """Run all test queries"""
        print("🚀 Starting Comprehensive Final Test Suite")
        print(f"📊 Testing {sum(len(queries) for queries in TEST_QUERIES.values())} queries across 5 categories")
        print("=" * 80)
        
        await self.setup_session()
        
        total_queries = 0
        category_results = {}
        
        try:
            for category, queries in TEST_QUERIES.items():
                print(f"\n🔍 Testing {category.upper()} ({len(queries)} queries)")
                print("-" * 50)
                
                category_results[category] = []
                
                for i, query in enumerate(queries, 1):
                    total_queries += 1
                    print(f"[{total_queries:2d}] Testing: {query[:60]}..." if len(query) > 60 else f"[{total_queries:2d}] Testing: {query}")
                    
                    result = await self.test_query(category, query, i)
                    category_results[category].append(result)
                    self.results.append(result)
                    
                    if result["success"]:
                        status = "✅" if result["has_useful_info"] else "⚠️"
                        print(f"     {status} Response: {result['answer_length']} chars, {result['response_time']:.2f}s")
                        if result.get("accuracy_indicators"):
                            print(f"     🎯 Accuracy: {', '.join(result['accuracy_indicators'][:2])}")
                    else:
                        print(f"     ❌ Error: {result['error']}")
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
        
        finally:
            await self.cleanup_session()
        
        # Generate comprehensive report
        self.generate_final_report(category_results)
        
        return self.results
    
    def generate_final_report(self, category_results: Dict[str, List[Dict]]):
        """Generate detailed final test report"""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE FINAL TEST REPORT")
        print("=" * 80)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        useful_responses = sum(1 for r in self.results if r.get("has_useful_info", False))
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"Total Tests: {total_tests}")
        print(f"Successful Responses: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"Useful Responses: {useful_responses} ({useful_responses/total_tests*100:.1f}%)")
        
        if successful_tests > 0:
            avg_response_time = sum(r['response_time'] for r in self.results if r['success'])/successful_tests
            avg_response_length = sum(r['answer_length'] for r in self.results if r['success'])/successful_tests
            print(f"Average Response Time: {avg_response_time:.2f}s")
            print(f"Average Response Length: {avg_response_length:.0f} characters")
        else:
            print("Average Response Time: N/A (no successful responses)")
            print("Average Response Length: N/A (no successful responses)")
        
        # Category breakdown
        print(f"\n📂 CATEGORY BREAKDOWN")
        for category, results in category_results.items():
            total = len(results)
            successful = sum(1 for r in results if r["success"])
            useful = sum(1 for r in results if r.get("has_useful_info", False))
            avg_relevance = sum(r.get("relevance_score", 0) for r in results if r["success"]) / max(successful, 1)
            avg_completeness = sum(r.get("completeness_score", 0) for r in results if r["success"]) / max(successful, 1)
            
            print(f"\n{category.upper()}:")
            print(f"  ✅ Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
            print(f"  🎯 Useful Responses: {useful}/{total} ({useful/total*100:.1f}%)")
            print(f"  📈 Avg Relevance Score: {avg_relevance:.1f}/100")
            print(f"  📝 Avg Completeness Score: {avg_completeness:.1f}/100")
        
        # Failed queries analysis
        failed_queries = [r for r in self.results if not r["success"] or not r.get("has_useful_info", False)]
        if failed_queries:
            print(f"\n❌ FAILED OR POOR QUALITY RESPONSES ({len(failed_queries)} queries)")
            for i, result in enumerate(failed_queries[:10], 1):  # Show first 10
                print(f"{i:2d}. [{result['category']}] {result['query']}")
                if result.get('error'):
                    print(f"    Error: {result['error']}")
                elif result.get('answer'):
                    print(f"    Response: {result['answer'][:100]}...")
        
        # Best performing queries
        good_queries = sorted([r for r in self.results if r["success"] and r.get("has_useful_info", False)], 
                             key=lambda x: x.get("relevance_score", 0), reverse=True)[:10]
        if good_queries:
            print(f"\n✨ TOP PERFORMING RESPONSES (Top 10)")
            for i, result in enumerate(good_queries, 1):
                print(f"{i:2d}. [{result['category']}] {result['query']}")
                print(f"    📊 Relevance: {result.get('relevance_score', 0):.1f}, Completeness: {result.get('completeness_score', 0):.1f}")
                if result.get('accuracy_indicators'):
                    print(f"    🎯 Accuracy: {result['accuracy_indicators'][0]}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"comprehensive_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_summary": {
                    "timestamp": timestamp,
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "useful_responses": useful_responses,
                    "success_rate": successful_tests/total_tests*100,
                    "usefulness_rate": useful_responses/total_tests*100
                },
                "category_results": category_results,
                "detailed_results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Final verdict
        overall_quality = useful_responses / total_tests * 100
        print(f"\n🏆 FINAL VERDICT")
        if overall_quality >= 85:
            print(f"🌟 EXCELLENT: {overall_quality:.1f}% useful responses - Production ready!")
        elif overall_quality >= 70:
            print(f"✅ GOOD: {overall_quality:.1f}% useful responses - Minor improvements needed")
        elif overall_quality >= 50:
            print(f"⚠️ FAIR: {overall_quality:.1f}% useful responses - Significant improvements needed")
        else:
            print(f"❌ POOR: {overall_quality:.1f}% useful responses - Major fixes required")


async def main():
    """Main test execution"""
    print("🔧 Checking backend connection...")
    
    # Test backend connection first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as response:
                if response.status != 200:
                    print("❌ Backend is not responding correctly. Please ensure the backend is running.")
                    sys.exit(1)
                print("✅ Backend connection verified")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Please ensure the backend server is running on http://localhost:8000")
        sys.exit(1)
    
    # Run comprehensive tests
    test_runner = ComprehensiveTestRunner()
    results = await test_runner.run_comprehensive_test()
    
    print(f"\n🎉 Test completed! Check the generated JSON report for detailed analysis.")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
