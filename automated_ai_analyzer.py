#!/usr/bin/env python3
"""
Automated AI Istanbul Response Analyzer
Automatically tests the AI chatbot and analyzes responses
"""

import json
import requests
import time
import datetime
import re
from typing import Dict, List, Any

class AutomatedAIAnalyzer:
    def __init__(self):
        self.test_questions = [
            # Transportation (1-16)
            "How do I get from Istanbul Airport to Sultanahmet?",
            "What's the best way to get from Taksim to Grand Bazaar?",
            "I want to take a ferry from Eminönü to Üsküdar. What's the schedule and cost?",
            "How can I get to Büyükada (Prince Islands) and what transport is available on the island?",
            "Is there a direct metro connection between European and Asian sides?",
            "What's the cheapest way to travel around Istanbul for a week?",
            "How do I get to Sabiha Gökçen Airport from Kadıköy at 4 AM?",
            "Can you explain the difference between dolmuş, minibüs, and regular buses?",
            "How does the tram system work in Istanbul?",
            "I need to get from Atatürk Airport area to Asian side during rush hour. Best route?",
            "What are the main ferry routes and which one is most scenic?",
            "How do I use Istanbul Kart and where can I buy one?",
            "What's the night transportation situation in Istanbul?",
            "How do I get to Belgrade Forest from city center?",
            "Are taxis expensive in Istanbul and how do I avoid scams?",
            "What's the best way to do a Bosphorus tour including both continents?",
            
            # Restaurants & Food (17-32)
            "Where can I find the best Turkish breakfast in Sultanahmet?",
            "I'm vegetarian. What traditional Turkish dishes can I eat?",
            "Can you recommend high-end Ottoman cuisine restaurants with historical ambiance?",
            "What street foods should I try and where are they safe to eat?",
            "I have celiac disease. Can you suggest gluten-free restaurants in Beyoğlu?",
            "What's the best Turkish dessert and where can I find it?",
            "I want to experience a traditional Turkish cooking class. Where can I find authentic ones?",
            "What's the difference between Turkish coffee houses in different districts?",
            "Can you recommend good seafood restaurants near the Bosphorus?",
            "I'm interested in the cultural significance of Turkish tea culture. Where can I experience it authentically?",
            "What are the best food markets in Istanbul and what should I buy?",
            "Is tap water safe to drink in Istanbul restaurants?",
            "Can you explain the etiquette and customs around dining in Turkish homes vs restaurants?",
            "What are the best budget-friendly local food spots that tourists usually miss?",
            "How much should I tip in Turkish restaurants?",
            "I want to understand regional Turkish cuisine differences. What should I look for in Istanbul?",
            
            # Museums & Cultural Sites (33-48)
            "What are the opening hours and ticket prices for Hagia Sophia?",
            "Can you explain the historical significance of Topkapi Palace and what to prioritize during a visit?",
            "I'm interested in Byzantine history. Beyond Hagia Sophia, what lesser-known sites should I visit?",
            "What's the difference between the Blue Mosque and other mosques in Istanbul?",
            "Are there any good art museums showcasing contemporary Turkish art?",
            "Can you recommend a cultural itinerary that shows Istanbul's evolution from Byzantine to Ottoman to modern?",
            "What should I know before visiting the Grand Bazaar?",
            "How can I learn about Ottoman architecture while exploring Istanbul?",
            "What are some hidden architectural gems that showcase Istanbul's multicultural past?",
            "Is the Basilica Cistern worth visiting and what should I expect?",
            "Can you suggest museums that are good for families with children?",
            "I'm researching Islamic calligraphy and ceramics. Which museums have the best collections?",
            "What's the best way to avoid crowds at popular tourist sites?",
            "Are there any archaeological sites within Istanbul city limits?",
            "How has Istanbul's cultural landscape changed in the past decade?",
            "What are the must-see cultural sites for a first-time visitor with only 2 days?",
            
            # Districts & Neighborhoods (49-64)
            "What's special about Sultanahmet district and what can I find there?",
            "I want to experience local life away from tourists. Which neighborhoods should I explore?",
            "Can you explain the character differences between Beyoğlu, Beşiktaş, and Şişli?",
            "What can I do in Kadıköy on the Asian side?",
            "Is Galata area worth staying in and what's the neighborhood like?",
            "I'm interested in Istanbul's gentrification process. Which areas are currently changing?",
            "What's the best area for nightlife and entertainment?",
            "Can you recommend family-friendly neighborhoods to explore with children?",
            "What's the socioeconomic profile of different Istanbul districts?",
            "Which area has the best shopping opportunities?",
            "What's unique about the Bosphorus waterfront neighborhoods?",
            "How do the European and Asian sides of Istanbul differ culturally and socially?",
            "Is it safe to walk around different neighborhoods at night?",
            "Which neighborhoods are best for street photography and why?",
            "How has neighborhood character in Istanbul changed due to Syrian refugee influx?",
            "What are the main characteristics of Ortaköy district?",
            
            # General Tips & Practical (65-80)
            "What's the weather like in Istanbul in March and what should I pack?",
            "What are the most important cultural etiquette rules I should follow?",
            "How do I navigate bureaucracy if I need to extend my visa or handle official matters?",
            "Is it safe for solo female travelers in Istanbul?",
            "What are the key Turkish phrases I should learn for daily interactions?",
            "How do I understand and respect Islamic customs during Ramadan?",
            "What's the best way to exchange money and avoid scams?",
            "How widespread is English and how can I communicate effectively?",
            "What should I know about Turkish business culture if I'm here for work?",
            "Are there any cultural taboos or things I should definitely avoid doing?",
            "What's the healthcare system like and how can I access medical care as a tourist?",
            "How do I understand and navigate Turkish social hierarchies and respect systems?",
            "What are the emergency numbers and basic safety information I should know?",
            "How do I handle haggling and price negotiations in markets?",
            "What are the environmental and sustainability challenges facing Istanbul?",
            "What should I do if I lose my passport or have other travel document emergencies?"
        ]
        
        self.results = []
        
    def get_category(self, test_num):
        """Get category name based on test number"""
        if 1 <= test_num <= 16:
            return "Transportation"
        elif 17 <= test_num <= 32:
            return "Restaurants & Food"
        elif 33 <= test_num <= 48:
            return "Museums & Culture"
        elif 49 <= test_num <= 64:
            return "Districts & Neighborhoods"
        else:
            return "General & Practical"
    
    def call_ai_chatbot(self, question: str) -> str:
        """
        Call your AI Istanbul chatbot API
        REPLACE THIS WITH YOUR ACTUAL API ENDPOINT
        """
        # This is a placeholder - replace with your actual API call
        try:
            # Example for local server
            response = requests.post(
                "http://localhost:3000/api/chat",  # Replace with your endpoint
                json={"message": question},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('response', 'No response received')
            else:
                return f"API Error: {response.status_code}"
        except Exception as e:
            # If API call fails, return placeholder for manual input
            return f"[API_CALL_FAILED] Please manually input AI response for: {question}"
    
    def analyze_accuracy(self, question: str, response: str) -> Dict[str, Any]:
        """Analyze response accuracy using keyword detection and patterns"""
        issues = []
        score = 3  # Start with perfect score, deduct for issues
        
        # Common accuracy issues to detect
        accuracy_checks = {
            # Price/Cost related
            "outdated_prices": [
                r"\b\d+\s*(lira|TL|Turkish\s+lira)\b",  # Look for specific prices
                r"\b(cheap|expensive|costs?\s+about)\b"
            ],
            
            # Time/Schedule related
            "time_schedules": [
                r"\b\d{1,2}:\d{2}\b",  # Time formats
                r"\b(open|closed|hours?|schedule)\b",
                r"\b(daily|weekdays?|weekends?)\b"
            ],
            
            # Location specific
            "locations": [
                r"\b(address|located|situated)\b",
                r"\b(district|neighborhood|area)\b",
                r"\b(near|close\s+to|walking\s+distance)\b"
            ],
            
            # Transportation specific
            "transport": [
                r"\b(metro|bus|tram|ferry|dolmuş)\b",
                r"\b(station|stop|terminal)\b",
                r"\b(line|route|connection)\b"
            ]
        }
        
        # Check for generic vs specific information
        if len(re.findall(r"\bIstanbul\b", response, re.IGNORECASE)) < 1:
            issues.append("Response lacks Istanbul-specific context")
            score -= 1
            
        # Check for vague language
        vague_patterns = [
            r"\b(some|many|several|various|different|numerous)\b",
            r"\b(might|could|may|perhaps|possibly)\b"
        ]
        
        vague_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) for pattern in vague_patterns)
        if vague_count > 5:
            issues.append("Response contains excessive vague language")
            score = max(0, score - 1)
        
        # Check response length (too short might be incomplete)
        if len(response.split()) < 20:
            issues.append("Response may be too brief/incomplete")
            score = max(0, score - 1)
        
        return {
            "accuracy_score": max(0, score),
            "issues_found": issues,
            "analysis_notes": f"Automated analysis detected {len(issues)} potential issues"
        }
    
    def analyze_completeness(self, question: str, response: str) -> int:
        """Analyze how completely the response answers the question"""
        score = 3
        
        # Check if question has multiple parts
        question_parts = len(re.findall(r'[.?!]', question))
        if question_parts > 1:
            # Multi-part question - check if response addresses multiple aspects
            response_parts = len(re.findall(r'[.!]', response))
            if response_parts < question_parts:
                score -= 1
        
        # Check for specific question words and if they're addressed
        question_words = {
            'how': r'\b(step|method|way|process|procedure)\b',
            'what': r'\b(is|are|include|consist)\b',
            'where': r'\b(location|address|place|area|district)\b',
            'when': r'\b(time|hour|schedule|open|closed)\b',
            'why': r'\b(because|reason|due to|purpose)\b',
            'which': r'\b(best|better|recommend|suggest)\b'
        }
        
        for q_word, expected_pattern in question_words.items():
            if q_word in question.lower():
                if not re.search(expected_pattern, response, re.IGNORECASE):
                    score = max(0, score - 1)
                    break
        
        return score
    
    def analyze_cultural_sensitivity(self, question: str, response: str) -> int:
        """Analyze cultural sensitivity of the response"""
        score = 2
        
        # Check for cultural context keywords
        cultural_keywords = [
            'turkish', 'ottoman', 'islamic', 'muslim', 'ramadan', 'prayer',
            'tradition', 'culture', 'custom', 'etiquette', 'respect'
        ]
        
        cultural_mentions = sum(1 for keyword in cultural_keywords 
                              if keyword in response.lower())
        
        if cultural_mentions == 0 and any(word in question.lower() for word in 
                                        ['culture', 'custom', 'etiquette', 'tradition']):
            score -= 1
        
        # Check for potentially insensitive language
        insensitive_patterns = [
            r'\b(weird|strange|odd|foreign)\b',
            r'\b(backward|primitive|undeveloped)\b'
        ]
        
        for pattern in insensitive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score = 0
                break
        
        return score
    
    def analyze_practical_usefulness(self, question: str, response: str) -> int:
        """Analyze how practical and useful the response is"""
        score = 2
        
        # Check for actionable information
        actionable_patterns = [
            r'\b(go to|visit|take|use|buy|book)\b',
            r'\b(step \d|first|then|next|finally)\b',
            r'\b(address|phone|website|email)\b',
            r'\b(cost|price|\d+\s*TL|lira)\b'
        ]
        
        actionable_count = sum(len(re.findall(pattern, response, re.IGNORECASE)) 
                              for pattern in actionable_patterns)
        
        if actionable_count < 2:
            score -= 1
        
        # Check for specific recommendations vs generic advice
        specific_patterns = [
            r'\b[A-Z][a-z]+\s+(Restaurant|Hotel|Museum|Street|Square)\b',
            r'\b\d+\s*(minute|hour|km|meter)\b',
            r'\b(Line \d+|M\d+|Bus \d+)\b'
        ]
        
        specific_count = sum(len(re.findall(pattern, response)) 
                           for pattern in specific_patterns)
        
        if specific_count == 0:
            score = max(0, score - 1)
        
        return score
    
    def analyze_response(self, test_num: int, question: str, response: str) -> Dict[str, Any]:
        """Perform complete analysis of a single response"""
        
        # Get accuracy analysis
        accuracy_analysis = self.analyze_accuracy(question, response)
        accuracy_score = accuracy_analysis["accuracy_score"]
        
        # Get other scores
        completeness_score = self.analyze_completeness(question, response)
        cultural_score = self.analyze_cultural_sensitivity(question, response)
        practical_score = self.analyze_practical_usefulness(question, response)
        
        total_score = accuracy_score + completeness_score + cultural_score + practical_score
        
        # Determine grade
        if total_score >= 9:
            grade = "Excellent"
        elif total_score >= 7:
            grade = "Very Good"
        elif total_score >= 5:
            grade = "Good"
        elif total_score >= 3:
            grade = "Fair"
        else:
            grade = "Poor"
        
        return {
            "test_number": test_num,
            "category": self.get_category(test_num),
            "question": question,
            "ai_response": response,
            "scores": {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "cultural_sensitivity": cultural_score,
                "practical_usefulness": practical_score,
                "total": total_score
            },
            "grade": grade,
            "issues_found": accuracy_analysis["issues_found"],
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "automated_analysis": True
        }
    
    def run_automated_evaluation(self):
        """Run the complete automated evaluation"""
        print("🚀 Starting Automated AI Istanbul Response Analysis")
        print("=" * 60)
        print(f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing {len(self.test_questions)} questions across 5 categories")
        print("=" * 60)
        
        results = []
        failed_api_calls = []
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n🧪 TEST {i}/80 - {self.get_category(i)}")
            print(f"❓ Question: {question[:60]}..." if len(question) > 60 else f"❓ Question: {question}")
            
            # Get AI response
            print("🤖 Calling AI chatbot...")
            ai_response = self.call_ai_chatbot(question)
            
            if "[API_CALL_FAILED]" in ai_response:
                failed_api_calls.append(i)
                print("❌ API call failed - manual input required")
                ai_response = input("Please paste the AI response here: ")
            
            # Analyze response
            print("📊 Analyzing response...")
            analysis = self.analyze_response(i, question, ai_response)
            results.append(analysis)
            
            # Display results
            scores = analysis["scores"]
            print(f"📈 Score: {scores['total']}/10 ({analysis['grade']})")
            print(f"   Accuracy: {scores['accuracy']}/3")
            print(f"   Completeness: {scores['completeness']}/3")
            print(f"   Cultural: {scores['cultural_sensitivity']}/2")
            print(f"   Practical: {scores['practical_usefulness']}/2")
            
            if analysis["issues_found"]:
                print(f"⚠️  Issues: {', '.join(analysis['issues_found'])}")
            
            # Progress update
            if i % 10 == 0:
                avg_score = sum(r["scores"]["total"] for r in results) / len(results)
                print(f"\n📊 Progress: {i}/80 completed. Average score so far: {avg_score:.2f}/10")
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
        
        # Save results
        self.save_results(results)
        
        # Generate summary
        self.generate_summary(results, failed_api_calls)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_istanbul_automated_analysis_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "analysis_date": datetime.datetime.now().isoformat(),
                "total_tests": len(results),
                "analysis_type": "automated",
                "version": "1.0"
            },
            "results": results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def generate_summary(self, results: List[Dict[str, Any]], failed_calls: List[int]):
        """Generate and display summary analysis"""
        print("\n" + "="*60)
        print("📈 AUTOMATED ANALYSIS SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_scores = [r["scores"]["total"] for r in results]
        avg_score = sum(total_scores) / len(total_scores)
        
        grade_counts = {}
        for result in results:
            grade = result["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        print(f"📊 Overall Results:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Average Score: {avg_score:.2f}/10")
        print(f"   Failed API Calls: {len(failed_calls)}")
        
        print(f"\n🎯 Grade Distribution:")
        for grade, count in sorted(grade_counts.items()):
            percentage = (count / len(results)) * 100
            print(f"   {grade}: {count} tests ({percentage:.1f}%)")
        
        # Category analysis
        print(f"\n📋 Category Performance:")
        categories = ["Transportation", "Restaurants & Food", "Museums & Culture", 
                     "Districts & Neighborhoods", "General & Practical"]
        
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            if cat_results:
                cat_avg = sum(r["scores"]["total"] for r in cat_results) / len(cat_results)
                print(f"   {category}: {cat_avg:.2f}/10 ({len(cat_results)} tests)")
        
        # Top issues
        all_issues = []
        for result in results:
            all_issues.extend(result["issues_found"])
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            print(f"\n⚠️  Top Issues Found:")
            for issue, count in issue_counts.most_common(5):
                print(f"   {issue}: {count} occurrences")
        
        # Recommendations
        print(f"\n💡 Automated Recommendations:")
        if avg_score < 6:
            print("   🔴 CRITICAL: Overall performance below acceptable threshold")
        if len([s for s in total_scores if s < 3]) > 5:
            print("   🟠 HIGH PRIORITY: Multiple responses with serious issues")
        if len(failed_calls) > 0:
            print(f"   🟡 MEDIUM: {len(failed_calls)} tests need manual review due to API failures")
        
        print("\n✅ Automated analysis completed!")

def main():
    """Main function to run automated analysis"""
    analyzer = AutomatedAIAnalyzer()
    
    print("🔧 AI Istanbul Automated Response Analyzer")
    print("=" * 50)
    print("⚙️  This script will automatically:")
    print("   1. Send all 80 test questions to your AI chatbot")
    print("   2. Analyze each response for accuracy, completeness, etc.")
    print("   3. Generate comprehensive analysis report")
    print("   4. Identify issues and provide recommendations")
    print("\n📋 NOTE: Make sure your AI chatbot API is running and accessible")
    print("   Update the API endpoint in the call_ai_chatbot() method")
    
    choice = input("\nProceed with automated analysis? (y/n): ").strip().lower()
    
    if choice == 'y':
        results = analyzer.run_automated_evaluation()
        print(f"\n🎉 Analysis complete! {len(results)} responses analyzed.")
    else:
        print("👋 Analysis cancelled.")

if __name__ == "__main__":
    main()
