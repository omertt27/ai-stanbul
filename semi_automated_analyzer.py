#!/usr/bin/env python3
"""
Semi-Automated AI Response Analyzer
Displays questions automatically and analyzes responses as you paste them
"""

import json
import datetime
import re
from typing import Dict, List, Any

class SemiAutomatedAnalyzer:
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
            "What's the best season to visit Istanbul and what should I pack?",
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
    
    def analyze_response_automatically(self, question: str, response: str) -> Dict[str, Any]:
        """Automatically analyze response quality"""
        
        # Initialize scores
        accuracy_score = 3
        completeness_score = 3
        cultural_score = 2
        practical_score = 2
        issues = []
        
        # ACCURACY ANALYSIS
        # Check for specific Istanbul references
        istanbul_refs = len(re.findall(r'\bIstanbul\b', response, re.IGNORECASE))
        if istanbul_refs == 0:
            issues.append("No specific Istanbul references - may be generic")
            accuracy_score -= 1
        
        # Check for specific places/locations
        specific_places = len(re.findall(r'\b[A-Z][a-z]+\s+(Street|Square|Museum|Palace|Mosque|District|Station)\b', response))
        if specific_places == 0 and any(word in question.lower() for word in ['where', 'which', 'what']):
            issues.append("Lacks specific location references")
            accuracy_score = max(0, accuracy_score - 1)
        
        # Check for price/cost information
        if 'cost' in question.lower() or 'price' in question.lower() or 'expensive' in question.lower():
            has_price_info = bool(re.search(r'\b\d+\s*(TL|lira|euro|dollar)\b', response, re.IGNORECASE))
            if not has_price_info:
                issues.append("Missing expected price/cost information")
                accuracy_score = max(0, accuracy_score - 1)
        
        # COMPLETENESS ANALYSIS
        # Check response length
        word_count = len(response.split())
        if word_count < 30:
            issues.append("Response may be too brief")
            completeness_score -= 1
        elif word_count > 300:
            issues.append("Response may be excessively long")
            completeness_score = max(0, completeness_score - 1)
        
        # Check for multiple aspects in multi-part questions
        question_markers = ['and', '&', ',', ';']
        if any(marker in question for marker in question_markers):
            # Multi-part question
            response_sentences = len(re.findall(r'[.!?]', response))
            if response_sentences < 2:
                issues.append("Multi-part question may not be fully addressed")
                completeness_score = max(0, completeness_score - 1)
        
        # CULTURAL SENSITIVITY ANALYSIS
        cultural_keywords = ['turkish', 'ottoman', 'islamic', 'muslim', 'ramadan', 'tradition', 'culture', 'custom', 'respect']
        cultural_mentions = sum(1 for keyword in cultural_keywords if keyword in response.lower())
        
        # Check if cultural context is needed and provided
        needs_cultural_context = any(word in question.lower() for word in 
                                   ['culture', 'custom', 'etiquette', 'respect', 'tradition', 'religious'])
        
        if needs_cultural_context and cultural_mentions == 0:
            issues.append("Missing cultural context where expected")
            cultural_score -= 1
        
        # Check for potentially insensitive language
        insensitive_words = ['weird', 'strange', 'backward', 'primitive']
        if any(word in response.lower() for word in insensitive_words):
            issues.append("Potentially insensitive language detected")
            cultural_score = 0
        
        # PRACTICAL USEFULNESS ANALYSIS
        # Check for actionable advice
        actionable_words = ['go to', 'visit', 'take', 'use', 'buy', 'book', 'call', 'check']
        actionable_count = sum(1 for phrase in actionable_words if phrase in response.lower())
        
        if actionable_count == 0:
            issues.append("Lacks actionable advice")
            practical_score -= 1
        
        # Check for specific details vs vague language
        vague_words = ['some', 'many', 'several', 'various', 'different', 'numerous', 'might', 'could', 'may']
        vague_count = sum(1 for word in vague_words if word in response.lower())
        
        if vague_count > 5:
            issues.append("Contains excessive vague language")
            practical_score = max(0, practical_score - 1)
        
        # Calculate total score
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
            "scores": {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "cultural_sensitivity": cultural_score,
                "practical_usefulness": practical_score,
                "total": total_score
            },
            "grade": grade,
            "issues_found": issues,
            "word_count": word_count,
            "analysis_confidence": "Medium (Automated)",
        }
    
    def run_semi_automated_analysis(self):
        """Run semi-automated analysis with manual response input"""
        
        print("🚀 AI ISTANBUL SEMI-AUTOMATED RESPONSE ANALYZER")
        print("=" * 60)
        print("📋 How this works:")
        print("   1. I'll show you each test question")
        print("   2. Copy the question to your AI chatbot")
        print("   3. Paste the AI response when prompted")
        print("   4. I'll automatically analyze and score the response")
        print("   5. Continue through all 80 tests")
        print("=" * 60)
        
        results = []
        start_from = 1
        
        # Option to resume from a specific test
        resume = input("Resume from a specific test number? (Enter number 1-80 or press Enter to start from 1): ").strip()
        if resume.isdigit() and 1 <= int(resume) <= 80:
            start_from = int(resume)
        
        for i in range(start_from - 1, len(self.test_questions)):
            test_num = i + 1
            question = self.test_questions[i]
            category = self.get_category(test_num)
            
            print(f"\n{'='*80}")
            print(f"🧪 TEST {test_num}/80 - {category.upper()}")
            print(f"{'='*80}")
            print(f"❓ COPY THIS QUESTION TO YOUR AI CHATBOT:")
            print(f"\n{question}")
            print(f"\n{'='*80}")
            
            # Get AI response from user
            print("🤖 Paste the AI response below (press Enter on empty line when finished):")
            response_lines = []
            while True:
                line = input()
                if line.strip() == "" and len(response_lines) > 0:
                    break
                response_lines.append(line)
            
            ai_response = "\n".join(response_lines)
            
            if not ai_response.strip():
                print("❌ No response entered. Skipping this test.")
                continue
            
            # Analyze response automatically
            print("📊 Analyzing response automatically...")
            analysis = self.analyze_response_automatically(question, ai_response)
            
            # Display analysis results
            scores = analysis["scores"]
            print(f"\n📈 AUTOMATED ANALYSIS RESULTS:")
            print(f"   Overall Score: {scores['total']}/10 ({analysis['grade']})")
            print(f"   • Accuracy: {scores['accuracy']}/3")
            print(f"   • Completeness: {scores['completeness']}/3")
            print(f"   • Cultural Sensitivity: {scores['cultural_sensitivity']}/2")
            print(f"   • Practical Usefulness: {scores['practical_usefulness']}/2")
            print(f"   • Word Count: {analysis['word_count']} words")
            
            if analysis["issues_found"]:
                print(f"\n⚠️  Issues Detected:")
                for issue in analysis["issues_found"]:
                    print(f"   • {issue}")
            
            # Allow manual override
            print(f"\n🤔 Do you agree with this analysis?")
            override = input("Press Enter to accept, or type 'override' to manually adjust scores: ").strip().lower()
            
            if override == 'override':
                print("Manual score override:")
                try:
                    scores["accuracy"] = int(input(f"Accuracy (0-3) [current: {scores['accuracy']}]: ") or scores['accuracy'])
                    scores["completeness"] = int(input(f"Completeness (0-3) [current: {scores['completeness']}]: ") or scores['completeness'])
                    scores["cultural_sensitivity"] = int(input(f"Cultural (0-2) [current: {scores['cultural_sensitivity']}]: ") or scores['cultural_sensitivity'])
                    scores["practical_usefulness"] = int(input(f"Practical (0-2) [current: {scores['practical_usefulness']}]: ") or scores['practical_usefulness'])
                    scores["total"] = sum(scores.values()) - scores["total"]  # Recalculate
                    analysis["analysis_confidence"] = "High (Manual Override)"
                except ValueError:
                    print("Invalid input, keeping automated scores")
            
            # Store result
            result = {
                "test_number": test_num,
                "category": category,
                "question": question,
                "ai_response": ai_response,
                "analysis": analysis,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Progress update
            if test_num % 10 == 0:
                avg_score = sum(r["analysis"]["scores"]["total"] for r in results) / len(results)
                print(f"\n📊 PROGRESS UPDATE: {test_num}/80 completed. Average score: {avg_score:.2f}/10")
            
            # Option to continue or stop
            if test_num < 80:
                continue_choice = input("\nPress Enter for next test, 's' to stop, 'save' to save progress: ").strip().lower()
                if continue_choice == 's':
                    break
                elif continue_choice == 'save':
                    self.save_results(results)
        
        # Final save and summary
        filename = self.save_results(results)
        self.generate_summary(results)
        
        print(f"\n🎉 Analysis completed! Results saved to: {filename}")
        
        return results
    
    def save_results(self, results: List[Dict]) -> str:
        """Save results to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_istanbul_semi_automated_results_{timestamp}.json"
        
        summary_data = {
            "metadata": {
                "analysis_date": datetime.datetime.now().isoformat(),
                "total_tests": len(results),
                "analysis_type": "semi_automated",
                "version": "1.0"
            },
            "results": results,
            "summary": self.calculate_summary(results)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def calculate_summary(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        if not results:
            return {}
        
        total_scores = [r["analysis"]["scores"]["total"] for r in results]
        avg_score = sum(total_scores) / len(total_scores)
        
        # Category averages
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result["analysis"]["scores"]["total"])
        
        category_averages = {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
        
        # Grade distribution
        grade_counts = {}
        for result in results:
            grade = result["analysis"]["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        return {
            "overall_average": round(avg_score, 2),
            "total_tests": len(results),
            "category_averages": {k: round(v, 2) for k, v in category_averages.items()},
            "grade_distribution": grade_counts
        }
    
    def generate_summary(self, results: List[Dict]):
        """Display summary analysis"""
        if not results:
            print("No results to summarize")
            return
        
        summary = self.calculate_summary(results)
        
        print(f"\n{'='*60}")
        print("📈 FINAL ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Overall Results:")
        print(f"   Tests Completed: {summary['total_tests']}/80")
        print(f"   Average Score: {summary['overall_average']}/10")
        
        print(f"\n📋 Category Performance:")
        for category, avg in summary['category_averages'].items():
            print(f"   {category}: {avg}/10")
        
        print(f"\n🎯 Grade Distribution:")
        for grade, count in summary['grade_distribution'].items():
            percentage = (count / summary['total_tests']) * 100
            print(f"   {grade}: {count} tests ({percentage:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if summary['overall_average'] < 6:
            print("   🔴 CRITICAL: Overall performance needs significant improvement")
        elif summary['overall_average'] < 7:
            print("   🟡 MODERATE: Performance is acceptable but has room for improvement")
        else:
            print("   🟢 GOOD: Performance meets target standards")

def main():
    analyzer = SemiAutomatedAnalyzer()
    
    print("Welcome to the AI Istanbul Semi-Automated Response Analyzer!")
    print("\nThis tool will help you systematically test and analyze all 80 responses.")
    print("Make sure you have access to your AI Istanbul chatbot in another window.")
    
    choice = input("\nReady to start the analysis? (y/n): ").strip().lower()
    
    if choice == 'y':
        analyzer.run_semi_automated_analysis()
    else:
        print("Analysis cancelled. Run the script again when ready.")

if __name__ == "__main__":
    main()
