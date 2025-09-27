#!/usr/bin/env python3
"""
AI Istanbul Comprehensive Response Evaluator
Interactive tool for systematically evaluating AI chatbot responses
"""

import json
import datetime
import os
from typing import Dict, List, Any

class AIResponseEvaluator:
    def __init__(self, template_file: str):
        """Initialize evaluator with template file"""
        self.template_file = template_file
        self.results_file = template_file.replace('.json', '_completed.json')
        self.current_test_index = 0
        
        # Load template
        with open(template_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Check if we have existing progress
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.find_current_test_index()
    
    def find_current_test_index(self):
        """Find the first incomplete test"""
        for i, test in enumerate(self.data['evaluation_results']):
            if test['ai_response'] == "" or test['scores']['total'] is None:
                self.current_test_index = i
                return
        self.current_test_index = len(self.data['evaluation_results'])
    
    def display_scoring_guide(self):
        """Display the scoring criteria"""
        print("\n" + "="*80)
        print("📊 SCORING GUIDE")
        print("="*80)
        
        criteria = self.data['scoring_criteria']
        for criterion, details in criteria.items():
            print(f"\n🎯 {criterion.upper().replace('_', ' ')} (0-{details['max_points']} points)")
            print(f"   {details['description']}")
            for score, desc in details['scale'].items():
                print(f"   {score}: {desc}")
    
    def display_test_info(self, test_index: int):
        """Display current test information"""
        test = self.data['evaluation_results'][test_index]
        
        print("\n" + "="*80)
        print(f"📝 TEST {test['test_id']}/80 - {test['category'].upper().replace('_', ' ')}")
        print("="*80)
        print(f"❓ QUESTION: {test['test_input']}")
        print("-"*80)
        
        if test['ai_response']:
            print(f"🤖 AI RESPONSE:\n{test['ai_response']}")
        else:
            print("🤖 AI RESPONSE: [Enter response below]")
        
        print("-"*80)
    
    def get_ai_response(self, test_index: int) -> str:
        """Get AI response for current test"""
        test = self.data['evaluation_results'][test_index]
        
        if test['ai_response']:
            print(f"✅ Response already recorded. Press Enter to keep, or type 'new' to re-enter:")
            choice = input().strip().lower()
            if choice != 'new':
                return test['ai_response']
        
        print("\n📥 Please input the AI response (press Enter twice when finished):")
        lines = []
        empty_lines = 0
        
        while empty_lines < 2:
            line = input()
            if line.strip() == "":
                empty_lines += 1
            else:
                empty_lines = 0
            lines.append(line)
        
        # Remove the trailing empty lines
        while lines and lines[-1].strip() == "":
            lines.pop()
            
        return "\n".join(lines)
    
    def get_scores(self, test_index: int) -> Dict[str, int]:
        """Get evaluation scores for current test"""
        test = self.data['evaluation_results'][test_index]
        scores = {}
        criteria = self.data['scoring_criteria']
        
        print("\n📊 SCORING:")
        
        for criterion, details in criteria.items():
            while True:
                try:
                    current_score = test['scores'].get(criterion)
                    prompt = f"{criterion.replace('_', ' ').title()} (0-{details['max_points']})"
                    if current_score is not None:
                        prompt += f" [current: {current_score}]"
                    prompt += ": "
                    
                    score_input = input(prompt).strip()
                    if score_input == "" and current_score is not None:
                        score = current_score
                    else:
                        score = int(score_input)
                        
                    if 0 <= score <= details['max_points']:
                        scores[criterion] = score
                        break
                    else:
                        print(f"❌ Score must be between 0 and {details['max_points']}")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        # Calculate total
        scores['total'] = sum(scores.values())
        return scores
    
    def get_evaluation_notes(self, test_index: int) -> Dict[str, Any]:
        """Get additional evaluation details"""
        test = self.data['evaluation_results'][test_index]
        
        print("\n📝 EVALUATION DETAILS:")
        
        # Issues found
        print("Issues found (comma-separated, or press Enter for none):")
        current_issues = ", ".join(test.get('issues_found', []))
        if current_issues:
            print(f"Current: {current_issues}")
        issues_input = input("Issues: ").strip()
        if issues_input == "" and current_issues:
            issues = test['issues_found']
        elif issues_input:
            issues = [issue.strip() for issue in issues_input.split(",")]
        else:
            issues = []
        
        # Recommendations
        print("\nRecommendations for improvement (or press Enter to skip):")
        current_rec = test.get('recommendations', '')
        if current_rec:
            print(f"Current: {current_rec}")
        recommendations = input("Recommendations: ").strip()
        if recommendations == "" and current_rec:
            recommendations = current_rec
        
        # Notes
        print("\nAdditional notes (or press Enter to skip):")
        current_notes = test.get('evaluation_notes', '')
        if current_notes:
            print(f"Current: {current_notes}")
        notes = input("Notes: ").strip()
        if notes == "" and current_notes:
            notes = current_notes
        
        return {
            'issues_found': issues,
            'recommendations': recommendations,
            'evaluation_notes': notes,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def save_progress(self):
        """Save current progress"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"✅ Progress saved to {self.results_file}")
    
    def display_progress(self):
        """Display current progress"""
        completed = sum(1 for test in self.data['evaluation_results'] 
                       if test['scores']['total'] is not None)
        total = len(self.data['evaluation_results'])
        percentage = (completed / total) * 100
        
        print(f"\n📈 PROGRESS: {completed}/{total} tests completed ({percentage:.1f}%)")
        
        if completed > 0:
            total_scores = [test['scores']['total'] for test in self.data['evaluation_results'] 
                           if test['scores']['total'] is not None]
            avg_score = sum(total_scores) / len(total_scores)
            print(f"📊 Current average score: {avg_score:.2f}/10")
    
    def evaluate_test(self, test_index: int):
        """Evaluate a single test"""
        if test_index >= len(self.data['evaluation_results']):
            print("✅ All tests completed!")
            return False
        
        test = self.data['evaluation_results'][test_index]
        
        # Display test info
        self.display_test_info(test_index)
        
        # Get AI response
        ai_response = self.get_ai_response(test_index)
        test['ai_response'] = ai_response
        
        # Display scoring guide
        self.display_scoring_guide()
        
        # Get scores
        scores = self.get_scores(test_index)
        test['scores'] = scores
        
        # Get evaluation details
        eval_details = self.get_evaluation_notes(test_index)
        test.update(eval_details)
        
        # Display summary
        print(f"\n✅ Test {test_index + 1} completed. Total score: {scores['total']}/10")
        
        # Save progress
        self.save_progress()
        
        return True
    
    def run_evaluation(self):
        """Run the interactive evaluation process"""
        print("🚀 AI Istanbul Comprehensive Response Evaluator")
        print("="*50)
        
        self.display_progress()
        
        if self.current_test_index >= len(self.data['evaluation_results']):
            print("✅ All evaluations completed!")
            self.generate_summary()
            return
        
        print(f"\nStarting from test {self.current_test_index + 1}")
        print("Commands: 'quit' to exit, 'skip' to skip current test, 'back' to go back")
        
        while self.current_test_index < len(self.data['evaluation_results']):
            print(f"\n" + "-"*50)
            
            # Check for user commands
            cmd = input(f"Press Enter to continue with test {self.current_test_index + 1}, or enter command: ").strip().lower()
            
            if cmd == 'quit':
                print("💾 Saving progress and exiting...")
                self.save_progress()
                break
            elif cmd == 'skip':
                print("⏭️ Skipping test...")
                self.current_test_index += 1
                continue
            elif cmd == 'back' and self.current_test_index > 0:
                self.current_test_index -= 1
                continue
            
            # Evaluate current test
            success = self.evaluate_test(self.current_test_index)
            if success:
                self.current_test_index += 1
                self.display_progress()
        
        if self.current_test_index >= len(self.data['evaluation_results']):
            print("\n🎉 All evaluations completed!")
            self.generate_summary()
    
    def generate_summary(self):
        """Generate summary analysis"""
        print("\n📈 Generating summary analysis...")
        
        # Calculate overall statistics
        completed_tests = [t for t in self.data['evaluation_results'] 
                          if t['scores']['total'] is not None]
        
        if not completed_tests:
            print("❌ No completed tests to analyze")
            return
        
        # Overall stats
        total_scores = [t['scores']['total'] for t in completed_tests]
        avg_score = sum(total_scores) / len(total_scores)
        
        grade_dist = {
            'excellent_9_10': len([s for s in total_scores if s >= 9]),
            'very_good_7_8': len([s for s in total_scores if 7 <= s < 9]),
            'good_5_6': len([s for s in total_scores if 5 <= s < 7]),
            'fair_3_4': len([s for s in total_scores if 3 <= s < 5]),
            'poor_1_2': len([s for s in total_scores if s < 3])
        }
        
        self.data['summary_analysis']['overall_statistics'] = {
            'total_tests_completed': len(completed_tests),
            'average_score': round(avg_score, 2),
            'grade_distribution': grade_dist
        }
        
        # Category analysis
        categories = self.data['test_categories']
        for cat_key, cat_info in categories.items():
            cat_tests = [t for t in completed_tests if t['category'] == cat_key]
            if cat_tests:
                cat_scores = [t['scores']['total'] for t in cat_tests]
                cat_avg = sum(cat_scores) / len(cat_scores)
                
                self.data['summary_analysis']['category_performance'][cat_key]['average_score'] = round(cat_avg, 2)
        
        self.save_progress()
        
        print(f"📊 Summary: {len(completed_tests)} tests completed")
        print(f"📈 Average score: {avg_score:.2f}/10")
        print(f"🎯 Grade distribution:")
        for grade, count in grade_dist.items():
            if count > 0:
                print(f"   {grade.replace('_', ' ').title()}: {count}")

def main():
    """Main function"""
    template_file = "ai_istanbul_comprehensive_evaluation_results.json"
    
    if not os.path.exists(template_file):
        print(f"❌ Template file not found: {template_file}")
        print("Please ensure the evaluation template file exists in the current directory.")
        return
    
    evaluator = AIResponseEvaluator(template_file)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
