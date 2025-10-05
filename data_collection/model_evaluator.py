"""
Istanbul Tourism Model Evaluation Script
Week 5-8 Implementation: Comprehensive evaluation of distilled model
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import logging
from datetime import datetime
from rouge_score import rouge_scorer
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IstanbulModelEvaluator:
    """Comprehensive evaluator for Istanbul tourism model"""
    
    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize evaluators
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Istanbul-specific test cases
        self.test_cases = self.create_istanbul_test_cases()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def create_istanbul_test_cases(self) -> List[Dict[str, str]]:
        """Create comprehensive Istanbul tourism test cases"""
        return [
            # Historical Sites
            {
                'category': 'historical_sites',
                'question': 'Tell me about Hagia Sophia',
                'expected_keywords': ['Byzantine', 'Ottoman', 'museum', 'cathedral', 'mosque', 'history']
            },
            {
                'category': 'historical_sites',
                'question': 'What can you tell me about Topkapi Palace?',
                'expected_keywords': ['Ottoman', 'sultan', 'palace', 'museum', 'treasures']
            },
            {
                'category': 'historical_sites',
                'question': 'How do I visit the Blue Mosque?',
                'expected_keywords': ['mosque', 'Sultanahmet', 'prayer times', 'dress code', 'free']
            },
            
            # Transportation
            {
                'category': 'transportation',
                'question': 'How do I get from the airport to Sultanahmet?',
                'expected_keywords': ['airport', 'metro', 'taxi', 'Sultanahmet', 'transportation']
            },
            {
                'category': 'transportation',
                'question': 'What is an Istanbulkart?',
                'expected_keywords': ['public transport', 'card', 'metro', 'bus', 'ferry']
            },
            {
                'category': 'transportation',
                'question': 'How does the Istanbul metro work?',
                'expected_keywords': ['metro', 'lines', 'stations', 'ticket', 'Istanbulkart']
            },
            
            # Food and Dining
            {
                'category': 'food_dining',
                'question': 'What should I eat in Istanbul?',
                'expected_keywords': ['Turkish', 'kebab', 'baklava', 'Turkish delight', 'street food']
            },
            {
                'category': 'food_dining',
                'question': 'Where can I find good Turkish breakfast?',
                'expected_keywords': ['breakfast', 'Turkish', 'cheese', 'olives', 'tea', 'restaurant']
            },
            {
                'category': 'food_dining',
                'question': 'What is Turkish tea culture like?',
                'expected_keywords': ['tea', 'çay', 'culture', 'Turkish', 'social']
            },
            
            # Neighborhoods
            {
                'category': 'neighborhoods',
                'question': 'What is special about Galata?',
                'expected_keywords': ['Galata Tower', 'neighborhood', 'views', 'historic', 'European']
            },
            {
                'category': 'neighborhoods',
                'question': 'Tell me about Beyoğlu district',
                'expected_keywords': ['Beyoğlu', 'Taksim', 'modern', 'nightlife', 'shopping']
            },
            
            # Shopping
            {
                'category': 'shopping',
                'question': 'Where should I go shopping in Istanbul?',
                'expected_keywords': ['Grand Bazaar', 'shopping', 'markets', 'souvenirs', 'bargaining']
            },
            {
                'category': 'shopping',
                'question': 'What can I buy at the Grand Bazaar?',
                'expected_keywords': ['carpets', 'jewelry', 'souvenirs', 'spices', 'traditional']
            },
            
            # Practical Information
            {
                'category': 'practical',
                'question': 'Do I need a visa to visit Istanbul?',
                'expected_keywords': ['visa', 'passport', 'Turkey', 'requirements', 'tourist']
            },
            {
                'category': 'practical',
                'question': 'What currency is used in Istanbul?',
                'expected_keywords': ['Turkish Lira', 'currency', 'money', 'exchange', 'payment']
            },
            
            # Cultural
            {
                'category': 'cultural',
                'question': 'What should I know about Turkish culture?',
                'expected_keywords': ['culture', 'hospitality', 'customs', 'respect', 'traditions']
            },
            {
                'category': 'cultural',
                'question': 'Are there any cultural etiquette tips for Istanbul?',
                'expected_keywords': ['etiquette', 'respect', 'mosque', 'dress', 'culture']
            }
        ]
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate response from the model"""
        # Format prompt
        formatted_prompt = f"Q: {prompt}\\nA:"
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        if "\\nA:" in full_response:
            response = full_response.split("\\nA:", 1)[1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        
        return response
    
    def evaluate_keyword_coverage(self, response: str, expected_keywords: List[str]) -> float:
        """Evaluate how many expected keywords are covered in the response"""
        response_lower = response.lower()
        covered_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                covered_keywords.append(keyword)
        
        coverage_ratio = len(covered_keywords) / len(expected_keywords) if expected_keywords else 0
        return coverage_ratio, covered_keywords
    
    def evaluate_istanbul_specificity(self, response: str) -> Dict[str, Any]:
        """Evaluate Istanbul-specific content in the response"""
        istanbul_terms = [
            'istanbul', 'sultanahmet', 'galata', 'beyoğlu', 'taksim', 'kadıköy',
            'bosphorus', 'golden horn', 'asia', 'europe', 'turkish', 'ottoman',
            'hagia sophia', 'blue mosque', 'topkapi', 'grand bazaar', 'galata tower'
        ]
        
        response_lower = response.lower()
        found_terms = []
        
        for term in istanbul_terms:
            if term in response_lower:
                found_terms.append(term)
        
        return {
            'istanbul_terms_found': found_terms,
            'istanbul_terms_count': len(found_terms),
            'specificity_score': len(found_terms) / len(istanbul_terms)
        }
    
    def evaluate_response_quality(self, response: str) -> Dict[str, float]:
        """Evaluate general response quality"""
        metrics = {}
        
        # Length metrics
        metrics['length'] = len(response)
        metrics['word_count'] = len(response.split())
        
        # Readability (simple heuristics)
        sentences = re.split(r'[.!?]+', response)
        metrics['sentence_count'] = len([s for s in sentences if s.strip()])
        metrics['avg_sentence_length'] = metrics['word_count'] / max(metrics['sentence_count'], 1)
        
        # Information density (presence of specific terms)
        info_indicators = ['located', 'built', 'established', 'famous', 'known', 'popular', 'recommended']
        info_count = sum(1 for indicator in info_indicators if indicator in response.lower())
        metrics['information_density'] = info_count / metrics['word_count'] if metrics['word_count'] > 0 else 0
        
        return metrics
    
    def evaluate_test_cases(self) -> Dict[str, Any]:
        """Evaluate all test cases"""
        results = {
            'test_results': [],
            'category_scores': {},
            'overall_metrics': {}
        }
        
        category_scores = {}
        all_keyword_coverage = []
        all_istanbul_specificity = []
        all_response_lengths = []
        
        logger.info(f"Evaluating {len(self.test_cases)} test cases...")
        
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(self.test_cases)}: {test_case['question'][:50]}...")
            
            # Generate response
            response = self.generate_response(test_case['question'])
            
            # Evaluate keyword coverage
            keyword_coverage, covered_keywords = self.evaluate_keyword_coverage(
                response, test_case['expected_keywords']
            )
            
            # Evaluate Istanbul specificity
            istanbul_metrics = self.evaluate_istanbul_specificity(response)
            
            # Evaluate response quality
            quality_metrics = self.evaluate_response_quality(response)
            
            # Store results
            test_result = {
                'test_case': test_case,
                'response': response,
                'keyword_coverage': keyword_coverage,
                'covered_keywords': covered_keywords,
                'istanbul_specificity': istanbul_metrics,
                'quality_metrics': quality_metrics
            }
            
            results['test_results'].append(test_result)
            
            # Track category performance
            category = test_case['category']
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(keyword_coverage)
            
            # Track overall metrics
            all_keyword_coverage.append(keyword_coverage)
            all_istanbul_specificity.append(istanbul_metrics['specificity_score'])
            all_response_lengths.append(quality_metrics['length'])
        
        # Calculate category averages
        for category, scores in category_scores.items():
            results['category_scores'][category] = {
                'avg_keyword_coverage': np.mean(scores),
                'test_count': len(scores)
            }
        
        # Calculate overall metrics
        results['overall_metrics'] = {
            'avg_keyword_coverage': np.mean(all_keyword_coverage),
            'avg_istanbul_specificity': np.mean(all_istanbul_specificity),
            'avg_response_length': np.mean(all_response_lengths),
            'total_test_cases': len(self.test_cases)
        }
        
        return results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("=" * 70)
        report.append("ISTANBUL TOURISM MODEL - EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model Path: {self.model_path}")
        report.append(f"Test Cases: {results['overall_metrics']['total_test_cases']}")
        report.append("")
        
        # Overall Performance
        report.append("📊 OVERALL PERFORMANCE")
        report.append("-" * 30)
        metrics = results['overall_metrics']
        report.append(f"Average Keyword Coverage: {metrics['avg_keyword_coverage']:.2%}")
        report.append(f"Average Istanbul Specificity: {metrics['avg_istanbul_specificity']:.2%}")
        report.append(f"Average Response Length: {metrics['avg_response_length']:.0f} characters")
        report.append("")
        
        # Category Performance
        report.append("📋 CATEGORY PERFORMANCE")
        report.append("-" * 30)
        for category, scores in results['category_scores'].items():
            report.append(f"{category.replace('_', ' ').title()}: {scores['avg_keyword_coverage']:.2%} "
                         f"({scores['test_count']} tests)")
        report.append("")
        
        # Top Performing Examples
        report.append("🌟 TOP PERFORMING EXAMPLES")
        report.append("-" * 30)
        sorted_results = sorted(results['test_results'], 
                               key=lambda x: x['keyword_coverage'], reverse=True)
        for result in sorted_results[:3]:
            report.append(f"Q: {result['test_case']['question']}")
            report.append(f"Coverage: {result['keyword_coverage']:.2%}")
            report.append(f"Istanbul Terms: {result['istanbul_specificity']['istanbul_terms_count']}")
            report.append(f"A: {result['response'][:100]}...")
            report.append("")
        
        # Areas for Improvement
        report.append("⚠️  AREAS FOR IMPROVEMENT")
        report.append("-" * 30)
        lowest_results = sorted(results['test_results'], 
                               key=lambda x: x['keyword_coverage'])
        for result in lowest_results[:3]:
            report.append(f"Q: {result['test_case']['question']}")
            report.append(f"Coverage: {result['keyword_coverage']:.2%}")
            report.append(f"Missing keywords: {set(result['test_case']['expected_keywords']) - set(result['covered_keywords'])}")
            report.append("")
        
        return "\\n".join(report)
    
    def interactive_evaluation(self):
        """Interactive evaluation mode"""
        print("\\n🤖 Istanbul Tourism Model - Interactive Evaluation")
        print("Type your questions about Istanbul (or 'quit' to exit)")
        print("-" * 50)
        
        while True:
            question = input("\\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                response = self.generate_response(question)
                istanbul_metrics = self.evaluate_istanbul_specificity(response)
                quality_metrics = self.evaluate_response_quality(response)
                
                print(f"\\nResponse: {response}")
                print(f"\\nAnalysis:")
                print(f"  Length: {quality_metrics['length']} chars, {quality_metrics['word_count']} words")
                print(f"  Istanbul Terms: {istanbul_metrics['istanbul_terms_count']} ({istanbul_metrics['istanbul_terms_found']})")
                print(f"  Specificity Score: {istanbul_metrics['specificity_score']:.2%}")
                print("-" * 50)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Istanbul Tourism Model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer (default: same as model)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive evaluation")
    parser.add_argument("--output_file", default="evaluation_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    print("🚀 Starting Istanbul Tourism Model Evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = IstanbulModelEvaluator(args.model_path, args.tokenizer_path)
        
        if args.interactive:
            evaluator.interactive_evaluation()
        else:
            # Run comprehensive evaluation
            results = evaluator.evaluate_test_cases()
            
            # Generate and save report
            report = evaluator.generate_evaluation_report(results)
            
            # Print summary
            print("\\n" + "=" * 50)
            print("EVALUATION COMPLETED")
            print("=" * 50)
            overall = results['overall_metrics']
            print(f"Overall Keyword Coverage: {overall['avg_keyword_coverage']:.2%}")
            print(f"Istanbul Specificity: {overall['avg_istanbul_specificity']:.2%}")
            print(f"Average Response Length: {overall['avg_response_length']:.0f} chars")
            
            # Save detailed results
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            with open(args.output_file.replace('.txt', '_detailed.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\\n📄 Detailed report saved to: {args.output_file}")
            print(f"📄 JSON results saved to: {args.output_file.replace('.txt', '_detailed.json')}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
