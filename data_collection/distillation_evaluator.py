"""
Istanbul Tourism Model Evaluation Script
Comprehensive evaluation of the distilled Istanbul tourism model
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from rouge_score import rouge_scorer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IstanbulModelEvaluator:
    """Comprehensive evaluator for Istanbul tourism model"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.load_model()
        self.setup_metrics()
        
    def load_model(self):
        """Load the trained Istanbul tourism model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to base model
            logger.info("Loading base GPT-2 medium as fallback")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        
    def setup_metrics(self):
        """Setup evaluation metrics"""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Istanbul-specific terms for domain evaluation
        self.istanbul_terms = [
            'istanbul', 'sultanahmet', 'hagia sophia', 'ayasofya', 'topkapi', 'topkapı',
            'galata', 'galata tower', 'bosphorus', 'boğaz', 'grand bazaar', 'kapalıçarşı',
            'blue mosque', 'taksim', 'beyoğlu', 'kadıköy', 'üsküdar', 'dolmabahçe',
            'basilica cistern', 'spice bazaar', 'eminönü', 'ortaköy', 'beşiktaş'
        ]
        
        # Tourism categories
        self.tourism_categories = {
            'attractions': ['museum', 'palace', 'mosque', 'church', 'tower', 'bazaar'],
            'transportation': ['metro', 'bus', 'ferry', 'taxi', 'airport', 'station'],
            'food': ['restaurant', 'café', 'kebab', 'turkish', 'cuisine', 'dining'],
            'accommodation': ['hotel', 'hostel', 'stay', 'booking', 'room'],
            'culture': ['ottoman', 'byzantine', 'history', 'culture', 'tradition']
        }
        
    def generate_response(self, prompt: str, max_length: int = 256, 
                         temperature: float = 0.7, do_sample: bool = True) -> str:
        """Generate response from model"""
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract only the generated part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        return response
    
    def evaluate_domain_knowledge(self, responses: List[str]) -> Dict[str, Any]:
        """Evaluate Istanbul tourism domain knowledge"""
        results = {
            'istanbul_term_coverage': 0,
            'category_coverage': defaultdict(int),
            'total_terms_found': 0,
            'term_frequency': defaultdict(int),
            'domain_relevance_score': 0
        }
        
        total_responses = len(responses)
        if total_responses == 0:
            return results
        
        responses_with_istanbul_terms = 0
        
        for response in responses:
            response_lower = response.lower()
            response_has_terms = False
            
            # Check for Istanbul-specific terms
            for term in self.istanbul_terms:
                if term in response_lower:
                    results['term_frequency'][term] += 1
                    results['total_terms_found'] += 1
                    response_has_terms = True
            
            if response_has_terms:
                responses_with_istanbul_terms += 1
            
            # Check category coverage
            for category, keywords in self.tourism_categories.items():
                for keyword in keywords:
                    if keyword in response_lower:
                        results['category_coverage'][category] += 1
        
        # Calculate metrics
        results['istanbul_term_coverage'] = responses_with_istanbul_terms / total_responses
        results['avg_terms_per_response'] = results['total_terms_found'] / total_responses
        
        # Domain relevance score (weighted combination)
        term_score = min(results['istanbul_term_coverage'], 1.0)
        category_score = len(results['category_coverage']) / len(self.tourism_categories)
        results['domain_relevance_score'] = 0.7 * term_score + 0.3 * category_score
        
        return results
    
    def run_comprehensive_evaluation(self, test_data_path: str = None) -> Dict[str, Any]:
        """Run comprehensive evaluation on test data"""
        logger.info("Starting comprehensive evaluation...")
        
        # Load test data
        if test_data_path and Path(test_data_path).exists():
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f if line.strip()]
        else:
            # Create sample test data
            test_data = self.create_sample_test_data()
        
        logger.info(f"Evaluating on {len(test_data)} test examples")
        
        # Generate responses
        prompts = []
        references = []
        predictions = []
        
        for item in test_data:
            if 'question' in item:
                prompt = item['question']
                reference = item.get('answer', '')
            elif 'input' in item:
                prompt = item['input']
                reference = item.get('output', '')
            else:
                continue
            
            prompts.append(prompt)
            references.append(reference)
            
            # Generate prediction
            prediction = self.generate_response(prompt)
            predictions.append(prediction)
        
        # Calculate metrics
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'test_samples': len(predictions),
            'metrics': {}
        }
        
        # Domain knowledge
        domain_results = self.evaluate_domain_knowledge(predictions)
        results['metrics']['domain'] = domain_results
        
        # Sample predictions for inspection
        results['sample_predictions'] = []
        for i in range(min(5, len(prompts))):
            results['sample_predictions'].append({
                'prompt': prompts[i],
                'reference': references[i] if i < len(references) else '',
                'prediction': predictions[i]
            })
        
        return results
    
    def create_sample_test_data(self) -> List[Dict]:
        """Create sample test data for evaluation"""
        sample_data = [
            {
                "question": "What is Hagia Sophia?",
                "answer": "Hagia Sophia is a former Byzantine church and Ottoman mosque, now a museum showcasing Istanbul's multicultural history."
            },
            {
                "question": "How do I get to Sultanahmet from the airport?",
                "answer": "You can take the metro to Zeytinburnu, then tram to Sultanahmet, or use taxi/airport shuttle services."
            },
            {
                "question": "What are the best Turkish foods to try in Istanbul?",
                "answer": "Must-try foods include kebabs, baklava, Turkish delight, döner, lahmacun, and Turkish tea or coffee."
            },
            {
                "question": "Where is the Grand Bazaar located?",
                "answer": "The Grand Bazaar is located in the Fatih district, near Sultanahmet and easily accessible by tram."
            },
            {
                "question": "What can I see from Galata Tower?",
                "answer": "From Galata Tower, you can enjoy panoramic views of Istanbul, including the Golden Horn, Bosphorus, and historic peninsula."
            }
        ]
        
        return sample_data
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("🎯 ISTANBUL TOURISM MODEL EVALUATION RESULTS")
        print("="*70)
        
        metrics = results['metrics']
        
        print(f"📊 Test Samples: {results['test_samples']}")
        print(f"📅 Evaluation Date: {results['timestamp']}")
        
        # Domain knowledge
        if 'domain' in metrics:
            print(f"\n🏛️ Istanbul Tourism Domain:")
            domain = metrics['domain']
            print(f"   Istanbul Term Coverage: {domain['istanbul_term_coverage']:.1%}")
            print(f"   Avg Terms per Response: {domain['avg_terms_per_response']:.1f}")
            print(f"   Domain Relevance Score: {domain['domain_relevance_score']:.3f}")
            print(f"   Categories Covered: {len(domain['category_coverage'])}/{len(self.tourism_categories)}")
        
        # Sample predictions
        if 'sample_predictions' in results:
            print(f"\n🔍 Sample Predictions:")
            for i, sample in enumerate(results['sample_predictions'][:3], 1):
                print(f"\n   Example {i}:")
                print(f"   Q: {sample['prompt']}")
                print(f"   A: {sample['prediction']}")
        
        print("="*70)

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Istanbul Tourism Model")
    parser.add_argument("--model_path", default="./training_environment/final", help="Path to trained model")
    parser.add_argument("--test_data", help="Path to test data file")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = IstanbulModelEvaluator(args.model_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(args.test_data)
    
    # Print summary
    evaluator.print_evaluation_summary(results)

if __name__ == "__main__":
    main()
