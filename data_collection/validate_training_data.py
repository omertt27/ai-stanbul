"""
Training Data Validation and Analysis Script
Validates the quality and format of training data for the Istanbul tourism model
"""

import json
import os
from pathlib import Path
import pandas as pd
from collections import Counter
import re
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingDataValidator:
    """Validate and analyze training data quality"""
    
    def __init__(self, data_directory: str = "./data/training"):
        self.data_dir = Path(data_directory)
        self.validation_results = {}
        
    def load_training_data(self) -> Dict[str, List[Dict]]:
        """Load all training data files"""
        data = {}
        
        # Load different data formats
        for file_type in ['qa_training_data.jsonl', 'conversation_training_data.jsonl', 'instruction_training_data.jsonl']:
            file_path = self.data_dir / file_type
            if file_path.exists():
                data[file_type] = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data[file_type].append(json.loads(line.strip()))
                print(f"Loaded {len(data[file_type])} examples from {file_type}")
            else:
                print(f"Warning: {file_type} not found")
        
        return data
    
    def validate_data_format(self, data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Validate data format and structure"""
        validation_results = {}
        
        for file_type, examples in data.items():
            results = {
                'total_examples': len(examples),
                'valid_examples': 0,
                'format_errors': [],
                'missing_fields': [],
                'empty_fields': []
            }
            
            for i, example in enumerate(examples):
                is_valid = True
                
                # Check required fields based on data type
                if file_type == 'qa_training_data.jsonl':
                    required_fields = ['question', 'answer']
                elif file_type == 'conversation_training_data.jsonl':
                    required_fields = ['conversation']  # Changed to match actual format
                elif file_type == 'instruction_training_data.jsonl':
                    required_fields = ['instruction', 'output']  # input can be empty
                else:
                    required_fields = ['text']
                
                # Check for missing fields
                missing = [field for field in required_fields if field not in example]
                if missing:
                    results['missing_fields'].extend(missing)
                    is_valid = False
                
                # Check for empty fields
                empty = []
                for field in required_fields:
                    if field in example:
                        if isinstance(example[field], str) and not example[field].strip():
                            empty.append(field)
                        elif isinstance(example[field], list) and not example[field]:
                            empty.append(field)
                if empty:
                    results['empty_fields'].extend(empty)
                    is_valid = False
                
                if is_valid:
                    results['valid_examples'] += 1
                else:
                    results['format_errors'].append(f"Example {i}: missing={missing}, empty={empty}")
            
            # Calculate validation rate
            results['validation_rate'] = results['valid_examples'] / results['total_examples'] if results['total_examples'] > 0 else 0
            
            validation_results[file_type] = results
        
        return validation_results
    
    def analyze_content_quality(self, data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        quality_results = {}
        
        for file_type, examples in data.items():
            all_text = []
            
            # Extract text content based on format
            for example in examples:
                if file_type == 'qa_training_data.jsonl':
                    all_text.extend([example.get('question', ''), example.get('answer', '')])
                elif file_type == 'conversation_training_data.jsonl':
                    # Handle conversation format
                    if 'conversation' in example:
                        for turn in example['conversation']:
                            all_text.append(turn.get('content', ''))
                    else:
                        all_text.append(example.get('text', ''))
                elif file_type == 'instruction_training_data.jsonl':
                    all_text.extend([
                        example.get('instruction', ''),
                        example.get('input', ''),
                        example.get('output', '')
                    ])
            
            # Remove empty strings
            all_text = [text for text in all_text if text.strip()]
            
            if not all_text:
                quality_results[file_type] = {'error': 'No valid text content found'}
                continue
            
            # Calculate quality metrics
            results = {
                'total_text_entries': len(all_text),
                'avg_length': sum(len(text) for text in all_text) / len(all_text),
                'min_length': min(len(text) for text in all_text),
                'max_length': max(len(text) for text in all_text),
                'turkish_content_ratio': self._estimate_turkish_ratio(all_text),
                'istanbul_terms_found': self._count_istanbul_terms(all_text),
                'question_patterns': self._count_question_patterns(all_text),
                'avg_words_per_entry': sum(len(text.split()) for text in all_text) / len(all_text)
            }
            
            quality_results[file_type] = results
        
        return quality_results
    
    def _estimate_turkish_ratio(self, texts: List[str]) -> float:
        """Estimate ratio of Turkish content"""
        turkish_indicators = [
            'istanbul', 'türkiye', 'nerede', 'nasıl', 'ne zaman', 'kaç', 'hangi',
            'çok', 'güzel', 'tarihi', 'müze', 'cami', 'saray', 'köprü',
            'ğ', 'ü', 'ö', 'ş', 'ı', 'ç'  # Turkish characters
        ]
        
        turkish_count = 0
        total_count = len(texts)
        
        for text in texts:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in turkish_indicators):
                turkish_count += 1
        
        return turkish_count / total_count if total_count > 0 else 0
    
    def _count_istanbul_terms(self, texts: List[str]) -> Dict[str, int]:
        """Count Istanbul-specific terms in the text"""
        istanbul_terms = [
            'istanbul', 'sultanahmet', 'galata', 'bosphorus', 'boğaz',
            'hagia sophia', 'ayasofya', 'topkapi', 'grand bazaar',
            'blue mosque', 'taksim', 'beyoglu', 'kadikoy'
        ]
        
        term_counts = Counter()
        
        for text in texts:
            text_lower = text.lower()
            for term in istanbul_terms:
                term_counts[term] += text_lower.count(term)
        
        return dict(term_counts.most_common(10))
    
    def _count_question_patterns(self, texts: List[str]) -> Dict[str, int]:
        """Count question patterns in the text"""
        question_patterns = {
            'what': r'\\b(what|ne)\\b',
            'where': r'\\b(where|nerede)\\b',
            'when': r'\\b(when|ne zaman)\\b',
            'how': r'\\b(how|nasıl)\\b',
            'which': r'\\b(which|hangi)\\b',
            'why': r'\\b(why|neden)\\b'
        }
        
        pattern_counts = {}
        
        for pattern_name, pattern in question_patterns.items():
            count = 0
            for text in texts:
                count += len(re.findall(pattern, text, re.IGNORECASE))
            pattern_counts[pattern_name] = count
        
        return pattern_counts
    
    def generate_report(self, data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\\n" + "="*60)
        print("TRAINING DATA VALIDATION REPORT")
        print("="*60)
        
        # Validate format
        format_results = self.validate_data_format(data)
        
        # Analyze quality
        quality_results = self.analyze_content_quality(data)
        
        # Summary statistics
        total_examples = sum(results['total_examples'] for results in format_results.values())
        total_valid = sum(results['valid_examples'] for results in format_results.values())
        overall_validation_rate = total_valid / total_examples if total_examples > 0 else 0
        
        print(f"\\nOVERALL SUMMARY:")
        print(f"Total Training Examples: {total_examples}")
        print(f"Valid Examples: {total_valid}")
        print(f"Overall Validation Rate: {overall_validation_rate:.2%}")
        
        # Detailed results by file type
        for file_type in format_results.keys():
            print(f"\\n{file_type.upper()}:")
            print(f"  Examples: {format_results[file_type]['total_examples']}")
            print(f"  Valid: {format_results[file_type]['valid_examples']}")
            print(f"  Validation Rate: {format_results[file_type]['validation_rate']:.2%}")
            
            if file_type in quality_results and 'error' not in quality_results[file_type]:
                qr = quality_results[file_type]
                print(f"  Avg Length: {qr['avg_length']:.0f} chars")
                print(f"  Avg Words: {qr['avg_words_per_entry']:.0f} words")
                print(f"  Turkish Content: {qr['turkish_content_ratio']:.2%}")
                
                # Show top Istanbul terms
                if qr['istanbul_terms_found']:
                    top_terms = list(qr['istanbul_terms_found'].items())[:3]
                    print(f"  Top Istanbul Terms: {', '.join([f'{term}({count})' for term, count in top_terms])}")
        
        # Compilation report
        report = {
            'validation_summary': {
                'total_examples': total_examples,
                'valid_examples': total_valid,
                'validation_rate': overall_validation_rate,
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'format_validation': format_results,
            'quality_analysis': quality_results,
            'recommendations': self._generate_recommendations(format_results, quality_results)
        }
        
        return report
    
    def _generate_recommendations(self, format_results: Dict, quality_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check validation rates
        for file_type, results in format_results.items():
            if results['validation_rate'] < 0.95:
                recommendations.append(f"Improve data quality for {file_type} (validation rate: {results['validation_rate']:.2%})")
        
        # Check content quality
        for file_type, results in quality_results.items():
            if 'error' in results:
                recommendations.append(f"Fix content issues in {file_type}: {results['error']}")
                continue
                
            if results.get('avg_length', 0) < 50:
                recommendations.append(f"Increase content length for {file_type} (avg: {results['avg_length']:.0f} chars)")
            
            if results.get('turkish_content_ratio', 0) < 0.3:
                recommendations.append(f"Add more Turkish content to {file_type} (current: {results['turkish_content_ratio']:.2%})")
        
        if not recommendations:
            recommendations.append("Training data quality looks good! Ready for model training.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], output_file: str = "training_data_validation_report.json"):
        """Save validation report to file"""
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\\nValidation report saved to: {output_path}")

def main():
    """Main validation function"""
    print("Starting training data validation...")
    
    # Initialize validator
    validator = TrainingDataValidator()
    
    # Load training data
    data = validator.load_training_data()
    
    if not data:
        print("No training data found. Please run the data collection pipeline first.")
        return
    
    # Generate validation report
    report = validator.generate_report(data)
    
    # Save report
    validator.save_report(report)
    
    # Print recommendations
    print(f"\\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\\nValidation complete!")

if __name__ == "__main__":
    main()
