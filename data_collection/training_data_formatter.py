"""
Training Data Formatter
Converts collected Istanbul tourism data into training formats for LLM fine-tuning
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TrainingDataFormatter:
    """Format collected data for LLM training"""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training templates for different query types
        self.query_templates = {
            'attractions': [
                "What can you tell me about {title}?",
                "I want to visit {title}. What should I know?",
                "Tell me about the attraction {title}",
                "What's special about {title}?",
                "Give me information about {title} in Istanbul"
            ],
            'dining': [
                "Where can I eat at {title}?",
                "Tell me about the restaurant {title}",
                "What kind of food does {title} serve?",
                "Is {title} a good restaurant?",
                "What should I know about dining at {title}?"
            ],
            'transportation': [
                "How do I use {title}?",
                "Tell me about {title} in Istanbul",
                "What's the schedule for {title}?",
                "How does {title} work?",
                "Give me information about {title}"
            ],
            'cultural_heritage': [
                "What's the history of {title}?",
                "Tell me about the historical significance of {title}",
                "What should I know about {title}?",
                "Explain the cultural importance of {title}",
                "Give me background on {title}"
            ],
            'general': [
                "Tell me about {title}",
                "What is {title}?",
                "Give me information about {title}",
                "I'm interested in {title}",
                "What should I know about {title}?"
            ]
        }
        
        # Conversation starters for context-aware training
        self.context_starters = [
            "I'm planning a trip to Istanbul.",
            "I'm visiting Istanbul next week.",
            "I'm looking for recommendations in Istanbul.",
            "I'm a first-time visitor to Istanbul.",
            "I'll be in Istanbul for a few days.",
            "I'm interested in Istanbul tourism."
        ]
    
    def format_for_training(self, data_file: Path) -> Dict[str, int]:
        """Convert collected data to training format"""
        logger.info("🔄 Formatting data for LLM training...")
        
        # Load cleaned dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Generate training samples
        training_samples = []
        conversation_samples = []
        qa_samples = []
        
        for item in dataset:
            # Generate Q&A pairs
            qa_pairs = self._generate_qa_pairs(item)
            qa_samples.extend(qa_pairs)
            
            # Generate conversation samples
            conv_samples = self._generate_conversation_samples(item)
            conversation_samples.extend(conv_samples)
            
            # Generate instruction-following samples
            instruction_samples = self._generate_instruction_samples(item)
            training_samples.extend(instruction_samples)
        
        # Save different training formats
        stats = {}
        stats['qa_pairs'] = len(qa_samples)
        stats['conversations'] = len(conversation_samples)
        stats['instructions'] = len(training_samples)
        
        # Save Q&A format (for general training)
        self._save_qa_format(qa_samples)
        
        # Save conversation format (for chat training)
        self._save_conversation_format(conversation_samples)
        
        # Save instruction format (for instruction tuning)
        self._save_instruction_format(training_samples)
        
        # Save distillation format (for knowledge distillation)
        self._save_distillation_format(dataset)
        
        # Generate training statistics
        self._generate_training_stats(stats, dataset)
        
        return stats
    
    def _generate_qa_pairs(self, item: Dict) -> List[Dict]:
        """Generate question-answer pairs from data item"""
        qa_pairs = []
        
        title = item.get('title', '')
        content = item.get('content', '')
        category = item.get('category', 'general')
        metadata = item.get('metadata', {})
        
        # Get appropriate templates for category
        templates = self.query_templates.get(category, self.query_templates['general'])
        
        # Generate basic Q&A
        for template in templates[:3]:  # Use first 3 templates
            question = template.format(title=title)
            answer = self._format_answer(content, metadata, category)
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'category': category,
                'source': item.get('source', 'unknown')
            })
        
        # Generate metadata-specific questions
        if metadata.get('address'):
            qa_pairs.append({
                'question': f"Where is {title} located?",
                'answer': f"{title} is located at {metadata['address']}.",
                'category': category,
                'source': item.get('source', 'unknown')
            })
        
        if metadata.get('opening_hours'):
            qa_pairs.append({
                'question': f"What are the opening hours for {title}?",
                'answer': f"The opening hours for {title} are {metadata['opening_hours']}.",
                'category': category,
                'source': item.get('source', 'unknown')
            })
        
        return qa_pairs
    
    def _generate_conversation_samples(self, item: Dict) -> List[Dict]:
        """Generate conversation samples with context"""
        conversations = []
        
        title = item.get('title', '')
        content = item.get('content', '')
        category = item.get('category', 'general')
        
        # Multi-turn conversation
        starter = random.choice(self.context_starters)
        template = random.choice(self.query_templates.get(category, self.query_templates['general']))
        question = template.format(title=title)
        
        conversation = {
            'conversation': [
                {'role': 'user', 'content': starter},
                {'role': 'assistant', 'content': "I'd be happy to help you with your Istanbul trip! What would you like to know?"},
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': self._format_conversational_answer(content, item.get('metadata', {}), category)}
            ],
            'category': category,
            'source': item.get('source', 'unknown')
        }
        
        conversations.append(conversation)
        
        return conversations
    
    def _generate_instruction_samples(self, item: Dict) -> List[Dict]:
        """Generate instruction-following samples"""
        instructions = []
        
        title = item.get('title', '')
        content = item.get('content', '')
        category = item.get('category', 'general')
        
        # Instruction format
        instruction_prompts = [
            f"Provide information about {title} for tourists visiting Istanbul.",
            f"Explain what visitors should know about {title}.",
            f"Give a detailed description of {title} in Istanbul.",
            f"Help a tourist understand {title}."
        ]
        
        for prompt in instruction_prompts[:2]:  # Use first 2 prompts
            instructions.append({
                'instruction': prompt,
                'input': '',
                'output': self._format_answer(content, item.get('metadata', {}), category),
                'category': category,
                'source': item.get('source', 'unknown')
            })
        
        return instructions
    
    def _format_answer(self, content: str, metadata: Dict, category: str) -> str:
        """Format answer with proper structure and metadata"""
        answer_parts = [content]
        
        # Add relevant metadata
        if metadata.get('address'):
            answer_parts.append(f"\nLocation: {metadata['address']}")
        
        if metadata.get('opening_hours'):
            answer_parts.append(f"\nOpening Hours: {metadata['opening_hours']}")
        
        if metadata.get('admission_fee'):
            answer_parts.append(f"\nAdmission: {metadata['admission_fee']}")
        
        if metadata.get('price_range'):
            answer_parts.append(f"\nPrice Range: {metadata['price_range']}")
        
        # Add practical tips based on category
        if category == 'attractions':
            answer_parts.append("\nTip: Check opening hours before visiting and consider buying tickets in advance during peak season.")
        elif category == 'dining':
            answer_parts.append("\nTip: Reservations are recommended, especially for dinner. Many restaurants accept both cash and cards.")
        elif category == 'transportation':
            answer_parts.append("\nTip: Keep your Istanbul Card handy for public transportation, and check current schedules as they may change.")
        
        return ' '.join(answer_parts)
    
    def _format_conversational_answer(self, content: str, metadata: Dict, category: str) -> str:
        """Format answer in conversational tone"""
        # Make content more conversational
        conversational_starters = [
            "Great choice!",
            "You'll love this!",
            "That's a fantastic spot!",
            "Perfect for your Istanbul visit!",
            "Excellent question!"
        ]
        
        starter = random.choice(conversational_starters)
        formatted_content = self._format_answer(content, metadata, category)
        
        return f"{starter} {formatted_content}"
    
    def _save_qa_format(self, qa_samples: List[Dict]):
        """Save Q&A format for training"""
        qa_file = self.output_dir / 'qa_training_data.jsonl'
        
        with open(qa_file, 'w', encoding='utf-8') as f:
            for sample in qa_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(qa_samples)} Q&A samples to {qa_file}")
    
    def _save_conversation_format(self, conversation_samples: List[Dict]):
        """Save conversation format for chat training"""
        conv_file = self.output_dir / 'conversation_training_data.jsonl'
        
        with open(conv_file, 'w', encoding='utf-8') as f:
            for sample in conversation_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(conversation_samples)} conversation samples to {conv_file}")
    
    def _save_instruction_format(self, instruction_samples: List[Dict]):
        """Save instruction format for instruction tuning"""
        inst_file = self.output_dir / 'instruction_training_data.jsonl'
        
        with open(inst_file, 'w', encoding='utf-8') as f:
            for sample in instruction_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(instruction_samples)} instruction samples to {inst_file}")
    
    def _save_distillation_format(self, dataset: List[Dict]):
        """Save format for knowledge distillation training"""
        distill_samples = []
        
        for item in dataset:
            # Create prompts for teacher model
            title = item.get('title', '')
            category = item.get('category', 'general')
            
            templates = self.query_templates.get(category, self.query_templates['general'])
            
            for template in templates:
                prompt = template.format(title=title)
                
                distill_samples.append({
                    'prompt': prompt,
                    'context': item.get('content', ''),
                    'metadata': item.get('metadata', {}),
                    'category': category,
                    'source': item.get('source', 'unknown')
                })
        
        distill_file = self.output_dir / 'distillation_prompts.jsonl'
        with open(distill_file, 'w', encoding='utf-8') as f:
            for sample in distill_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(distill_samples)} distillation prompts to {distill_file}")
    
    def _generate_training_stats(self, stats: Dict[str, int], dataset: List[Dict]):
        """Generate training data statistics"""
        category_counts = {}
        source_counts = {}
        
        for item in dataset:
            category = item.get('category', 'unknown')
            source = item.get('source', 'unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        training_stats = {
            'generation_date': datetime.now().isoformat(),
            'total_original_items': len(dataset),
            'generated_samples': {
                'qa_pairs': stats['qa_pairs'],
                'conversations': stats['conversations'],
                'instructions': stats['instructions'],
                'total': stats['qa_pairs'] + stats['conversations'] + stats['instructions']
            },
            'data_amplification_ratio': (stats['qa_pairs'] + stats['conversations'] + stats['instructions']) / len(dataset) if dataset else 0,
            'category_distribution': category_counts,
            'source_distribution': source_counts
        }
        
        stats_file = self.output_dir / 'training_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("🎯 TRAINING DATA GENERATION SUMMARY")
        print("="*50)
        print(f"Original items: {len(dataset)}")
        print(f"Generated Q&A pairs: {stats['qa_pairs']}")
        print(f"Generated conversations: {stats['conversations']}")
        print(f"Generated instructions: {stats['instructions']}")
        print(f"Total training samples: {stats['qa_pairs'] + stats['conversations'] + stats['instructions']}")
        print(f"Data amplification: {(stats['qa_pairs'] + stats['conversations'] + stats['instructions']) / len(dataset):.1f}x" if dataset else "0x")
        print("\nCategory distribution:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        print("="*50)

def main():
    """Run training data formatting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Format curated data for training")
    parser.add_argument("--input_dir", default="./data/validated", help="Input directory with curated data")
    parser.add_argument("--output_dir", default="./data/training", help="Output directory for training data")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Input directory {input_dir} not found.")
        return
    
    # Find the latest curated data file
    curated_files = list(input_dir.glob("curated_istanbul_data_*.json"))
    if not curated_files:
        print("❌ No curated data files found. Please run data collection and curation first.")
        return
    
    # Use the most recent file
    latest_file = max(curated_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using curated data file: {latest_file}")
    
    formatter = TrainingDataFormatter()
    formatter.output_dir = Path(args.output_dir)
    stats = formatter.format_for_training(latest_file)
    
    total_samples = sum(stats.values())
    print(f"✅ Training data formatting completed. Generated {total_samples} training samples.")
    print(f"📊 Stats: {stats}")

if __name__ == "__main__":
    main()
