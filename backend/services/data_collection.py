"""
Chat Data Collection for Model Fine-tuning
==========================================
Logs all chat interactions for Llama 3.1 fine-tuning dataset

Features:
- Logs user queries and LLM responses
- Tracks user feedback (thumbs up/down)
- Records response quality metrics
- Exports to JSONL format for training
- Privacy-compliant (anonymized data)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib

logger = logging.getLogger(__name__)

# Data collection settings
DATA_DIR = Path("training_data")
DATA_DIR.mkdir(exist_ok=True)

CHAT_LOGS_FILE = DATA_DIR / "chat_logs.jsonl"
FEEDBACK_FILE = DATA_DIR / "user_feedback.jsonl"
STATS_FILE = DATA_DIR / "collection_stats.json"


class DataCollector:
    """Collects chat data for model fine-tuning"""
    
    def __init__(self):
        self.stats = self.load_stats()
        
    def load_stats(self) -> Dict[str, Any]:
        """Load collection statistics"""
        if STATS_FILE.exists():
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        return {
            'total_interactions': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'languages': {},
            'intents': {},
            'started_at': datetime.now().isoformat()
        }
    
    def save_stats(self):
        """Save collection statistics"""
        self.stats['updated_at'] = datetime.now().isoformat()
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID with hash"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def log_interaction(
        self,
        user_query: str,
        llm_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a chat interaction for training data
        
        Args:
            user_query: User's question/query
            llm_response: LLM's response
            metadata: Additional context (language, intent, etc.)
            
        Returns:
            Interaction ID for feedback tracking
        """
        metadata = metadata or {}
        
        # Generate interaction ID
        interaction_id = hashlib.sha256(
            f"{user_query}{llm_response}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Anonymize user ID if present
        if 'user_id' in metadata:
            metadata['user_id'] = self.anonymize_user_id(metadata['user_id'])
        
        # Build log entry
        log_entry = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query.strip(),
            'llm_response': llm_response.strip(),
            'language': metadata.get('language', 'en'),
            'intent': metadata.get('intent', 'general'),
            'response_time': metadata.get('response_time'),
            'cached': metadata.get('cached', False),
            'method': metadata.get('method', 'llm'),
            'query_length': len(user_query),
            'response_length': len(llm_response),
            'has_map_data': metadata.get('has_map_data', False),
            'session_id': metadata.get('session_id'),
            'user_id': metadata.get('user_id')
        }
        
        # Append to JSONL file
        with open(CHAT_LOGS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Update stats
        self.stats['total_interactions'] += 1
        
        language = metadata.get('language', 'en')
        self.stats['languages'][language] = self.stats['languages'].get(language, 0) + 1
        
        intent = metadata.get('intent', 'general')
        self.stats['intents'][intent] = self.stats['intents'].get(intent, 0) + 1
        
        self.save_stats()
        
        logger.info(f"Logged interaction {interaction_id} for training data")
        return interaction_id
    
    def log_feedback(
        self,
        interaction_id: str,
        feedback_type: str,
        feedback_value: Optional[str] = None,
        comment: Optional[str] = None
    ):
        """
        Log user feedback on a response
        
        Args:
            interaction_id: ID from log_interaction
            feedback_type: 'thumbs_up', 'thumbs_down', 'rating', 'comment'
            feedback_value: Value for rating or additional data
            comment: User comment (optional)
        """
        feedback_entry = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'comment': comment
        }
        
        # Append to feedback file
        with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        # Update stats
        if feedback_type == 'thumbs_up':
            self.stats['positive_feedback'] += 1
        elif feedback_type == 'thumbs_down':
            self.stats['negative_feedback'] += 1
        
        self.save_stats()
        
        logger.info(f"Logged {feedback_type} feedback for {interaction_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            **self.stats,
            'feedback_rate': (
                (self.stats['positive_feedback'] + self.stats['negative_feedback']) 
                / max(self.stats['total_interactions'], 1) * 100
            ),
            'positive_rate': (
                self.stats['positive_feedback'] 
                / max(self.stats['positive_feedback'] + self.stats['negative_feedback'], 1) * 100
            )
        }
    
    def export_training_data(
        self,
        output_file: str = None,
        filter_positive_only: bool = False,
        min_response_length: int = 20,
        max_response_length: int = 500
    ) -> int:
        """
        Export collected data to training format
        
        Args:
            output_file: Output JSONL file path
            filter_positive_only: Only include positively rated responses
            min_response_length: Minimum response length
            max_response_length: Maximum response length
            
        Returns:
            Number of training examples exported
        """
        output_file = output_file or str(DATA_DIR / "training_dataset.jsonl")
        
        # Load feedback data
        feedback_map = {}
        if FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    fb = json.loads(line)
                    interaction_id = fb['interaction_id']
                    if interaction_id not in feedback_map:
                        feedback_map[interaction_id] = []
                    feedback_map[interaction_id].append(fb)
        
        # Process chat logs
        count = 0
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with open(CHAT_LOGS_FILE, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    log = json.loads(line)
                    
                    # Apply filters
                    response_len = log['response_length']
                    if response_len < min_response_length or response_len > max_response_length:
                        continue
                    
                    # Check feedback filter
                    if filter_positive_only:
                        feedbacks = feedback_map.get(log['interaction_id'], [])
                        has_positive = any(
                            fb['feedback_type'] == 'thumbs_up' for fb in feedbacks
                        )
                        if not has_positive:
                            continue
                    
                    # Format for training (Alpaca/Instruction format)
                    training_example = {
                        'instruction': log['user_query'],
                        'output': log['llm_response'],
                        'language': log['language'],
                        'intent': log['intent'],
                        'metadata': {
                            'interaction_id': log['interaction_id'],
                            'response_time': log['response_time'],
                            'has_map_data': log['has_map_data']
                        }
                    }
                    
                    out_f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
                    count += 1
        
        logger.info(f"Exported {count} training examples to {output_file}")
        return count


# Global instance
data_collector = DataCollector()


def log_chat_interaction(
    user_query: str,
    llm_response: str,
    **metadata
) -> str:
    """Convenience function to log chat interaction"""
    return data_collector.log_interaction(user_query, llm_response, metadata)


def log_user_feedback(
    interaction_id: str,
    feedback_type: str,
    **kwargs
):
    """Convenience function to log user feedback"""
    data_collector.log_feedback(interaction_id, feedback_type, **kwargs)


def get_collection_stats() -> Dict[str, Any]:
    """Convenience function to get statistics"""
    return data_collector.get_stats()


def export_training_dataset(**kwargs) -> int:
    """Convenience function to export training data"""
    return data_collector.export_training_data(**kwargs)
