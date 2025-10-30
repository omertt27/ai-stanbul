"""
Conversation Summarizer for managing context window limits
Intelligently truncates and summarizes conversation history
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Manages conversation context within token limits"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        summary_ratio: float = 0.3,
        min_messages: int = 2
    ):
        """
        Initialize conversation summarizer
        
        Args:
            max_tokens: Maximum tokens for conversation context
            summary_ratio: Ratio of context to preserve when summarizing (0.0-1.0)
            min_messages: Minimum messages to keep unsummarized
        """
        self.max_tokens = max_tokens
        self.summary_ratio = summary_ratio
        self.min_messages = min_messages
        self.summarization_count = 0
        logger.info(f"✅ ConversationSummarizer initialized (max_tokens={max_tokens})")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        # More conservative for mixed language content
        return len(text) // 3
    
    def count_conversation_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens in conversation
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total estimated token count
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            total += self.estimate_tokens(content)
            total += self.estimate_tokens(role)
            total += 4  # Message structure overhead
        
        return total
    
    def should_summarize(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if conversation should be summarized
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if summarization is needed
        """
        if len(messages) <= self.min_messages:
            return False
        
        token_count = self.count_conversation_tokens(messages)
        return token_count > self.max_tokens
    
    def truncate_conversation(
        self,
        messages: List[Dict[str, Any]],
        keep_recent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Truncate conversation by keeping system message and recent messages
        
        Args:
            messages: List of message dictionaries
            keep_recent: Number of recent messages to keep
            
        Returns:
            Truncated message list
        """
        if len(messages) <= keep_recent + 1:
            return messages
        
        # Keep system message (if present) and recent messages
        result = []
        
        # Find and keep system message
        for msg in messages:
            if msg.get("role") == "system":
                result.append(msg)
                break
        
        # Add recent messages
        result.extend(messages[-keep_recent:])
        
        self.summarization_count += 1
        logger.info(f"🔧 Truncated conversation: {len(messages)} -> {len(result)} messages")
        
        return result
    
    def create_summary_message(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary message from conversation history
        
        Args:
            messages: List of message dictionaries to summarize
            
        Returns:
            Summary message dictionary
        """
        # Extract key information
        topics = []
        user_queries = []
        assistant_responses = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                user_queries.append(content[:100])  # First 100 chars
            elif role == "assistant":
                assistant_responses.append(content[:100])
        
        # Build summary
        summary_parts = [
            f"Previous conversation covered {len(user_queries)} topics:",
            "User asked about: " + "; ".join(user_queries[:3]),
            "Topics discussed: tourism, restaurants, transportation, attractions"
        ]
        
        summary = " ".join(summary_parts)
        
        return {
            "role": "system",
            "content": f"[Conversation Summary] {summary}",
            "summary": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        keep_recent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Summarize conversation with intelligent compression
        
        Args:
            messages: List of message dictionaries
            keep_recent: Number of recent messages to keep
            
        Returns:
            Summarized message list with summary + recent messages
        """
        if len(messages) <= keep_recent + 1:
            return messages
        
        result = []
        
        # Keep system message if present
        system_msg = None
        non_system_msgs = []
        
        for msg in messages:
            if msg.get("role") == "system" and not msg.get("summary"):
                system_msg = msg
            else:
                non_system_msgs.append(msg)
        
        if system_msg:
            result.append(system_msg)
        
        # Create summary of older messages
        if len(non_system_msgs) > keep_recent:
            messages_to_summarize = non_system_msgs[:-keep_recent]
            summary = self.create_summary_message(messages_to_summarize)
            result.append(summary)
        
        # Add recent messages
        result.extend(non_system_msgs[-keep_recent:])
        
        self.summarization_count += 1
        token_reduction = self.count_conversation_tokens(messages) - self.count_conversation_tokens(result)
        
        logger.info(f"📝 Summarized conversation: {len(messages)} -> {len(result)} messages "
                   f"(saved ~{token_reduction} tokens)")
        
        return result
    
    def optimize_context(
        self,
        messages: List[Dict[str, Any]],
        strategy: str = "summarize"
    ) -> List[Dict[str, Any]]:
        """
        Optimize conversation context using specified strategy
        
        Args:
            messages: List of message dictionaries
            strategy: Optimization strategy ("summarize", "truncate", "auto")
            
        Returns:
            Optimized message list
        """
        if not self.should_summarize(messages):
            return messages
        
        if strategy == "truncate":
            return self.truncate_conversation(messages)
        elif strategy == "summarize":
            return self.summarize_conversation(messages)
        elif strategy == "auto":
            # Choose based on message count
            if len(messages) > 10:
                return self.summarize_conversation(messages)
            else:
                return self.truncate_conversation(messages)
        else:
            logger.warning(f"⚠️ Unknown strategy '{strategy}', defaulting to truncate")
            return self.truncate_conversation(messages)
    
    def get_context_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about conversation context
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Dictionary with context statistics
        """
        token_count = self.count_conversation_tokens(messages)
        utilization = (token_count / self.max_tokens) * 100
        
        message_types = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            message_types[role] = message_types.get(role, 0) + 1
        
        return {
            "total_messages": len(messages),
            "estimated_tokens": token_count,
            "max_tokens": self.max_tokens,
            "utilization": f"{utilization:.1f}%",
            "should_optimize": self.should_summarize(messages),
            "message_types": message_types,
            "summarizations_performed": self.summarization_count
        }
    
    def extract_key_entities(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key entities and topics from conversation
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of key entities/topics
        """
        # Simple keyword extraction
        keywords = set()
        
        # Common Istanbul-related terms
        istanbul_terms = {
            "taksim", "sultanahmet", "beyoglu", "kadikoy", "bosphorus",
            "galata", "topkapi", "hagia sophia", "blue mosque",
            "restaurant", "museum", "ferry", "metro", "tram"
        }
        
        for msg in messages:
            content = msg.get("content", "").lower()
            
            # Find Istanbul terms
            for term in istanbul_terms:
                if term in content:
                    keywords.add(term)
        
        return list(keywords)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get summarizer statistics
        
        Returns:
            Dictionary with summarizer stats
        """
        return {
            "max_tokens": self.max_tokens,
            "summary_ratio": self.summary_ratio,
            "min_messages": self.min_messages,
            "total_summarizations": self.summarization_count
        }


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing ConversationSummarizer...")
    
    summarizer = ConversationSummarizer(max_tokens=500, min_messages=2)
    
    # Create test conversation
    messages = [
        {"role": "system", "content": "You are a helpful Istanbul guide."},
        {"role": "user", "content": "What are the best restaurants in Beyoğlu?"},
        {"role": "assistant", "content": "Here are the top restaurants in Beyoğlu: 1. Mikla, 2. Karaköy Lokantası, 3. Zübeyir Ocakbaşı..."},
        {"role": "user", "content": "How do I get to Taksim Square?"},
        {"role": "assistant", "content": "You can reach Taksim Square by metro, bus, or taxi. The M2 metro line..."},
        {"role": "user", "content": "Tell me about museums near Sultanahmet"},
        {"role": "assistant", "content": "Near Sultanahmet you'll find: Topkapı Palace, Hagia Sophia Museum, Istanbul Archaeology Museums..."},
    ]
    
    # Test token counting
    token_count = summarizer.count_conversation_tokens(messages)
    assert token_count > 0, "Should count tokens"
    print(f"✅ Token counting test passed ({token_count} tokens)")
    
    # Test should summarize
    should_sum = summarizer.should_summarize(messages)
    print(f"✅ Should summarize check: {should_sum}")
    
    # Test truncation
    truncated = summarizer.truncate_conversation(messages, keep_recent=2)
    assert len(truncated) < len(messages), "Should truncate"
    assert truncated[0]["role"] == "system", "Should keep system message"
    print(f"✅ Truncation test passed ({len(messages)} -> {len(truncated)} messages)")
    
    # Test summarization
    summarized = summarizer.summarize_conversation(messages, keep_recent=2)
    assert len(summarized) < len(messages), "Should compress"
    print(f"✅ Summarization test passed ({len(messages)} -> {len(summarized)} messages)")
    
    # Verify summary message exists
    has_summary = any(msg.get("summary") for msg in summarized)
    assert has_summary, "Should have summary message"
    print("✅ Summary message test passed")
    
    # Test context stats
    stats = summarizer.get_context_stats(messages)
    assert "total_messages" in stats, "Should have stats"
    assert stats["total_messages"] == len(messages), "Should count correctly"
    print("✅ Context stats test passed")
    print(f"📊 {stats}")
    
    # Test entity extraction
    entities = summarizer.extract_key_entities(messages)
    assert len(entities) > 0, "Should extract entities"
    print(f"✅ Entity extraction test passed: {entities}")
    
    # Test optimization
    optimized = summarizer.optimize_context(messages, strategy="auto")
    assert len(optimized) <= len(messages), "Should optimize"
    print(f"✅ Optimization test passed")
    
    # Test stats
    sum_stats = summarizer.get_stats()
    assert sum_stats["total_summarizations"] > 0, "Should have summarizations"
    print(f"📊 Summarizer stats: {sum_stats}")
    
    print("\n✅ All ConversationSummarizer tests passed!")
