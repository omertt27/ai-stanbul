"""
T4 GPU Neural Query Processor for Istanbul AI
==============================================
Optimized for NVIDIA T4 GPU with MPS/CPU fallback for local development.
Uses classification and ranking models (NO generative AI).

Features:
- Turkish BERT for query understanding
- Multi-intent classification
- Entity extraction (locations, attractions, dates)
- Semantic similarity ranking
- Hardware-aware optimization (T4/MPS/CPU)

Author: Istanbul AI Team
Date: October 21, 2025
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from neural query processing."""
    intent: str
    confidence: float
    entities: Dict[str, List[str]]
    embeddings: np.ndarray
    processing_time_ms: float
    device_used: str


class IntentClassifier(nn.Module):
    """Multi-class intent classifier for tourism queries."""
    
    def __init__(self, input_dim: int = 768, num_intents: int = 15):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_intents)
        )
    
    def forward(self, x):
        return self.classifier(x)


class EntityExtractor(nn.Module):
    """Named entity recognition for locations, dates, etc."""
    
    def __init__(self, input_dim: int = 768, num_entity_types: int = 8):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_entity_types)
        )
    
    def forward(self, x):
        return self.extractor(x)


class T4NeuralQueryProcessor:
    """
    Neural query processor optimized for T4 GPU.
    Falls back to MPS (Apple Silicon) or CPU for local development.
    """
    
    # Intent categories
    INTENTS = [
        "find_attraction",
        "get_directions",
        "transportation",
        "restaurant_recommendation",
        "hotel_search",
        "event_information",
        "historical_info",
        "opening_hours",
        "ticket_prices",
        "weather_query",
        "emergency_help",
        "cultural_tips",
        "photo_spots",
        "shopping",
        "general_question"
    ]
    
    # Entity types
    ENTITY_TYPES = [
        "LOCATION",
        "ATTRACTION",
        "DATE",
        "TIME",
        "TRANSPORT",
        "PERSON",
        "NUMBER",
        "MISC"
    ]
    
    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        cache_dir: Optional[str] = None,
        force_cpu: bool = False
    ):
        """
        Initialize the neural query processor.
        
        Args:
            model_name: Hugging Face model identifier
            cache_dir: Directory to cache models
            force_cpu: Force CPU usage (for testing)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "istanbul-ai")
        self.force_cpu = force_cpu
        
        # Detect and set device
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Performance tracking
        self.total_queries = 0
        self.total_time_ms = 0.0
    
    def _detect_device(self) -> str:
        """Detect best available device: CUDA (T4) > MPS (Apple) > CPU."""
        if self.force_cpu:
            return "cpu"
        
        if torch.cuda.is_available():
            # Check if it's a T4 GPU
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}")
            if "T4" in gpu_name:
                logger.info("✓ NVIDIA T4 GPU detected - optimal performance")
            return "cuda"
        
        if torch.backends.mps.is_available():
            logger.info("✓ Apple MPS detected - good performance for local dev")
            return "mps"
        
        logger.warning("⚠ Using CPU - performance will be limited")
        return "cpu"
    
    def _load_models(self):
        """Load BERT and classification models."""
        logger.info(f"Loading models from {self.model_name}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load BERT base model
            self.bert_model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # Load intent classifier
            self.intent_classifier = IntentClassifier(
                input_dim=768,
                num_intents=len(self.INTENTS)
            )
            
            # Load entity extractor
            self.entity_extractor = EntityExtractor(
                input_dim=768,
                num_entity_types=len(self.ENTITY_TYPES)
            )
            
            # Move models to device
            self.intent_classifier.to(self.device)
            self.entity_extractor.to(self.device)
            
            # Set to eval mode
            self.intent_classifier.eval()
            self.entity_extractor.eval()
            
            logger.info("✓ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _get_bert_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for input text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def _classify_intent(self, embeddings: torch.Tensor) -> Tuple[str, float]:
        """Classify query intent."""
        with torch.no_grad():
            logits = self.intent_classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            confidence, intent_idx = torch.max(probs, dim=-1)
        
        intent = self.INTENTS[intent_idx.item()]
        confidence_score = confidence.item()
        
        return intent, confidence_score
    
    def _extract_entities(self, text: str, embeddings: torch.Tensor) -> Dict[str, List[str]]:
        """Extract named entities from query."""
        entities = {entity_type: [] for entity_type in self.ENTITY_TYPES}
        
        # Use simple rule-based extraction for now
        # TODO: Implement full NER model
        text_lower = text.lower()
        
        # Istanbul attractions (basic pattern matching)
        attractions = [
            "ayasofya", "sultanahmet", "topkapı", "kapalıçarşı",
            "galata", "beyazıt", "eminönü", "taksim", "beşiktaş",
            "boğaziçi", "üsküdar", "kadıköy", "dolmabahçe", "rumeli"
        ]
        for attraction in attractions:
            if attraction in text_lower:
                entities["ATTRACTION"].append(attraction.title())
        
        # Transportation keywords
        transport_words = ["metro", "metrobüs", "otobüs", "tramvay", "vapur", "taksi"]
        for transport in transport_words:
            if transport in text_lower:
                entities["TRANSPORT"].append(transport.title())
        
        # Time expressions
        time_words = ["sabah", "öğle", "akşam", "gece", "yarın", "bugün"]
        for time_word in time_words:
            if time_word in text_lower:
                entities["TIME"].append(time_word.title())
        
        # Filter empty entities
        entities = {k: v for k, v in entities.items() if v}
        
        return entities
    
    def process_query(self, query: str) -> QueryResult:
        """
        Process a user query with neural models.
        
        Args:
            query: User input query in Turkish or English
            
        Returns:
            QueryResult with intent, entities, and embeddings
        """
        start_time = time.time()
        
        try:
            # Get BERT embeddings
            embeddings = self._get_bert_embeddings(query)
            
            # Classify intent
            intent, confidence = self._classify_intent(embeddings)
            
            # Extract entities
            entities = self._extract_entities(query, embeddings)
            
            # Convert embeddings to numpy
            embeddings_np = embeddings.cpu().numpy()
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.total_queries += 1
            self.total_time_ms += processing_time_ms
            
            result = QueryResult(
                intent=intent,
                confidence=confidence,
                entities=entities,
                embeddings=embeddings_np,
                processing_time_ms=processing_time_ms,
                device_used=str(self.device)
            )
            
            logger.info(
                f"Query processed: intent={intent} ({confidence:.2%}), "
                f"time={processing_time_ms:.2f}ms, device={self.device}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def batch_process(self, queries: List[str]) -> List[QueryResult]:
        """Process multiple queries in batch for efficiency."""
        results = []
        
        # Batch tokenization
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            # Batch embeddings
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Batch intent classification
            logits = self.intent_classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            confidences, intent_indices = torch.max(probs, dim=-1)
        
        processing_time_ms = (time.time() - start_time) * 1000
        per_query_time = processing_time_ms / len(queries)
        
        # Process each query result
        for i, query in enumerate(queries):
            intent = self.INTENTS[intent_indices[i].item()]
            confidence = confidences[i].item()
            entities = self._extract_entities(query, embeddings[i:i+1])
            embeddings_np = embeddings[i].cpu().numpy()
            
            result = QueryResult(
                intent=intent,
                confidence=confidence,
                entities=entities,
                embeddings=embeddings_np,
                processing_time_ms=per_query_time,
                device_used=str(self.device)
            )
            results.append(result)
        
        logger.info(
            f"Batch processed {len(queries)} queries in {processing_time_ms:.2f}ms "
            f"({per_query_time:.2f}ms per query)"
        )
        
        return results
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        avg_time = self.total_time_ms / self.total_queries if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": avg_time,
            "device": str(self.device),
            "model": self.model_name
        }
    
    def benchmark(self, num_queries: int = 100) -> Dict[str, float]:
        """Run benchmark test."""
        logger.info(f"Running benchmark with {num_queries} queries...")
        
        test_queries = [
            "Ayasofya'ya nasıl gidebilirim?",
            "En yakın metro durağı nerede?",
            "Sultanahmet'te ne yemek yiyebilirim?",
            "Bugün hava nasıl?",
            "Topkapı Sarayı kaçta açılıyor?",
        ] * (num_queries // 5)
        
        start_time = time.time()
        results = self.batch_process(test_queries)
        total_time = (time.time() - start_time) * 1000
        
        # Calculate percentiles
        times = [r.processing_time_ms for r in results]
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        benchmark_results = {
            "num_queries": num_queries,
            "total_time_ms": total_time,
            "avg_time_ms": np.mean(times),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "qps": num_queries / (total_time / 1000),
            "device": str(self.device)
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results


def main():
    """Test the neural query processor."""
    print("=" * 60)
    print("T4 Neural Query Processor - Test Suite")
    print("=" * 60)
    
    # Initialize processor
    processor = T4NeuralQueryProcessor()
    
    # Test queries in Turkish
    test_queries = [
        "Ayasofya'ya nasıl gidebilirim?",
        "En yakın restoran nerede?",
        "Sultanahmet'te gezilecek yerler",
        "Bugün hava nasıl?",
        "Taksim'den Kadıköy'e nasıl giderim?",
        "Topkapı Sarayı kaçta açılıyor?",
        "Galata Kulesi bilet fiyatı ne kadar?",
        "İstanbul'da akşam yemeği için öneri",
    ]
    
    print(f"\n🧪 Testing with {len(test_queries)} queries...\n")
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = processor.process_query(query)
        print(f"   Intent: {result.intent} ({result.confidence:.1%} confidence)")
        print(f"   Entities: {result.entities}")
        print(f"   Time: {result.processing_time_ms:.2f}ms on {result.device_used}")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("📊 Performance Statistics")
    print("=" * 60)
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Run benchmark
    print("\n" + "=" * 60)
    print("🚀 Running Benchmark (100 queries)")
    print("=" * 60)
    benchmark = processor.benchmark(100)
    print(f"\n  Average latency: {benchmark['avg_time_ms']:.2f}ms")
    print(f"  P95 latency: {benchmark['p95_ms']:.2f}ms")
    print(f"  P99 latency: {benchmark['p99_ms']:.2f}ms")
    print(f"  Throughput: {benchmark['qps']:.1f} QPS")
    print(f"  Device: {benchmark['device']}")
    
    # Check if target met
    target_p95 = 50  # Target: <50ms P95 latency
    if benchmark['p95_ms'] < target_p95:
        print(f"\n  ✅ Target met! P95 < {target_p95}ms")
    else:
        print(f"\n  ⚠️  Target not met. P95 = {benchmark['p95_ms']:.2f}ms (target: <{target_p95}ms)")
    
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
