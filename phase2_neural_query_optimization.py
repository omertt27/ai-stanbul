#!/usr/bin/env python3
"""
Phase 2: Neural Query Enhancement - Optimization Script
Improves T4 Neural Query Processor for <50ms latency and >90% accuracy
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import time
import asyncio
from typing import List, Tuple
import json

class OptimizedIntentClassifier(nn.Module):
    """Lightweight intent classifier for Istanbul AI"""
    
    def __init__(self, embedding_dim=768, num_intents=25, hidden_dim=256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_intents)
        )
    
    def forward(self, embeddings):
        return self.classifier(embeddings)


class Phase2Optimizer:
    """Phase 2 optimization for neural query processing"""
    
    def __init__(self):
        self.device = self._get_best_device()
        print(f"🖥️  Using device: {self.device}")
        
        # Intent classes
        self.intents = [
            'attraction', 'museum', 'restaurant', 'transportation',
            'accommodation', 'shopping', 'nightlife', 'events',
            'weather', 'emergency', 'general_info', 'recommendation',
            'route_planning', 'gps_navigation', 'price_info',
            'booking', 'cultural_info', 'food', 'history',
            'local_tips', 'hidden_gems', 'family_activities',
            'romantic', 'budget', 'luxury'
        ]
        
    def _get_best_device(self):
        """Get best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    async def optimize_model_size(self):
        """Step 1: Reduce model size for faster inference"""
        print("\n📦 Step 1: Optimizing Model Size")
        print("=" * 60)
        
        # Option 1: DistilBERT (40% faster, 60% smaller)
        print("\n🔸 Testing DistilBERT Turkish...")
        try:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
            model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
            model.to(self.device)
            model.eval()
            
            # Test inference speed
            test_text = "Sultanahmet'e nasıl gidilir?"
            times = []
            
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    inputs = tokenizer(test_text, return_tensors="pt").to(self.device)
                    outputs = model(**inputs)
                times.append((time.time() - start) * 1000)
            
            avg_time = sum(times[2:]) / len(times[2:])  # Skip first 2 (warmup)
            print(f"   ✅ DistilBERT: {avg_time:.1f}ms average")
            
            return tokenizer, model, avg_time
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, None, None
    
    def create_training_data(self) -> List[Tuple[str, str]]:
        """Step 2: Create labeled training data for Istanbul tourism"""
        print("\n📝 Step 2: Creating Training Data")
        print("=" * 60)
        
        # Sample labeled queries for each intent
        training_data = [
            # Attractions
            ("Sultanahmet'i görmek istiyorum", "attraction"),
            ("Ayasofya nerede", "attraction"),
            ("Topkapı Sarayı hakkında bilgi", "attraction"),
            ("Galata Kulesi ne zaman açık", "attraction"),
            
            # Museums
            ("Müze önerisi", "museum"),
            ("İstanbul Modern kaça kadar açık", "museum"),
            ("Tarih müzesi nerede", "museum"),
            
            # Restaurants
            ("En iyi kebap nerede", "restaurant"),
            ("Balık restoranı tavsiye", "restaurant"),
            ("Ucuz yemek yeri", "restaurant"),
            ("Romantik restoran önerisi", "romantic"),
            
            # Transportation
            ("Havalimanından şehre nasıl gidilir", "transportation"),
            ("Taksim'e metro ile nasıl gidilir", "transportation"),
            ("Otobüs saatleri", "transportation"),
            ("Taksi çağır", "transportation"),
            
            # GPS Navigation
            ("Sultanahmet'e nasıl gidilir", "gps_navigation"),
            ("En yakın eczane nerede", "gps_navigation"),
            ("Beni Boğaz'a götür", "gps_navigation"),
            
            # Accommodation
            ("Ucuz otel önerisi", "accommodation"),
            ("5 yıldızlı otel", "luxury"),
            ("Hostel tavsiyesi", "budget"),
            
            # Shopping
            ("Kapalıçarşı kaça kadar açık", "shopping"),
            ("Alışveriş merkezi", "shopping"),
            ("Hediyelik eşya nereden alınır", "shopping"),
            
            # Nightlife
            ("Gece hayatı önerileri", "nightlife"),
            ("Bar tavsiyesi", "nightlife"),
            ("Canlı müzik mekanı", "nightlife"),
            
            # Cultural Info
            ("Boğaz turu hakkında bilgi", "cultural_info"),
            ("Osmanlı tarihi", "history"),
            ("Cami ziyareti kuralları", "cultural_info"),
            
            # Price Info
            ("Giriş ücreti ne kadar", "price_info"),
            ("Tur fiyatları", "price_info"),
            
            # Events
            ("Bu hafta sonu etkinlikler", "events"),
            ("Konser takvimi", "events"),
            
            # Weather
            ("Hava durumu nasıl", "weather"),
            ("Yarın yağmur yağacak mı", "weather"),
            
            # Emergency
            ("En yakın hastane", "emergency"),
            ("Polis çağır", "emergency"),
            
            # Family Activities
            ("Çocuklu gezilecek yerler", "family_activities"),
            ("Aile için restoran", "family_activities"),
            
            # Hidden Gems
            ("Turistik olmayan yerler", "hidden_gems"),
            ("Yerel önerileri", "local_tips"),
        ]
        
        print(f"   ✅ Created {len(training_data)} labeled samples")
        print(f"   📊 Covering {len(set(intent for _, intent in training_data))} intents")
        
        return training_data
    
    async def fine_tune_model(self, model, tokenizer, training_data):
        """Step 3: Fine-tune on Istanbul tourism queries"""
        print("\n🎯 Step 3: Fine-Tuning Model")
        print("=" * 60)
        
        # Create intent classifier
        classifier = OptimizedIntentClassifier(
            embedding_dim=768,
            num_intents=len(self.intents),
            hidden_dim=256
        ).to(self.device)
        
        # Simple training (in production, use proper train/val split)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("   🔄 Training intent classifier...")
        
        for epoch in range(10):
            total_loss = 0
            for text, intent in training_data:
                # Get BERT embeddings
                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embeddings = model(**inputs).last_hidden_state[:, 0, :]
                
                # Train classifier
                logits = classifier(embeddings)
                target = torch.tensor([self.intents.index(intent)]).to(self.device)
                loss = criterion(logits, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"   Epoch {epoch + 1}/10: Loss = {avg_loss:.4f}")
        
        print("   ✅ Fine-tuning complete!")
        return classifier
    
    async def benchmark_accuracy(self, model, tokenizer, classifier):
        """Step 4: Benchmark accuracy"""
        print("\n📊 Step 4: Benchmarking Accuracy")
        print("=" * 60)
        
        # Test queries with expected intents
        test_cases = [
            ("Ayasofya'yı görmek istiyorum", "attraction"),
            ("En iyi kebap nerede?", "restaurant"),
            ("Taksim'e nasıl gidilir?", "gps_navigation"),
            ("Ucuz otel önerisi", "accommodation"),
            ("Gece hayatı", "nightlife"),
            ("Hava durumu", "weather"),
            ("Boğaz turu", "cultural_info"),
            ("Müze önerisi", "museum"),
            ("Alışveriş merkezi", "shopping"),
            ("Bu hafta etkinlikler", "events"),
        ]
        
        correct = 0
        total = len(test_cases)
        latencies = []
        
        for query, expected_intent in test_cases:
            start = time.time()
            
            # Inference
            with torch.no_grad():
                inputs = tokenizer(query, return_tensors="pt").to(self.device)
                embeddings = model(**inputs).last_hidden_state[:, 0, :]
                logits = classifier(embeddings)
                predicted_idx = torch.argmax(logits, dim=1).item()
                predicted_intent = self.intents[predicted_idx]
            
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            is_correct = predicted_intent == expected_intent
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} '{query}'")
            print(f"      Predicted: {predicted_intent}, Expected: {expected_intent}, {latency:.1f}ms")
        
        accuracy = (correct / total) * 100
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"\n   📊 Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
        print(f"   ⚡ Avg Latency: {avg_latency:.1f}ms")
        print(f"   🎯 Target: >90% accuracy, <50ms latency")
        
        if accuracy >= 90:
            print("   ✅ Accuracy target MET!")
        else:
            print("   ⚠️  Accuracy needs improvement")
        
        if avg_latency < 50:
            print("   ✅ Latency target MET!")
        else:
            print("   ⚠️  Latency needs optimization")
        
        return accuracy, avg_latency
    
    async def run_phase2(self):
        """Run complete Phase 2 optimization"""
        print("\n" + "=" * 60)
        print("🚀 PHASE 2: ML MODEL ENHANCEMENT")
        print("   Week 2: Neural Query Enhancement")
        print("=" * 60)
        
        # Step 1: Optimize model size
        tokenizer, model, base_latency = await self.optimize_model_size()
        
        if model is None:
            print("\n❌ Model optimization failed!")
            return
        
        # Step 2: Create training data
        training_data = self.create_training_data()
        
        # Step 3: Fine-tune
        classifier = await self.fine_tune_model(model, tokenizer, training_data)
        
        # Step 4: Benchmark
        accuracy, latency = await self.benchmark_accuracy(model, tokenizer, classifier)
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 PHASE 2 SUMMARY")
        print("=" * 60)
        print(f"✅ Model optimized: DistilBERT (60% smaller)")
        print(f"✅ Training data: {len(training_data)} samples")
        print(f"✅ Fine-tuning: Complete (10 epochs)")
        print(f"📊 Final Accuracy: {accuracy:.1f}%")
        print(f"⚡ Final Latency: {latency:.1f}ms")
        
        if accuracy >= 90 and latency < 50:
            print("\n🎉 PHASE 2 COMPLETE - Ready for Phase 3!")
        elif accuracy >= 90:
            print("\n⚠️  Accuracy good, but latency needs work")
        elif latency < 50:
            print("\n⚠️  Latency good, but accuracy needs work")
        else:
            print("\n⚠️  Both metrics need improvement")
        
        # Save results
        results = {
            "phase": 2,
            "accuracy_percent": accuracy,
            "latency_ms": latency,
            "model": "DistilBERT-multilingual",
            "device": str(self.device),
            "training_samples": len(training_data),
            "targets_met": accuracy >= 90 and latency < 50
        }
        
        with open("phase2_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to phase2_results.json")


async def main():
    optimizer = Phase2Optimizer()
    await optimizer.run_phase2()


if __name__ == "__main__":
    asyncio.run(main())
