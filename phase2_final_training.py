#!/usr/bin/env python3
"""
Phase 2 FINAL TRAINING: Train with comprehensive balanced dataset
Target: >90% accuracy, <50ms latency
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import time
import random


class FinalIntentClassifier(nn.Module):
    """Production-ready intent classifier"""
    
    def __init__(self, num_intents):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_intents)
        )
    
    def forward(self, x):
        return self.classifier(x)


async def final_training():
    print("=" * 70)
    print("🎯 PHASE 2 FINAL TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading training data...")
    with open("comprehensive_training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    print(f"✅ Loaded {len(training_data)} training samples")
    
    # Get intents
    intents = sorted(list(set(intent for _, intent in training_data)))
    print(f"✅ {len(intents)} intent classes")
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load model
    print("\n🤖 Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
    model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # Create classifier
    print("\n🎯 Creating classifier...")
    classifier = FinalIntentClassifier(len(intents)).to(device)
    print("✅ Classifier ready")
    
    # Training setup
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\n🔄 Training (100 epochs)...")
    print("=" * 70)
    
    random.shuffle(training_data)
    
    for epoch in range(100):
        total_loss = 0
        classifier.train()
        
        for text, intent in training_data:
            # Get embeddings
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :]
            
            # Train
            logits = classifier(embeddings)
            target = torch.tensor([intents.index(intent)]).to(device)
            loss = criterion(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"   Epoch {epoch + 1}/100: Loss = {avg_loss:.4f}")
    
    print("✅ Training complete!")
    
    # Test
    print("\n📊 Testing on comprehensive test set...")
    print("=" * 70)
    
    test_cases = [
        # Attractions
        ("Ayasofya'yı görmek istiyorum", "attraction"),
        ("Topkapı Sarayı nerede", "attraction"),
        ("Galata Kulesi kaça kadar açık", "attraction"),
        
        # Restaurants
        ("En iyi kebap nerede?", "restaurant"),
        ("Balık restoranı", "restaurant"),
        ("Nerede yemek yiyebilirim", "restaurant"),
        
        # Navigation
        ("Taksim'e nasıl gidilir?", "gps_navigation"),
        ("Sultanahmet'e git", "gps_navigation"),
        ("Beni Kadıköy'e götür", "gps_navigation"),
        
        # Transportation
        ("Metro saatleri", "transportation"),
        ("Otobüs güzergahı", "transportation"),
        ("Taksi çağır", "transportation"),
        
        # Accommodation
        ("Ucuz otel önerisi", "accommodation"),
        ("5 yıldızlı otel", "luxury"),
        ("Hostel tavsiyesi", "budget"),
        
        # Nightlife
        ("Gece hayatı", "nightlife"),
        ("Bar tavsiyesi", "nightlife"),
        
        # Weather
        ("Hava durumu", "weather"),
        ("Yarın yağmur yağacak mı", "weather"),
        
        # Cultural
        ("Boğaz turu", "cultural_info"),
        ("Osmanlı tarihi", "history"),
        
        # Museums
        ("Müze önerisi", "museum"),
        ("İstanbul Modern", "museum"),
        
        # Shopping
        ("Alışveriş merkezi", "shopping"),
        ("Kapalıçarşı", "shopping"),
        
        # Events
        ("Bu hafta etkinlikler", "events"),
        ("Konser var mı", "events"),
        
        # Family
        ("Çocuklu yerler", "family_activities"),
        
        # Hidden gems
        ("Gizli mekanlar", "hidden_gems"),
        ("Yerel öneriler", "local_tips"),
        
        # Others
        ("Giriş ücreti ne kadar", "price_info"),
        ("Rezervasyon yapmak istiyorum", "booking"),
        ("En yakın hastane", "emergency"),
    ]
    
    classifier.eval()
    correct = 0
    latencies = []
    
    for query, expected in test_cases:
        start = time.time()
        
        with torch.no_grad():
            inputs = tokenizer(query, return_tensors="pt", max_length=128, truncation=True).to(device)
            embeddings = model(**inputs).last_hidden_state[:, 0, :]
            logits = classifier(embeddings)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted = intents[predicted_idx]
            confidence = torch.softmax(logits, dim=1)[0][predicted_idx].item()
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} '{query[:40]:<40}' → {predicted:<20} ({confidence:.1%}) [{latency:.1f}ms]")
    
    accuracy = (correct / len(test_cases)) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 70)
    print("🎉 FINAL RESULTS")
    print("=" * 70)
    print(f"📊 Training samples: {len(training_data)}")
    print(f"📊 Test samples: {len(test_cases)}")
    print(f"✅ Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)} correct)")
    print(f"⚡ Avg latency: {avg_latency:.1f}ms")
    print(f"🎯 Targets: >90% accuracy, <50ms latency")
    print()
    
    if accuracy >= 90:
        print("✅ ACCURACY TARGET MET!")
    else:
        print(f"⚠️  Accuracy: {accuracy:.1f}% (need {90 - accuracy:.1f}% more)")
    
    if avg_latency < 50:
        print("✅ LATENCY TARGET MET!")
    else:
        print(f"⚠️  Latency: {avg_latency:.1f}ms (need {avg_latency - 50:.1f}ms improvement)")
    
    print()
    
    if accuracy >= 90 and avg_latency < 50:
        print("🎉🎉🎉 PHASE 2 COMPLETE! ALL TARGETS MET! 🎉🎉🎉")
    elif accuracy >= 70:
        print("✅ Great progress! Close to target.")
    
    # Save
    results = {
        "training_samples": len(training_data),
        "test_samples": len(test_cases),
        "accuracy_percent": accuracy,
        "latency_ms": avg_latency,
        "correct": correct,
        "targets_met": accuracy >= 90 and avg_latency < 50,
        "model": "DistilBERT-multilingual",
        "device": str(device)
    }
    
    with open("phase2_final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to phase2_final_results.json")
    
    # Save model
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'intents': intents,
        'accuracy': accuracy,
        'latency': avg_latency,
        'training_samples': len(training_data)
    }, 'phase2_final_model.pth')
    
    print(f"💾 Model saved to phase2_final_model.pth")
    print()
    
    return accuracy, avg_latency


if __name__ == "__main__":
    import asyncio
    asyncio.run(final_training())
