#!/usr/bin/env python3
"""
Phase 2: Extended Training - 150 More Epochs
Continue training from saved model to reach 85%+ accuracy
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


async def extended_training():
    print("=" * 70)
    print("🚀 EXTENDED TRAINING: 150 MORE EPOCHS")
    print("   Goal: Push accuracy from 75.8% → 85%+")
    print("=" * 70)
    
    # Load previous model
    print("\n📂 Loading previous model...")
    checkpoint = torch.load('phase2_final_model.pth')
    intents = checkpoint['intents']
    previous_accuracy = checkpoint['accuracy']
    print(f"✅ Previous accuracy: {previous_accuracy:.1f}%")
    
    # Load data
    print("\n📂 Loading training data...")
    with open("comprehensive_training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    print(f"✅ Loaded {len(training_data)} training samples")
    
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
    
    # Load classifier with previous weights
    print("\n🎯 Loading classifier with previous training...")
    classifier = FinalIntentClassifier(len(intents)).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    print("✅ Classifier loaded")
    
    # Training setup with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)  # Lower LR
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\n🔄 Training 150 more epochs...")
    print("   Using learning rate: 0.0005 (lower for fine-tuning)")
    print("=" * 70)
    
    random.shuffle(training_data)
    
    start_time = time.time()
    
    for epoch in range(150):
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
        
        if (epoch + 1) % 15 == 0:
            avg_loss = total_loss / len(training_data)
            elapsed = time.time() - start_time
            remaining = (elapsed / (epoch + 1)) * (150 - epoch - 1)
            print(f"   Epoch {epoch + 1}/150: Loss = {avg_loss:.4f} | "
                  f"Elapsed: {elapsed/60:.1f}m | Remaining: {remaining/60:.1f}m")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Total time: {total_time/60:.1f} minutes")
    
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
        
        # Additional tests for problematic intents
        ("Kebap yemek istiyorum", "restaurant"),
        ("Topkapı Sarayı bilgileri", "attraction"),
        ("Konser bileti", "events"),
        ("Mısır Çarşısı", "shopping"),
    ]
    
    classifier.eval()
    correct = 0
    latencies = []
    errors = []
    
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
        else:
            errors.append((query, expected, predicted, confidence))
        
        status = "✅" if is_correct else "❌"
        print(f"{status} '{query[:40]:<40}' → {predicted:<20} ({confidence:.1%}) [{latency:.1f}ms]")
    
    accuracy = (correct / len(test_cases)) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 70)
    print("🎉 EXTENDED TRAINING RESULTS")
    print("=" * 70)
    print(f"📊 Training: 100 + 150 = 250 total epochs")
    print(f"📊 Test samples: {len(test_cases)}")
    print(f"✅ Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)} correct)")
    print(f"⚡ Avg latency: {avg_latency:.1f}ms")
    print(f"📈 Improvement: {accuracy - previous_accuracy:+.1f}%")
    print()
    
    if accuracy >= 90:
        print("🎉 ACCURACY TARGET MET! (>90%)")
    elif accuracy >= 85:
        print("✅ EXCELLENT! Accuracy >85% - Ready for production!")
    elif accuracy >= 80:
        print("✅ GOOD! Accuracy >80% - Acceptable for deployment")
    else:
        print(f"⚠️  Accuracy: {accuracy:.1f}% (need more improvement)")
    
    if avg_latency < 50:
        print("✅ LATENCY TARGET MET! (<50ms)")
    
    print()
    
    # Show errors
    if errors:
        print(f"\n📋 Errors to analyze ({len(errors)} total):")
        print("=" * 70)
        for query, expected, predicted, conf in errors[:10]:  # Show first 10
            print(f"❌ '{query}'")
            print(f"   Expected: {expected} | Got: {predicted} ({conf:.1%})")
    
    # Save
    results = {
        "training_epochs": 250,
        "training_samples": len(training_data),
        "test_samples": len(test_cases),
        "accuracy_percent": accuracy,
        "previous_accuracy": previous_accuracy,
        "improvement": accuracy - previous_accuracy,
        "latency_ms": avg_latency,
        "correct": correct,
        "errors": len(errors),
        "targets_met": accuracy >= 85 and avg_latency < 50,
        "model": "DistilBERT-multilingual",
        "device": str(device),
        "training_time_minutes": total_time / 60
    }
    
    with open("phase2_extended_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to phase2_extended_results.json")
    
    # Save improved model
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'intents': intents,
        'accuracy': accuracy,
        'latency': avg_latency,
        'training_samples': len(training_data),
        'total_epochs': 250
    }, 'phase2_extended_model.pth')
    
    print(f"💾 Improved model saved to phase2_extended_model.pth")
    
    # Decision
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS RECOMMENDATION")
    print("=" * 70)
    
    if accuracy >= 85:
        print("✅ READY FOR DEPLOYMENT!")
        print("   Your model is production-ready.")
        print("   Next: Integrate with main system (Phase 2 Day 2)")
    elif accuracy >= 80:
        print("✅ GOOD PROGRESS!")
        print("   Options:")
        print("   1. Deploy now (80%+ is acceptable)")
        print("   2. Add more data to push to 85%+")
    else:
        print("⚠️  NEEDS MORE WORK")
        print("   Recommendation: Add 100-200 more training samples")
        print("   Focus on confused intents:")
        for query, expected, predicted, conf in errors[:5]:
            print(f"      • {expected} vs {predicted}")
    
    print()
    
    return accuracy, avg_latency


if __name__ == "__main__":
    import asyncio
    asyncio.run(extended_training())
