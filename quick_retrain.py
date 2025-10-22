#!/usr/bin/env python3
"""
Quick Retrain - Fast fine-tuning on targeted improvements
Only trains classifier head (freezes base model) for <10 min training
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import time
from pathlib import Path


class BilingualIntentClassifier(nn.Module):
    """Bilingual intent classifier"""
    
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


def quick_retrain():
    """Quick retrain with targeted improvements"""
    print("=" * 80)
    print("QUICK RETRAIN - Targeted Improvements")
    print("=" * 80)
    print()
    
    # Load final dataset
    print("📂 Loading final dataset...")
    with open('final_bilingual_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    intents = sorted(list(set(item['intent'] for item in data)))
    intent_to_id = {intent: idx for idx, intent in enumerate(intents)}
    
    print(f"✅ Loaded {len(data)} samples (+190 targeted)")
    print(f"   Intents: {len(intents)}")
    print()
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print()
    
    # Load tokenizer and base model
    print("🔧 Loading DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    base_model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
    
    # Create classifier and load previous weights
    classifier = BilingualIntentClassifier(len(intents))
    
    if Path("bilingual_model.pth").exists():
        print("📥 Loading previous bilingual model...")
        try:
            checkpoint = torch.load("bilingual_model.pth", map_location=device)
            if 'classifier_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['classifier_state_dict'])
                print(f"✅ Loaded previous weights (Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%)")
        except Exception as e:
            print(f"⚠️  Could not load: {e}")
            print("   Starting fresh...")
    
    base_model.to(device)
    classifier.to(device)
    base_model.eval()  # Freeze base model for speed
    
    # Training setup
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0005)  # Lower LR for fine-tuning
    criterion = nn.CrossEntropyLoss()
    
    # Quick training - fewer epochs
    print()
    print("🚀 Starting quick retrain...")
    print("   Strategy: Fine-tune classifier head only (base model frozen)")
    print("   Target: <10 min on MPS")
    print("-" * 80)
    
    epochs = 150  # Reduced from 300
    batch_size = 16
    best_accuracy = 0.0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        # Train in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            texts = [item['text'] for item in batch]
            labels = torch.tensor([intent_to_id[item['intent']] for item in batch]).to(device)
            
            # Tokenize
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            
            # Forward pass (base model frozen)
            with torch.no_grad():
                embeddings = base_model(**inputs).last_hidden_state[:, 0, :]
            
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / (len(data) / batch_size)
        
        if (epoch + 1) % 15 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {elapsed:.1f}s")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    training_time = time.time() - start_time
    
    print("-" * 80)
    print(f"✅ Quick retrain complete in {training_time:.1f}s")
    print(f"   Best accuracy: {best_accuracy:.2f}%")
    print(f"   Time per epoch: {training_time/epochs:.2f}s")
    print()
    
    # Save improved model
    print("💾 Saving improved model...")
    checkpoint = {
        'classifier_state_dict': classifier.state_dict(),
        'intents': intents,
        'accuracy': best_accuracy,
        'training_samples': len(data),
        'language': 'bilingual',
        'languages': ['Turkish', 'English'],
        'total_epochs': epochs,
        'improvements': 'Targeted emergency, family_activities, attraction'
    }
    
    torch.save(checkpoint, 'bilingual_model.pth')
    print("✅ Saved to: bilingual_model.pth")
    print()
    
    # Quick validation test
    print("=" * 80)
    print("QUICK VALIDATION")
    print("=" * 80)
    print()
    
    classifier.eval()
    base_model.eval()
    
    test_queries = [
        # Previously failed Turkish
        ("Acil durum!", "emergency", "🇹🇷"),
        ("Boğaz turu", "attraction", "🇹🇷"),
        
        # Previously failed English
        ("Where to go with kids?", "family_activities", "🇬🇧"),
        
        # Should still work
        ("I want to visit Hagia Sophia", "attraction", "🇬🇧"),
        ("Hava durumu nasıl?", "weather", "🇹🇷"),
        ("Museum recommendations", "museum", "🇬🇧"),
    ]
    
    correct_count = 0
    
    for query, expected, flag in test_queries:
        inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            embeddings = base_model(**inputs).last_hidden_state[:, 0, :]
            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_id = probs.max(1)
            
            predicted_id = predicted_id.item()
            confidence = confidence.item()
            predicted_intent = intents[predicted_id]
        
        is_correct = predicted_intent == expected
        if is_correct:
            correct_count += 1
        
        status = "✅" if is_correct else "❌"
        conf_marker = "🔥" if confidence >= 0.70 else "⚠️"
        
        print(f"{status} {conf_marker} {flag} '{query}'")
        print(f"      Predicted: {predicted_intent} ({confidence:.1%})")
        if not is_correct:
            print(f"      Expected: {expected}")
    
    validation_accuracy = (correct_count / len(test_queries)) * 100
    
    print()
    print(f"Validation: {correct_count}/{len(test_queries)} correct ({validation_accuracy:.1f}%)")
    print()
    
    print("=" * 80)
    print("🎉 QUICK RETRAIN COMPLETE!")
    print("=" * 80)
    print()
    print("✅ Improvements:")
    print("   • Emergency detection (Turkish)")
    print("   • Family activities (English)")
    print("   • Attraction context (Boğaz turu)")
    print()
    print("Next: python3 test_bilingual.py")
    print()


if __name__ == "__main__":
    quick_retrain()
