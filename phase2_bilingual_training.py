#!/usr/bin/env python3
"""
Quick Bilingual Training - Improve existing model with English data
Uses the existing Turkish model and fine-tunes with bilingual data
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import time
from pathlib import Path


class BilingualIntentClassifier(nn.Module):
    """Bilingual intent classifier (same architecture as before)"""
    
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


def quick_bilingual_training():
    """Quick training on bilingual data"""
    print("=" * 80)
    print("BILINGUAL TRAINING - Turkish + English")
    print("=" * 80)
    print()
    
    # Load bilingual dataset
    print("📂 Loading bilingual dataset...")
    dataset_file = 'enhanced_bilingual_dataset.json' if Path('enhanced_bilingual_dataset.json').exists() else 'bilingual_training_dataset.json'
    print(f"   Using: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get unique intents
    intents = sorted(list(set(item['intent'] for item in data)))
    intent_to_id = {intent: idx for idx, intent in enumerate(intents)}
    
    print(f"✅ Loaded {len(data)} samples")
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
    
    # Create classifier
    classifier = BilingualIntentClassifier(len(intents))
    
    # Try to load existing weights if available
    if Path("phase2_extended_model.pth").exists():
        print("📥 Loading existing model weights...")
        try:
            checkpoint = torch.load("phase2_extended_model.pth", map_location=device)
            if 'classifier_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['classifier_state_dict'])
                print("✅ Loaded existing weights - will fine-tune!")
        except Exception as e:
            print(f"⚠️  Could not load existing weights: {e}")
            print("   Starting fresh training...")
    
    base_model.to(device)
    classifier.to(device)
    base_model.eval()  # Freeze base model
    
    # Training setup
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print()
    print("🚀 Starting training...")
    print("-" * 80)
    
    epochs = 300  # Increase from 100 to 300
    batch_size = 16
    best_loss = float('inf')
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
            
            # Forward pass
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
        
        if (epoch + 1) % 20 == 0:  # Report every 20 epochs
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:  # Track best accuracy instead of loss
            best_accuracy = accuracy
            best_loss = avg_loss
    
    training_time = time.time() - start_time
    
    print("-" * 80)
    print(f"✅ Training complete in {training_time:.1f}s")
    print(f"   Best accuracy: {best_accuracy:.2f}%")
    print()
    
    # Save model
    print("💾 Saving bilingual model...")
    checkpoint = {
        'classifier_state_dict': classifier.state_dict(),
        'intents': intents,
        'accuracy': best_accuracy,
        'training_samples': len(data),
        'language': 'bilingual',
        'languages': ['Turkish', 'English'],
        'total_epochs': epochs
    }
    
    torch.save(checkpoint, 'bilingual_model.pth')
    print("✅ Saved to: bilingual_model.pth")
    print()
    
    # Quick test
    print("=" * 80)
    print("QUICK TEST")
    print("=" * 80)
    print()
    
    classifier.eval()
    base_model.eval()
    
    test_queries = [
        # Turkish
        ("Ayasofya'yı görmek istiyorum", "attraction"),
        ("Hava durumu nasıl?", "weather"),
        ("En yakın restoran", "restaurant"),
        # English
        ("I want to visit Hagia Sophia", "attraction"),
        ("What's the weather?", "weather"),
        ("Nearest restaurant", "restaurant"),
    ]
    
    for query, expected in test_queries:
        inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            embeddings = base_model(**inputs).last_hidden_state[:, 0, :]
            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_id = probs.max(1)
            
            predicted_id = predicted_id.item()
            confidence = confidence.item()
            predicted_intent = intents[predicted_id]
        
        status = "✅" if predicted_intent == expected else "❌"
        print(f"{status} '{query}'")
        print(f"   Predicted: {predicted_intent} ({confidence:.1%})")
        print(f"   Expected: {expected}")
        print()
    
    print("=" * 80)
    print("🎉 BILINGUAL MODEL READY!")
    print("=" * 80)
    print()
    print("To use the bilingual model:")
    print("1. Update neural_query_classifier.py to use 'bilingual_model.pth'")
    print("2. Run: python3 integration_test.py")
    print("3. Test with both Turkish and English queries")
    print()


if __name__ == "__main__":
    quick_bilingual_training()
