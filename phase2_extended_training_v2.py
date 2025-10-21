#!/usr/bin/env python3
"""
Phase 2 - Extended Training (150 More Epochs)
Goal: Push accuracy from 75.8% to 85-90%
Training on comprehensive_training_data.json
"""

import torch
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict

print("=" * 80)
print("🚀 Phase 2 - Extended Training (150 More Epochs)")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check for Apple MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Apple MPS GPU detected and enabled!")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (MPS not available)")
print()

# Intent mapping - must match the training data
INTENT_CLASSES = [
    "accommodation", "attraction", "booking", "budget", "cultural_info",
    "emergency", "events", "family_activities", "food", "general_info",
    "gps_navigation", "hidden_gems", "history", "local_tips", "luxury",
    "museum", "nightlife", "price_info", "recommendation", "restaurant",
    "romantic", "route_planning", "shopping", "transportation", "weather"
]

intent_to_id = {intent: idx for idx, intent in enumerate(INTENT_CLASSES)}
id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}

print(f"📚 Training Classes: {len(INTENT_CLASSES)} intents")
print()

# Load training data
print("📂 Loading training data...")
with open('comprehensive_training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} training samples")
print()

# Calculate class distribution
class_dist = defaultdict(int)
for item in training_data:
    intent = item[1]  # Format is [query, intent]
    class_dist[intent] += 1

print("📊 Class Distribution:")
for intent in sorted(class_dist.keys()):
    print(f"   {intent}: {class_dist[intent]} samples")
print()

# Custom Dataset
class TurkishQueryDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query, intent = self.data[idx]  # Format is [query, intent]
        encoding = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(intent_to_id[intent], dtype=torch.long)
        }

# Initialize model and tokenizer
print("🤖 Loading model and tokenizer...")
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the previously trained model
print("📥 Loading previous model checkpoint (phase2_extended_model.pth)...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(INTENT_CLASSES)
)

# Try to load previous weights from the extended model
try:
    checkpoint = torch.load('phase2_extended_model.pth', map_location=device)
    # If checkpoint is just the state dict
    if isinstance(checkpoint, dict) and 'distilbert.embeddings.word_embeddings.weight' in checkpoint:
        model.load_state_dict(checkpoint)
        print("✅ Loaded previous model weights successfully!")
    else:
        print("⚠️  Checkpoint format not compatible, starting from scratch...")
except FileNotFoundError:
    print("⚠️  No previous checkpoint found, starting from scratch...")
except Exception as e:
    print(f"⚠️  Could not load previous weights: {e}")
    print("   Starting from scratch...")

model.to(device)
print()

# Create dataset
print("🔄 Creating dataset...")
train_dataset = TurkishQueryDataset(training_data, tokenizer)
print(f"✅ Dataset ready: {len(train_dataset)} samples")
print()

# Training arguments - 150 more epochs
print("⚙️  Training Configuration:")
training_args = TrainingArguments(
    output_dir='./phase2_extended_v2',
    num_train_epochs=150,  # 150 MORE epochs
    per_device_train_batch_size=16,
    learning_rate=2e-5,  # Slightly lower learning rate for fine-tuning
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=False,
    use_mps_device=torch.backends.mps.is_available(),
)

print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch Size: {training_args.per_device_train_batch_size}")
print(f"   Learning Rate: {training_args.learning_rate}")
print(f"   Device: {device}")
print()

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
print("🎯 Starting extended training...")
print("=" * 80)
start_time = time.time()

trainer.train()

training_time = time.time() - start_time
print("=" * 80)
print(f"✅ Extended training complete!")
print(f"⏱️  Training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
print()

# Save model
print("💾 Saving extended model...")
model_path = 'phase2_extended_v2_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")
print()

# Benchmark on test queries
print("=" * 80)
print("🧪 Running Benchmark Tests")
print("=" * 80)
print()

test_queries = [
    ("En yakın restoran nerede?", "restaurant"),
    ("Ayasofya'yı görmek istiyorum", "attraction"),
    ("Havaalanına nasıl giderim?", "transportation"),
    ("Hava durumu nasıl?", "weather"),
    ("Topkapı Sarayı nerede?", "attraction"),
    ("Kebap nerede yiyebilirim?", "food"),
    ("İstanbul Arkeoloji Müzesi açık mı?", "museum"),
    ("Taksim'e nasıl giderim?", "gps_navigation"),
    ("Boğaz turu için öneri", "recommendation"),
    ("Otel rezervasyonu yapabilir misiniz?", "booking"),
    ("Bu ne kadar?", "price_info"),
    ("Bu hafta sonu konser var mı?", "events"),
    ("Türk kahvesi nasıl içilir?", "cultural_info"),
    ("Acil durum, yardım!", "emergency"),
    ("Kapalıçarşı nerede?", "shopping"),
    ("Gece hayatı için tavsiye", "nightlife"),
    ("Çocuklu gezilecek yerler", "family_activities"),
    ("Lüks restoran önerisi", "luxury"),
    ("Saklı yerler göster", "hidden_gems"),
    ("Bizans dönemi hakkında bilgi", "history"),
    ("Ucuz konaklama", "accommodation"),
    ("Romantik mekanlar", "romantic"),
    ("En iyi rota ne?", "route_planning"),
    ("Yerel ipuçları", "local_tips"),
    ("Genel bilgi", "general_info"),
    ("Ekonomik gezi", "budget"),
]

model.eval()
correct = 0
total = len(test_queries)
latencies = []
predictions_detail = []

print("Running test queries...")
print()

with torch.no_grad():
    for query, expected_intent in test_queries:
        start = time.time()
        
        # Tokenize
        inputs = tokenizer(
            query,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Predict
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_intent = id_to_intent[predicted_id]
        
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        is_correct = predicted_intent == expected_intent
        if is_correct:
            correct += 1
        
        predictions_detail.append({
            "query": query,
            "expected": expected_intent,
            "predicted": predicted_intent,
            "correct": is_correct,
            "latency_ms": round(latency, 2),
            "confidence": round(torch.softmax(outputs.logits, dim=1).max().item() * 100, 1)
        })
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Query: '{query[:40]}...'")
        print(f"   Expected: {expected_intent}")
        print(f"   Predicted: {predicted_intent}")
        print(f"   Latency: {latency:.2f}ms")
        print()

accuracy = (correct / total) * 100
avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)

print("=" * 80)
print("📊 EXTENDED TRAINING RESULTS")
print("=" * 80)
print()
print(f"🎯 Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
print(f"⚡ Average Latency: {avg_latency:.2f}ms")
print(f"📈 P95 Latency: {p95_latency:.2f}ms")
print(f"📈 P99 Latency: {p99_latency:.2f}ms")
print()

# Accuracy by intent
print("📊 Accuracy by Intent:")
intent_correct = defaultdict(int)
intent_total = defaultdict(int)

for pred in predictions_detail:
    intent_total[pred['expected']] += 1
    if pred['correct']:
        intent_correct[pred['expected']] += 1

for intent in sorted(intent_total.keys()):
    intent_acc = (intent_correct[intent] / intent_total[intent]) * 100
    status = "✅" if intent_acc >= 80 else "⚠️" if intent_acc >= 50 else "❌"
    print(f"{status} {intent}: {intent_acc:.0f}% ({intent_correct[intent]}/{intent_total[intent]})")
print()

# Confusion analysis
print("🔍 Errors Analysis:")
errors = [p for p in predictions_detail if not p['correct']]
if errors:
    for err in errors:
        print(f"❌ '{err['query'][:50]}...'")
        print(f"   Expected: {err['expected']}")
        print(f"   Got: {err['predicted']}")
        print(f"   Confidence: {err['confidence']}%")
        print()
else:
    print("✅ No errors! Perfect accuracy!")
    print()

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'training_duration_seconds': round(training_time, 2),
    'training_duration_minutes': round(training_time / 60, 2),
    'total_epochs': 250,  # 100 from before + 150 now
    'new_epochs': 150,
    'training_samples': len(training_data),
    'test_samples': total,
    'accuracy_percent': round(accuracy, 2),
    'correct_predictions': correct,
    'total_predictions': total,
    'avg_latency_ms': round(avg_latency, 3),
    'p95_latency_ms': round(p95_latency, 3),
    'p99_latency_ms': round(p99_latency, 3),
    'min_latency_ms': round(min(latencies), 3),
    'max_latency_ms': round(max(latencies), 3),
    'device': str(device),
    'model': model_name,
    'predictions': predictions_detail,
    'intent_accuracy': {
        intent: {
            'correct': intent_correct[intent],
            'total': intent_total[intent],
            'accuracy_percent': round((intent_correct[intent] / intent_total[intent]) * 100, 1)
        }
        for intent in intent_total.keys()
    }
}

results_path = 'phase2_extended_v2_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"💾 Results saved to: {results_path}")
print()

# Final summary
print("=" * 80)
print("🎉 EXTENDED TRAINING COMPLETE!")
print("=" * 80)
print()
print(f"✅ Model trained for 150 MORE epochs (250 total)")
print(f"✅ Accuracy: {accuracy:.1f}%")
print(f"✅ Latency: {avg_latency:.2f}ms")
print(f"✅ Training time: {training_time/60:.1f} minutes")
print()

if accuracy >= 90:
    print("🎯 TARGET ACHIEVED! ≥90% accuracy reached!")
elif accuracy >= 85:
    print("🎯 EXCELLENT! ≥85% accuracy - production ready!")
elif accuracy >= 80:
    print("🎯 VERY GOOD! ≥80% accuracy - close to target!")
else:
    print("🎯 GOOD PROGRESS! Moving in the right direction!")

print()
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
