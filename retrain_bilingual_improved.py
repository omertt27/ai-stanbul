#!/usr/bin/env python3
"""
Quick Retrain Script for Bilingual Classifier
Focuses on improving English accuracy while maintaining Turkish performance
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Intents
INTENTS = [
    "accommodation", "attraction", "booking", "budget", "cultural_info",
    "emergency", "events", "family_activities", "food", "general_info",
    "gps_navigation", "hidden_gems", "history", "local_tips", "luxury",
    "museum", "nightlife", "price_info", "recommendation", "restaurant",
    "romantic", "route_planning", "shopping", "transportation", "weather"
]

intent_to_id = {intent: idx for idx, intent in enumerate(INTENTS)}
id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}

class IntentDataset(Dataset):
    """Dataset for intent classification"""
    
    def __init__(self, texts, intents, tokenizer, max_length=64):
        self.texts = texts
        self.intents = intents
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        intent = self.intents[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent': torch.tensor(intent_to_id[intent], dtype=torch.long)
        }

class IntentClassifierHead(nn.Module):
    """Classifier head matching production architecture"""
    
    def __init__(self, num_intents=25):
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

def load_dataset(filename="final_bilingual_dataset.json"):
    """Load training dataset"""
    print(f"\n📂 Loading dataset: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    texts = [s['text'] for s in samples]
    intents = [s['intent'] for s in samples]
    
    print(f"   Total samples: {len(samples)}")
    print(f"   Unique intents: {len(set(intents))}")
    
    return texts, intents

def train_epoch(model, base_model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    base_model.eval()  # Keep base model frozen
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        intents = batch['intent'].to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings from frozen base model
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Forward through classifier head
        logits = model(embeddings)
        loss = criterion(logits, intents)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += intents.size(0)
        correct += (predicted == intents).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, base_model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    base_model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intents = batch['intent'].to(device)
            
            # Get embeddings
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Forward through classifier
            logits = model(embeddings)
            loss = criterion(logits, intents)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += intents.size(0)
            correct += (predicted == intents).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def main():
    print("="*60)
    print("🚀 QUICK RETRAIN - Bilingual Classifier")
    print("="*60)
    
    # Load dataset
    texts, intents = load_dataset()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, intents, test_size=0.15, random_state=42, stratify=intents
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Load tokenizer and base model
    print(f"\n🔧 Loading DistilBERT multilingual...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    base_model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
    base_model = base_model.to(device)
    base_model.eval()  # Freeze base model
    
    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create classifier head
    print(f"🎯 Creating classifier head...")
    classifier = IntentClassifierHead(num_intents=len(INTENTS))
    classifier = classifier.to(device)
    
    # Create datasets and dataloaders
    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    val_dataset = IntentDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Training loop
    num_epochs = 15
    best_val_acc = 0
    best_model_state = None
    
    print(f"\n🎓 Training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            classifier, base_model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            classifier, base_model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - start_time
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = classifier.state_dict().copy()
            print(f"⭐ Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_acc:.1%} | Val: {val_acc:.1%} ⭐ NEW BEST!")
        else:
            print(f"   Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_acc:.1%} | Val: {val_acc:.1%}")
    
    print("="*60)
    print(f"✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2%}")
    
    # Save best model
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    save_path = "bilingual_model.pth"
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'intents': INTENTS,
        'accuracy': best_val_acc * 100,
        'training_samples': len(texts),
        'timestamp': datetime.now().isoformat()
    }, save_path)
    
    print(f"\n💾 Model saved to: {save_path}")
    print(f"   Accuracy: {best_val_acc:.2%}")
    print(f"   Ready for production!")

if __name__ == "__main__":
    main()
