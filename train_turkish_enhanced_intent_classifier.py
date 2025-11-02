#!/usr/bin/env python3
"""
Turkish-Enhanced Neural Intent Classifier Training Script
==========================================================

This script trains a DistilBERT model on the enhanced Turkish training data
for improved bilingual intent classification.

Features:
- Works with simple [query, intent] format training data
- Multilingual DistilBERT for Turkish/English support
- Automatic train/validation split
- Comprehensive evaluation metrics
- Model checkpointing
- Training metadata tracking
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleIntentDataset(Dataset):
    """Dataset for intent classification from [query, intent] format"""
    
    def __init__(self, data, tokenizer, max_length=128):
        """
        Args:
            data: List of [query, intent] pairs
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract unique intents and create mappings
        self.intents = sorted(list(set(item[1] for item in self.data)))
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(self.intents)}
        self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        
        logger.info(f"✅ Dataset created with {len(self.data)} examples")
        logger.info(f"📊 Unique intents: {len(self.intents)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        query, intent = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.intent_to_idx[intent], dtype=torch.long)
        }


def load_training_data(data_file):
    """Load training data from various formats (list or dict with 'training_data' key)"""
    logger.info(f"📥 Loading training data from: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different data formats
    if isinstance(raw_data, dict):
        # New format with metadata
        if 'training_data' in raw_data:
            training_examples = raw_data['training_data']
            metadata = raw_data.get('metadata', {})
            logger.info(f"📊 Metadata: {metadata}")
        else:
            raise ValueError("Dictionary format must have 'training_data' key")
    elif isinstance(raw_data, list):
        # Old format - direct list
        training_examples = raw_data
    else:
        raise ValueError(f"Unexpected data format: {type(raw_data)}")
    
    # Normalize to [query, intent] pairs
    data = []
    for item in training_examples:
        if isinstance(item, dict):
            # Dict format: {'text': '...', 'intent': '...'}
            text = item.get('text') or item.get('query')
            intent = item.get('intent') or item.get('label')
            if text and intent:
                data.append([text, intent])
        elif isinstance(item, list) and len(item) == 2:
            # Already in [query, intent] format
            data.append(item)
    
    logger.info(f"✅ Loaded {len(data)} training examples")
    
    # Analyze intent distribution
    intent_counts = Counter(item[1] for item in data)
    logger.info(f"\n📊 Intent Distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(data) * 100
        logger.info(f"   {intent:25s}: {count:4d} examples ({pct:5.1f}%)")
    
    # Detect language balance (simple heuristic)
    english_keywords = ['what', 'where', 'how', 'when', 'can', 'is', 'are', 'the', 'best', 'nearest']
    turkish_count = sum(1 for query, _ in data if not any(kw in query.lower() for kw in english_keywords))
    english_count = len(data) - turkish_count
    
    logger.info(f"\n🌍 Estimated Language Distribution:")
    logger.info(f"   Turkish: {turkish_count} ({turkish_count/len(data)*100:.1f}%)")
    logger.info(f"   English: {english_count} ({english_count/len(data)*100:.1f}%)")
    
    return data


def evaluate_model(model, dataloader, device, idx_to_intent):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels


def train_intent_classifier(
    data_file="comprehensive_training_data.json",
    output_dir="models/istanbul_intent_classifier_finetuned",
    model_name="distilbert-base-multilingual-cased",
    epochs=5,
    batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    validation_split=0.15,
    max_length=128,
    random_seed=42
):
    """
    Train the DistilBERT intent classifier on enhanced Turkish data
    
    Args:
        data_file: Path to training data JSON (simple [query, intent] format)
        output_dir: Directory to save trained model
        model_name: Pretrained model to use
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        warmup_ratio: Warmup ratio for learning rate scheduler
        validation_split: Fraction of data for validation
        max_length: Maximum sequence length
        random_seed: Random seed for reproducibility
    """
    
    print("\n" + "=" * 70)
    print("🚀 TURKISH-ENHANCED INTENT CLASSIFIER TRAINING")
    print("=" * 70 + "\n")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🍎 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("🚀 Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU (consider using GPU for faster training)")
    
    # Load training data
    data = load_training_data(data_file)
    
    # Split into train and validation
    train_data, val_data = train_test_split(
        data, 
        test_size=validation_split, 
        random_state=random_seed,
        stratify=[intent for _, intent in data]  # Stratified split
    )
    
    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Training: {len(train_data)} examples")
    logger.info(f"   Validation: {len(val_data)} examples")
    
    # Load tokenizer and model
    logger.info(f"\n🤖 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = SimpleIntentDataset(train_data, tokenizer, max_length)
    val_dataset = SimpleIntentDataset(val_data, tokenizer, max_length)
    
    # Get number of labels from dataset
    num_labels = len(train_dataset.intents)
    logger.info(f"📊 Number of intent classes: {num_labels}")
    
    # Load model with correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues on macOS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"\n🎯 Training Configuration:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Total steps: {total_steps}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    # Training loop
    logger.info(f"\n🏋️ Starting training...\n")
    
    best_val_acc = 0.0
    training_history = []
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}\n")
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        val_loss, val_acc, val_preds, val_labels = evaluate_model(
            model, val_loader, device, train_dataset.idx_to_intent
        )
        
        # Log results
        print(f"\n📊 Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"   ✨ New best validation accuracy: {val_acc:.4f}")
            logger.info(f"   💾 Saving model checkpoint...")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save intent mappings
            intent_mapping = {
                'intents': train_dataset.intents,
                'intent_to_idx': train_dataset.intent_to_idx,
                'idx_to_intent': {str(k): v for k, v in train_dataset.idx_to_intent.items()}
            }
            
            with open(f"{output_dir}/intent_mapping.json", 'w', encoding='utf-8') as f:
                json.dump(intent_mapping, f, ensure_ascii=False, indent=2)
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
    
    # Final evaluation with classification report
    print(f"\n{'='*70}")
    print("📈 FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    _, _, val_preds, val_labels = evaluate_model(
        model, val_loader, device, train_dataset.idx_to_intent
    )
    
    # Classification report
    intent_names = [train_dataset.idx_to_intent[i] for i in range(num_labels)]
    report = classification_report(
        val_labels, 
        val_preds, 
        target_names=intent_names,
        zero_division=0
    )
    
    print("Classification Report:")
    print(report)
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'model_name': model_name,
        'dataset_file': data_file,
        'dataset_size': len(data),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'num_intents': num_labels,
        'intents': train_dataset.intents,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length': max_length,
        'final_train_accuracy': training_history[-1]['train_acc'],
        'final_val_accuracy': training_history[-1]['val_acc'],
        'best_val_accuracy': best_val_acc,
        'training_history': training_history,
        'device': str(device)
    }
    
    with open(f"{output_dir}/training_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to: {output_dir}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Test the model with real queries")
    print(f"   2. Update neural_query_classifier.py to use this model")
    print(f"   3. Run comprehensive evaluation")
    print(f"   4. Update documentation")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Turkish-Enhanced Intent Classifier')
    parser.add_argument('--data-file', type=str, default='comprehensive_training_data.json',
                        help='Path to training data JSON file')
    parser.add_argument('--num-intents', type=int, default=None,
                        help='Number of intent classes (auto-detected if not specified)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/istanbul_intent_classifier_finetuned',
                        help='Output directory for trained model')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("🚀 TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Data file: {args.data_file}")
    print(f"Num intents: {args.num_intents if args.num_intents else 'Auto-detect'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    train_intent_classifier(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
