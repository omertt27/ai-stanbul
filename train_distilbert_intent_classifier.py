#!/usr/bin/env python3
"""
Train DistilBERT Intent Classifier for AI Istanbul
===================================================

This script trains a multilingual DistilBERT model for robust intent classification
in both Turkish and English, covering all required categories.
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
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntentDataset(Dataset):
    """Dataset for intent classification training"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create intent mapping
        self.intents = sorted(list(set(item['intent'] for item in self.data)))
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(self.intents)}
        self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        
        logger.info(f"✅ Loaded {len(self.data)} training examples")
        logger.info(f"📊 Found {len(self.intents)} unique intents: {', '.join(self.intents)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        intent = item['intent']
        
        encoding = self.tokenizer(
            text,
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
    """Load and prepare training data"""
    logger.info(f"📥 Loading training data from: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle multiple formats
    if isinstance(data, dict):
        if 'train_data' in data:
            # New format with train_data and test_data
            training_data = data['train_data']
            logger.info(f"📊 Found train_data with {len(training_data)} examples")
        elif 'training_data' in data:
            # Legacy format with training_data key
            training_data = data['training_data']
            logger.info(f"📊 Found training_data with {len(training_data)} examples")
        else:
            # Dict is the data itself - shouldn't reach here now
            logger.warning("⚠️  Data is a dict but has no train_data or training_data key")
            training_data = []
    elif isinstance(data, list):
        # Direct list format
        training_data = data
        logger.info(f"📊 Found direct list with {len(training_data)} examples")
    else:
        logger.error(f"❌ Unknown data format: {type(data)}")
        training_data = []
    
    logger.info(f"✅ Loaded {len(training_data)} examples")
    
    # Print intent distribution
    from collections import Counter
    intent_counts = Counter(item['intent'] for item in training_data)
    logger.info(f"📊 Intent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        logger.info(f"   {intent}: {count} examples")
    
    return training_data


def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels


def train_intent_classifier(
    data_file="comprehensive_intent_training_data.json",
    output_dir="models/distilbert_intent_classifier",
    model_name="distilbert-base-multilingual-cased",
    epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    validation_split=0.15,
    max_length=128
):
    """
    Train the DistilBERT intent classifier
    
    Args:
        data_file: Path to training data JSON
        output_dir: Directory to save trained model
        model_name: Pretrained model to use
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        warmup_ratio: Warmup ratio for learning rate scheduler
        validation_split: Fraction of data to use for validation
        max_length: Maximum sequence length
    
    Returns:
        Path to saved model, intent mapping
    """
    
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 TRAINING DISTILBERT INTENT CLASSIFIER FOR AI ISTANBUL")
    print("="*80)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")
    if device.type == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    training_data = load_training_data(data_file)
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(training_data))
    val_size = int(len(training_data) * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_data = [training_data[i] for i in train_indices]
    val_data = [training_data[i] for i in val_indices]
    
    logger.info(f"📊 Dataset split:")
    logger.info(f"   Training: {len(train_data)} examples")
    logger.info(f"   Validation: {len(val_data)} examples")
    
    # Load tokenizer and create datasets
    logger.info(f"📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = IntentDataset(train_data, tokenizer, max_length)
    val_dataset = IntentDataset(val_data, tokenizer, max_length)
    
    # Initialize model
    num_labels = len(train_dataset.intents)
    logger.info(f"🏗️  Initializing model with {num_labels} intent classes...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_labels = len(train_dataset.intents)
    logger.info(f"🏗️  Initializing model with {num_labels} intent classes...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.to(device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info(f"🏋️  Starting training for {epochs} epochs...")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Total training steps: {total_steps}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Evaluation
        val_loss, val_accuracy, val_preds, val_labels = evaluate_model(model, val_dataloader, device)
        
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            logger.info(f"  🎯 New best accuracy! Saving model...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    logger.info("\n📊 Final Evaluation:")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best validation accuracy: {best_accuracy:.4f}")
    
    # Load best model and generate detailed predictions
    logger.info("� Loading best model for detailed evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    
    _, _, val_preds, val_labels = evaluate_model(model, val_dataloader, device)
    
    # Save intent mapping
    intent_mapping = {
        'intent_to_idx': train_dataset.intent_to_idx,
        'idx_to_intent': train_dataset.idx_to_intent,
        'intents': train_dataset.intents
    }
    
    intent_mapping_file = output_path / 'intent_mapping.json'
    with open(intent_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(intent_mapping, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved intent mapping to {intent_mapping_file}")
    
    # Classification report
    intent_names = [train_dataset.idx_to_intent[i] for i in range(num_labels)]
    report = classification_report(
        val_labels, 
        val_preds, 
        target_names=intent_names,
        zero_division=0
    )
    
    logger.info("\n📊 Classification Report:")
    print(report)
    
    # Save report
    output_path = Path(output_dir)
    report_file = output_path / 'classification_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Training summary
    elapsed_time = time.time() - start_time
    
    summary = {
        'model_name': model_name,
        'output_dir': output_dir,
        'num_intents': num_labels,
        'intents': train_dataset.intents,
        'training_examples': len(train_data),
        'validation_examples': len(val_data),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'validation_accuracy': float(best_accuracy),
        'best_epoch': best_epoch,
        'training_time_seconds': elapsed_time,
        'trained_at': datetime.now().isoformat(),
        'device': str(device)
    }
    
    summary_file = output_path / 'training_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved training summary to {summary_file}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
    print(f"🎯 Best validation accuracy: {best_accuracy:.4f} (epoch {best_epoch})")
    print(f"📂 Model saved to: {output_dir}")
    print(f"🏷️  Intent mapping saved to: {intent_mapping_file}")
    print("="*80 + "\n")
    
    return output_dir, intent_mapping


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DistilBERT Intent Classifier')
    parser.add_argument('--data-file', type=str, 
                       default='comprehensive_intent_training_data.json',
                       help='Path to training data JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='models/distilbert_intent_classifier',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.15,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    train_intent_classifier(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split
    )
