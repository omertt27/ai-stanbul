"""
Fine-tune Intent Classifier for AI Istanbul
Takes the training data and fine-tunes DistilBERT on Istanbul-specific intents
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
from pathlib import Path
import logging
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IstanbulIntentDataset(Dataset):
    """Dataset for Istanbul intent training"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 128):
        logger.info(f"Loading training data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create intent mapping
        self.intents = sorted(list(set(item['intent'] for item in self.data)))
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(self.intents)}
        self.idx_to_intent = {idx: intent for intent, idx in self.intent_to_idx.items()}
        
        logger.info(f"✅ Loaded {len(self.data)} training examples")
        logger.info(f"📊 Found {len(self.intents)} unique intents")
        logger.info(f"🏷️  Intents: {', '.join(self.intents[:5])}{'...' if len(self.intents) > 5 else ''}")
    
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


def finetune_intent_classifier(
    data_file: str = "data/intent_training_data_augmented.json",
    output_dir: str = "models/istanbul_intent_classifier_finetuned",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    validation_split: float = 0.15
):
    """
    Fine-tune the intent classifier on Istanbul data
    
    Args:
        data_file: Path to training data JSON
        output_dir: Directory to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        validation_split: Fraction of data to use for validation
    
    Returns:
        Path to saved model
    """
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print("🚀 FINE-TUNING INTENT CLASSIFIER FOR AI ISTANBUL")
    print("="*70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")
    if device.type == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer and model (same as your current system)
    model_name = "distilbert-base-multilingual-cased"
    logger.info(f"📥 Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset
    dataset = IstanbulIntentDataset(data_file, tokenizer)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"📊 Dataset split:")
    logger.info(f"   Training: {train_size} examples")
    logger.info(f"   Validation: {val_size} examples")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    logger.info(f"🏗️  Initializing model with {len(dataset.intents)} intent classes...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(dataset.intents)
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"⚙️  Training configuration:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Total steps: {total_steps}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    # Training loop
    print("\n" + "="*70)
    print("🎓 STARTING FINE-TUNING")
    print("="*70 + "\n")
    
    best_val_accuracy = 0.0
    training_history = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        print(f"\n📚 Epoch {epoch+1}/{epochs} - Training...")
        
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                current_acc = correct_train / total_train
                print(f"   Batch {batch_idx+1}/{len(train_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {current_acc:.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train / total_train
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        print(f"\n🔍 Epoch {epoch+1}/{epochs} - Validation...")
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct_val / total_val
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch results
        print(f"\n📊 Epoch {epoch+1}/{epochs} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'time': epoch_time
        })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"   ⭐ New best validation accuracy! Saving checkpoint...")
    
    # Save final fine-tuned model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving fine-tuned model to: {output_path}")
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save intent mapping
    with open(output_path / "intent_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({
            'intents': dataset.intents,
            'intent_to_idx': dataset.intent_to_idx,
            'idx_to_intent': {str(k): v for k, v in dataset.idx_to_intent.items()}
        }, f, indent=2, ensure_ascii=False)
    
    # Save training history
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save training metadata
    training_metadata = {
        'model_name': model_name,
        'training_date': datetime.now().isoformat(),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'dataset_size': dataset_size,
        'train_size': train_size,
        'val_size': val_size,
        'num_intents': len(dataset.intents),
        'final_train_accuracy': float(train_accuracy),
        'final_val_accuracy': float(val_accuracy),
        'best_val_accuracy': float(best_val_accuracy),
        'total_training_time': time.time() - start_time,
        'device': str(device)
    }
    
    with open(output_path / "training_metadata.json", 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"📁 Model saved to: {output_path}")
    print(f"📊 Final Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"📊 Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"⭐ Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"⏱️  Total Training Time: {total_time/60:.2f} minutes")
    print(f"🎯 Intents: {len(dataset.intents)}")
    print("\n🔧 Next Steps:")
    print("   1. Test the fine-tuned model")
    print("   2. Update main_system_neural_integration.py to use this model")
    print(f"   3. Set model_path = '{output_path}'")
    print("="*70 + "\n")
    
    return output_path


def test_model(model_path: str, test_queries: list = None):
    """Quick test of the fine-tuned model"""
    
    print("\n" + "="*70)
    print("🧪 TESTING FINE-TUNED MODEL")
    print("="*70 + "\n")
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load intent mapping
    with open(Path(model_path) / "intent_mapping.json", 'r') as f:
        mapping = json.load(f)
        idx_to_intent = {int(k): v for k, v in mapping['idx_to_intent'].items()}
    
    # Test queries
    if test_queries is None:
        test_queries = [
            "Best seafood restaurants in Istanbul",
            "How to get to Blue Mosque",
            "What's the weather today?",
            "Hello, I need help",
            "Cultural events this weekend",
            "Hotels in Sultanahmet",
            "Tell me about Beyoğlu",
        ]
    
    print("Testing with sample queries:\n")
    
    with torch.no_grad():
        for query in test_queries:
            encoding = tokenizer(
                query,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encoding)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_idx = torch.max(probs, dim=-1)
            
            intent = idx_to_intent[predicted_idx.item()]
            conf = confidence.item()
            
            print(f"Query: '{query}'")
            print(f"  → Intent: {intent} (confidence: {conf:.4f})")
            print()
    
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Intent Classifier for AI Istanbul")
    parser.add_argument("--data", type=str, default="data/intent_training_data_augmented.json", 
                       help="Path to training data")
    parser.add_argument("--output", type=str, default="models/istanbul_intent_classifier_finetuned",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after training")
    
    args = parser.parse_args()
    
    # Run fine-tuning
    model_path = finetune_intent_classifier(
        data_file=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Test if requested
    if args.test:
        test_model(str(model_path))
