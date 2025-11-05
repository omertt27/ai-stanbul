"""
NCF Training Script
Complete training pipeline for Lightweight NCF

Week 5-6 Implementation - Budget-Optimized Roadmap

Cross-Platform Support:
- Production: T4 GPU with CUDA + FP16 mixed precision
- Development: M2 Pro with MPS + FP32
- Fallback: CPU with FP32

Usage:
    # Auto-detect device (recommended)
    python train_ncf.py --epochs 10 --batch_size 2048
    
    # Force specific device
    python train_ncf.py --device mps --epochs 5
    python train_ncf.py --device cuda --mixed_precision
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import pickle

from backend.ml.models.lightweight_ncf import LightweightNCF
from backend.ml.training.lightweight_trainer import LightweightTrainer, NCFDataset
from backend.ml.utils.device_utils import (
    get_optimal_device, 
    print_device_info,
    get_dataloader_config
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = './data/ncf'):
    """
    Load prepared training data
    
    Args:
        data_dir: Directory containing prepared data
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata)
    """
    logger.info(f"📂 Loading data from {data_dir}...")
    
    data_path = Path(data_dir)
    
    # Load numpy arrays
    train_users = np.load(data_path / 'train_users.npy')
    train_items = np.load(data_path / 'train_items.npy')
    train_labels = np.load(data_path / 'train_labels.npy')
    
    val_users = np.load(data_path / 'val_users.npy')
    val_items = np.load(data_path / 'val_items.npy')
    val_labels = np.load(data_path / 'val_labels.npy')
    
    test_users = np.load(data_path / 'test_users.npy')
    test_items = np.load(data_path / 'test_items.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    # Load mappings
    with open(data_path / 'mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    logger.info(
        f"✅ Data loaded:\n"
        f"   Train: {len(train_users):,} samples\n"
        f"   Val:   {len(val_users):,} samples\n"
        f"   Test:  {len(test_users):,} samples\n"
        f"   Users: {mappings['num_users']:,}\n"
        f"   Items: {mappings['num_items']:,}"
    )
    
    # Create datasets
    train_dataset = NCFDataset(train_users, train_items, train_labels)
    val_dataset = NCFDataset(val_users, val_items, val_labels)
    test_dataset = NCFDataset(test_users, test_items, test_labels)
    
    return train_dataset, val_dataset, test_dataset, mappings


def train_ncf(args):
    """
    Main training function
    
    Args:
        args: Command-line arguments
    """
    logger.info("🚀 Starting NCF training...")
    logger.info(f"Configuration: {vars(args)}")
    
    # Print device information
    print_device_info()
    
    # Auto-detect optimal device if not specified
    if args.device == 'auto':
        device, supports_fp16 = get_optimal_device()
        # Override mixed precision based on device capability
        if args.mixed_precision and not supports_fp16:
            logger.warning("⚠️  Mixed precision not supported on this device, disabling...")
            args.mixed_precision = False
    else:
        device = args.device
        supports_fp16 = (device == 'cuda')
        if args.mixed_precision and not supports_fp16:
            logger.warning(f"⚠️  Mixed precision not supported on {device}, disabling...")
            args.mixed_precision = False
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    train_dataset, val_dataset, test_dataset, mappings = load_data(args.data_dir)
    
    # Get optimal DataLoader config for this device
    dataloader_config = get_dataloader_config(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_config
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        **dataloader_config
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **dataloader_config
    )
    
    # Create model
    model = LightweightNCF(
        num_users=mappings['num_users'],
        num_items=mappings['num_items'],
        embedding_dim=args.embedding_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout
    )
    
    logger.info(f"📊 Model: {model.get_model_size()['total_mb']:.2f} MB")
    logger.info(f"   Parameters: {model.get_model_size()['parameters']:,}")
    
    # Create trainer
    trainer = LightweightTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        use_mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping=args.early_stopping
    )
    
    # Evaluate on test set
    logger.info("🧪 Evaluating on test set...")
    test_loss = trainer.validate(test_loader)
    logger.info(f"✅ Test loss: {test_loss:.4f}")
    
    # Save final model
    final_model_path = Path(args.checkpoint_dir) / 'final_model.pth'
    model.save(str(final_model_path))
    logger.info(f"💾 Final model saved to {final_model_path}")
    
    # Save training history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'test_loss': test_loss,
            'config': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"📊 Training history saved to {history_path}")
    
    # Print summary
    logger.info(
        f"\n{'='*60}\n"
        f"🎉 Training Complete!\n"
        f"{'='*60}\n"
        f"Best validation loss: {trainer.best_val_loss:.4f}\n"
        f"Test loss: {test_loss:.4f}\n"
        f"Model size: {model.get_model_size()['total_mb']:.2f} MB\n"
        f"Checkpoint dir: {args.checkpoint_dir}\n"
        f"{'='*60}"
    )


def main():
    """Parse arguments and run training"""
    parser = argparse.ArgumentParser(description='Train Lightweight NCF')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/ncf',
                        help='Directory containing prepared data')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Embedding dimension')
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[64, 32, 16],
                        help='MLP layer sizes')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to train on (auto-detect recommended)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (auto-detect if None)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/ncf',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_ncf(args)


if __name__ == "__main__":
    main()
