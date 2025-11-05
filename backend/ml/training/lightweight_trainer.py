"""
Lightweight Trainer for Single T4 GPU
Efficient training with mixed precision, gradient accumulation, and memory optimization

Week 5-6 Implementation - Budget-Optimized Roadmap

Cross-Platform Support:
- CUDA (T4 Production): FP16 mixed precision
- MPS (M2 Pro Dev): FP32 only
- CPU (Fallback): FP32 only
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time
import logging
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.device_utils import get_optimal_device, setup_device_optimizations, move_batch_to_device

logger = logging.getLogger(__name__)


class NCFDataset(Dataset):
    """
    Dataset for NCF training
    Supports both explicit and implicit feedback
    """
    
    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset
        
        Args:
            user_ids: User ID array
            item_ids: Item ID array
            labels: Rating/interaction labels
            features: Optional context features
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)
        
        if features is not None:
            self.features = torch.FloatTensor(features)
        else:
            self.features = None
    
    def __len__(self) -> int:
        return len(self.user_ids)
    
    def __getitem__(self, idx: int) -> Tuple:
        if self.features is not None:
            return (
                self.user_ids[idx],
                self.item_ids[idx],
                self.labels[idx],
                self.features[idx]
            )
        else:
            return (
                self.user_ids[idx],
                self.item_ids[idx],
                self.labels[idx]
            )


class LightweightTrainer:
    """
    Efficient trainer for single T4 GPU
    
    Features:
    - Mixed precision (FP16) training for 2x speedup
    - Gradient accumulation for larger effective batch sizes
    - Early stopping to prevent overfitting
    - Learning rate scheduling
    - GPU memory optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        accumulation_steps: int = 4,
        use_mixed_precision: Optional[bool] = None,
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Initialize trainer with cross-platform support
        
        Args:
            model: NCF model to train
            device: Device to train on ('cuda', 'mps', 'cpu', or None for auto-detect)
            learning_rate: Initial learning rate
            batch_size: Batch size for training
            accumulation_steps: Gradient accumulation steps (effective batch = batch_size * accumulation_steps)
            use_mixed_precision: Use FP16 mixed precision (None for auto-detect based on device)
            checkpoint_dir: Directory to save checkpoints
        """
        # Auto-detect optimal device if not specified
        if device is None:
            device, supports_fp16 = get_optimal_device()
            if use_mixed_precision is None:
                use_mixed_precision = supports_fp16
        else:
            supports_fp16 = (device == 'cuda')
            if use_mixed_precision is None:
                use_mixed_precision = supports_fp16
        
        # Setup device optimizations
        setup_device_optimizations(device)
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        
        # Only enable mixed precision on CUDA (T4)
        # MPS (M2 Pro) doesn't support FP16 well
        self.use_mixed_precision = use_mixed_precision and (device == 'cuda')
        
        if use_mixed_precision and device != 'cuda':
            logger.warning(
                f"⚠️  Mixed precision requested but device is '{device}'. "
                f"FP16 is only supported on CUDA. Using FP32 instead."
            )
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Mixed precision scaler (only for CUDA)
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info("✅ Mixed precision (FP16) enabled for CUDA")
        else:
            self.scaler = None
            logger.info(f"ℹ️  Using FP32 precision on {device}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5
        
        logger.info(f"🔧 Trainer initialized on {device}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Accumulation steps: {accumulation_steps}")
        logger.info(f"   Effective batch: {batch_size * accumulation_steps}")
        
        logger.info(
            f"🔧 Trainer initialized: "
            f"batch_size={batch_size}, "
            f"effective_batch={batch_size * accumulation_steps}, "
            f"mixed_precision={use_mixed_precision}"
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                user_ids, item_ids, labels = batch
                features = None
            else:
                user_ids, item_ids, labels, features = batch
            
            # Move to device
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            labels = labels.to(self.device)
            
            if features is not None:
                features = features.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    if features is not None:
                        predictions = self.model(user_ids, item_ids, features)
                    else:
                        predictions = self.model(user_ids, item_ids)
                    
                    loss = self.criterion(predictions, labels)
                    loss = loss / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            
            else:
                # Standard training
                if features is not None:
                    predictions = self.model(user_ids, item_ids, features)
                else:
                    predictions = self.model(user_ids, item_ids)
                
                loss = self.criterion(predictions, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item() * self.accumulation_steps:.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    user_ids, item_ids, labels = batch
                    features = None
                else:
                    user_ids, item_ids, labels, features = batch
                
                # Move to device
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)
                
                if features is not None:
                    features = features.to(self.device)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        if features is not None:
                            predictions = self.model(user_ids, item_ids, features)
                        else:
                            predictions = self.model(user_ids, item_ids)
                        loss = self.criterion(predictions, labels)
                else:
                    if features is not None:
                        predictions = self.model(user_ids, item_ids, features)
                    else:
                        predictions = self.model(user_ids, item_ids)
                    loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping: bool = True
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping: Enable early stopping
        
        Returns:
            Training history dictionary
        """
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"lr={current_lr:.6f}, "
                f"time={epoch_time:.1f}s"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                self.model.save(str(checkpoint_path))
                logger.info(f"💾 Best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if early_stopping and self.patience_counter >= self.max_patience:
                logger.info(
                    f"⚠️ Early stopping triggered "
                    f"(no improvement for {self.max_patience} epochs)"
                )
                break
        
        total_time = time.time() - start_time
        logger.info(
            f"✅ Training complete! "
            f"Total time: {total_time / 60:.1f} minutes, "
            f"Best val_loss: {self.best_val_loss:.4f}"
        )
        
        return self.history
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"📂 Checkpoint loaded from {path}")
        return checkpoint['epoch']


if __name__ == "__main__":
    """Test the trainer"""
    from backend.ml.models.lightweight_ncf import LightweightNCF
    
    print("🧪 Testing Lightweight Trainer...")
    
    # Create dummy data
    num_samples = 10000
    num_users = 1000
    num_items = 500
    
    user_ids = np.random.randint(0, num_users, num_samples)
    item_ids = np.random.randint(0, num_items, num_samples)
    labels = np.random.rand(num_samples)
    
    # Create datasets
    train_dataset = NCFDataset(user_ids, item_ids, labels)
    val_dataset = NCFDataset(
        user_ids[:1000],
        item_ids[:1000],
        labels[:1000]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # Create model
    model = LightweightNCF(num_users=num_users, num_items=num_items)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = LightweightTrainer(
        model=model,
        device=device,
        batch_size=256,
        use_mixed_precision=(device == 'cuda')
    )
    
    # Train for 2 epochs (test)
    print("\n🏃 Running test training...")
    history = trainer.train(train_loader, val_loader, epochs=2)
    
    print(f"\n✅ Training test passed!")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {history['val_loss'][-1]:.4f}")
