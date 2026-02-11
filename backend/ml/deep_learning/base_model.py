"""
Base Model Class for Deep Learning Models

Provides common functionality for all deep learning models:
- Model lifecycle management (train, validate, predict)
- Checkpointing and versioning
- Metrics tracking
- Model serialization

Author: AI Istanbul Team
Date: February 10, 2026
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all deep learning models.
    
    All Phase 2 models (NCF, Wide&Deep, BERT4Rec) inherit from this class.
    """
    
    def __init__(
        self,
        model_name: str,
        version: str = "1.0.0",
        model_dir: str = "./models"
    ):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model (e.g., 'ncf', 'wide_and_deep')
            version: Model version
            model_dir: Directory to save/load models
        """
        self.model_name = model_name
        self.version = version
        self.model_dir = model_dir
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"✅ Initialized {model_name} v{version}")
    
    @abstractmethod
    def build(self, **kwargs) -> None:
        """
        Build the model architecture.
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_data: Any,
        validation_data: Optional[Any] = None,
        epochs: int = 10,
        batch_size: int = 256,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional training parameters
            
        Returns:
            Training history (loss, metrics per epoch)
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        inputs: Any,
        batch_size: int = 256,
        **kwargs
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            inputs: Input data for prediction
            batch_size: Batch size for inference
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    def evaluate(
        self,
        test_data: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test dataset
            metrics: List of metrics to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ['accuracy', 'auc', 'ndcg@10']
        
        logger.info(f"Evaluating {self.model_name} on test data...")
        
        # Get predictions
        predictions = self.predict(test_data, **kwargs)
        
        # Compute metrics
        results = {}
        for metric_name in metrics:
            metric_value = self._compute_metric(
                metric_name,
                test_data,
                predictions
            )
            results[metric_name] = metric_value
        
        self.metrics = results
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model. If None, uses default path.
            
        Returns:
            Path where model was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_v{self.version}_{timestamp}.pth"
            filepath = os.path.join(self.model_dir, filename)
        
        # Save model state
        model_state = {
            'model_name': self.model_name,
            'version': self.version,
            'model_state_dict': self._get_model_state(),
            'training_history': self.training_history,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_checkpoint(filepath, model_state)
        
        # Save metadata
        metadata_path = filepath.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'version': self.version,
                'metrics': self.metrics,
                'timestamp': model_state['timestamp']
            }, f, indent=2)
        
        logger.info(f"✅ Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        logger.info(f"Loading model from {filepath}...")
        
        checkpoint = self._load_checkpoint(filepath)
        
        self.model_name = checkpoint['model_name']
        self.version = checkpoint['version']
        self.training_history = checkpoint['training_history']
        self.metrics = checkpoint['metrics']
        self.is_trained = checkpoint['is_trained']
        
        self._set_model_state(checkpoint['model_state_dict'])
        
        logger.info(f"✅ Model loaded successfully")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving."""
        pass
    
    @abstractmethod
    def _set_model_state(self, state_dict: Dict[str, Any]) -> None:
        """Set model state from loaded checkpoint."""
        pass
    
    @abstractmethod
    def _save_checkpoint(self, filepath: str, state: Dict[str, Any]) -> None:
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def _load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        pass
    
    def _compute_metric(
        self,
        metric_name: str,
        true_data: Any,
        predictions: np.ndarray
    ) -> float:
        """
        Compute a specific metric.
        
        Args:
            metric_name: Name of the metric to compute
            true_data: Ground truth data
            predictions: Model predictions
            
        Returns:
            Metric value
        """
        # This is a placeholder - subclasses should override
        # or implement specific metrics
        
        if metric_name == 'accuracy':
            # Binary accuracy
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_preds == true_data)
            return float(accuracy)
        
        elif metric_name == 'auc':
            # AUC-ROC (placeholder)
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(true_data, predictions)
                return float(auc)
            except:
                return 0.0
        
        elif metric_name.startswith('ndcg@'):
            # NDCG@K (placeholder)
            k = int(metric_name.split('@')[1])
            # Implement NDCG calculation
            return 0.0
        
        else:
            logger.warning(f"Unknown metric: {metric_name}")
            return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'training_epochs': len(self.training_history)
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, version={self.version}, trained={self.is_trained})"
