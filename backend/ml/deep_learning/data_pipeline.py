"""
Data Pipeline for Deep Learning Models

Handles data loading, preprocessing, and negative sampling
for recommendation systems.

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for recommendation model training.
    
    Features:
    - User-item interaction processing
    - Train/val/test splitting
    - Negative sampling
    - Data normalization
    """
    
    def __init__(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        negative_samples_ratio: int = 4,
        random_seed: int = 42
    ):
        """
        Initialize data pipeline.
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            test_ratio: Ratio of test data
            val_ratio: Ratio of validation data
            negative_samples_ratio: Ratio of negative to positive samples
            random_seed: Random seed for reproducibility
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.negative_samples_ratio = negative_samples_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Mapping dictionaries
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        
        # Statistics
        self.num_users = 0
        self.num_items = 0
        self.num_interactions = 0
        
        logger.info("✅ DataPipeline initialized")
    
    def load_from_feedback(
        self,
        feedback_data: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from feedback events.
        
        Args:
            feedback_data: List of feedback events with user_id, item_id, rating/feedback
            
        Returns:
            Tuple of (user_ids, item_ids, labels)
        """
        logger.info(f"Loading {len(feedback_data)} feedback events...")
        
        # Extract interactions
        interactions = []
        for event in feedback_data:
            user_id = event.get('user_id', event.get('session_id'))
            item_id = event.get('item_id', event.get('place_id', event.get('venue_id')))
            
            # Convert feedback to binary label
            feedback = event.get('feedback', event.get('rating', event.get('score', 0)))
            
            # Positive label if feedback is good
            if isinstance(feedback, str):
                label = 1 if feedback in ['positive', 'click', 'accept', 'like'] else 0
            else:
                label = 1 if feedback > 0.5 else 0
            
            if user_id and item_id:
                interactions.append((user_id, item_id, label))
        
        logger.info(f"Extracted {len(interactions)} valid interactions")
        
        # Filter by minimum interactions
        interactions = self._filter_interactions(interactions)
        
        # Create mappings
        self._create_mappings(interactions)
        
        # Convert to indices
        user_ids = np.array([self.user_to_idx[u] for u, i, l in interactions])
        item_ids = np.array([self.item_to_idx[i] for u, i, l in interactions])
        labels = np.array([l for u, i, l in interactions])
        
        self.num_interactions = len(interactions)
        
        logger.info(
            f"✅ Data loaded: {self.num_users} users, "
            f"{self.num_items} items, {self.num_interactions} interactions"
        )
        
        return user_ids, item_ids, labels
    
    def _filter_interactions(
        self,
        interactions: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int]]:
        """
        Filter interactions by minimum user and item interactions.
        
        Args:
            interactions: List of (user_id, item_id, label) tuples
            
        Returns:
            Filtered interactions
        """
        # Count interactions per user and item
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for user_id, item_id, _ in interactions:
            user_counts[user_id] += 1
            item_counts[item_id] += 1
        
        # Filter
        filtered = [
            (u, i, l) for u, i, l in interactions
            if user_counts[u] >= self.min_user_interactions
            and item_counts[i] >= self.min_item_interactions
        ]
        
        logger.info(
            f"Filtered from {len(interactions)} to {len(filtered)} interactions "
            f"(min_user={self.min_user_interactions}, min_item={self.min_item_interactions})"
        )
        
        return filtered
    
    def _create_mappings(self, interactions: List[Tuple[str, str, int]]) -> None:
        """
        Create user and item ID mappings.
        
        Args:
            interactions: List of (user_id, item_id, label) tuples
        """
        unique_users = sorted(set(u for u, i, l in interactions))
        unique_items = sorted(set(i for u, i, l in interactions))
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
    
    def split_data(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray,
        strategy: str = "random"
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            labels: Array of labels
            strategy: Split strategy ('random' or 'temporal')
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(user_ids)
        
        if strategy == "random":
            # Random shuffle
            indices = np.random.permutation(n_samples)
            
            # Calculate split points
            test_size = int(n_samples * self.test_ratio)
            val_size = int(n_samples * self.val_ratio)
            
            test_indices = indices[:test_size]
            val_indices = indices[test_size:test_size + val_size]
            train_indices = indices[test_size + val_size:]
            
        elif strategy == "temporal":
            # Temporal split (last interactions for test)
            indices = np.arange(n_samples)
            
            test_size = int(n_samples * self.test_ratio)
            val_size = int(n_samples * self.val_ratio)
            
            test_indices = indices[-test_size:]
            val_indices = indices[-(test_size + val_size):-test_size]
            train_indices = indices[:-(test_size + val_size)]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Create splits
        train_data = (
            user_ids[train_indices],
            item_ids[train_indices],
            labels[train_indices]
        )
        
        val_data = (
            user_ids[val_indices],
            item_ids[val_indices],
            labels[val_indices]
        )
        
        test_data = (
            user_ids[test_indices],
            item_ids[test_indices],
            labels[test_indices]
        )
        
        logger.info(
            f"✅ Data split: Train={len(train_indices)}, "
            f"Val={len(val_indices)}, Test={len(test_indices)}"
        )
        
        return train_data, val_data, test_data
    
    def generate_negative_samples(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate negative samples for implicit feedback.
        
        For each positive user-item interaction, sample N negative items
        that the user hasn't interacted with.
        
        Args:
            user_ids: Array of user indices
            item_ids: Array of item indices
            labels: Array of labels (should be all 1s for positive samples)
            
        Returns:
            Augmented (user_ids, item_ids, labels) with negative samples
        """
        logger.info(f"Generating negative samples (ratio={self.negative_samples_ratio})...")
        
        # Build user interaction sets
        user_items: Dict[int, Set[int]] = defaultdict(set)
        for user_id, item_id in zip(user_ids, item_ids):
            user_items[user_id].add(item_id)
        
        # All items
        all_items = set(range(self.num_items))
        
        # Generate negative samples
        neg_user_ids = []
        neg_item_ids = []
        
        for user_id in user_ids:
            # Items user hasn't interacted with
            uninteracted_items = list(all_items - user_items[user_id])
            
            if len(uninteracted_items) < self.negative_samples_ratio:
                # Not enough items, sample with replacement
                neg_samples = np.random.choice(
                    uninteracted_items,
                    size=self.negative_samples_ratio,
                    replace=True
                )
            else:
                # Sample without replacement
                neg_samples = np.random.choice(
                    uninteracted_items,
                    size=self.negative_samples_ratio,
                    replace=False
                )
            
            for neg_item in neg_samples:
                neg_user_ids.append(user_id)
                neg_item_ids.append(neg_item)
        
        # Combine positive and negative samples
        combined_user_ids = np.concatenate([user_ids, neg_user_ids])
        combined_item_ids = np.concatenate([item_ids, neg_item_ids])
        combined_labels = np.concatenate([
            labels,
            np.zeros(len(neg_user_ids))
        ])
        
        logger.info(
            f"✅ Generated {len(neg_user_ids)} negative samples "
            f"({len(combined_labels)} total samples)"
        )
        
        return combined_user_ids, combined_item_ids, combined_labels
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions,
            'sparsity': 1 - (self.num_interactions / (self.num_users * self.num_items)),
            'avg_interactions_per_user': self.num_interactions / max(self.num_users, 1),
            'avg_interactions_per_item': self.num_interactions / max(self.num_items, 1)
        }
