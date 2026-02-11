"""
Training Data Collector Service

Collects and aggregates user interaction data for NCF model retraining.

Collects:
- User views (implicit rating: 0.5)
- User clicks (implicit rating: 1.0)
- User conversions/purchases (implicit rating: 2.0)

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects and prepares training data for NCF model retraining.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize data collector.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.min_interactions = 1000
        self.min_users = 100
        self.min_items = 50
    
    def collect_interactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_interactions_per_user: int = 3
    ) -> pd.DataFrame:
        """
        Collect user interactions from database.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            min_interactions_per_user: Minimum interactions per user to include
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        # Default to last 30 days if not specified
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.info(f"📊 Collecting interactions from {start_date} to {end_date}")
        
        # Query user interactions
        query = text("""
            SELECT 
                user_id,
                item_id,
                interaction_type,
                implicit_rating,
                timestamp
            FROM user_interactions
            WHERE timestamp >= :start_date 
                AND timestamp <= :end_date
            ORDER BY timestamp ASC
        """)
        
        try:
            result = self.db.execute(
                query,
                {"start_date": start_date, "end_date": end_date}
            )
            
            interactions = []
            for row in result:
                interactions.append({
                    'user_id': row.user_id,
                    'item_id': row.item_id,
                    'interaction_type': row.interaction_type,
                    'rating': row.implicit_rating,
                    'timestamp': row.timestamp
                })
            
            if not interactions:
                logger.warning("⚠️ No interactions found in the specified period")
                return pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
            
            df = pd.DataFrame(interactions)
            logger.info(f"✅ Collected {len(df)} raw interactions")
            
            # Filter users with min interactions
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_interactions_per_user].index
            df = df[df['user_id'].isin(valid_users)]
            
            logger.info(f"✅ Filtered to {len(df)} interactions from {len(valid_users)} active users")
            
            return df[['user_id', 'item_id', 'rating', 'timestamp']]
            
        except Exception as e:
            logger.error(f"❌ Failed to collect interactions: {e}")
            raise
    
    def aggregate_duplicate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate duplicate user-item interactions.
        
        Takes the maximum rating if user interacted with item multiple times.
        
        Args:
            df: DataFrame with user interactions
            
        Returns:
            Aggregated DataFrame
        """
        logger.info("🔄 Aggregating duplicate interactions...")
        
        # Group by user-item and take max rating + latest timestamp
        aggregated = df.groupby(['user_id', 'item_id']).agg({
            'rating': 'max',  # Take highest interaction level
            'timestamp': 'max'  # Take latest timestamp
        }).reset_index()
        
        reduction = len(df) - len(aggregated)
        logger.info(f"✅ Reduced {reduction} duplicate interactions")
        
        return aggregated
    
    def create_user_item_mappings(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Create user and item ID mappings for the model.
        
        Args:
            df: DataFrame with user_id and item_id
            
        Returns:
            Tuple of (user_to_idx, item_to_idx) mappings
        """
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        logger.info(f"📋 Created mappings: {len(user_to_idx)} users, {len(item_to_idx)} items")
        
        return user_to_idx, item_to_idx
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Uses temporal split: last 20% of each user's interactions for testing.
        
        Args:
            df: DataFrame with interactions
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"🔀 Creating train/test split (test_size={test_size})...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        train_data = []
        test_data = []
        
        # For each user, use temporal split
        for user_id, user_df in df.groupby('user_id'):
            n_interactions = len(user_df)
            n_test = max(1, int(n_interactions * test_size))
            
            # Last n_test interactions for testing
            test_data.append(user_df.iloc[-n_test:])
            train_data.append(user_df.iloc[:-n_test])
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        logger.info(f"✅ Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")
        
        return train_df, test_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate that collected data meets quality thresholds.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data quality is acceptable
        """
        n_interactions = len(df)
        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        
        # Calculate sparsity
        sparsity = 1 - (n_interactions / (n_users * n_items))
        
        logger.info("🔍 Data Quality Check:")
        logger.info(f"  📊 Interactions: {n_interactions}")
        logger.info(f"  👥 Unique users: {n_users}")
        logger.info(f"  📦 Unique items: {n_items}")
        logger.info(f"  🎯 Sparsity: {sparsity:.2%}")
        
        # Check minimum thresholds
        checks = {
            'min_interactions': n_interactions >= self.min_interactions,
            'min_users': n_users >= self.min_users,
            'min_items': n_items >= self.min_items,
            'max_sparsity': sparsity < 0.999  # At least 0.1% density
        }
        
        all_passed = all(checks.values())
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}: {passed}")
        
        if not all_passed:
            logger.warning("⚠️ Data quality checks failed!")
        
        return all_passed
    
    def prepare_training_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline to prepare training data.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with train/test data and mappings
        """
        logger.info("🚀 Starting training data preparation...")
        
        # Step 1: Collect interactions
        df = self.collect_interactions(start_date, end_date)
        
        if df.empty:
            raise ValueError("No interactions collected")
        
        # Step 2: Aggregate duplicates
        df = self.aggregate_duplicate_interactions(df)
        
        # Step 3: Validate quality
        if not self.validate_data_quality(df):
            raise ValueError("Data quality validation failed")
        
        # Step 4: Create mappings
        user_to_idx, item_to_idx = self.create_user_item_mappings(df)
        
        # Step 5: Split into train/test
        train_df, test_df = self.create_train_test_split(df)
        
        # Step 6: Convert to integer indices
        train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
        train_df['item_idx'] = train_df['item_id'].map(item_to_idx)
        test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
        test_df['item_idx'] = test_df['item_id'].map(item_to_idx)
        
        logger.info("✅ Training data preparation complete!")
        
        return {
            'train': train_df[['user_idx', 'item_idx', 'rating']].to_numpy(),
            'test': test_df[['user_idx', 'item_idx', 'rating']].to_numpy(),
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'num_users': len(user_to_idx),
            'num_items': len(item_to_idx),
            'metadata': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'collection_timestamp': datetime.utcnow().isoformat()
            }
        }


def get_training_data_collector(db_session: Session) -> TrainingDataCollector:
    """
    Factory function to get training data collector.
    
    Args:
        db_session: Database session
        
    Returns:
        TrainingDataCollector instance
    """
    return TrainingDataCollector(db_session)
