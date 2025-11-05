"""
Data Preparation for NCF Training
Extracts, processes, and prepares training data from feedback database

Week 5-6 Implementation - Budget-Optimized Roadmap
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
import asyncpg
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCFDataPreparator:
    """
    Prepare training data for NCF from feedback database
    """
    
    def __init__(self, db_url: str):
        """
        Initialize data preparator
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
    
    async def extract_feedback_data(
        self,
        days_back: int = 90,
        min_interactions: int = 5
    ) -> pd.DataFrame:
        """
        Extract feedback data from database
        
        Args:
            days_back: Number of days to look back
            min_interactions: Minimum interactions per user
        
        Returns:
            DataFrame with user_id, item_id, rating, timestamp
        """
        logger.info(f"📊 Extracting feedback data (last {days_back} days)...")
        
        conn = await asyncpg.connect(self.db_url)
        
        try:
            # Get feedback events
            query = """
            SELECT 
                user_id,
                item_id,
                event_type,
                reward,
                timestamp
            FROM feedback_events
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            ORDER BY timestamp
            """
            
            rows = await conn.fetch(query, days_back)
            
            if not rows:
                logger.warning("⚠️ No feedback data found!")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'event_type', 'reward', 'timestamp'])
            
            logger.info(f"✅ Extracted {len(df)} feedback events")
            
            # Filter users with minimum interactions
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            df = df[df['user_id'].isin(valid_users)]
            
            logger.info(
                f"✅ Filtered to {len(df)} events from "
                f"{df['user_id'].nunique()} users and "
                f"{df['item_id'].nunique()} items"
            )
            
            return df
        
        finally:
            await conn.close()
    
    def create_id_mappings(
        self,
        df: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """
        Create user and item ID mappings
        
        Args:
            df: Feedback DataFrame
        
        Returns:
            Tuple of (user_id_map, item_id_map)
        """
        logger.info("🔢 Creating ID mappings...")
        
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        logger.info(
            f"✅ Mapped {len(self.user_id_map)} users and "
            f"{len(self.item_id_map)} items"
        )
        
        return self.user_id_map, self.item_id_map
    
    def convert_to_implicit_feedback(
        self,
        df: pd.DataFrame,
        positive_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Convert explicit feedback to implicit (binary) feedback
        
        Args:
            df: Feedback DataFrame with 'reward' column
            positive_threshold: Threshold for positive interaction
        
        Returns:
            DataFrame with binary labels
        """
        logger.info(f"🔄 Converting to implicit feedback (threshold={positive_threshold})...")
        
        # Create binary label
        df['label'] = (df['reward'] >= positive_threshold).astype(float)
        
        # Map IDs
        df['user_idx'] = df['user_id'].map(self.user_id_map)
        df['item_idx'] = df['item_id'].map(self.item_id_map)
        
        # Remove unmapped entries
        df = df.dropna(subset=['user_idx', 'item_idx'])
        df['user_idx'] = df['user_idx'].astype(int)
        df['item_idx'] = df['item_idx'].astype(int)
        
        positive_ratio = df['label'].mean()
        logger.info(f"✅ Positive interactions: {positive_ratio:.2%}")
        
        return df
    
    def add_negative_samples(
        self,
        df: pd.DataFrame,
        num_negatives: int = 4
    ) -> pd.DataFrame:
        """
        Add negative samples for each positive interaction
        
        Args:
            df: Feedback DataFrame with positive samples
            num_negatives: Number of negative samples per positive
        
        Returns:
            DataFrame with negative samples added
        """
        logger.info(f"➕ Adding negative samples (ratio 1:{num_negatives})...")
        
        positive_df = df[df['label'] == 1.0].copy()
        
        # Get all items
        all_items = set(self.item_id_map.values())
        
        negative_samples = []
        
        for _, row in positive_df.iterrows():
            user_idx = row['user_idx']
            
            # Get items the user has interacted with
            user_items = set(df[df['user_idx'] == user_idx]['item_idx'].unique())
            
            # Sample negative items
            negative_items = list(all_items - user_items)
            
            if len(negative_items) < num_negatives:
                sampled_negatives = negative_items
            else:
                sampled_negatives = np.random.choice(
                    negative_items,
                    size=num_negatives,
                    replace=False
                )
            
            for item_idx in sampled_negatives:
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'label': 0.0,
                    'timestamp': row['timestamp']
                })
        
        # Combine positive and negative samples
        negative_df = pd.DataFrame(negative_samples)
        combined_df = pd.concat([positive_df[['user_idx', 'item_idx', 'label', 'timestamp']], negative_df], ignore_index=True)
        
        logger.info(
            f"✅ Created dataset with {len(positive_df)} positive and "
            f"{len(negative_df)} negative samples"
        )
        
        return combined_df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Full dataset
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"✂️ Splitting data (test={test_size}, val={val_size})...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=train_val_df['label']
        )
        
        logger.info(
            f"✅ Split complete:\n"
            f"   Train: {len(train_df)} samples\n"
            f"   Val:   {len(val_df)} samples\n"
            f"   Test:  {len(test_df)} samples"
        )
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = './data/ncf'
    ):
        """
        Save processed data and mappings
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Output directory
        """
        logger.info(f"💾 Saving processed data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrames
        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        val_df.to_csv(f"{output_dir}/val.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        # Save as numpy arrays (for faster loading)
        np.save(f"{output_dir}/train_users.npy", train_df['user_idx'].values)
        np.save(f"{output_dir}/train_items.npy", train_df['item_idx'].values)
        np.save(f"{output_dir}/train_labels.npy", train_df['label'].values)
        
        np.save(f"{output_dir}/val_users.npy", val_df['user_idx'].values)
        np.save(f"{output_dir}/val_items.npy", val_df['item_idx'].values)
        np.save(f"{output_dir}/val_labels.npy", val_df['label'].values)
        
        np.save(f"{output_dir}/test_users.npy", test_df['user_idx'].values)
        np.save(f"{output_dir}/test_items.npy", test_df['item_idx'].values)
        np.save(f"{output_dir}/test_labels.npy", test_df['label'].values)
        
        # Save mappings
        with open(f"{output_dir}/mappings.pkl", 'wb') as f:
            pickle.dump({
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'reverse_user_map': self.reverse_user_map,
                'reverse_item_map': self.reverse_item_map,
                'num_users': len(self.user_id_map),
                'num_items': len(self.item_id_map)
            }, f)
        
        logger.info("✅ All data saved successfully!")
    
    async def prepare_full_pipeline(
        self,
        output_dir: str = './data/ncf',
        days_back: int = 90,
        min_interactions: int = 5,
        num_negatives: int = 4
    ):
        """
        Run full data preparation pipeline
        
        Args:
            output_dir: Output directory
            days_back: Days of history to use
            min_interactions: Minimum interactions per user
            num_negatives: Negative samples per positive
        """
        logger.info("🚀 Starting full data preparation pipeline...")
        start_time = datetime.now()
        
        # 1. Extract data
        df = await self.extract_feedback_data(days_back, min_interactions)
        
        if len(df) == 0:
            logger.error("❌ No data to process!")
            return
        
        # 2. Create ID mappings
        self.create_id_mappings(df)
        
        # 3. Convert to implicit feedback
        df = self.convert_to_implicit_feedback(df)
        
        # 4. Add negative samples
        df = self.add_negative_samples(df, num_negatives)
        
        # 5. Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # 6. Save data
        self.save_processed_data(train_df, val_df, test_df, output_dir)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Pipeline complete in {elapsed:.1f}s!")
        
        # Print summary
        logger.info(
            f"\n📊 Data Summary:\n"
            f"   Users: {len(self.user_id_map)}\n"
            f"   Items: {len(self.item_id_map)}\n"
            f"   Train samples: {len(train_df)}\n"
            f"   Val samples: {len(val_df)}\n"
            f"   Test samples: {len(test_df)}\n"
            f"   Positive ratio: {train_df['label'].mean():.2%}"
        )


async def main():
    """Main function"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/aiistanbul')
    
    preparator = NCFDataPreparator(db_url)
    
    await preparator.prepare_full_pipeline(
        output_dir='./data/ncf',
        days_back=90,
        min_interactions=5,
        num_negatives=4
    )


if __name__ == "__main__":
    asyncio.run(main())
