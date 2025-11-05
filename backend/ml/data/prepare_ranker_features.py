"""
Feature Engineering for LightGBM Ranker
Week 7-8: Create rich features for ranking model

Features extracted:
1. User features: demographics, interaction history
2. Item features: popularity, category, ratings
3. User-Item features: interaction type, recency
4. Context features: time of day, day of week
5. NCF embeddings: precomputed embeddings from NCF model
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import pickle


class RankerFeatureEngineer:
    """
    Feature engineering for LightGBM ranker
    """
    
    def __init__(
        self,
        ncf_embeddings_path: Optional[str] = None,
        min_interactions: int = 3
    ):
        """
        Initialize feature engineer
        
        Args:
            ncf_embeddings_path: Path to precomputed NCF embeddings
            min_interactions: Minimum interactions for user/item
        """
        self.ncf_embeddings_path = ncf_embeddings_path
        self.min_interactions = min_interactions
        
        # Feature statistics (for normalization)
        self.feature_stats: Dict = {}
        self.ncf_embeddings: Optional[Dict] = None
        
        if ncf_embeddings_path and Path(ncf_embeddings_path).exists():
            self._load_ncf_embeddings()
    
    def _load_ncf_embeddings(self):
        """Load precomputed NCF embeddings"""
        print(f"📦 Loading NCF embeddings from {self.ncf_embeddings_path}")
        with open(self.ncf_embeddings_path, 'rb') as f:
            self.ncf_embeddings = pickle.load(f)
        print(f"✅ Loaded {len(self.ncf_embeddings.get('user', {}))} user and "
              f"{len(self.ncf_embeddings.get('item', {}))} item embeddings")
    
    def prepare_features(
        self,
        interactions_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        items_df: Optional[pd.DataFrame] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for training and validation
        
        Args:
            interactions_df: Dataframe with columns [user_id, item_id, rating, timestamp]
            users_df: Optional user features dataframe
            items_df: Optional item features dataframe
            test_size: Validation split ratio
            
        Returns:
            X_train, y_train, group_train, X_val, y_val, group_val, feature_names
        """
        print("🔧 Starting feature engineering...")
        
        # Add temporal features
        interactions_df = self._add_temporal_features(interactions_df)
        
        # Compute user statistics
        user_stats = self._compute_user_stats(interactions_df)
        
        # Compute item statistics
        item_stats = self._compute_item_stats(interactions_df)
        
        # Split by user (to prevent leakage)
        unique_users = interactions_df['user_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_users)
        
        split_idx = int(len(unique_users) * (1 - test_size))
        train_users = set(unique_users[:split_idx])
        val_users = set(unique_users[split_idx:])
        
        train_df = interactions_df[interactions_df['user_id'].isin(train_users)].copy()
        val_df = interactions_df[interactions_df['user_id'].isin(val_users)].copy()
        
        print(f"📊 Train: {len(train_df)} interactions from {len(train_users)} users")
        print(f"📊 Val: {len(val_df)} interactions from {len(val_users)} users")
        
        # Extract features
        X_train, y_train, group_train, feature_names = self._extract_features(
            train_df, user_stats, item_stats, users_df, items_df, is_train=True
        )
        
        X_val, y_val, group_val, _ = self._extract_features(
            val_df, user_stats, item_stats, users_df, items_df, is_train=False
        )
        
        print(f"✅ Features prepared: {X_train.shape[1]} features")
        print(f"✅ Train groups: {len(group_train)}, Val groups: {len(group_val)}")
        
        return X_train, y_train, group_train, X_val, y_val, group_val, feature_names
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df['datetime'].dt.month
        else:
            # Use current time as default
            now = datetime.now()
            df['hour'] = now.hour
            df['day_of_week'] = now.weekday()
            df['is_weekend'] = int(now.weekday() in [5, 6])
            df['month'] = now.month
        
        return df
    
    def _compute_user_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute user-level statistics"""
        user_stats = df.groupby('user_id').agg({
            'item_id': 'count',  # Number of interactions
            'rating': ['mean', 'std', 'min', 'max'],  # Rating statistics
            'timestamp': ['min', 'max']  # First and last interaction
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = ['user_id', 'n_interactions', 'avg_rating', 
                             'std_rating', 'min_rating', 'max_rating',
                             'first_timestamp', 'last_timestamp']
        
        # Compute user activity recency
        if 'timestamp' in df.columns:
            max_ts = df['timestamp'].max()
            user_stats['days_since_last_interaction'] = (
                max_ts - user_stats['last_timestamp']
            ) / (24 * 3600)  # Convert to days
        else:
            user_stats['days_since_last_interaction'] = 0
        
        # Fill NaN in std_rating
        user_stats['std_rating'] = user_stats['std_rating'].fillna(0)
        
        return user_stats
    
    def _compute_item_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute item-level statistics"""
        item_stats = df.groupby('item_id').agg({
            'user_id': 'count',  # Popularity (number of interactions)
            'rating': ['mean', 'std', 'min', 'max'],  # Rating statistics
            'timestamp': 'max'  # Last interaction
        }).reset_index()
        
        # Flatten column names
        item_stats.columns = ['item_id', 'popularity', 'avg_rating',
                             'std_rating', 'min_rating', 'max_rating',
                             'last_timestamp']
        
        # Fill NaN in std_rating
        item_stats['std_rating'] = item_stats['std_rating'].fillna(0)
        
        # Log-scale popularity
        item_stats['log_popularity'] = np.log1p(item_stats['popularity'])
        
        return item_stats
    
    def _extract_features(
        self,
        df: pd.DataFrame,
        user_stats: pd.DataFrame,
        item_stats: pd.DataFrame,
        users_df: Optional[pd.DataFrame],
        items_df: Optional[pd.DataFrame],
        is_train: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from interaction data
        
        Returns:
            X: Feature matrix
            y: Labels (ratings)
            group: Group sizes for ranking
            feature_names: Names of features
        """
        # Merge with statistics
        df = df.merge(user_stats, on='user_id', how='left', suffixes=('', '_user'))
        df = df.merge(item_stats, on='item_id', how='left', suffixes=('', '_item'))
        
        feature_list = []
        feature_names = []
        
        # 1. User features
        user_features = [
            'n_interactions',
            'avg_rating_user',
            'std_rating_user',
            'days_since_last_interaction',
        ]
        
        for feat in user_features:
            if feat in df.columns:
                feature_list.append(df[feat].values.reshape(-1, 1))
                feature_names.append(f'user_{feat}')
        
        # 2. Item features
        item_features = [
            'popularity',
            'log_popularity',
            'avg_rating_item',
            'std_rating_item',
        ]
        
        for feat in item_features:
            if feat in df.columns:
                feature_list.append(df[feat].values.reshape(-1, 1))
                feature_names.append(f'item_{feat}')
        
        # 3. Temporal features
        temporal_features = [
            'hour',
            'day_of_week',
            'is_weekend',
            'month',
        ]
        
        for feat in temporal_features:
            if feat in df.columns:
                feature_list.append(df[feat].values.reshape(-1, 1))
                feature_names.append(f'temporal_{feat}')
        
        # 4. User-Item interaction features
        # Rating deviation from user mean
        if 'avg_rating_user' in df.columns:
            user_rating_dev = df['rating'] - df['avg_rating_user']
            feature_list.append(user_rating_dev.values.reshape(-1, 1))
            feature_names.append('user_item_rating_deviation')
        
        # Rating deviation from item mean
        if 'avg_rating_item' in df.columns:
            item_rating_dev = df['rating'] - df['avg_rating_item']
            feature_list.append(item_rating_dev.values.reshape(-1, 1))
            feature_names.append('item_rating_deviation')
        
        # 5. NCF embeddings (if available)
        if self.ncf_embeddings is not None:
            user_emb_features = self._get_ncf_embeddings(
                df['user_id'].values,
                df['item_id'].values
            )
            if user_emb_features is not None:
                feature_list.append(user_emb_features)
                emb_dim = user_emb_features.shape[1]
                feature_names.extend([f'ncf_emb_{i}' for i in range(emb_dim)])
        
        # Concatenate all features
        X = np.hstack(feature_list).astype(np.float32)
        
        # Normalize features (fit on train, transform on both)
        if is_train:
            self.feature_stats = {
                'mean': X.mean(axis=0),
                'std': X.std(axis=0) + 1e-8  # Avoid division by zero
            }
        
        X = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Labels (ratings) - convert to integer relevance for ranking
        # LightGBM lambdarank requires integer labels
        y_raw = df['rating'].values.astype(np.float32)
        
        # Convert continuous ratings (1-5) to discrete relevance levels (0-4)
        # 1.0-1.5 -> 0, 1.5-2.5 -> 1, 2.5-3.5 -> 2, 3.5-4.5 -> 3, 4.5-5.0 -> 4
        y = np.floor(y_raw).astype(np.int32)
        y = np.clip(y - 1, 0, 4)  # Shift to 0-4 range
        
        # Group by user (each user is a query in ranking)
        group = df.groupby('user_id').size().values
        
        return X, y, group, feature_names
    
    def _get_ncf_embeddings(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Get NCF embeddings and compute interaction features
        
        Returns:
            Embedding features (e.g., dot product, cosine similarity)
        """
        if self.ncf_embeddings is None:
            return None
        
        user_emb_dict = self.ncf_embeddings.get('user', {})
        item_emb_dict = self.ncf_embeddings.get('item', {})
        
        features = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            if user_id in user_emb_dict and item_id in item_emb_dict:
                user_emb = user_emb_dict[user_id]
                item_emb = item_emb_dict[item_id]
                
                # Compute embedding features
                dot_product = np.dot(user_emb, item_emb)
                
                # Cosine similarity
                user_norm = np.linalg.norm(user_emb)
                item_norm = np.linalg.norm(item_emb)
                cosine_sim = dot_product / (user_norm * item_norm + 1e-8)
                
                # Element-wise product (Hadamard)
                hadamard = user_emb * item_emb
                
                # Concatenate all embedding features
                emb_features = np.concatenate([
                    [dot_product, cosine_sim],
                    hadamard
                ])
                
                features.append(emb_features)
            else:
                # Use zero features if embedding not found
                emb_dim = len(next(iter(user_emb_dict.values())))
                features.append(np.zeros(emb_dim + 2))
        
        return np.array(features, dtype=np.float32)
    
    def save_feature_stats(self, output_path: str):
        """Save feature statistics for inference"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.feature_stats, f)
        
        print(f"💾 Feature statistics saved to {output_path}")
    
    def load_feature_stats(self, input_path: str):
        """Load feature statistics for inference"""
        with open(input_path, 'rb') as f:
            self.feature_stats = pickle.load(f)
        
        print(f"📦 Feature statistics loaded from {input_path}")


def create_synthetic_data(
    n_users: int = 1000,
    n_items: int = 500,
    n_interactions: int = 10000,
    output_dir: str = "data/ranker"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic interaction data for testing
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_interactions: Number of interactions
        output_dir: Directory to save data
        
    Returns:
        interactions_df, users_df, items_df
    """
    print(f"🎲 Creating synthetic data...")
    print(f"   Users: {n_users}, Items: {n_items}, Interactions: {n_interactions}")
    
    np.random.seed(42)
    
    # Create interactions
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # Ratings with some structure (popular items get higher ratings)
    item_popularity = np.random.pareto(1.5, n_items)
    item_base_rating = 3 + 2 * (item_popularity / item_popularity.max())
    
    ratings = []
    for item_id in item_ids:
        base = item_base_rating[item_id]
        rating = np.clip(base + np.random.normal(0, 0.5), 1, 5)
        ratings.append(rating)
    
    # Timestamps (last 90 days)
    now = int(datetime.now().timestamp())
    timestamps = now - np.random.randint(0, 90 * 24 * 3600, n_interactions)
    
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Remove duplicates
    interactions_df = interactions_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # Create user features (optional)
    users_df = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 70, n_users),
        'gender': np.random.choice(['M', 'F', 'Other'], n_users),
    })
    
    # Create item features (optional)
    items_df = pd.DataFrame({
        'item_id': range(n_items),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_items),
        'price': np.random.uniform(10, 100, n_items),
    })
    
    # Save data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    interactions_df.to_csv(output_path / 'interactions.csv', index=False)
    users_df.to_csv(output_path / 'users.csv', index=False)
    items_df.to_csv(output_path / 'items.csv', index=False)
    
    print(f"💾 Synthetic data saved to {output_dir}")
    print(f"   Interactions: {len(interactions_df)}")
    
    return interactions_df, users_df, items_df


if __name__ == "__main__":
    """
    Test feature engineering pipeline
    """
    print("="*60)
    print("🧪 Testing LightGBM Ranker Feature Engineering")
    print("="*60)
    
    # Create synthetic data
    interactions_df, users_df, items_df = create_synthetic_data(
        n_users=500,
        n_items=200,
        n_interactions=5000
    )
    
    # Initialize feature engineer
    engineer = RankerFeatureEngineer()
    
    # Prepare features
    X_train, y_train, group_train, X_val, y_val, group_val, feature_names = \
        engineer.prepare_features(interactions_df, users_df, items_df)
    
    print("\n" + "="*60)
    print("📊 Feature Engineering Results")
    print("="*60)
    print(f"✅ Training features shape: {X_train.shape}")
    print(f"✅ Validation features shape: {X_val.shape}")
    print(f"✅ Number of features: {len(feature_names)}")
    print(f"✅ Training groups: {len(group_train)} (sum={group_train.sum()})")
    print(f"✅ Validation groups: {len(group_val)} (sum={group_val.sum()})")
    
    print("\n📋 Feature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    print("\n📈 Label statistics:")
    print(f"  Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"  Val - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
    
    # Save feature stats
    engineer.save_feature_stats('data/ranker/feature_stats.pkl')
    
    print("\n✅ Feature engineering test completed successfully!")
