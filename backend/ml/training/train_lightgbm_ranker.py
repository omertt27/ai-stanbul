"""
Training script for LightGBM Ranker
Week 7-8: Train the ranking model

Usage:
    python train_lightgbm_ranker.py --data_dir data/ranker --output_dir models/ranker
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add ML directory to path
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

# Import from ML subdirectories
from models.lightgbm_ranker import LightGBMRanker
from data.prepare_ranker_features import RankerFeatureEngineer, create_synthetic_data


def train_ranker(
    data_dir: str,
    output_dir: str,
    use_synthetic: bool = False,
    ncf_embeddings_path: str = None,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    early_stopping_rounds: int = 10
):
    """
    Train LightGBM ranker model
    
    Args:
        data_dir: Directory containing interaction data
        output_dir: Directory to save trained model
        use_synthetic: Whether to use synthetic data
        ncf_embeddings_path: Path to NCF embeddings (optional)
        num_leaves: LightGBM num_leaves parameter
        learning_rate: Learning rate
        n_estimators: Number of boosting rounds
        early_stopping_rounds: Early stopping patience
    """
    print("="*80)
    print("🚀 LightGBM Ranker Training")
    print("="*80)
    print(f"📂 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🤖 Use synthetic data: {use_synthetic}")
    if ncf_embeddings_path:
        print(f"🧠 NCF embeddings: {ncf_embeddings_path}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    data_path = Path(data_dir)
    
    if use_synthetic or not (data_path / 'interactions.csv').exists():
        print("📊 Creating synthetic data for testing...")
        interactions_df, users_df, items_df = create_synthetic_data(
            n_users=1000,
            n_items=500,
            n_interactions=10000,
            output_dir=data_dir
        )
    else:
        print("📊 Loading interaction data...")
        interactions_df = pd.read_csv(data_path / 'interactions.csv')
        
        users_df = None
        if (data_path / 'users.csv').exists():
            users_df = pd.read_csv(data_path / 'users.csv')
        
        items_df = None
        if (data_path / 'items.csv').exists():
            items_df = pd.read_csv(data_path / 'items.csv')
    
    print(f"✅ Loaded {len(interactions_df)} interactions")
    print(f"✅ Users: {interactions_df['user_id'].nunique()}")
    print(f"✅ Items: {interactions_df['item_id'].nunique()}")
    print()
    
    # Feature engineering
    print("🔧 Starting feature engineering...")
    engineer = RankerFeatureEngineer(
        ncf_embeddings_path=ncf_embeddings_path,
        min_interactions=3
    )
    
    X_train, y_train, group_train, X_val, y_val, group_val, feature_names = \
        engineer.prepare_features(
            interactions_df,
            users_df,
            items_df,
            test_size=0.2
        )
    
    print(f"\n📊 Training data: {X_train.shape}")
    print(f"📊 Validation data: {X_val.shape}")
    print(f"📊 Features: {len(feature_names)}")
    print()
    
    # Initialize ranker
    print("🤖 Initializing LightGBM ranker...")
    ranker = LightGBMRanker(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_threads=8,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        verbose=1
    )
    
    # Train model
    print("\n" + "="*80)
    print("🏋️  Training LightGBM Ranker")
    print("="*80)
    
    training_start = datetime.now()
    
    metrics = ranker.train(
        X_train=X_train,
        y_train=y_train,
        group_train=group_train,
        X_val=X_val,
        y_val=y_val,
        group_val=group_val,
        feature_names=feature_names,
        early_stopping_rounds=early_stopping_rounds
    )
    
    training_time = (datetime.now() - training_start).total_seconds()
    
    print(f"\n⏱️  Training completed in {training_time:.2f} seconds")
    print()
    
    # Feature importance
    print("="*80)
    print("📊 Top 10 Most Important Features")
    print("="*80)
    importance_dict = ranker.get_feature_importance()
    importance_df = pd.DataFrame([
        {'feature': feat, 'importance': imp}
        for feat, imp in importance_dict.items()
    ]).head(10)
    print(importance_df.to_string(index=False))
    print()
    
    # Validation metrics
    print("="*80)
    print("📈 Validation Metrics")
    print("="*80)
    
    val_predictions = ranker.predict(X_val)
    
    # Compute NDCG per group
    ndcg_scores = []
    start_idx = 0
    
    for group_size in group_val:
        end_idx = start_idx + group_size
        
        group_labels = y_val[start_idx:end_idx]
        group_preds = val_predictions[start_idx:end_idx]
        
        # Sort by predictions
        sorted_indices = np.argsort(-group_preds)
        sorted_labels = group_labels[sorted_indices]
        
        # Compute DCG@10
        k = min(10, len(sorted_labels))
        dcg = np.sum((2 ** sorted_labels[:k] - 1) / np.log2(np.arange(2, k + 2)))
        
        # Compute IDCG@10
        ideal_labels = np.sort(group_labels)[::-1]
        idcg = np.sum((2 ** ideal_labels[:k] - 1) / np.log2(np.arange(2, k + 2)))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        start_idx = end_idx
    
    avg_ndcg = np.mean(ndcg_scores)
    print(f"📊 Average NDCG@10: {avg_ndcg:.4f}")
    
    # Prediction statistics
    print(f"📊 Prediction range: [{val_predictions.min():.2f}, {val_predictions.max():.2f}]")
    print(f"📊 Prediction mean: {val_predictions.mean():.2f}")
    print(f"📊 Prediction std: {val_predictions.std():.2f}")
    print()
    
    # Save model
    print("="*80)
    print("💾 Saving Model")
    print("="*80)
    
    model_path = output_path / 'lightgbm_ranker.pkl'
    ranker.save(str(model_path))
    
    # Save feature engineer stats
    feature_stats_path = output_path / 'feature_stats.pkl'
    engineer.save_feature_stats(str(feature_stats_path))
    
    # Save feature importance
    importance_path = output_path / 'feature_importance.csv'
    importance_full_df = pd.DataFrame([
        {'feature': feat, 'importance': imp}
        for feat, imp in importance_dict.items()
    ])
    importance_full_df.to_csv(importance_path, index=False)
    print(f"💾 Feature importance saved to {importance_path}")
    
    # Save training metadata
    metadata = {
        'training_time_seconds': training_time,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'avg_ndcg': float(avg_ndcg),
        'model_params': {
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
        },
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    metadata_path = output_path / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Training metadata saved to {metadata_path}")
    
    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"📂 Model saved to: {model_path}")
    print(f"📂 Feature stats saved to: {feature_stats_path}")
    print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"📈 Validation NDCG@10: {avg_ndcg:.4f}")
    
    return ranker, engineer, metrics


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM Ranker')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/ranker',
        help='Directory containing interaction data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/ranker',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--use_synthetic',
        action='store_true',
        help='Use synthetic data for testing'
    )
    
    parser.add_argument(
        '--ncf_embeddings',
        type=str,
        default=None,
        help='Path to precomputed NCF embeddings'
    )
    
    parser.add_argument(
        '--num_leaves',
        type=int,
        default=31,
        help='LightGBM num_leaves parameter'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of boosting rounds'
    )
    
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    
    args = parser.parse_args()
    
    train_ranker(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_synthetic=args.use_synthetic,
        ncf_embeddings_path=args.ncf_embeddings,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds
    )


if __name__ == "__main__":
    main()
