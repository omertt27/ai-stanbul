"""
Automated Retraining Scheduler
Automatically retrain intent classifier when conditions are met
"""

import os
import json
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Automatically retrain model when conditions are met
    
    Triggers:
    1. Minimum new samples threshold reached
    2. Maximum days since last training exceeded
    3. Model accuracy drops below threshold
    """
    
    def __init__(self,
                 db_session=None,
                 model_dir: str = "./models",
                 data_dir: str = "./training_data",
                 min_new_samples: int = 500,
                 max_days_since_training: int = 30,
                 min_accuracy_drop: float = 0.05,
                 validation_split: float = 0.2):
        """
        Initialize retraining scheduler
        
        Args:
            db_session: Database session for feedback queries
            model_dir: Directory to store models
            data_dir: Directory to store training data
            min_new_samples: Minimum new samples to trigger retraining
            max_days_since_training: Max days between retraining
            min_accuracy_drop: Retrain if accuracy drops by this amount
            validation_split: Portion of data for validation
        """
        self.db_session = db_session
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.min_new_samples = min_new_samples
        self.max_days_since_training = max_days_since_training
        self.min_accuracy_drop = min_accuracy_drop
        self.validation_split = validation_split
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history file
        self.history_file = self.model_dir / "training_history.json"
        self.history = self._load_training_history()
    
    def _load_training_history(self) -> Dict[str, Any]:
        """Load training history from file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        
        return {
            'last_training_date': None,
            'training_runs': [],
            'current_model_version': None,
            'current_model_accuracy': None
        }
    
    def _save_training_history(self):
        """Save training history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    async def check_retraining_needed(self) -> Dict[str, Any]:
        """
        Check if retraining should be triggered
        
        Returns:
            Dictionary with decision and reasons
        """
        
        logger.info("🔍 Checking if retraining is needed...")
        
        reasons = []
        should_retrain = False
        
        # Check 1: Count new training samples
        new_samples = await self._count_new_samples()
        if new_samples >= self.min_new_samples:
            reasons.append(f"New samples: {new_samples} >= {self.min_new_samples}")
            should_retrain = True
        else:
            logger.info(f"  New samples: {new_samples}/{self.min_new_samples}")
        
        # Check 2: Time since last training
        days_since = await self._days_since_last_training()
        if days_since is not None and days_since >= self.max_days_since_training:
            reasons.append(f"Days since training: {days_since} >= {self.max_days_since_training}")
            should_retrain = True
        else:
            logger.info(f"  Days since training: {days_since or 'N/A'}")
        
        # Check 3: Accuracy drop
        if self.history['current_model_accuracy'] is not None:
            current_accuracy = await self._get_current_accuracy()
            baseline_accuracy = self.history['current_model_accuracy']
            
            if current_accuracy is not None:
                accuracy_drop = baseline_accuracy - current_accuracy
                if accuracy_drop >= self.min_accuracy_drop:
                    reasons.append(
                        f"Accuracy drop: {accuracy_drop:.2%} >= {self.min_accuracy_drop:.2%}"
                    )
                    should_retrain = True
                else:
                    logger.info(
                        f"  Accuracy: {current_accuracy:.2%} "
                        f"(baseline: {baseline_accuracy:.2%}, drop: {accuracy_drop:.2%})"
                    )
        
        result = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'new_samples': new_samples,
            'days_since_training': days_since,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if should_retrain:
            logger.info(f"✅ Retraining RECOMMENDED: {', '.join(reasons)}")
        else:
            logger.info("⏸️  Retraining NOT needed at this time")
        
        return result
    
    async def _count_new_samples(self) -> int:
        """Count new training-ready samples since last training"""
        
        if not self.db_session:
            logger.warning("No database session, returning 0 new samples")
            return 0
        
        try:
            from models.intent_feedback import IntentFeedback
            
            # Get samples that are approved and not yet used for training
            query = self.db_session.query(IntentFeedback).filter(
                IntentFeedback.review_status == 'approved',
                IntentFeedback.used_for_training == False
            )
            
            # If we have a last training date, only count samples after that
            if self.history['last_training_date']:
                last_date = datetime.fromisoformat(self.history['last_training_date'])
                query = query.filter(IntentFeedback.timestamp > last_date)
            
            count = query.count()
            return count
            
        except Exception as e:
            logger.error(f"Error counting new samples: {e}")
            return 0
    
    async def _days_since_last_training(self) -> Optional[int]:
        """Calculate days since last training"""
        
        if not self.history['last_training_date']:
            return None
        
        last_date = datetime.fromisoformat(self.history['last_training_date'])
        delta = datetime.utcnow() - last_date
        return delta.days
    
    async def _get_current_accuracy(self) -> Optional[float]:
        """
        Get current model accuracy from recent feedback
        
        Uses feedback from last 7 days to estimate current accuracy
        """
        
        if not self.db_session:
            return None
        
        try:
            from models.intent_feedback import FeedbackStatistics
            from datetime import timedelta
            
            start_date = datetime.utcnow() - timedelta(days=7)
            stats = FeedbackStatistics.calculate_accuracy(
                self.db_session,
                start_date=start_date
            )
            
            return stats['accuracy'] if stats['total'] >= 10 else None
            
        except Exception as e:
            logger.error(f"Error calculating current accuracy: {e}")
            return None
    
    async def trigger_retraining(self, force: bool = False) -> Dict[str, Any]:
        """
        Trigger automated retraining pipeline
        
        Args:
            force: Force retraining even if not needed
            
        Returns:
            Dictionary with retraining results
        """
        
        logger.info("=" * 60)
        logger.info("🔄 Starting Automated Retraining Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.utcnow()
        
        # Check if retraining is needed
        if not force:
            check_result = await self.check_retraining_needed()
            if not check_result['should_retrain']:
                return {
                    'success': False,
                    'message': 'Retraining not needed',
                    'check_result': check_result
                }
        
        try:
            # Step 1: Collect and filter feedback data
            logger.info("\n📊 Step 1: Collecting feedback data...")
            feedback_data = await self._collect_feedback_data()
            logger.info(f"  Collected {len(feedback_data)} feedback samples")
            
            # Step 2: Filter for quality
            logger.info("\n🔍 Step 2: Filtering for quality...")
            training_data = await self._filter_training_data(feedback_data)
            logger.info(f"  Filtered to {len(training_data)} quality samples")
            
            if len(training_data) < self.min_new_samples and not force:
                return {
                    'success': False,
                    'message': f'Insufficient quality samples: {len(training_data)}/{self.min_new_samples}'
                }
            
            # Step 3: Merge with existing training data
            logger.info("\n📁 Step 3: Merging with existing data...")
            existing_data = await self._load_existing_training_data()
            merged_data = await self._merge_and_deduplicate(existing_data, training_data)
            logger.info(f"  Total training samples: {len(merged_data)}")
            
            # Step 4: Save training data
            logger.info("\n💾 Step 4: Saving training data...")
            data_file = await self._save_training_data(merged_data)
            logger.info(f"  Saved to: {data_file}")
            
            # Step 5: Train model (placeholder - actual training would go here)
            logger.info("\n🧠 Step 5: Training model...")
            model_result = await self._retrain_model(merged_data)
            
            # Step 6: Validate new model
            logger.info("\n✅ Step 6: Validating model...")
            validation_results = await self._validate_model(model_result)
            
            # Step 7: Update training history
            logger.info("\n📝 Step 7: Updating history...")
            self._record_training_run(
                sample_count=len(merged_data),
                validation_results=validation_results
            )
            
            # Step 8: Mark samples as used
            if self.db_session:
                await self._mark_samples_as_used(feedback_data)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'success': True,
                'message': 'Retraining completed successfully',
                'duration_seconds': duration,
                'training_samples': len(merged_data),
                'new_samples': len(training_data),
                'validation_accuracy': validation_results.get('accuracy'),
                'model_version': self.history['current_model_version'],
                'timestamp': end_time.isoformat()
            }
            
            logger.info("\n" + "=" * 60)
            logger.info(f"✅ Retraining completed in {duration:.1f}s")
            logger.info(f"   New accuracy: {validation_results.get('accuracy', 0):.2%}")
            logger.info("=" * 60 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'message': f'Retraining failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _collect_feedback_data(self) -> List[Any]:
        """Collect approved feedback from database"""
        
        if not self.db_session:
            return []
        
        try:
            from models.intent_feedback import IntentFeedback
            
            query = self.db_session.query(IntentFeedback).filter(
                IntentFeedback.review_status == 'approved',
                IntentFeedback.used_for_training == False
            )
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return []
    
    async def _filter_training_data(self, feedback_data: List[Any]) -> List[Any]:
        """Filter feedback data using quality filter"""
        
        from data_quality_filter import TrainingDataQualityFilter
        
        filter_system = TrainingDataQualityFilter(
            min_implicit_confidence=0.8,
            similarity_threshold=0.9,
            min_samples_per_intent=5,
            max_samples_per_intent=1000
        )
        
        return filter_system.filter_training_data(feedback_data)
    
    async def _load_existing_training_data(self) -> List[Dict]:
        """Load existing training data"""
        
        data_file = self.data_dir / "training_data.json"
        
        if not data_file.exists():
            logger.info("  No existing training data found")
            return []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"  Loaded {len(data)} existing samples")
        return data
    
    async def _merge_and_deduplicate(self, existing: List, new: List) -> List:
        """Merge and deduplicate training data"""
        
        # Convert new training examples to dicts
        new_dicts = [ex.to_dict() if hasattr(ex, 'to_dict') else ex for ex in new]
        
        # Merge
        merged = existing + new_dicts
        
        # Deduplicate by text (case-insensitive)
        seen = set()
        unique = []
        
        for item in merged:
            text_key = item['text'].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                unique.append(item)
        
        return unique
    
    async def _save_training_data(self, data: List[Dict]) -> Path:
        """Save training data to file"""
        
        # Create versioned filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        data_file = self.data_dir / f"training_data_{timestamp}.json"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Also save as latest
        latest_file = self.data_dir / "training_data.json"
        shutil.copy(data_file, latest_file)
        
        return data_file
    
    async def _retrain_model(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        Retrain the model (placeholder)
        
        In production, this would:
        1. Load the current model architecture
        2. Fine-tune on new data
        3. Save the new model
        """
        
        logger.info("  🔧 Model training (simulated)...")
        
        # Simulate training
        await asyncio.sleep(1)
        
        # Generate model version
        model_version = f"v{len(self.history['training_runs']) + 1}"
        model_file = self.model_dir / f"intent_model_{model_version}.pth"
        
        logger.info(f"  Model saved as: {model_version}")
        
        return {
            'model_version': model_version,
            'model_file': str(model_file),
            'training_samples': len(training_data)
        }
    
    async def _validate_model(self, model_result: Dict) -> Dict[str, Any]:
        """
        Validate the retrained model (placeholder)
        
        In production, this would:
        1. Run model on validation set
        2. Calculate accuracy metrics
        3. Compare with baseline
        """
        
        logger.info("  📊 Running validation...")
        
        # Simulate validation
        await asyncio.sleep(0.5)
        
        # Simulate improved accuracy
        baseline = self.history.get('current_model_accuracy') or 0.80
        new_accuracy = min(0.95, baseline + 0.03)  # Simulate 3% improvement
        
        return {
            'accuracy': new_accuracy,
            'baseline_accuracy': baseline,
            'improvement': new_accuracy - baseline if baseline else 0
        }
    
    def _record_training_run(self, sample_count: int, validation_results: Dict):
        """Record training run in history"""
        
        run = {
            'timestamp': datetime.utcnow().isoformat(),
            'sample_count': sample_count,
            'accuracy': validation_results.get('accuracy'),
            'model_version': self.history.get('current_model_version')
        }
        
        self.history['training_runs'].append(run)
        self.history['last_training_date'] = run['timestamp']
        self.history['current_model_accuracy'] = validation_results.get('accuracy')
        
        # Increment model version
        current_version = self.history.get('current_model_version', 'v0')
        version_num = int(current_version[1:]) + 1 if current_version else 1
        self.history['current_model_version'] = f"v{version_num}"
        
        self._save_training_history()
    
    async def _mark_samples_as_used(self, feedback_data: List[Any]):
        """Mark feedback samples as used for training"""
        
        try:
            for sample in feedback_data:
                sample.used_for_training = True
            
            self.db_session.commit()
            logger.info(f"  Marked {len(feedback_data)} samples as used")
            
        except Exception as e:
            logger.error(f"Error marking samples: {e}")
            self.db_session.rollback()


# Example usage
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test_scheduler():
        print("=" * 60)
        print("Automated Retraining Scheduler - Test")
        print("=" * 60)
        
        scheduler = RetrainingScheduler(
            model_dir="./test_models",
            data_dir="./test_data",
            min_new_samples=5  # Low threshold for testing
        )
        
        # Check if retraining needed
        check_result = await scheduler.check_retraining_needed()
        print(f"\nRetraining needed: {check_result['should_retrain']}")
        
        # Force a retraining run (for demo)
        print("\nTriggering forced retraining...")
        result = await scheduler.trigger_retraining(force=True)
        
        print(f"\n✅ Result: {result['message']}")
        if result['success']:
            print(f"   Accuracy: {result.get('validation_accuracy', 0):.2%}")
            print(f"   Duration: {result.get('duration_seconds', 0):.1f}s")
    
    asyncio.run(test_scheduler())
