"""
Create Intent Feedback Table Migration
Run this to create the intent_feedback table in your database
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from sqlalchemy import create_engine
from models.intent_feedback import Base, IntentFeedback, create_tables
from database import DATABASE_URL, engine

def migrate_intent_feedback_table():
    """Create intent_feedback table if it doesn't exist"""
    
    print("=" * 60)
    print("Intent Feedback Table Migration")
    print("=" * 60)
    
    print(f"\n📊 Database: {DATABASE_URL[:50]}...")
    
    try:
        # Create tables
        print("\n🔧 Creating intent_feedback table...")
        create_tables(engine)
        
        print("\n✅ Migration completed successfully!")
        print("\n📋 Table Schema:")
        print("   - id (Primary Key)")
        print("   - session_id (Indexed)")
        print("   - user_id (Indexed)")
        print("   - original_query")
        print("   - language (Indexed)")
        print("   - predicted_intent (Indexed)")
        print("   - predicted_confidence")
        print("   - classification_method")
        print("   - latency_ms")
        print("   - is_correct (Indexed)")
        print("   - actual_intent (Indexed)")
        print("   - feedback_type (Indexed)")
        print("   - timestamp (Indexed)")
        print("   - user_action")
        print("   - used_for_training (Indexed)")
        print("   - review_status (Indexed)")
        print("   - context_data (JSON)")
        
        print("\n🔍 Indexes created:")
        print("   - idx_feedback_status (review_status, used_for_training)")
        print("   - idx_feedback_quality (feedback_type, is_correct, timestamp)")
        print("   - idx_training_data (used_for_training, review_status, predicted_intent)")
        
        print("\n🚀 Ready to collect feedback!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = migrate_intent_feedback_table()
    sys.exit(0 if success else 1)
