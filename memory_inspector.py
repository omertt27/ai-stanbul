#!/usr/bin/env python3
"""
AI Istanbul - Memory Database Inspector
======================================

Inspect what memories and preferences have been stored in the database
"""

import sys
sys.path.append('backend')

from database import SessionLocal
from models import UserMemory, UserPreference, ConversationContext, UserSession
from datetime import datetime
import json

def inspect_session_memory(session_id: str):
    """Inspect the memories stored for a specific session"""
    db = SessionLocal()
    
    try:
        print(f"🧠 Memory Database Inspector")
        print(f"📧 Session ID: {session_id}")
        print("=" * 80)
        
        # Check if session exists
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if session:
            print(f"✅ Session found: Created {session.created_at}, Last active {session.last_activity}")
        else:
            print(f"⚠️ Session not found in user_sessions table")
        
        # Get memories
        memories = db.query(UserMemory).filter(UserMemory.session_id == session_id).all()
        print(f"\n📚 MEMORIES STORED: {len(memories)}")
        
        if memories:
            for i, memory in enumerate(memories, 1):
                print(f"\n--- Memory {i} ---")
                print(f"Type: {memory.memory_type}")
                print(f"Key: {memory.memory_key}")
                print(f"Value: {memory.memory_value}")
                print(f"Confidence: {memory.confidence_score}")
                print(f"Created: {memory.created_at}")
                print(f"References: {memory.reference_count}")
                if memory.memory_context:
                    print(f"Context: {memory.memory_context}")
        
        # Get preferences
        preferences = db.query(UserPreference).filter(UserPreference.session_id == session_id).all()
        print(f"\n🎯 PREFERENCES STORED: {len(preferences)}")
        
        if preferences:
            for i, pref in enumerate(preferences, 1):
                print(f"\n--- Preference {i} ---")
                print(f"Category: {pref.category}")
                print(f"Name: {pref.preference_name}")
                print(f"Value: {pref.preference_value}")
                print(f"Strength: {pref.strength}")
                print(f"Created: {pref.created_at}")
                if pref.context_info:
                    print(f"Context: {pref.context_info}")
                if pref.inferred_from:
                    print(f"Inferred from: {pref.inferred_from}")
        
        # Get conversation context
        contexts = db.query(ConversationContext).filter(ConversationContext.session_id == session_id).all()
        print(f"\n💬 CONVERSATION CONTEXTS: {len(contexts)}")
        
        if contexts:
            for i, ctx in enumerate(contexts, 1):
                print(f"\n--- Context {i} ---")
                print(f"Current Topic: {ctx.current_topic}")
                print(f"Travel Stage: {ctx.travel_stage}")
                print(f"Travel Style: {ctx.travel_style}")
                print(f"Visit Duration: {ctx.visit_duration}")
                print(f"Current Need: {ctx.current_need}")
                print(f"Last Location: {ctx.last_location_discussed}")
                print(f"Topics Discussed: {ctx.topics_discussed}")
                print(f"Places Mentioned: {ctx.places_mentioned}")
                print(f"Updated: {ctx.updated_at}")
        
        print(f"\n✅ Memory inspection complete")
        
    except Exception as e:
        print(f"❌ Error inspecting memory: {e}")
    finally:
        db.close()

def list_recent_sessions():
    """List recent sessions with memory data"""
    db = SessionLocal()
    
    try:
        print(f"📊 RECENT SESSIONS WITH MEMORY DATA")
        print("=" * 80)
        
        # Get sessions that have memories or preferences
        sessions_with_memories = db.query(UserMemory.session_id).distinct().all()
        sessions_with_preferences = db.query(UserPreference.session_id).distinct().all()
        
        all_sessions = set([s[0] for s in sessions_with_memories] + [s[0] for s in sessions_with_preferences])
        
        print(f"Found {len(all_sessions)} sessions with memory data:\n")
        
        for session_id in sorted(all_sessions):
            memory_count = db.query(UserMemory).filter(UserMemory.session_id == session_id).count()
            pref_count = db.query(UserPreference).filter(UserPreference.session_id == session_id).count()
            context_count = db.query(ConversationContext).filter(ConversationContext.session_id == session_id).count()
            
            print(f"📧 {session_id}")
            print(f"   Memories: {memory_count}, Preferences: {pref_count}, Contexts: {context_count}")
            
            # Get latest memory for this session
            latest_memory = db.query(UserMemory).filter(UserMemory.session_id == session_id).order_by(UserMemory.created_at.desc()).first()
            if latest_memory:
                print(f"   Latest: {latest_memory.created_at} - {latest_memory.memory_type}: {latest_memory.memory_key}")
            print()
        
    except Exception as e:
        print(f"❌ Error listing sessions: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
        inspect_session_memory(session_id)
    else:
        list_recent_sessions()
