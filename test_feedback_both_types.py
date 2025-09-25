#!/usr/bin/env python3
"""
Test script to verify that both liked and disliked sessions are saved to the database.
"""

import requests
import json
import time
import uuid
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_feedback_session_creation():
    """Test that both liked and disliked feedback creates sessions in the database."""
    print("🧪 Testing Feedback Session Creation for Both Like and Dislike")
    print("=" * 60)
    
    # Test data
    test_cases = [
        {
            "sessionId": f"test-like-{uuid.uuid4().hex[:8]}",
            "feedbackType": "like",
            "userQuery": "Where can I find the best baklava in Istanbul?",
            "messageText": "For the best baklava in Istanbul, I recommend: 🧿 **Karaköy Güllüoğlu** - Famous for their traditional baklava since 1949, located in Karaköy. 🧿 **Hafiz Mustafa** - Historic confectionery with multiple locations, known for their pistachio baklava."
        },
        {
            "sessionId": f"test-dislike-{uuid.uuid4().hex[:8]}",
            "feedbackType": "dislike",
            "userQuery": "How do I get from Taksim to Galata Tower?",
            "messageText": "You can walk from Taksim to Galata Tower in about 10-15 minutes. Head down İstiklal Street towards Galata, then turn right at the end to reach the tower area."
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: Sending {test_case['feedbackType']} feedback")
        print(f"Session ID: {test_case['sessionId']}")
        print(f"Query: {test_case['userQuery']}")
        
        try:
            # Send feedback
            response = requests.post(
                f"{BASE_URL}/feedback",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Feedback sent successfully")
                result_data = response.json()
                print(f"Response: {result_data}")
                results.append({
                    "test_id": i,
                    "session_id": test_case['sessionId'],
                    "feedback_type": test_case['feedbackType'],
                    "status": "success"
                })
            else:
                print(f"❌ Failed to send feedback: {response.status_code}")
                print(f"Response: {response.text}")
                results.append({
                    "test_id": i,
                    "session_id": test_case['sessionId'],
                    "feedback_type": test_case['feedbackType'],
                    "status": "failed",
                    "error": response.text
                })
                
        except Exception as e:
            print(f"❌ Error sending feedback: {e}")
            results.append({
                "test_id": i,
                "session_id": test_case['sessionId'],
                "feedback_type": test_case['feedbackType'],
                "status": "error",
                "error": str(e)
            })
    
    # Wait a moment for database operations
    print(f"\n⏳ Waiting 2 seconds for database operations...")
    time.sleep(2)
    
    # Check if sessions were created by querying the API
    print(f"\n🔍 Checking if sessions were created in the database...")
    
    try:
        sessions_response = requests.get(f"{BASE_URL}/api/chat-sessions")
        if sessions_response.status_code == 200:
            sessions_data = sessions_response.json()
            all_sessions = sessions_data.get('sessions', [])
            
            print(f"📊 Total sessions in database: {len(all_sessions)}")
            
            # Check for our test sessions
            for result in results:
                session_id = result['session_id']
                feedback_type = result['feedback_type']
                
                # Find our test session
                test_session = None
                for session in all_sessions:
                    if session['id'] == session_id:
                        test_session = session
                        break
                
                if test_session:
                    print(f"✅ {feedback_type.upper()} session found: {session_id}")
                    print(f"   Title: {test_session.get('title', 'N/A')}")
                    print(f"   Message Count: {test_session.get('message_count', 0)}")
                    print(f"   Saved: {test_session.get('saved_at', 'N/A')}")
                    
                    # Check conversation history
                    history = test_session.get('conversation_history', [])
                    if history and len(history) > 0:
                        last_entry = history[-1]
                        entry_feedback = last_entry.get('feedback', 'N/A')
                        print(f"   Last feedback: {entry_feedback}")
                    
                    result['found_in_db'] = True
                else:
                    print(f"❌ {feedback_type.upper()} session NOT found: {session_id}")
                    result['found_in_db'] = False
                    
        else:
            print(f"❌ Failed to fetch sessions: {sessions_response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking sessions: {e}")
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 40)
    
    like_tests = [r for r in results if r['feedback_type'] == 'like']
    dislike_tests = [r for r in results if r['feedback_type'] == 'dislike']
    
    like_success = len([r for r in like_tests if r.get('found_in_db', False)])
    dislike_success = len([r for r in dislike_tests if r.get('found_in_db', False)])
    
    print(f"LIKE feedback sessions saved: {like_success}/{len(like_tests)}")
    print(f"DISLIKE feedback sessions saved: {dislike_success}/{len(dislike_tests)}")
    
    if like_success == len(like_tests) and dislike_success == len(dislike_tests):
        print(f"🎉 ALL TESTS PASSED: Both liked and disliked sessions are being saved!")
    else:
        print(f"❌ SOME TESTS FAILED: Check the implementation")
    
    return results

if __name__ == "__main__":
    test_feedback_session_creation()
