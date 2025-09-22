#!/usr/bin/env python3
"""
Test script for blog like functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"
BLOG_API_URL = f"{BASE_URL}/blog"

def test_like_functionality():
    """Test the complete like functionality"""
    print("🧪 Testing Blog Like Functionality")
    print("=" * 50)
    
    try:
        # 1. Get list of blog posts
        print("\n📝 Step 1: Getting blog posts...")
        response = requests.get(f"{BLOG_API_URL}/posts")
        if response.status_code == 200:
            posts_data = response.json()
            posts = posts_data.get('posts', [])
            if posts:
                post_id = posts[0]['id']
                print(f"✅ Found post ID: {post_id}")
                initial_likes = posts[0].get('likes_count', 0)
                print(f"   Initial likes: {initial_likes}")
            else:
                print("❌ No posts found")
                return
        else:
            print(f"❌ Failed to get posts: {response.status_code}")
            return
        
        # 2. Check initial like status
        print(f"\n👁️ Step 2: Checking like status for post {post_id}...")
        user_identifier = "test_user_123"
        response = requests.get(f"{BLOG_API_URL}/posts/{post_id}/like-status", 
                               params={"user_identifier": user_identifier})
        if response.status_code == 200:
            like_status = response.json()
            print(f"✅ Initial like status: {like_status}")
        else:
            print(f"❌ Failed to check like status: {response.status_code}")
            return
        
        # 3. Like the post
        print(f"\n❤️ Step 3: Liking post {post_id}...")
        response = requests.post(f"{BLOG_API_URL}/posts/{post_id}/like")
        if response.status_code == 200:
            like_response = response.json()
            print(f"✅ Like successful: {like_response}")
            new_likes_count = like_response.get('likes_count', 0)
            print(f"   New likes count: {new_likes_count}")
        else:
            print(f"❌ Failed to like post: {response.status_code}")
            return
        
        # 4. Check like status again
        print(f"\n🔄 Step 4: Checking like status again...")
        response = requests.get(f"{BLOG_API_URL}/posts/{post_id}/like-status", 
                               params={"user_identifier": user_identifier})
        if response.status_code == 200:
            final_like_status = response.json()
            print(f"✅ Final like status: {final_like_status}")
            
            # Verify the count increased
            if final_like_status['likes'] > initial_likes:
                print("🎉 SUCCESS: Like count increased correctly!")
            else:
                print("⚠️ WARNING: Like count did not increase as expected")
        else:
            print(f"❌ Failed to check final like status: {response.status_code}")
        
        print("\n✅ Like functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_like_functionality()
