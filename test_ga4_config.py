#!/usr/bin/env python3
"""
Test Google Analytics 4 API Configuration
"""
import os
import sys
import json
from datetime import datetime

# Add the backend directory to the path
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul/backend')

def test_ga4_configuration():
    """Test if GA4 API is properly configured"""
    print("🔍 Testing Google Analytics 4 API Configuration")
    print("=" * 50)
    
    # Check environment variables
    property_id = os.getenv('GOOGLE_ANALYTICS_PROPERTY_ID')
    service_account_path = os.getenv('GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH')
    
    print(f"Property ID: {property_id or 'NOT SET'}")
    print(f"Service Account Path: {service_account_path or 'NOT SET'}")
    
    if not property_id:
        print("❌ GOOGLE_ANALYTICS_PROPERTY_ID not set in .env file")
        return False
    
    if not service_account_path:
        print("❌ GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH not set in .env file")
        return False
    
    # Check if service account file exists
    if not os.path.exists(service_account_path):
        print(f"❌ Service account file not found: {service_account_path}")
        return False
    
    # Check if it's valid JSON
    try:
        with open(service_account_path, 'r') as f:
            credentials = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in credentials]
        
        if missing_fields:
            print(f"❌ Invalid service account JSON. Missing fields: {missing_fields}")
            return False
        
        print(f"✅ Service account file is valid")
        print(f"   Project ID: {credentials['project_id']}")
        print(f"   Client Email: {credentials['client_email']}")
        
    except Exception as e:
        print(f"❌ Error reading service account file: {e}")
        return False
    
    # Test Google Analytics API import
    try:
        from api_clients.google_analytics_api import google_analytics_service
        print(f"✅ Google Analytics API service imported successfully")
        
        if google_analytics_service.enabled:
            print(f"✅ Google Analytics service is ENABLED and ready!")
            return True
        else:
            print(f"⚠️ Google Analytics service imported but not enabled. Check credentials.")
            return False
            
    except Exception as e:
        print(f"❌ Error importing Google Analytics service: {e}")
        return False

def show_next_steps():
    """Show what to do after configuration"""
    print("\n🚀 Next Steps:")
    print("1. Restart your backend server:")
    print("   cd backend && python start_server.py")
    print("\n2. Check the logs for:")
    print("   ✅ Google Analytics API initialized successfully")
    print("\n3. Visit admin dashboard:")
    print("   http://localhost:3000/admin")
    print("\n4. Look for 'Data Source: Google Analytics 4' at the bottom")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('/Users/omer/Desktop/ai-stanbul/.env')
    
    success = test_ga4_configuration()
    
    if success:
        print("\n🎉 Configuration Test: PASSED")
        print("Your Google Analytics 4 API is properly configured!")
    else:
        print("\n❌ Configuration Test: FAILED")
        print("Please check the issues above and try again.")
    
    show_next_steps()
