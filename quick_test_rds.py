#!/usr/bin/env python3
"""
Quick RDS Connection Tester
Tests different username combinations to find the correct one
"""

import psycopg2
import sys

# RDS Details
HOST = "database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com"
PORT = "5432"
DATABASE = "postgres"

def test_connection(username, password):
    """Test a single connection attempt"""
    try:
        print(f"🔄 Testing username: {username}...", end=" ")
        conn = psycopg2.connect(
            host=HOST,
            port=PORT,
            database=DATABASE,
            user=username,
            password=password,
            connect_timeout=10
        )
        conn.close()
        print("✅ SUCCESS!")
        return True
    except psycopg2.OperationalError as e:
        error_msg = str(e)
        if "authentication failed" in error_msg.lower():
            print("❌ Wrong password")
        elif "does not exist" in error_msg.lower():
            print("❌ Username doesn't exist")
        else:
            print(f"❌ {error_msg}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Test different common usernames"""
    
    # Get password from user
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = input("Enter your AWS RDS password: ").strip()
    
    if not password:
        print("❌ Password cannot be empty")
        return
    
    print(f"\n🧪 Testing RDS Connection: {HOST}")
    print("="*60)
    
    # Common PostgreSQL usernames
    usernames = [
        "postgres",
        "admin",
        "root",
        "masteruser",
        "dbadmin"
    ]
    
    found = False
    for username in usernames:
        if test_connection(username, password):
            found = True
            print("\n" + "="*60)
            print(f"✅ SUCCESS! Use this username: {username}")
            print("="*60)
            print("\n📝 Update your .env file with:")
            print(f"DATABASE_URL=postgresql://{username}:{password}@{HOST}:{PORT}/{DATABASE}")
            break
    
    if not found:
        print("\n" + "="*60)
        print("❌ Could not connect with common usernames")
        print("\n💡 Next steps:")
        print("1. Check AWS Console for the exact master username")
        print("2. Or reset your password in AWS RDS Console")
        print("="*60)

if __name__ == "__main__":
    main()
