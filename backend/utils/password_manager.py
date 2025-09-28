#!/usr/bin/env python3
"""
Password Manager Utility for AI Istanbul Backend
Utility to generate secure password hashes for admin users
"""

import bcrypt
import getpass
import os
import sys
from pathlib import Path

def hash_password(password: str) -> str:
    """Generate bcrypt hash for password"""
    salt = bcrypt.gensalt(rounds=12)  # Higher rounds for better security
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception as e:
        print(f"Verification error: {e}")
        return False

def generate_admin_password():
    """Interactive password generation for admin"""
    print("AI Istanbul - Admin Password Generator")
    print("=" * 40)
    
    # Get password securely
    while True:
        password = getpass.getpass("Enter new admin password: ")
        confirm_password = getpass.getpass("Confirm password: ")
        
        if password != confirm_password:
            print("Passwords don't match. Try again.")
            continue
            
        if len(password) < 8:
            print("Password must be at least 8 characters long.")
            continue
            
        break
    
    # Generate hash
    password_hash = hash_password(password)
    
    # Verify hash works
    if verify_password(password, password_hash):
        print("\n✅ Password hash generated successfully!")
        print(f"Hash: {password_hash}")
        print("\n📝 Add this to your .env file:")
        print(f"ADMIN_PASSWORD_HASH={password_hash}")
        
        # Optionally update .env file
        env_path = Path("../backend/.env")
        if env_path.exists():
            update_env = input("\nUpdate .env file automatically? (y/N): ")
            if update_env.lower() == 'y':
                update_env_file(env_path, password_hash)
    else:
        print("❌ Error: Generated hash verification failed!")
        sys.exit(1)

def update_env_file(env_path: Path, new_hash: str):
    """Update .env file with new password hash"""
    try:
        # Read current content
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update or add ADMIN_PASSWORD_HASH
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('ADMIN_PASSWORD_HASH='):
                lines[i] = f'ADMIN_PASSWORD_HASH={new_hash}\n'
                updated = True
                break
        
        if not updated:
            lines.append(f'ADMIN_PASSWORD_HASH={new_hash}\n')
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        print(f"✅ Updated {env_path}")
        print("🔄 Restart the server to apply changes.")
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")

if __name__ == "__main__":
    generate_admin_password()
