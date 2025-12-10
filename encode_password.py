#!/usr/bin/env python3
"""
URL-encode the password for DATABASE_URL
"""

from urllib.parse import quote_plus

username = "postgres"
password = "*iwP#MDmX5dn8V:1LExE|70:O>|i"
host = "database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com"
port = "5432"
database = "postgres"

# URL-encode the password
encoded_password = quote_plus(password)

# Create the connection string
database_url = f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}"

print("="*60)
print("🔑 URL-Encoded DATABASE_URL")
print("="*60)
print(database_url)
print("="*60)
print("\n✅ Copy this to your .env file as DATABASE_URL")
print("\n⚠️  Special characters in password have been URL-encoded")
print(f"Original password: {password}")
print(f"Encoded password: {encoded_password}")
