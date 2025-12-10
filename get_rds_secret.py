#!/usr/bin/env python3
"""
Retrieve RDS Credentials from AWS Secrets Manager
"""

import json
import boto3
from botocore.exceptions import ClientError

def get_secret():
    """Retrieve RDS credentials from AWS Secrets Manager"""
    
    secret_name = "rds!db-fb4ca2fc-e451-4c36-add1-66f80ff7f5c8"
    region_name = "eu-central-1"
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        print(f"🔄 Retrieving secret: {secret_name}")
        print(f"📍 Region: {region_name}\n")
        
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
        # Decrypts secret using the associated KMS key
        secret = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret)
        
        print("✅ Secret retrieved successfully!\n")
        print("="*60)
        print("🔑 RDS CREDENTIALS")
        print("="*60)
        print(f"Username: {secret_dict.get('username', 'NOT FOUND')}")
        print(f"Password: {secret_dict.get('password', 'NOT FOUND')}")
        print(f"Engine: {secret_dict.get('engine', 'postgres')}")
        print(f"Host: {secret_dict.get('host', 'database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com')}")
        print(f"Port: {secret_dict.get('port', '5432')}")
        print(f"Database: {secret_dict.get('dbname', 'postgres')}")
        print("="*60)
        
        # Generate connection string
        username = secret_dict.get('username')
        password = secret_dict.get('password')
        host = secret_dict.get('host', 'database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com')
        port = secret_dict.get('port', '5432')
        dbname = secret_dict.get('dbname', 'postgres')
        
        if username and password:
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{dbname}"
            print("\n📝 DATABASE_URL for .env file:")
            print("="*60)
            print(connection_string)
            print("="*60)
            
            # Save to a file for easy copying
            with open('aws_rds_credentials.txt', 'w') as f:
                f.write("AWS RDS Credentials\n")
                f.write("="*60 + "\n")
                f.write(f"Username: {username}\n")
                f.write(f"Password: {password}\n")
                f.write(f"Host: {host}\n")
                f.write(f"Port: {port}\n")
                f.write(f"Database: {dbname}\n")
                f.write("="*60 + "\n\n")
                f.write("DATABASE_URL:\n")
                f.write(connection_string + "\n")
            
            print("\n💾 Credentials saved to: aws_rds_credentials.txt")
            print("⚠️  Remember to delete this file after updating your .env!")
            
            return secret_dict
        else:
            print("\n❌ Username or password not found in secret")
            return None
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'DecryptionFailureException':
            print("❌ Secrets Manager can't decrypt the protected secret text using the provided KMS key")
        elif error_code == 'InternalServiceErrorException':
            print("❌ An error occurred on the server side")
        elif error_code == 'InvalidParameterException':
            print("❌ You provided an invalid value for a parameter")
        elif error_code == 'InvalidRequestException':
            print("❌ You provided a parameter value that is not valid for the current state of the resource")
        elif error_code == 'ResourceNotFoundException':
            print("❌ We can't find the resource that you asked for")
        elif error_code == 'AccessDeniedException':
            print("❌ Access Denied - You need to configure AWS credentials")
            print("\n💡 Configure AWS CLI:")
            print("   aws configure")
            print("   Then enter your AWS Access Key ID and Secret Access Key")
        else:
            print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    print("🔐 AWS RDS Credentials Retriever")
    print("="*60)
    get_secret()
