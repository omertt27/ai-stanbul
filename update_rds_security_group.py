#!/usr/bin/env python3
"""
AWS RDS Security Group Updater
Adds your current IP to RDS security group for database access
"""

import boto3
import requests
import sys

def get_current_ip():
    """Get current public IP address"""
    try:
        print("   Checking ifconfig.me...", end=" ")
        response = requests.get('https://ifconfig.me', timeout=5)
        ip = response.text.strip()
        print(f"✅ {ip}")
        return ip
    except Exception as e:
        print(f"❌ Failed")
        try:
            print("   Trying ipify.org...", end=" ")
            response = requests.get('https://api.ipify.org', timeout=5)
            ip = response.text.strip()
            print(f"✅ {ip}")
            return ip
        except:
            print(f"❌ Failed")
            return None

def update_security_group(security_group_id, ip_address):
    """Add IP to security group inbound rules"""
    try:
        ec2 = boto3.client('ec2', region_name='eu-central-1')
        
        print(f"   Adding rule to {security_group_id}...", end=" ")
        
        # Add inbound rule for PostgreSQL
        response = ec2.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 5432,
                    'ToPort': 5432,
                    'IpRanges': [
                        {
                            'CidrIp': f'{ip_address}/32',
                            'Description': 'Development access from local machine'
                        }
                    ]
                }
            ]
        )
        
        print("✅ Added")
        return True
        
    except ec2.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidPermission.Duplicate':
            print("ℹ️  Already exists")
            return True
        else:
            print(f"❌ Error: {e.response['Error']['Message']}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🔧 AWS RDS Security Group Updater")
    print("="*70)
    
    # Get current IP
    print("\n1️⃣  Getting your current public IP address...")
    current_ip = get_current_ip()
    
    if not current_ip:
        print("\n❌ Could not automatically detect your IP address")
        print("\n💡 Manual steps:")
        print("   1. Get your IP: curl ifconfig.me")
        print("   2. Go to AWS Console → EC2 → Security Groups")
        print("   3. Find sg-097205416c4b1f1e9")
        print("   4. Add inbound rule: PostgreSQL (5432) from YOUR_IP/32")
        sys.exit(1)
    
    # Security group ID for RDS
    security_group_id = "sg-097205416c4b1f1e9"
    
    # Update security group
    print(f"\n2️⃣  Updating RDS security group...")
    
    success = update_security_group(security_group_id, current_ip)
    
    print("\n" + "="*70)
    
    if success:
        print("✅ SUCCESS! Security group updated")
        print("="*70)
        print(f"\n📋 Added inbound rule:")
        print(f"   • Type: PostgreSQL")
        print(f"   • Port: 5432")
        print(f"   • Source: {current_ip}/32")
        print(f"   • Security Group: {security_group_id}")
        
        print("\n⏱️  Wait 1-2 minutes for changes to propagate")
        
        print("\n🚀 Next steps:")
        print("   1. Test connection:")
        print("      python3 test_rds_connection.py")
        print("\n   2. Run migration:")
        print("      python3 migrate_render_to_aws.py")
        
    else:
        print("❌ Could not update security group automatically")
        print("="*70)
        print("\n📝 Please update manually:")
        print(f"\n   1. Go to AWS Console:")
        print(f"      https://console.aws.amazon.com/ec2/v2/home?region=eu-central-1#SecurityGroups:search={security_group_id}")
        print(f"\n   2. Click on security group: {security_group_id}")
        print(f"\n   3. Click 'Edit inbound rules'")
        print(f"\n   4. Click 'Add rule':")
        print(f"      • Type: PostgreSQL")
        print(f"      • Port: 5432")
        print(f"      • Source: {current_ip}/32")
        print(f"      • Description: Development access")
        print(f"\n   5. Click 'Save rules'")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
