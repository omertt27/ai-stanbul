#!/usr/bin/env python3
"""
Check AWS Resources - ElastiCache Redis, EC2, VPC, etc.
This script checks if you have AWS credentials and lists available AWS resources.
"""

import os
import sys

def check_env_credentials():
    """Check if AWS credentials are set in environment"""
    print("=" * 60)
    print("🔍 CHECKING AWS CREDENTIALS")
    print("=" * 60)
    
    aws_key = os.getenv('AWS_ACCESS_KEY_ID', '')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    aws_region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', ''))
    
    if aws_key:
        print(f"✅ AWS_ACCESS_KEY_ID: ***{aws_key[-4:]}")
    else:
        print("❌ AWS_ACCESS_KEY_ID: Not set")
    
    if aws_secret:
        print(f"✅ AWS_SECRET_ACCESS_KEY: ***{aws_secret[-4:]}")
    else:
        print("❌ AWS_SECRET_ACCESS_KEY: Not set")
    
    if aws_region:
        print(f"✅ AWS_REGION: {aws_region}")
    else:
        print("⚠️  AWS_REGION: Not set (will use default)")
    
    print()
    
    return bool(aws_key and aws_secret)

def check_boto3():
    """Check if boto3 is installed"""
    try:
        import boto3
        print(f"✅ boto3 is installed (version {boto3.__version__})")
        return True
    except ImportError:
        print("❌ boto3 is NOT installed")
        print("   Install with: pip install boto3")
        return False

def list_elasticache_clusters():
    """List all ElastiCache Redis clusters"""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        print("\n" + "=" * 60)
        print("🔍 CHECKING AWS ELASTICACHE REDIS CLUSTERS")
        print("=" * 60)
        
        # Get region from env or use default
        region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        print(f"Region: {region}\n")
        
        client = boto3.client('elasticache', region_name=region)
        
        # List all cache clusters
        response = client.describe_cache_clusters(ShowCacheNodeInfo=True)
        clusters = response.get('CacheClusters', [])
        
        if not clusters:
            print("❌ No ElastiCache clusters found in this region")
            print(f"\n💡 Try checking other regions:")
            print("   - us-east-1 (N. Virginia)")
            print("   - us-west-2 (Oregon)")
            print("   - eu-west-1 (Ireland)")
            print("   - eu-central-1 (Frankfurt)")
            return None
        
        print(f"✅ Found {len(clusters)} ElastiCache cluster(s):\n")
        
        redis_clusters = []
        for cluster in clusters:
            cluster_id = cluster.get('CacheClusterId', 'N/A')
            engine = cluster.get('Engine', 'N/A')
            status = cluster.get('CacheClusterStatus', 'N/A')
            
            print(f"Cluster: {cluster_id}")
            print(f"  Engine: {engine}")
            print(f"  Status: {status}")
            
            if engine.lower() == 'redis':
                redis_clusters.append(cluster)
                
                # Get endpoint
                nodes = cluster.get('CacheNodes', [])
                if nodes:
                    endpoint = nodes[0].get('Endpoint', {})
                    address = endpoint.get('Address', 'N/A')
                    port = endpoint.get('Port', 6379)
                    print(f"  ✅ Redis Endpoint: {address}:{port}")
                    print(f"  📋 REDIS_URL format: redis://{address}:{port}/0")
                else:
                    print(f"  ⚠️  No endpoint available")
                
                # Check if encryption and auth are enabled
                transit_encryption = cluster.get('TransitEncryptionEnabled', False)
                auth_enabled = cluster.get('AuthTokenEnabled', False)
                
                if transit_encryption:
                    print(f"  🔒 Transit Encryption: Enabled (use rediss://)")
                else:
                    print(f"  🔓 Transit Encryption: Disabled (use redis://)")
                
                if auth_enabled:
                    print(f"  🔑 Auth Token: Enabled (add password to URL)")
                else:
                    print(f"  🔓 Auth Token: Disabled (no password needed)")
                
            print()
        
        if redis_clusters:
            print(f"\n✅ Found {len(redis_clusters)} Redis cluster(s)!")
            return redis_clusters[0]  # Return first Redis cluster
        else:
            print("\n❌ No Redis clusters found (only other cache types)")
            return None
            
    except NoCredentialsError:
        print("❌ No AWS credentials found!")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return None
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_msg = e.response['Error']['Message']
        print(f"❌ AWS Error: {error_code}")
        print(f"   {error_msg}")
        
        if error_code == 'InvalidClientTokenId':
            print("\n💡 Your AWS credentials are invalid or expired")
        elif error_code == 'AccessDenied':
            print("\n💡 Your AWS credentials don't have ElastiCache permissions")
            print("   Required permissions: elasticache:DescribeCacheClusters")
        
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def check_other_regions():
    """Check ElastiCache in multiple AWS regions"""
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        print("\n" + "=" * 60)
        print("🌍 SCANNING MULTIPLE AWS REGIONS FOR REDIS")
        print("=" * 60)
        
        regions = [
            'us-east-1',      # N. Virginia
            'us-west-2',      # Oregon
            'eu-west-1',      # Ireland
            'eu-central-1',   # Frankfurt
            'ap-southeast-1', # Singapore
        ]
        
        found_clusters = {}
        
        for region in regions:
            print(f"\n📍 Checking {region}...")
            try:
                client = boto3.client('elasticache', region_name=region)
                response = client.describe_cache_clusters()
                clusters = response.get('CacheClusters', [])
                
                redis_count = sum(1 for c in clusters if c.get('Engine', '').lower() == 'redis')
                
                if redis_count > 0:
                    print(f"   ✅ Found {redis_count} cluster(s)")
                    found_clusters[region] = redis_count
                else:
                    print(f"   ❌ No clusters")
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'AccessDenied':
                    print(f"   ⚠️  Access denied")
                else:
                    print(f"   ❌ Error: {e.response['Error']['Code']}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:50]}")
        
        if found_clusters:
            print("\n" + "=" * 60)
            print("✅ REDIS CLUSTERS FOUND IN:")
            for region, count in found_clusters.items():
                print(f"   • {region}: {count} cluster(s)")
            print("=" * 60)
        else:
            print("\n❌ No Redis clusters found in any scanned region")
        
        return found_clusters
        
    except ImportError:
        print("❌ boto3 not installed, skipping region scan")
        return {}

def generate_setup_instructions(cluster=None):
    """Generate setup instructions based on findings"""
    print("\n" + "=" * 60)
    print("📝 NEXT STEPS")
    print("=" * 60)
    
    if cluster:
        # Get endpoint info
        nodes = cluster.get('CacheNodes', [])
        if nodes:
            endpoint = nodes[0].get('Endpoint', {})
            address = endpoint.get('Address', 'YOUR_ENDPOINT')
            port = endpoint.get('Port', 6379)
            
            transit_encryption = cluster.get('TransitEncryptionEnabled', False)
            auth_enabled = cluster.get('AuthTokenEnabled', False)
            
            protocol = 'rediss' if transit_encryption else 'redis'
            auth_part = ':YOUR_AUTH_TOKEN@' if auth_enabled else '://'
            
            redis_url = f"{protocol}{auth_part}{address}:{port}/0"
            
            print(f"\n✅ You have a Redis cluster! Here's how to connect:\n")
            print(f"1. Set environment variables:\n")
            print(f"   export REDIS_URL=\"{redis_url}\"")
            print(f"   export ENABLE_REDIS_CACHE=true")
            
            if auth_enabled:
                print(f"\n2. Get your auth token from AWS Console:")
                print(f"   AWS Console → ElastiCache → {cluster.get('CacheClusterId')} → Modify")
                print(f"   Replace YOUR_AUTH_TOKEN in the URL above")
            
            print(f"\n3. Test the connection:")
            print(f"   python3 test_redis_connection.py")
            
            print(f"\n4. Check security group rules:")
            print(f"   - Allow inbound Redis (port {port}) from your IP")
            print(f"   - Or configure VPC peering for Cloud Run")
            
    else:
        print("\n❌ No Redis cluster found. You have two options:\n")
        print("Option 1: Create AWS ElastiCache Redis")
        print("   • AWS Console → ElastiCache → Create")
        print("   • Choose Redis")
        print("   • Select instance type (cache.t3.micro for testing)")
        print("   • Configure VPC and security groups")
        print("   • Enable auth token if needed")
        
        print("\nOption 2: Use Local Redis (for development)")
        print("   • Install: brew install redis")
        print("   • Start: brew services start redis")
        print("   • Test: redis-cli ping")
        print("   • Set: export REDIS_URL=\"redis://localhost:6379/0\"")
        print("   • Enable: export ENABLE_REDIS_CACHE=true")

def main():
    """Main check function"""
    print("🔍 AWS REDIS RESOURCE CHECKER")
    print("=" * 60)
    print("This script will check if you have:")
    print("  • AWS credentials configured")
    print("  • ElastiCache Redis clusters")
    print("  • Connection information")
    print("=" * 60)
    print()
    
    # Step 1: Check credentials
    has_creds = check_env_credentials()
    
    if not has_creds:
        print("\n" + "=" * 60)
        print("⚠️  NO AWS CREDENTIALS FOUND")
        print("=" * 60)
        print("\nTo set AWS credentials:")
        print("  export AWS_ACCESS_KEY_ID='your_access_key'")
        print("  export AWS_SECRET_ACCESS_KEY='your_secret_key'")
        print("  export AWS_REGION='us-east-1'  # or your preferred region")
        print("\nOr configure AWS CLI:")
        print("  aws configure")
        print("=" * 60)
        return
    
    # Step 2: Check boto3
    if not check_boto3():
        print("\n💡 Install boto3 to check AWS resources:")
        print("   pip install boto3")
        return
    
    # Step 3: List ElastiCache clusters in current region
    cluster = list_elasticache_clusters()
    
    # Step 4: If nothing found, scan other regions
    if not cluster:
        print("\n💡 Scanning other common regions...")
        found_regions = check_other_regions()
        
        if found_regions:
            print("\n💡 Re-run this script with the correct region:")
            for region in found_regions.keys():
                print(f"   export AWS_REGION={region}")
                print(f"   python3 check_aws_resources.py")
                break
    
    # Step 5: Generate setup instructions
    generate_setup_instructions(cluster)
    
    print("\n" + "=" * 60)
    print("✅ Check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
