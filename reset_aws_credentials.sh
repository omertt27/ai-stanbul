#!/bin/bash

echo "🔧 AWS Credentials Reset Script"
echo "================================"
echo ""

echo "Step 1: Backing up old credentials..."
if [ -d ~/.aws ]; then
    cp -r ~/.aws ~/.aws.backup.$(date +%Y%m%d_%H%M%S)
    echo "✅ Backup created at ~/.aws.backup.$(date +%Y%m%d_%H%M%S)"
else
    echo "⚠️  No existing AWS config found"
fi
echo ""

echo "Step 2: Removing invalid credentials..."
rm -f ~/.aws/credentials
rm -f ~/.aws/config
echo "✅ Old credentials removed"
echo ""

echo "Step 3: Creating fresh AWS config directory..."
mkdir -p ~/.aws
chmod 700 ~/.aws
echo "✅ Directory created"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Get NEW AWS Access Keys:"
echo "   → Go to: https://console.aws.amazon.com/iam"
echo "   → Click: Users → Your Username → Security credentials"
echo "   → Click: Create access key"
echo "   → Download the CSV file with your keys"
echo ""
echo "2. Run AWS Configure:"
echo "   → Type: aws configure"
echo "   → Paste your NEW Access Key ID"
echo "   → Paste your NEW Secret Access Key"
echo "   → Region: eu-central-1"
echo "   → Output: json"
echo ""
echo "3. Verify It Works:"
echo "   → Type: aws sts get-caller-identity"
echo "   → You should see your account info"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: You MUST create NEW access keys in AWS Console!"
echo "   Old keys ending in '...0767' are invalid."
echo ""
echo "Ready to configure? Run: aws configure"
