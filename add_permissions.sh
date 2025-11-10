#!/bin/bash
# Add ECR and related permissions to render-mvp-user

set -e

echo "🔐 Adding permissions to render-mvp-user"
echo "=========================================="
echo ""

USER_NAME="render-mvp-user"

echo "📋 Attaching IAM policies..."
echo ""

# ECR Full Access
echo "1/4 Adding ECR permissions..."
aws iam attach-user-policy \
  --user-name $USER_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
echo "✅ ECR permissions added"

# IAM Read Only
echo "2/4 Adding IAM read permissions..."
aws iam attach-user-policy \
  --user-name $USER_NAME \
  --policy-arn arn:aws:iam::aws:policy/IAMReadOnlyAccess
echo "✅ IAM read permissions added"

# Secrets Manager
echo "3/4 Adding Secrets Manager permissions..."
aws iam attach-user-policy \
  --user-name $USER_NAME \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
echo "✅ Secrets Manager permissions added"

# AWS Batch
echo "4/4 Adding Batch permissions..."
aws iam attach-user-policy \
  --user-name $USER_NAME \
  --policy-arn arn:aws:iam::aws:policy/AWSBatchFullAccess
echo "✅ Batch permissions added"

echo ""
echo "=========================================="
echo "✅ ALL PERMISSIONS ADDED SUCCESSFULLY!"
echo "=========================================="
echo ""

# List attached policies to verify
echo "📋 Verifying attached policies..."
echo ""
aws iam list-attached-user-policies --user-name $USER_NAME

echo ""
echo "🎉 Done! You can now run:"
echo "   ./push_to_ecr.sh"
echo ""
