#!/bin/bash
# Push Docker image to AWS ECR

set -e

echo "🚀 Pushing Docker image to AWS ECR"
echo "===================================="
echo ""

# Get AWS account info
echo "📋 Getting AWS account information..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="eu-central-1"
ECR_REPO="ai-istanbul-llm-4bit"

echo "✅ Account ID: $AWS_ACCOUNT_ID"
echo "✅ Region: $AWS_REGION"
echo ""

# ECR URI
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
echo "📦 ECR URI: $ECR_URI"
echo ""

# Login to ECR
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "✅ Logged in to ECR"
echo ""

# Tag image
echo "🏷️  Tagging image..."
docker tag ai-istanbul-llm-4bit:latest ${ECR_URI}:latest
docker tag ai-istanbul-llm-4bit:latest ${ECR_URI}:v1.0

echo "✅ Image tagged"
echo ""

# Push to ECR
echo "⬆️  Pushing to ECR (this will take 5-10 minutes)..."
echo "Image size: ~5-6 GB"
echo ""

docker push ${ECR_URI}:latest
docker push ${ECR_URI}:v1.0

echo ""
echo "✅ Image pushed successfully!"
echo ""
echo "=========================================="
echo "IMAGE URI (copy this for AWS Console):"
echo "${ECR_URI}:latest"
echo "=========================================="
echo ""

# Generate full configuration
echo "📝 Generating full configuration..."
echo ""

SECRET_ARN=$(aws secretsmanager describe-secret --secret-id ai-istanbul/hf-token --region $AWS_REGION --query ARN --output text)
EXECUTION_ROLE_ARN=$(aws iam get-role --role-name aiIstanbulECSTaskExecutionRole --query Role.Arn --output text)
TASK_ROLE_ARN=$(aws iam get-role --role-name aiIstanbulECSTaskRole --query Role.Arn --output text)

cat > ECS_DEPLOYMENT_CONFIG.txt << EOF
========================================
ECS/BATCH DEPLOYMENT CONFIGURATION
========================================

✅ ALL RESOURCES CREATED SUCCESSFULLY!

1️⃣ CONTAINER IMAGE
-------------------
${ECR_URI}:latest

2️⃣ EXECUTION ROLE ARN
----------------------
${EXECUTION_ROLE_ARN}

3️⃣ TASK ROLE ARN
-----------------
${TASK_ROLE_ARN}

4️⃣ SECRET ARN (HF_TOKEN)
-------------------------
${SECRET_ARN}

5️⃣ RESOURCE REQUIREMENTS
-------------------------
vCPUs: 8
Memory (MiB): 32768
GPU: 1

6️⃣ ENVIRONMENT VARIABLES
-------------------------
PORT=8000
MODEL_NAME=meta-llama/Llama-3.1-8B
QUANTIZATION_BITS=4
DEVICE=cuda
MAX_TOKENS=250
BATCH_SIZE=1
TORCH_DTYPE=float16
LOW_CPU_MEM_USAGE=true
USE_CACHE=true

========================================
NEXT STEPS
========================================

1. Go to AWS Batch Console:
   https://console.aws.amazon.com/batch

2. Create Job Definition:
   - Copy values from above
   - Follow: ECS_FORM_FILLING_GUIDE.md

3. Create Compute Environment:
   - Instance type: g5.2xlarge (24GB VRAM A10G GPU)
   - Use SPOT for 70% savings

4. Create Job Queue

5. Submit Job & Test!

========================================
EOF

cat ECS_DEPLOYMENT_CONFIG.txt

echo ""
echo "✅ Configuration saved to: ECS_DEPLOYMENT_CONFIG.txt"
echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo ""
echo "📋 Next: Open AWS Batch Console and create your job definition"
echo "    Use values from ECS_DEPLOYMENT_CONFIG.txt"
echo ""
