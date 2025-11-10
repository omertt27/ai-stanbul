# Fix ECR Permissions for render-mvp-user

## ✅ Good News: Your AWS credentials work!

The error shows:
```
User: arn:aws:iam::701893740767:user/render-mvp-user is not authorized to perform: ecr:GetAuthorizationToken
```

This means your IAM user needs ECR permissions.

## Quick Fix: Add ECR Permissions to Your IAM User

### Option 1: AWS Console (Easiest - 2 minutes)

1. **Go to IAM Console:**
   ```
   https://console.aws.amazon.com/iam/home#/users/render-mvp-user
   ```

2. **Click "Add permissions" → "Attach policies directly"**

3. **Search and select these policies:**
   - ✅ `AmazonEC2ContainerRegistryFullAccess` (for ECR push/pull)
   - ✅ `IAMReadOnlyAccess` (to read role ARNs)
   - ✅ `SecretsManagerReadWrite` (to read HF token)
   - ✅ `AWSBatchFullAccess` (to create batch jobs)

4. **Click "Next" → "Add permissions"**

5. **Test immediately:**
   ```bash
   cd ~/Desktop/ai-stanbul
   ./push_to_ecr.sh
   ```

---

### Option 2: AWS CLI (Advanced)

Run these commands to attach the policies:

```bash
# Attach ECR permissions
aws iam attach-user-policy \
  --user-name render-mvp-user \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

# Attach IAM read permissions
aws iam attach-user-policy \
  --user-name render-mvp-user \
  --policy-arn arn:aws:iam::aws:policy/IAMReadOnlyAccess

# Attach Secrets Manager permissions
aws iam attach-user-policy \
  --user-name render-mvp-user \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite

# Attach Batch permissions
aws iam attach-user-policy \
  --user-name render-mvp-user \
  --policy-arn arn:aws:iam::aws:policy/AWSBatchFullAccess
```

Then test:
```bash
./push_to_ecr.sh
```

---

### Option 3: Create Custom Policy (Most Secure)

If you want minimal permissions, create a custom policy:

1. **Go to IAM Console → Policies → Create Policy**

2. **Use JSON tab and paste:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:GetRepositoryPolicy",
        "ecr:DescribeRepositories",
        "ecr:ListImages",
        "ecr:DescribeImages",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:GetRole",
        "iam:ListRoles"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:701893740767:secret:ai-istanbul/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "batch:*"
      ],
      "Resource": "*"
    }
  ]
}
```

3. **Name it:** `AIIstanbulDeploymentPolicy`

4. **Attach to user:** IAM → Users → render-mvp-user → Add permissions → Attach policy → AIIstanbulDeploymentPolicy

---

## What Each Policy Does

| Policy | Why Needed |
|--------|-----------|
| `AmazonEC2ContainerRegistryFullAccess` | Push Docker image to ECR |
| `IAMReadOnlyAccess` | Read IAM role ARNs for config |
| `SecretsManagerReadWrite` | Read HuggingFace token |
| `AWSBatchFullAccess` | Create and manage Batch jobs |

---

## After Adding Permissions

Run the push script again:

```bash
cd ~/Desktop/ai-stanbul
./push_to_ecr.sh
```

Expected output:
```
🚀 Pushing Docker image to AWS ECR
====================================

📋 Getting AWS account information...
✅ Account ID: 701893740767
✅ Region: eu-central-1

🔐 Logging into ECR...
✅ Logged in to ECR

🏷️  Tagging image...
✅ Image tagged

⬆️  Pushing to ECR...
```

---

## Troubleshooting

### If you still get permission errors:

1. **Check policies are attached:**
   ```bash
   aws iam list-attached-user-policies --user-name render-mvp-user
   ```

2. **Wait 30 seconds** for permissions to propagate

3. **Try again:**
   ```bash
   ./push_to_ecr.sh
   ```

### If ECR repository doesn't exist:

Create it first:
```bash
aws ecr create-repository \
  --repository-name ai-istanbul-llm-4bit \
  --region eu-central-1
```

---

## Next Steps After Permissions Fixed

1. ✅ Add IAM permissions (use Option 1 - AWS Console)
2. ✅ Run `./push_to_ecr.sh`
3. ✅ Wait 5-10 minutes for Docker image to upload
4. ✅ Use generated `ECS_DEPLOYMENT_CONFIG.txt` for AWS Batch setup

---

**Quick Link:** Go add permissions now:
https://console.aws.amazon.com/iam/home#/users/render-mvp-user?section=permissions

Let me know once you've added the permissions and I'll help verify the push works! 🚀
