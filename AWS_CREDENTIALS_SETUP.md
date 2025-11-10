# 🔑 AWS Credentials Setup Guide

## Problem
You're seeing: `Unable to locate credentials. You can configure credentials by running "aws configure"`

This means you haven't set up valid AWS credentials yet.

---

## ✅ Solution: Get Real AWS Credentials

### **Step 1: Log into AWS Console**
1. Go to: https://console.aws.amazon.com/
2. Sign in with your AWS account

### **Step 2: Create Access Keys**

#### **Option A: IAM User (Recommended for Personal Use)**
1. Go to **IAM** → **Users** → Click your username
2. Click **"Security credentials"** tab
3. Scroll to **"Access keys"** section
4. Click **"Create access key"**
5. Select **"Command Line Interface (CLI)"**
6. Check the confirmation box
7. Click **"Next"** → **"Create access key"**
8. ⚠️ **IMPORTANT**: Save both:
   - **Access Key ID** (e.g., `AKIAIOSFODNN7EXAMPLE`)
   - **Secret Access Key** (e.g., `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)

#### **Option B: Root User (Not Recommended)**
1. Click your account name (top right) → **Security credentials**
2. Expand **"Access keys"**
3. Click **"Create access key"**
4. ⚠️ Save the Access Key ID and Secret Access Key

⚠️ **WARNING**: The Secret Access Key is shown ONLY ONCE. Save it immediately!

---

### **Step 3: Configure AWS CLI**

Run this command in your terminal:

```bash
aws configure
```

It will prompt you for:

```
AWS Access Key ID [None]: PASTE_YOUR_ACCESS_KEY_HERE
AWS Secret Access Key [None]: PASTE_YOUR_SECRET_KEY_HERE
Default region name [None]: eu-central-1
Default output format [None]: json
```

#### **Example (with fake credentials):**
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: eu-central-1
Default output format [None]: json
```

---

### **Step 4: Verify Configuration**

After configuring, test it:

```bash
# Test 1: Check your AWS account ID
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "AIDACKCEVSQ6C2EXAMPLE",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/YourUsername"
# }

# Test 2: List S3 buckets (as a basic permission test)
aws s3 ls

# Test 3: Check ECR repositories
aws ecr describe-repositories --region eu-central-1
```

---

## 🎯 **Your Deployment Region**

Based on your command, you're deploying to: **`eu-central-1`** (Frankfurt, Germany)

Make sure to use this region consistently in all commands.

---

## 📝 **What to Do After Getting Credentials**

### **1. Configure AWS CLI**
```bash
aws configure
# Enter your real Access Key ID
# Enter your real Secret Access Key
# Region: eu-central-1
# Format: json
```

### **2. Verify Configuration**
```bash
aws sts get-caller-identity
```

### **3. Run Deployment Script**
```bash
cd /Users/omer/Desktop/ai-stanbul
chmod +x deploy_to_ecs.sh
./deploy_to_ecs.sh
```

---

## 🚨 **Security Best Practices**

### **DO:**
✅ Create an IAM user with only necessary permissions
✅ Enable MFA (Multi-Factor Authentication) on your AWS account
✅ Rotate access keys regularly
✅ Never share your Secret Access Key
✅ Use AWS Secrets Manager for application secrets

### **DON'T:**
❌ Use root user access keys (create IAM user instead)
❌ Commit credentials to Git
❌ Share credentials via email/Slack
❌ Use overly permissive policies (e.g., `AdministratorAccess` unless needed)

---

## 🔐 **Minimal IAM Permissions Required**

If you want to create a dedicated IAM user for this deployment, attach these policies:

### **Custom Policy (Copy this to IAM → Policies → Create policy → JSON):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:*",
        "ecs:*",
        "batch:*",
        "iam:GetRole",
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "iam:PassRole",
        "secretsmanager:*",
        "logs:*",
        "ec2:DescribeInstances",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeSubnets",
        "ec2:DescribeVpcs"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## 🆘 **Still Having Issues?**

### **Error: "Unable to locate credentials"**
- You haven't run `aws configure` yet, or the credentials file is empty
- Run: `cat ~/.aws/credentials` to check if credentials exist

### **Error: "The security token included in the request is invalid"**
- You entered incorrect Access Key or Secret Key
- Re-run `aws configure` with correct values

### **Error: "Access Denied"**
- Your IAM user doesn't have required permissions
- Attach `AdministratorAccess` policy (or custom policy above) to your user

---

## 📞 **Next Steps After Fixing Credentials**

1. ✅ Get AWS Access Keys from IAM Console
2. ✅ Run `aws configure` with real credentials
3. ✅ Test: `aws sts get-caller-identity`
4. ✅ Run: `./deploy_to_ecs.sh`
5. ✅ Follow ECS_FORM_FILLING_GUIDE.md

---

**Status:** 🔴 Credentials Not Configured  
**Action Required:** Get real AWS credentials and run `aws configure`  
**Last Updated:** November 10, 2025
