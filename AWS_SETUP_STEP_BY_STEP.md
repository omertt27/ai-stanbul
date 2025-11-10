# 🔑 AWS Credentials Setup - Step-by-Step Guide

## ❌ Current Problem
```
Unable to locate credentials. You can configure credentials by running "aws configure".
```

**This means:** Your AWS CLI doesn't have access keys configured yet.

---

## ✅ SOLUTION: Get Your AWS Access Keys

### **Option 1: AWS Console (Recommended for Beginners)** 🌟

#### **Step 1: Log into AWS Console**
1. Go to: https://console.aws.amazon.com/
2. Sign in with your AWS account credentials

#### **Step 2: Navigate to IAM**
1. In the search bar at the top, type: `IAM`
2. Click on **IAM** (Identity and Access Management)

#### **Step 3: Create Access Keys**
1. In the left sidebar, click **Users**
2. Click on your username (or create a new user if needed)
3. Click the **Security credentials** tab
4. Scroll down to **Access keys** section
5. Click **Create access key**
6. Select use case: **Command Line Interface (CLI)**
7. Check the box: "I understand the above recommendation..."
8. Click **Next**
9. (Optional) Add description: "Local development on MacBook"
10. Click **Create access key**

#### **Step 4: Save Your Keys** ⚠️ IMPORTANT
You'll see two values:
- **Access key ID**: `AKIAIOSFODNN7EXAMPLE`
- **Secret access key**: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

**⚠️ SAVE THESE IMMEDIATELY!**
- The secret key is only shown ONCE
- Download the CSV file or copy both values
- Store them securely (password manager recommended)

---

### **Option 2: Use AWS CloudShell (No Credentials Needed!)** ☁️

If you're having trouble with credentials, use AWS CloudShell instead:

1. Log into AWS Console: https://console.aws.amazon.com/
2. Click the **CloudShell** icon (>_) in the top navigation bar
3. Wait for CloudShell to initialize (~30 seconds)
4. You now have a pre-authenticated terminal in your browser!
5. Run all your AWS commands directly in CloudShell

**Advantages:**
- ✅ No credentials needed
- ✅ Always authenticated
- ✅ Works from any browser
- ✅ Free to use

---

## 🛠️ CONFIGURE AWS CLI (After Getting Keys)

### **On Your MacBook Terminal:**

```bash
# Run this command
aws configure
```

**You'll be prompted for 4 values:**

```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: eu-central-1
Default output format [None]: json
```

**Enter:**
1. **Access Key ID**: Paste your access key ID
2. **Secret Access Key**: Paste your secret access key
3. **Default region**: `eu-central-1` (or your preferred region)
4. **Output format**: `json` (or press Enter for default)

---

## ✅ VERIFY SETUP

### **Test 1: Check Identity**
```bash
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDAJ...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### **Test 2: List Regions**
```bash
aws ec2 describe-regions --output table
```

**If you see a table of regions:** ✅ Success!

---

## 🚀 AFTER CREDENTIALS ARE CONFIGURED

### **Continue with Deployment:**

```bash
# 1. Make deployment script executable
chmod +x deploy_to_ecs.sh

# 2. Run deployment script
./deploy_to_ecs.sh

# 3. Check generated config
cat ECS_DEPLOYMENT_CONFIG.txt
```

---

## 🔄 ALTERNATIVE: Use CloudShell for Everything

If you want to skip credential setup entirely:

### **Steps:**
1. Log into AWS Console
2. Open CloudShell (click >_ icon)
3. Upload your project files:
   ```bash
   # In CloudShell, run:
   git clone https://github.com/your-repo/ai-stanbul.git
   cd ai-stanbul
   ```
4. Run deployment:
   ```bash
   chmod +x deploy_to_ecs.sh
   ./deploy_to_ecs.sh
   ```

**All AWS commands will work automatically in CloudShell!**

---

## 📊 COMPARISON: Local vs CloudShell

| Feature | Local MacBook | AWS CloudShell |
|---------|---------------|----------------|
| Credentials needed | ✅ Yes | ❌ No |
| Setup time | 5-10 minutes | 30 seconds |
| Access from anywhere | ❌ Only your Mac | ✅ Any browser |
| Cost | Free | Free |
| File storage | Unlimited | 1 GB |
| Session timeout | Never | 20 min idle |

---

## 🚨 COMMON ISSUES

### **Issue 1: "InvalidClientTokenId"**
**Cause:** Wrong access key or secret key

**Fix:**
```bash
# Remove old config
rm ~/.aws/credentials
rm ~/.aws/config

# Reconfigure with correct keys
aws configure
```

### **Issue 2: "Access Denied"**
**Cause:** User doesn't have required permissions

**Fix:**
1. Log into AWS Console
2. Go to IAM → Users → Your user
3. Click **Permissions** tab
4. Click **Add permissions**
5. Attach these policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `IAMFullAccess`
   - `SecretsManagerReadWrite`
   - `AmazonECS_FullAccess`

### **Issue 3: "Credentials file not found"**
**Cause:** `~/.aws` directory doesn't exist

**Fix:**
```bash
# Create directory
mkdir -p ~/.aws

# Run configure
aws configure
```

---

## 📝 QUICK START CHECKLIST

- [ ] Log into AWS Console
- [ ] Go to IAM → Users → Security credentials
- [ ] Create access key
- [ ] Save both keys (Access Key ID + Secret Access Key)
- [ ] Run `aws configure` in terminal
- [ ] Paste both keys when prompted
- [ ] Set region: `eu-central-1`
- [ ] Test: `aws sts get-caller-identity`
- [ ] ✅ Success! Continue with deployment

---

## 🎯 RECOMMENDED WORKFLOW

### **For First-Time Setup:**
Use **AWS CloudShell** to avoid credential issues:
1. Open CloudShell in AWS Console
2. Upload/clone your project
3. Run deployment script
4. Everything works automatically

### **For Long-Term Development:**
Configure credentials locally:
1. Create access keys in IAM
2. Run `aws configure` on your Mac
3. Verify with `aws sts get-caller-identity`
4. Run deployment scripts locally

---

## 💡 NEED MORE HELP?

### **Option A: Use CloudShell (Fastest)**
1. Go to: https://console.aws.amazon.com/cloudshell
2. Click the CloudShell icon (>_)
3. Start running commands immediately

### **Option B: Get 1-on-1 Help**
If you're stuck:
1. Share the exact error message
2. Run: `ls -la ~/.aws` and share output
3. Run: `cat ~/.aws/config` (if file exists)
4. I'll provide specific troubleshooting steps

---

**Status:** ✅ Complete AWS Setup Guide  
**Last Updated:** November 10, 2025  
**Region Used:** eu-central-1
