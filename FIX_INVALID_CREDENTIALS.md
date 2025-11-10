# 🚨 FIX: Invalid AWS Credentials

## ❌ Error You're Seeing
```
An error occurred (InvalidClientTokenId) when calling the GetCallerIdentity operation: 
The security token included in the request is invalid.
```

## 🔍 What This Means
Your AWS CLI **has** credentials configured, but they are:
- ❌ Incorrect (typo when pasting)
- ❌ Deleted (removed from AWS IAM)
- ❌ From a different account
- ❌ Expired or deactivated

---

## ✅ QUICK FIX - Step by Step

### **Step 1: Remove Old/Invalid Credentials**

```bash
# Check what's currently configured
cat ~/.aws/credentials

# Remove the invalid credentials
rm ~/.aws/credentials
rm ~/.aws/config

# Verify they're gone
ls -la ~/.aws
```

---

### **Step 2: Get NEW Valid Access Keys**

#### **Option A: Create New Access Keys in AWS Console** 🌟

1. **Log into AWS Console:**
   - Go to: https://console.aws.amazon.com/
   - Sign in with your AWS account

2. **Navigate to IAM:**
   - Search for "IAM" in the top search bar
   - Click on **IAM** service

3. **Go to Your User:**
   - Left sidebar → Click **Users**
   - Click on **your username**

4. **Check Existing Keys:**
   - Click **Security credentials** tab
   - Scroll to **Access keys** section
   - You should see your old keys listed

5. **DELETE Old Keys** (Important!):
   - Find the old/invalid key
   - Click **Actions** → **Deactivate** or **Delete**
   - Confirm deletion

6. **Create NEW Access Key:**
   - Click **Create access key**
   - Select use case: **Command Line Interface (CLI)**
   - Check: "I understand the above recommendation..."
   - Click **Next**
   - (Optional) Description: "MacBook - November 2025"
   - Click **Create access key**

7. **SAVE YOUR NEW KEYS** ⚠️
   - **Access key ID**: `AKIA...` (20 characters)
   - **Secret access key**: `wJalr...` (40 characters)
   - Click **Download .csv file** (IMPORTANT!)
   - Or copy both values to a secure note

---

### **Step 3: Configure AWS CLI with NEW Keys**

```bash
# Run configuration
aws configure

# You'll be prompted - paste your NEW keys:
```

**Prompts and Answers:**
```
AWS Access Key ID [None]: PASTE_YOUR_NEW_ACCESS_KEY_ID_HERE
AWS Secret Access Key [None]: PASTE_YOUR_NEW_SECRET_ACCESS_KEY_HERE
Default region name [None]: eu-central-1
Default output format [None]: json
```

**⚠️ IMPORTANT:**
- **Copy-paste carefully** - no extra spaces or line breaks
- The **Access Key ID** starts with `AKIA`
- The **Secret Access Key** is about 40 characters long
- Use **eu-central-1** for region (or your preferred region)

---

### **Step 4: Verify New Credentials Work**

```bash
# Test 1: Get your account info
aws sts get-caller-identity

# Expected output (SUCCESS):
{
    "UserId": "AIDAJ...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

```bash
# Test 2: List your ECR repositories
aws ecr describe-repositories --region eu-central-1

# Expected: List of repositories (or empty list if none exist)
```

**If both tests pass:** ✅ You're good to go!

---

## 🔄 ALTERNATIVE: Use AWS CloudShell Instead

If you keep having credential issues, **skip credentials entirely** and use CloudShell:

### **Why CloudShell?**
- ✅ No credentials needed
- ✅ Always authenticated
- ✅ Works in your browser
- ✅ Free to use

### **How to Use CloudShell:**

1. **Open CloudShell:**
   - Go to: https://console.aws.amazon.com/
   - Click the **CloudShell icon** (>_) in top navigation
   - Wait 30 seconds for it to start

2. **Upload Your Files:**
   - Click **Actions** → **Upload file**
   - Upload these files:
     - `Dockerfile.4bit`
     - `llm_api_server_4bit.py`
     - `deploy_to_ecs.sh`
     - `requirements.txt` (if you have one)

3. **Run Deployment:**
   ```bash
   # Make script executable
   chmod +x deploy_to_ecs.sh
   
   # Run it
   ./deploy_to_ecs.sh
   ```

4. **Get Your Config:**
   ```bash
   # View generated config
   cat ECS_DEPLOYMENT_CONFIG.txt
   ```

**Everything works automatically in CloudShell!** 🎉

---

## 🛠️ TROUBLESHOOTING

### **Problem: "Access Denied" Error**

**Cause:** Your IAM user doesn't have required permissions

**Fix:**
1. Log into AWS Console
2. Go to **IAM** → **Users** → Your user
3. Click **Permissions** tab
4. Click **Add permissions** → **Attach policies directly**
5. Attach these policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `IAMFullAccess`
   - `SecretsManagerReadWrite`
   - `AmazonECS_FullAccess`
   - `AWSBatchFullAccess`
6. Click **Add permissions**

---

### **Problem: "Credentials not found" After Running aws configure**

**Cause:** Configuration didn't save properly

**Fix:**
```bash
# Check if credentials file exists
cat ~/.aws/credentials

# If empty or not found, manually create it
mkdir -p ~/.aws

# Edit credentials file
nano ~/.aws/credentials
```

**Add this content:**
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

**Edit config file:**
```bash
nano ~/.aws/config
```

**Add this content:**
```ini
[default]
region = eu-central-1
output = json
```

**Save and test:**
```bash
# Save: Ctrl+O, Enter, Ctrl+X
# Test:
aws sts get-caller-identity
```

---

### **Problem: Still Getting "InvalidClientTokenId"**

**Possible causes:**
1. ❌ Typo when pasting keys
2. ❌ Extra spaces or line breaks in keys
3. ❌ Keys belong to different AWS account
4. ❌ Keys were deleted/deactivated

**Fix:**
```bash
# Show exactly what's configured (hide secret)
aws configure list

# Output should show:
      Name                    Value             Type    Location
      ----                    -----             ----    --------
   profile                <not set>             None    None
access_key     ****************ABCD shared-credentials-file    
secret_key     ****************WXYZ shared-credentials-file    
    region           eu-central-1      config-file    ~/.aws/config
```

**If access_key shows asterisks:** Config exists, but keys are wrong
**If access_key shows <not set>:** Config is empty

**Solution:**
1. Delete and recreate keys in AWS Console
2. Run `aws configure` again with NEW keys
3. **Triple-check** you copy-pasted correctly

---

## 📋 DIAGNOSTIC SCRIPT

Run this to check your AWS CLI setup:

```bash
echo "=== AWS CLI Diagnostics ==="
echo ""
echo "1. AWS CLI Version:"
aws --version
echo ""
echo "2. AWS Config Location:"
ls -la ~/.aws/
echo ""
echo "3. Credentials File (redacted):"
if [ -f ~/.aws/credentials ]; then
    echo "✅ Credentials file exists"
    grep -E "^\[|^aws_access_key_id" ~/.aws/credentials | head -5
else
    echo "❌ No credentials file found"
fi
echo ""
echo "4. Config File:"
if [ -f ~/.aws/config ]; then
    echo "✅ Config file exists"
    cat ~/.aws/config
else
    echo "❌ No config file found"
fi
echo ""
echo "5. AWS Configuration:"
aws configure list
echo ""
echo "6. Test Connection:"
aws sts get-caller-identity 2>&1
```

**Save this as `check_aws.sh` and run:**
```bash
chmod +x check_aws.sh
./check_aws.sh
```

Share the output if you need more help!

---

## ✅ QUICK CHECKLIST

- [ ] Delete old/invalid credentials: `rm ~/.aws/credentials ~/.aws/config`
- [ ] Log into AWS Console
- [ ] Go to IAM → Users → Your user → Security credentials
- [ ] Delete old access keys
- [ ] Create NEW access key
- [ ] Download CSV file with keys
- [ ] Run `aws configure` with NEW keys
- [ ] Test: `aws sts get-caller-identity`
- [ ] ✅ Success! Continue with deployment

---

## 🎯 RECOMMENDED APPROACH

### **For Fastest Success:**

**Option 1: Use CloudShell** (Recommended) ☁️
- No credentials needed
- Works immediately
- Go to: https://console.aws.amazon.com/cloudshell

**Option 2: Fix Local Credentials** ⚙️
- Create NEW keys in IAM
- Run `aws configure` with correct keys
- Verify with `aws sts get-caller-identity`

---

## 💡 NEXT STEPS

**After credentials are working:**

```bash
# 1. Test AWS access
aws sts get-caller-identity

# 2. Run deployment script
chmod +x deploy_to_ecs.sh
./deploy_to_ecs.sh

# 3. Check generated config
cat ECS_DEPLOYMENT_CONFIG.txt
```

---

**Need immediate help?**
- Show me the output of: `aws configure list`
- Show me: `ls -la ~/.aws/`
- Tell me if you want to use CloudShell instead

---

**Status:** 🔧 Credential Troubleshooting Guide  
**Last Updated:** November 10, 2025  
**Issue:** InvalidClientTokenId Error
