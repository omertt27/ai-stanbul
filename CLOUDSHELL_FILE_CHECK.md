# 🔍 CloudShell File Verification Guide

## 📋 3 Required Files for CloudShell

You need to upload these 3 files to AWS CloudShell:

### **1️⃣ Dockerfile.4bit**
- **Purpose:** Container definition for 4-bit quantized Llama model
- **Size:** ~1-2 KB
- **Location:** `/Users/omer/Desktop/ai-stanbul/Dockerfile.4bit`

### **2️⃣ llm_api_server_4bit.py**
- **Purpose:** Python API server code
- **Size:** ~10-15 KB
- **Location:** `/Users/omer/Desktop/ai-stanbul/llm_api_server_4bit.py`

### **3️⃣ deploy_to_ecs.sh**
- **Purpose:** Deployment automation script
- **Size:** ~7-8 KB
- **Location:** `/Users/omer/Desktop/ai-stanbul/deploy_to_ecs.sh`

---

## 🌐 How to Upload to CloudShell

### **Step 1: Open AWS CloudShell**
1. Go to: https://console.aws.amazon.com/
2. Log in to your AWS account
3. Click the **CloudShell icon** (>_) in the top-right corner
4. Wait ~30 seconds for CloudShell to initialize

### **Step 2: Upload Files**
1. In CloudShell, click **Actions** → **Upload file**
2. Select all 3 files:
   - `Dockerfile.4bit`
   - `llm_api_server_4bit.py`
   - `deploy_to_ecs.sh`
3. Click **Upload**
4. Wait for upload to complete (~5-10 seconds)

### **Step 3: Verify Files in CloudShell**
```bash
# Check files exist
ls -lh

# Should see:
# Dockerfile.4bit
# llm_api_server_4bit.py
# deploy_to_ecs.sh
```

---

## ✅ Verification Commands (Run in CloudShell)

### **Quick Check:**
```bash
# List all files
ls -lh

# Count files
ls | wc -l
# Should show: 3
```

### **Detailed Check:**
```bash
# Check if all files exist
echo "Checking Dockerfile.4bit..."
[ -f "Dockerfile.4bit" ] && echo "✅ Found" || echo "❌ Missing"

echo "Checking llm_api_server_4bit.py..."
[ -f "llm_api_server_4bit.py" ] && echo "✅ Found" || echo "❌ Missing"

echo "Checking deploy_to_ecs.sh..."
[ -f "deploy_to_ecs.sh" ] && echo "✅ Found" || echo "❌ Missing"
```

### **Check File Contents:**
```bash
# Preview Dockerfile
head -20 Dockerfile.4bit

# Preview Python server
head -30 llm_api_server_4bit.py

# Preview deployment script
head -30 deploy_to_ecs.sh
```

### **Check File Sizes:**
```bash
# Should show all 3 files with their sizes
ls -lh Dockerfile.4bit llm_api_server_4bit.py deploy_to_ecs.sh
```

---

## 🛠️ Run Verification Script

I've created a script to automatically check everything:

### **In CloudShell, run:**
```bash
# First, upload check_cloudshell_files.sh to CloudShell
# Then make it executable:
chmod +x check_cloudshell_files.sh

# Run verification:
./check_cloudshell_files.sh
```

**This will:**
- ✅ Check if all 3 files exist
- ✅ Show file sizes
- ✅ Preview first lines of each file
- ✅ Verify deployment script is executable
- ✅ Provide next steps

---

## 📊 Expected Output

After uploading, you should see:

```bash
$ ls -lh
total 28K
-rw-r--r-- 1 cloudshell-user cloudshell-user 1.2K Nov 10 14:30 Dockerfile.4bit
-rw-r--r-- 1 cloudshell-user cloudshell-user  13K Nov 10 14:30 llm_api_server_4bit.py
-rw-r--r-- 1 cloudshell-user cloudshell-user 7.8K Nov 10 14:30 deploy_to_ecs.sh
```

---

## 🚨 Troubleshooting

### **Problem: File Not Found After Upload**

**Check CloudShell home directory:**
```bash
cd ~
ls -la
```

**If files are in a different location:**
```bash
# Find the files
find ~ -name "*.4bit" -o -name "*.py" -o -name "*.sh"
```

### **Problem: File is Corrupted or Wrong Size**

**Re-upload the file:**
1. Delete the corrupted file: `rm filename`
2. Upload again from AWS CloudShell Actions menu

### **Problem: deploy_to_ecs.sh Won't Run**

**Make it executable:**
```bash
chmod +x deploy_to_ecs.sh

# Verify:
ls -l deploy_to_ecs.sh
# Should show: -rwxr-xr-x (x = executable)
```

---

## 🎯 After Verification

### **If All Files Are Present:**
```bash
# Make script executable
chmod +x deploy_to_ecs.sh

# Run deployment
./deploy_to_ecs.sh
```

### **If Files Are Missing:**
1. Go back to AWS Console
2. CloudShell → Actions → Upload file
3. Select missing files
4. Upload again

---

## 📝 File Content Checklist

### **Dockerfile.4bit should contain:**
- `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- `RUN pip3 install transformers bitsandbytes`
- `CMD ["python3", "llm_api_server_4bit.py"]`

### **llm_api_server_4bit.py should contain:**
- `from transformers import AutoTokenizer, AutoModelForCausalLM`
- `from flask import Flask, request, jsonify`
- `load_in_4bit=True`
- `@app.route('/health')`

### **deploy_to_ecs.sh should contain:**
- `#!/bin/bash`
- `ECR_REPO_NAME="ai-istanbul-llm-4bit"`
- `aws ecr create-repository`
- `docker build -f Dockerfile.4bit`

---

## 🔍 Manual Verification Commands

### **In CloudShell:**
```bash
# 1. Check you're in the right directory
pwd
# Should show: /home/cloudshell-user

# 2. List all files
ls -lh

# 3. Check file types
file Dockerfile.4bit
file llm_api_server_4bit.py
file deploy_to_ecs.sh

# 4. Count lines in each file
wc -l Dockerfile.4bit llm_api_server_4bit.py deploy_to_ecs.sh

# 5. Check for syntax errors in Python
python3 -m py_compile llm_api_server_4bit.py
echo $?  # Should show: 0 (success)

# 6. Check for syntax errors in Bash
bash -n deploy_to_ecs.sh
echo $?  # Should show: 0 (success)
```

---

## 💡 Quick Reference

| File | Purpose | Check Command |
|------|---------|---------------|
| `Dockerfile.4bit` | Container definition | `head -10 Dockerfile.4bit` |
| `llm_api_server_4bit.py` | API server | `python3 -m py_compile llm_api_server_4bit.py` |
| `deploy_to_ecs.sh` | Deployment script | `bash -n deploy_to_ecs.sh` |

---

## 🚀 Next Steps After Verification

Once all 3 files are verified in CloudShell:

```bash
# 1. Make deployment script executable
chmod +x deploy_to_ecs.sh

# 2. Verify AWS credentials
aws sts get-caller-identity

# 3. Run deployment
./deploy_to_ecs.sh

# 4. Follow prompts:
#    - Enter region: eu-central-1
#    - Enter HuggingFace token: hf_xxxxxxxxxxxxx
#    - Confirm: y
```

---

## 📞 Need Help?

If files are not showing up in CloudShell:

1. **Check upload location:**
   ```bash
   cd ~
   ls -la
   ```

2. **Check upload history:**
   ```bash
   history | grep -i upload
   ```

3. **Re-upload from local machine:**
   - The files are at: `/Users/omer/Desktop/ai-stanbul/`
   - Upload via CloudShell Actions menu

---

**Status:** ✅ CloudShell File Verification Guide  
**Last Updated:** November 10, 2025  
**Required Files:** 3 (Dockerfile.4bit, llm_api_server_4bit.py, deploy_to_ecs.sh)
