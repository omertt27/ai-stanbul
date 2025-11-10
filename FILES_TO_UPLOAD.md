# 📦 Files to Upload to AWS CloudShell

## ✅ 3 Required Files

Upload these 3 files from your Mac to AWS CloudShell:

### **1️⃣ Dockerfile.4bit**
```
📁 Full Path: /Users/omer/Desktop/ai-stanbul/Dockerfile.4bit
📊 Size: ~1.2 KB
📝 Purpose: Docker container definition for 4-bit quantized Llama model
```

### **2️⃣ llm_api_server_4bit.py**
```
📁 Full Path: /Users/omer/Desktop/ai-stanbul/llm_api_server_4bit.py
📊 Size: ~13 KB
📝 Purpose: Python Flask API server for LLM inference
```

### **3️⃣ deploy_to_ecs.sh**
```
📁 Full Path: /Users/omer/Desktop/ai-stanbul/deploy_to_ecs.sh
📊 Size: ~7.7 KB
📝 Purpose: Automated deployment script for AWS ECS
```

---

## 🌐 How to Upload to CloudShell

### **Step 1: Open CloudShell**
1. Go to: https://console.aws.amazon.com/cloudshell
2. Click the CloudShell icon (>_) in the top-right corner
3. Wait 30 seconds for initialization

### **Step 2: Upload Files**
1. In CloudShell window, click: **Actions** → **Upload file**
2. Click **Select file** or drag and drop
3. Navigate to: `/Users/omer/Desktop/ai-stanbul/`
4. Select **all 3 files**:
   - ✅ `Dockerfile.4bit`
   - ✅ `llm_api_server_4bit.py`
   - ✅ `deploy_to_ecs.sh`
5. Click **Upload**
6. Wait for "Successfully uploaded X files" message

### **Step 3: Verify Upload**
In CloudShell, run:
```bash
ls -lh
```

You should see:
```
-rw-r--r-- 1 cloudshell-user cloudshell-user 1.2K Nov 10 XX:XX Dockerfile.4bit
-rw-r--r-- 1 cloudshell-user cloudshell-user  13K Nov 10 XX:XX llm_api_server_4bit.py
-rw-r--r-- 1 cloudshell-user cloudshell-user 7.7K Nov 10 XX:XX deploy_to_ecs.sh
```

### **Step 4: Make Script Executable**
```bash
chmod +x deploy_to_ecs.sh
```

### **Step 5: Run Deployment**
```bash
./deploy_to_ecs.sh
```

When prompted:
```
AWS Region: eu-central-1
HuggingFace Token: [paste from https://huggingface.co/settings/tokens]
Continue: y
```

---

## 📋 Quick Checklist

Before uploading, verify these files exist on your Mac:

```bash
# Run in your Mac terminal to verify:
ls -lh /Users/omer/Desktop/ai-stanbul/Dockerfile.4bit
ls -lh /Users/omer/Desktop/ai-stanbul/llm_api_server_4bit.py
ls -lh /Users/omer/Desktop/ai-stanbul/deploy_to_ecs.sh
```

All 3 commands should show file information (no "No such file" errors).

---

## 🚀 After Upload

Once files are uploaded and verified in CloudShell:

1. ✅ Run: `chmod +x deploy_to_ecs.sh`
2. ✅ Run: `./deploy_to_ecs.sh`
3. ✅ Enter region: `eu-central-1`
4. ✅ Enter HuggingFace token
5. ✅ Confirm: `y`
6. ⏳ Wait ~20 minutes for deployment
7. ✅ Get config: `cat ECS_DEPLOYMENT_CONFIG.txt`

---

## 💡 Pro Tip: Upload Multiple Files at Once

In CloudShell's upload dialog:
1. Click "Select file"
2. Hold **Cmd** (Mac) or **Ctrl** (Windows)
3. Click all 3 files
4. Click "Open"
5. All 3 files upload together!

---

## 🎯 Summary

| File | Location | Size |
|------|----------|------|
| `Dockerfile.4bit` | `/Users/omer/Desktop/ai-stanbul/` | 1.2 KB |
| `llm_api_server_4bit.py` | `/Users/omer/Desktop/ai-stanbul/` | 13 KB |
| `deploy_to_ecs.sh` | `/Users/omer/Desktop/ai-stanbul/` | 7.7 KB |

**Total upload size: ~22 KB** (takes <5 seconds)

---

**Ready?** Open CloudShell and upload these 3 files! 🚀

CloudShell URL: https://console.aws.amazon.com/cloudshell
