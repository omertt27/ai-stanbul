# 📋 Where to Put Configuration Values

## 🎯 OVERVIEW

After running `./deploy_to_ecs.sh`, you'll get a file called:
```
ECS_DEPLOYMENT_CONFIG.txt
```

This file contains all the values you need. Here's where to use them:

---

## 📂 FILES IN YOUR PROJECT

### **1️⃣ ECS_DEPLOYMENT_CONFIG.txt** (Auto-generated)
**Created by:** `deploy_to_ecs.sh` script  
**Purpose:** Contains all ARNs, URIs, and IDs you need  
**When created:** After you run `./deploy_to_ecs.sh`

**Example contents:**
```
========================================
ECS/Batch Deployment Configuration
========================================

AWS Account ID: 123456789012
AWS Region: eu-central-1

ECR Repository URI:
123456789012.dkr.ecr.eu-central-1.amazonaws.com/ai-istanbul-llm-4bit:latest

Execution Role ARN:
arn:aws:iam::123456789012:role/aiIstanbulECSTaskExecutionRole

Task Role ARN:
arn:aws:iam::123456789012:role/aiIstanbulECSTaskRole

HuggingFace Token Secret ARN:
arn:aws:secretsmanager:eu-central-1:123456789012:secret:ai-istanbul/hf-token-AbCdEf
```

### **2️⃣ ECS_FORM_FILLING_GUIDE.md** (Already exists)
**Purpose:** Instructions on how to fill the AWS Console form  
**Use it:** Open side-by-side with AWS Console

### **3️⃣ Dockerfile.4bit** (Already exists)
**Purpose:** Docker container definition  
**Use it:** Gets built by `deploy_to_ecs.sh`

### **4️⃣ llm_api_server_4bit.py** (Already exists)
**Purpose:** Your LLM API server code  
**Use it:** Gets packaged into Docker container

---

## 🌐 WHERE TO PASTE CONFIGURATION (AWS Console)

After you have `ECS_DEPLOYMENT_CONFIG.txt`, you'll paste values into AWS Console:

### **Step 1: Go to AWS Batch Console**
```
https://console.aws.amazon.com/batch
```

### **Step 2: Navigate to Job Definitions**
1. Click **Job definitions** in left sidebar
2. Click **Create** button
3. Select **EC2** (not Fargate - you need GPU)
4. Fill in the form using values from `ECS_DEPLOYMENT_CONFIG.txt`

### **Step 3: Form Fields Mapping**

| AWS Console Field | Value from ECS_DEPLOYMENT_CONFIG.txt |
|-------------------|--------------------------------------|
| **Job definition name** | Type: `ai-istanbul-llm-4bit` |
| **Image** | Copy: ECR Repository URI |
| **Repository credentials** | Leave empty |
| **Command** | Leave empty |
| **Execution role** | Copy: Execution Role ARN |
| **Job role** | Copy: Task Role ARN |
| **vCPUs** | Type: `4` |
| **Memory (MiB)** | Type: `16384` |
| **GPU** | Type: `1` |

### **Step 4: Add Secret**
Click "Add secret" button:

| Field | Value |
|-------|-------|
| **Key** | `HF_TOKEN` |
| **Value from** | Select: "Secrets Manager" |
| **Secret ARN** | Copy: HuggingFace Token Secret ARN from config file |

### **Step 5: Add Environment Variables**
Click "Add environment variable" for each:

| Key | Value |
|-----|-------|
| `PORT` | `8000` |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B` |
| `QUANTIZATION_BITS` | `4` |
| `DEVICE` | `cuda` |
| `MAX_TOKENS` | `250` |
| `BATCH_SIZE` | `1` |
| `TORCH_DTYPE` | `float16` |
| `LOW_CPU_MEM_USAGE` | `true` |
| `USE_CACHE` | `true` |

---

## 🔄 WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────┐
│  1. Run: ./deploy_to_ecs.sh            │
│                                         │
│  Creates:                               │
│  • ECR repository                       │
│  • IAM roles                            │
│  • Secrets in Secrets Manager          │
│  • Builds & pushes Docker image        │
│  • Generates ECS_DEPLOYMENT_CONFIG.txt │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  2. Open: ECS_DEPLOYMENT_CONFIG.txt     │
│                                         │
│  Read all the ARNs and URIs            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  3. Go to AWS Batch Console             │
│     https://console.aws.amazon.com/batch│
│                                         │
│  Create → Job Definition → EC2          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  4. Copy-paste values from              │
│     ECS_DEPLOYMENT_CONFIG.txt           │
│     into AWS Console form fields        │
│                                         │
│  Use ECS_FORM_FILLING_GUIDE.md          │
│  as reference for which field is which  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  5. Submit form → Create Job Definition │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  6. Create Compute Environment          │
│     (GPU instance: g4dn.xlarge)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  7. Create Job Queue                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  8. Submit Job → Test API               │
└─────────────────────────────────────────┘
```

---

## 📝 QUICK COMMAND REFERENCE

### **View your configuration:**
```bash
cat ECS_DEPLOYMENT_CONFIG.txt
```

### **Copy a specific value:**
```bash
# Copy ECR URI
grep "ECR Repository URI" -A 1 ECS_DEPLOYMENT_CONFIG.txt | tail -1

# Copy Execution Role ARN
grep "Execution Role ARN" -A 1 ECS_DEPLOYMENT_CONFIG.txt | tail -1

# Copy Task Role ARN
grep "Task Role ARN" -A 1 ECS_DEPLOYMENT_CONFIG.txt | tail -1

# Copy Secret ARN
grep "Secret ARN" -A 1 ECS_DEPLOYMENT_CONFIG.txt | tail -1
```

---

## 🎯 SUMMARY: YOU DON'T NEED TO EDIT ANY FILES!

✅ **All configuration is auto-generated**
- The `deploy_to_ecs.sh` script creates everything
- The `ECS_DEPLOYMENT_CONFIG.txt` file has all your values
- You just **copy-paste** from that file into AWS Console

❌ **You DON'T manually edit:**
- Dockerfile.4bit (already correct)
- llm_api_server_4bit.py (already correct)
- deploy_to_ecs.sh (already correct)

📋 **You ONLY need to:**
1. Run `./deploy_to_ecs.sh` (once)
2. Open `ECS_DEPLOYMENT_CONFIG.txt`
3. Copy values from file → Paste into AWS Console form
4. Submit the form

---

## 💡 HELPFUL TIPS

### **Tip 1: Open Files Side-by-Side**
```
Terminal 1: cat ECS_DEPLOYMENT_CONFIG.txt
Terminal 2: Open AWS Console in browser
```
Copy-paste from terminal to browser!

### **Tip 2: Use Your Text Editor**
```bash
# Open config in VS Code
code ECS_DEPLOYMENT_CONFIG.txt

# Open guide in VS Code
code ECS_FORM_FILLING_GUIDE.md
```

### **Tip 3: Print a Checklist**
```bash
# Create a checklist
cat ECS_DEPLOYMENT_CONFIG.txt
echo ""
echo "✅ Checklist:"
echo "[ ] Copied ECR URI to 'Image' field"
echo "[ ] Copied Execution Role ARN to 'Execution role' field"
echo "[ ] Copied Task Role ARN to 'Job role' field"
echo "[ ] Copied Secret ARN to 'HF_TOKEN' secret"
echo "[ ] Added all 9 environment variables"
echo "[ ] Set vCPUs = 4"
echo "[ ] Set Memory = 16384"
echo "[ ] Set GPU = 1"
```

---

## 🚨 COMMON QUESTIONS

### **Q: Where do I put my HuggingFace token?**
**A:** You already put it in the deployment script or it will prompt you. The script stores it in AWS Secrets Manager automatically.

### **Q: Do I need to edit Dockerfile.4bit?**
**A:** No! It's already configured correctly.

### **Q: Where do I put the ECR URI?**
**A:** In AWS Console → Job Definition form → "Image" field

### **Q: Can I edit ECS_DEPLOYMENT_CONFIG.txt?**
**A:** You can, but you shouldn't need to. It's auto-generated with correct values.

### **Q: What if I lose the config file?**
**A:** Just run `./deploy_to_ecs.sh` again (it will skip existing resources and regenerate the file)

---

## 🎬 NEXT STEPS

**Right now, you should:**
1. ⏳ Wait for `./deploy_to_ecs.sh` to finish
2. ✅ Check that `ECS_DEPLOYMENT_CONFIG.txt` exists
3. 📖 Read the config file
4. 🌐 Open AWS Batch Console
5. 📋 Fill form using values from config file

---

**Status:** ✅ Configuration Location Guide  
**Last Updated:** November 10, 2025  
**Key File:** ECS_DEPLOYMENT_CONFIG.txt (auto-generated)
