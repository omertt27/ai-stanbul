# 📋 ECS CONTAINER CONFIGURATION - QUICK REFERENCE
**Copy-Paste Values for AWS ECS Console**

---

## 🔴 REPLACE THESE PLACEHOLDERS

Before using, replace:
- `123456789012` → Your AWS Account ID
- `us-east-1` → Your AWS Region
- `AbCdEf` → Your Secret suffix (from Secrets Manager)

---

## 📝 FILL OUT FORM

### **1️⃣ Container Configuration**

**Image:**
```
123456789012.dkr.ecr.us-east-1.amazonaws.com/ai-istanbul-llm-4bit:latest
```

**Repository credentials:**
```
(leave empty)
```

**Command:**
```
(leave empty)
```

---

### **2️⃣ Execution Role**

**Execution role - optional:**
```
arn:aws:iam::123456789012:role/aiIstanbulECSTaskExecutionRole
```

**Enable ECS execute command:**
```
✅ Checked
```

---

### **3️⃣ Job Role Configuration**

**Job role:**
```
arn:aws:iam::123456789012:role/aiIstanbulECSTaskRole
```

---

### **4️⃣ Environment Configuration**

**vCPUs:**
```
4
```

**Memory (MiB):**
```
16384
```

**GPU:**
```
1
```

---

### **5️⃣ Secrets**

Click **"Add Secret"** and enter:

**Key:**
```
HF_TOKEN
```

**Value from:**
```
Secrets Manager
```

**Secret ARN:**
```
arn:aws:secretsmanager:us-east-1:123456789012:secret:ai-istanbul/hf-token-AbCdEf
```

---

### **6️⃣ Environment Variables**

Click **"Add Environment Variable"** for each row:

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

### **7️⃣ Parameters**

```
(leave empty)
```

---

## ✅ BEFORE YOU START

Run these commands first to get your actual values:

```bash
# 1. Get AWS Account ID
aws sts get-caller-identity --query Account --output text

# 2. Get AWS Region (from your config)
aws configure get region

# 3. Build and push Docker image (see ECS_4BIT_DEPLOYMENT_GUIDE.md)

# 4. Create HuggingFace secret
aws secretsmanager create-secret \
  --name ai-istanbul/hf-token \
  --secret-string "hf_YOUR_TOKEN_HERE" \
  --region us-east-1

# 5. Get Secret ARN
aws secretsmanager describe-secret \
  --secret-id ai-istanbul/hf-token \
  --query ARN \
  --output text

# 6. Create IAM roles (see ECS_4BIT_DEPLOYMENT_GUIDE.md)
```

---

## 🎯 RECOMMENDED INSTANCE TYPES

For GPU requirement, use one of these EC2 instance types:

### **Budget Option (Recommended)**
- **g4dn.xlarge** - NVIDIA T4 GPU, 4 vCPUs, 16 GB RAM
- **Cost:** ~$0.526/hr (on-demand) or ~$0.158/hr (spot)

### **Better Performance**
- **g5.xlarge** - NVIDIA A10G GPU, 4 vCPUs, 16 GB RAM
- **Cost:** ~$1.01/hr (on-demand) or ~$0.23/hr (spot)

**💡 Pro Tip:** Use **Spot instances** for 70% cost savings!

---

## 🚨 COMMON MISTAKES TO AVOID

1. ❌ **Forgetting to create ECR repository first**
   - ✅ Run: `aws ecr create-repository --repository-name ai-istanbul-llm-4bit`

2. ❌ **Not pushing Docker image to ECR**
   - ✅ Follow Step 1 in `ECS_4BIT_DEPLOYMENT_GUIDE.md`

3. ❌ **Wrong Secret ARN (missing suffix)**
   - ✅ Copy exact ARN from Secrets Manager console

4. ❌ **Not creating IAM roles**
   - ✅ Follow Step 3 in `ECS_4BIT_DEPLOYMENT_GUIDE.md`

5. ❌ **Forgetting to enable GPU**
   - ✅ Set GPU to `1` in Environment Configuration

6. ❌ **Insufficient memory**
   - ✅ Use at least 16384 MiB (16 GB)

---

## 📞 NEED HELP?

1. Check logs: AWS Console → ECS → Your Task → Logs tab
2. Or CLI: `aws logs tail /ecs/ai-istanbul-llm-4bit --follow`
3. Health check: `curl http://TASK_IP:8000/health`
4. Full guide: See `ECS_4BIT_DEPLOYMENT_GUIDE.md`

---

**Ready to deploy?** Follow `ECS_4BIT_DEPLOYMENT_GUIDE.md` for step-by-step instructions!
