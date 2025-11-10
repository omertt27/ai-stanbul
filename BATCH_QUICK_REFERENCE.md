# 📋 AWS Batch - Quick Copy-Paste Values

Use these exact values when filling the AWS Batch form.

---

## 🎯 JOB DEFINITION

**Console Link:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#job-definition

### Basic Info
```
Name: ai-istanbul-llm-4bit
Timeout: 3600
Retry: 1
Priority: 1
```

### Container Image
```
701893740767.dkr.ecr.eu-central-1.amazonaws.com/ai-istanbul-llm-4bit:latest
```

### IAM Roles
```
Execution Role: arn:aws:iam::701893740767:role/aiIstanbulECSTaskExecutionRole
Job Role: arn:aws:iam::701893740767:role/aiIstanbulECSTaskRole
```

### Secret (HF_TOKEN)
```
Key: HF_TOKEN
ValueFrom: arn:aws:secretsmanager:eu-central-1:701893740767:secret:ai-istanbul/hf-token-0fEVwA
```

### Environment Variables (copy one by one)
```
PORT=8000
MODEL_NAME=meta-llama/Llama-3.1-8B
QUANTIZATION_BITS=4
DEVICE=cuda
MAX_TOKENS=250
BATCH_SIZE=1
TORCH_DTYPE=float16
LOW_CPU_MEM_USAGE=true
USE_CACHE=true
```

### Resources (CORRECTED FOR LLAMA 3.1 8B 4-BIT!)
```
vCPUs: 8
Memory: 32768
GPU: 1
```

⚠️ **CRITICAL:** These requirements are MINIMUM for Llama 3.1 8B 4-bit!

---

## 🖥️ COMPUTE ENVIRONMENT

**Console Link:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#compute-environments

```
Name: ai-istanbul-gpu-compute
Provisioning: Spot (cheaper) or On-Demand
Instance type: g5.2xlarge (REQUIRED! 24GB VRAM A10G GPU)
Min vCPUs: 0
Desired vCPUs: 0
Max vCPUs: 8
```

⚠️ **DO NOT USE g4dn.xlarge** - Only 16GB VRAM, will crash!
✅ **MUST USE g5.2xlarge** - 24GB VRAM, 8 vCPUs, 32GB RAM

---

## 📊 JOB QUEUE

**Console Link:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#queues

```
Name: ai-istanbul-gpu-queue
Priority: 1
Compute Environment: ai-istanbul-gpu-compute
```

---

## 🚀 SUBMIT JOB

**Console Link:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#jobs

```
Name: ai-istanbul-test-1
Job Definition: ai-istanbul-llm-4bit:1
Job Queue: ai-istanbul-gpu-queue
```

---

## 🧪 TEST API (after job is RUNNING)

```bash
# Get public IP from EC2 Console, then:

# Health check
curl http://<PUBLIC_IP>:8000/health

# Test generation
curl -X POST http://<PUBLIC_IP>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello Istanbul!", "max_tokens": 50}'
```

---

## 💡 TIPS

1. **Use g5.2xlarge ONLY** → g4dn.xlarge will OOM crash!
2. **Use Spot Instances** → Save 70% on costs
3. **Set Timeout** → Prevent runaway costs
4. **Monitor CloudWatch** → Check logs for errors
5. **Stop When Done** → Terminate job to save money

---

## 💰 COST ESTIMATE (CORRECTED)

**g5.2xlarge (24GB VRAM - REQUIRED):**
- On-Demand: ~$1.21/hour
- Spot: ~$0.36/hour (70% savings!)

**DO NOT USE g4dn.xlarge:**
- Will fail with OOM errors
- Wastes time and money on failed jobs

---

## 📈 EXPECTED TIMELINE

```
Submit Job
    ↓ (1-2 min)
Instance Provisioning
    ↓ (2-5 min)
Docker Pull
    ↓ (3-5 min)
Model Loading
    ↓ (5-10 min)
🎉 API READY!
```

**Total:** ~15-20 minutes from submission to ready

---

## 🆘 NEED HELP?

Read the full guide: `AWS_BATCH_SETUP_GUIDE.md`

Check config file: `ECS_DEPLOYMENT_CONFIG.txt`

---

**Start here:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#job-definition 🚀
