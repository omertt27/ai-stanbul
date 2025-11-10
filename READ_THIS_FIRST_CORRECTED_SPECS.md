# ⚠️ READ THIS FIRST: Critical Resource Corrections

## 🚨 IMPORTANT: Previous specs were TOO LOW!

This file contains the **CORRECTED** configuration for Llama 3.1 8B 4-bit.

---

## ✅ USE THESE VALUES (CORRECTED!)

### Job Definition Resources:
```
vCPUs: 8          ← Was 4, too low!
Memory: 32768     ← Was 16384, too low!
GPU: 1            ← Correct
```

### Compute Environment:
```
Instance Type: g5.2xlarge    ← Was g4dn.xlarge, TOO SMALL!
Max vCPUs: 8                 ← Was 4, too low!
```

---

## 📋 Quick Copy-Paste (CORRECTED VALUES)

### For AWS Batch Job Definition Form:

**Image:**
```
701893740767.dkr.ecr.eu-central-1.amazonaws.com/ai-istanbul-llm-4bit:latest
```

**Execution Role:**
```
arn:aws:iam::701893740767:role/aiIstanbulECSTaskExecutionRole
```

**Job Role:**
```
arn:aws:iam::701893740767:role/aiIstanbulECSTaskRole
```

**Secret (HF_TOKEN):**
```
Key: HF_TOKEN
ValueFrom: arn:aws:secretsmanager:eu-central-1:701893740767:secret:ai-istanbul/hf-token-0fEVwA
```

**Resources (CORRECTED!):**
```
vCPUs: 8
Memory (MiB): 32768
GPU: 1
```

**Environment Variables:**
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

### For Compute Environment:

```
Name: ai-istanbul-gpu-compute
Instance Type: g5.2xlarge       ← CRITICAL: DO NOT USE g4dn.xlarge!
Provisioning: Spot (recommended)
Min vCPUs: 0
Desired vCPUs: 0
Max vCPUs: 8
```

---

## ❌ Why g4dn.xlarge FAILS

| Spec | g4dn.xlarge | Needed | Result |
|------|-------------|--------|--------|
| GPU VRAM | 16 GB | 24 GB | ❌ OOM Crash |
| System RAM | 16 GB | 32 GB | ❌ OOM Crash |
| vCPUs | 4 | 8 | ❌ CPU Bottleneck |

**Symptoms:**
- Job fails immediately after starting
- "CUDA out of memory" in CloudWatch logs
- Container exits with error code
- Wasted time and money

---

## ✅ Why g5.2xlarge WORKS

| Spec | g5.2xlarge | Needed | Result |
|------|------------|--------|--------|
| GPU | A10G | A10G/T4/V100 | ✅ Perfect |
| GPU VRAM | 24 GB | 24 GB | ✅ Fits perfectly |
| System RAM | 32 GB | 32 GB | ✅ No OOM |
| vCPUs | 8 | 8 | ✅ Smooth operation |

**Cost (eu-central-1):**
- Spot: ~$0.36/hour (70% off!)
- On-Demand: ~$1.21/hour

---

## 📚 Updated Documentation

All guides have been corrected with these values:

| File | Status |
|------|--------|
| `ECS_DEPLOYMENT_CONFIG.txt` | ✅ Updated |
| `BATCH_QUICK_REFERENCE.md` | ✅ Updated |
| `AWS_BATCH_SETUP_GUIDE.md` | ✅ Updated |
| `DEPLOYMENT_COMPLETE.md` | ✅ Updated |
| `CORRECTED_RESOURCE_REQUIREMENTS.md` | ✅ New (detailed explanation) |
| `push_to_ecr.sh` | ✅ Updated |

**READ FIRST:** `CORRECTED_RESOURCE_REQUIREMENTS.md` for full explanation

---

## 🚀 Next Steps

1. **Read:** `CORRECTED_RESOURCE_REQUIREMENTS.md` (understand why)
2. **Follow:** `BATCH_QUICK_REFERENCE.md` (quick copy-paste values)
3. **Reference:** `AWS_BATCH_SETUP_GUIDE.md` (step-by-step)
4. **Use:** Values from this file when creating resources

---

## 🎯 Success Checklist

- [ ] Job definition uses: vCPUs=8, Memory=32768, GPU=1
- [ ] Compute environment uses: g5.2xlarge instance
- [ ] Job submitted and status = RUNNING
- [ ] CloudWatch logs show "Model loaded successfully!"
- [ ] API health check returns 200 OK
- [ ] Text generation works

---

## 💡 Remember

- ✅ Use g5.2xlarge ONLY
- ✅ Set vCPUs to 8
- ✅ Set Memory to 32768
- ✅ Use Spot for 70% savings
- ❌ Never use g4dn.xlarge for this workload

---

**Start Here:** https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#job-definition

Use the corrected values above! 🚀
