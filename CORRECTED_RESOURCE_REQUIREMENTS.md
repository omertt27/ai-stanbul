# ⚠️ CRITICAL: Corrected Resource Requirements for Llama 3.1 8B 4-bit

## 🚨 PREVIOUS CONFIG WAS TOO LOW - USE THESE VALUES!

---

## ✅ CORRECTED RESOURCE REQUIREMENTS

### Job Definition Resources:
```
vCPUs: 8
Memory (MiB): 32768 (32 GB)
GPU: 1
```

### Compute Environment Instance:
```
Instance Type: g5.2xlarge (ONLY!)
```

---

## ❌ DO NOT USE g4dn.xlarge

### Why g4dn.xlarge Will FAIL:

| Spec | g4dn.xlarge | Required for Llama 3.1 8B |
|------|-------------|---------------------------|
| GPU VRAM | 16 GB | 24 GB minimum |
| System RAM | 16 GB | 32 GB minimum |
| vCPUs | 4 | 8 minimum |
| Result | **OOM CRASH** | ✅ Works |

### What Will Happen with g4dn.xlarge:
1. ❌ Container starts
2. ❌ Model begins loading
3. ❌ CUDA out of memory error
4. ❌ Job fails immediately
5. ❌ Wasted time and money

---

## ✅ USE g5.2xlarge

### Why g5.2xlarge Works Perfectly:

| Spec | g5.2xlarge | Llama 3.1 8B 4-bit Needs |
|------|------------|--------------------------|
| GPU | A10G | ✅ Perfect |
| GPU VRAM | 24 GB | ✅ 24 GB minimum |
| System RAM | 32 GB | ✅ 32 GB minimum |
| vCPUs | 8 | ✅ 8 minimum |
| Result | **WORKS!** | ✅ Optimal |

### Cost (eu-central-1):
- **On-Demand:** ~$1.21/hour
- **Spot:** ~$0.36/hour (70% savings!)

---

## 📊 Why These Requirements?

### vCPUs: 8 (not 4)
- Llama tokenizer is CPU-intensive
- Transformers library needs parallel processing
- 4 cores will bottleneck throughput
- 8 cores provides smooth operation

### Memory: 32 GB (not 16 GB)
Even in 4-bit quantization:
- Model weights: ~5 GB
- Tokenizer: ~1 GB
- CUDA kernels: ~3 GB
- PyTorch overhead: ~2 GB
- FastAPI + Python: ~2 GB
- Buffer for requests: ~3 GB
- **Total:** ~16-20 GB minimum
- **Safe:** 32 GB provides headroom

### GPU VRAM: 24 GB (not 16 GB)
- Model in 4-bit: ~5 GB
- Activation memory: ~4 GB
- KV cache: ~6 GB
- CUDA operations: ~3 GB
- Batch processing: ~2 GB
- **Total:** ~20 GB under load
- **Safe:** 24 GB provides buffer

---

## 🔧 How to Fix Your Configuration

### If You Already Created Job Definition:

1. **Go to AWS Batch Console:**
   ```
   https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#job-definition
   ```

2. **Create New Revision:**
   - Click your job definition
   - Click "Create new revision"
   - Update resources:
     - vCPUs: 8
     - Memory: 32768
     - GPU: 1

3. **Save**

### If You Already Created Compute Environment:

1. **Go to Compute Environments:**
   ```
   https://eu-central-1.console.aws.amazon.com/batch/home?region=eu-central-1#compute-environments
   ```

2. **Either:**
   - **Option A:** Edit existing environment
     - Change instance type to: `g5.2xlarge`
     - Change max vCPUs to: `8`
   
   - **Option B:** Create new environment
     - Name: `ai-istanbul-gpu-compute-g5`
     - Instance: `g5.2xlarge`
     - Max vCPUs: `8`
     - Update job queue to use new environment

---

## 📋 Corrected Quick Reference

### Job Definition Form Values:
```
Image: 701893740767.dkr.ecr.eu-central-1.amazonaws.com/ai-istanbul-llm-4bit:latest
Execution Role: arn:aws:iam::701893740767:role/aiIstanbulECSTaskExecutionRole
Job Role: arn:aws:iam::701893740767:role/aiIstanbulECSTaskRole

Secret:
  Key: HF_TOKEN
  ValueFrom: arn:aws:secretsmanager:eu-central-1:701893740767:secret:ai-istanbul/hf-token-0fEVwA

Resources:
  vCPUs: 8
  Memory: 32768
  GPU: 1

Environment Variables:
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

### Compute Environment Form Values:
```
Name: ai-istanbul-gpu-compute
Provisioning: Spot (recommended) or On-Demand
Instance Type: g5.2xlarge (REQUIRED!)
Min vCPUs: 0
Desired vCPUs: 0
Max vCPUs: 8
```

---

## ✅ Success Indicators

After correcting, you should see:

1. ✅ Job status: RUNNING (not FAILED)
2. ✅ CloudWatch logs: "Model loaded successfully!"
3. ✅ No OOM errors in logs
4. ✅ API responds to health check
5. ✅ Text generation works smoothly

---

## 🧪 Test After Deployment

Once job is running on g5.2xlarge:

```bash
# Get public IP from EC2 console, then:

# Health check
curl http://<PUBLIC_IP>:8000/health

# Should return:
# {"status": "healthy", "model": "meta-llama/Llama-3.1-8B", "device": "cuda"}

# Test generation
curl -X POST http://<PUBLIC_IP>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the best places to visit in Istanbul?",
    "max_tokens": 100
  }'

# Should return text about Istanbul attractions!
```

---

## 💡 Why This Matters

Using the wrong instance type:
- ❌ Wastes 2-3 hours debugging
- ❌ Multiple failed job attempts = wasted money
- ❌ Frustration and confusion
- ❌ Doubt about whether the Docker image works

Using the right instance type:
- ✅ Works first try
- ✅ Model loads in 5-10 minutes
- ✅ API responds immediately
- ✅ Confidence in the deployment
- ✅ Can focus on integration, not infrastructure

---

## 📚 References

- **AWS g5 Instances:** https://aws.amazon.com/ec2/instance-types/g5/
- **Llama 3.1 Requirements:** https://huggingface.co/meta-llama/Llama-3.1-8B
- **4-bit Quantization:** https://huggingface.co/docs/transformers/main_classes/quantization

---

## 🎯 Action Items

- [ ] Update job definition to 8 vCPUs, 32768 MB
- [ ] Update compute environment to g5.2xlarge
- [ ] Resubmit job
- [ ] Monitor CloudWatch logs
- [ ] Test API once RUNNING
- [ ] Celebrate when it works! 🎉

---

**Bottom Line:** Use g5.2xlarge with 8 vCPUs and 32 GB RAM. Nothing less will work reliably!
