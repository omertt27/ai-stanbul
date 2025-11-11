# 🎯 READ THIS FIRST: RunPod Deployment Corrected Specs

**AI Istanbul LLM - Final Configuration**  
**Last Updated:** January 2025  
**Status:** ✅ All documentation updated and verified

---

## 🚨 **CRITICAL: WHAT WAS CORRECTED**

### **Port Configuration** ✅
- **INCORRECT (old):** Port 8000
- **CORRECT (now):** Port 8888
- **Why:** FastAPI server runs on 8888, RunPod proxy URL must match

**All endpoint references updated to:**
```
https://<pod-id>-8888.proxy.runpod.net
```

### **Container Image** ✅
- **RECOMMENDED:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **Source:** Docker Hub (public, no login required)
- **Alternative:** Private ECR image (requires AWS credentials)

**Why public image is better:**
- No registry authentication needed
- Faster pod startup
- Easier to iterate and test
- Install dependencies via pip on first run

### **Storage Configuration** ✅
- **Container Disk:** 30GB (temporary, cleared on restart)
- **Volume Disk:** 50GB (persistent, mounted to `/workspace`)

**Critical:** Always save to `/workspace` for persistence:
- Model cache
- Logs
- Application data
- User files

### **Security Configuration** ✅
- **HF_TOKEN:** Use RunPod Secrets (not plain environment variables)
- **Why:** Secrets are encrypted and not exposed in logs/UI
- **How:** In RunPod pod settings → Secrets → Add Secret

---

## 📋 **VERIFIED DOCUMENTATION FILES**

All these files are now **CORRECT** and **CONSISTENT**:

1. ✅ **RUNPOD_DEPLOYMENT_GUIDE.md** - Complete step-by-step guide
2. ✅ **DEPLOYMENT_STATUS.md** - Current deployment status and checklist
3. ✅ **TRANSITION_TO_RUNPOD.md** - Migration from AWS Batch
4. ✅ **RUNPOD_CONFIGURATION_SUMMARY.md** - Single source of truth for config
5. ✅ **RUNPOD_QUICK_REFERENCE.md** - Quick start and cheat sheet
6. ✅ **READ_THIS_FIRST_CORRECTED_SPECS.md** - This file!

**All port 8000 references have been replaced with port 8888.**  
**All container image references are correct.**  
**All storage paths are verified.**  
**All security practices are documented.**

---

## 🚀 **5-MINUTE DEPLOYMENT (CORRECT STEPS)**

### **0. SSH Key Setup (One-Time, 2 Minutes)**

```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -C "omertahtoko@gmail.com"
# Press Enter for default location, set passphrase (optional)

# 2. Copy your public key
cat ~/.ssh/id_ed25519.pub

# 3. Add to RunPod
# Go to: https://www.runpod.io/console → Settings → SSH Keys
# Click "Add SSH Key" and paste your public key

# ✅ Done! Ready to deploy pods
```

### **0.5 HuggingFace Model Access (One-Time, 2 Minutes)**

```bash
# 1. Request model access
# Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# Click "Request Access" → Accept license → Usually instant approval

# 2. Get your HuggingFace token
# Go to: https://huggingface.co/settings/tokens
# Create new token (Read access) → Copy token (hf_...)

# ✅ Model name formats (all work):
# - meta-llama/Llama-3.1-8B-Instruct
# - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# - Just paste the URL from your browser - RunPod handles it!
```

### **1. Create RunPod Pod**
- Go to https://runpod.io/console/pods
- Click "Deploy" → "GPU Pods"
- **GPU:** RTX 4090 (Spot, EUR-IS-1 or EUR-NO-1)
- **Container Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **Container Disk:** 30GB
- **Volume Disk:** 50GB (mounted to `/workspace`)
- **Expose Port:** HTTP 8888 ⚠️ (NOT 8000!)
- **Secret:** Add `HF_TOKEN` = `hf_xxxxxxxxxxxxx`

### **2. SSH into Pod**
```bash
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519
```

### **3. Install Dependencies**
```bash
pip install transformers accelerate bitsandbytes fastapi uvicorn
```

### **4. Login to HuggingFace**
```bash
huggingface-cli login --token $HF_TOKEN
```

### **5. Deploy Your LLM Server**
```bash
cd /workspace

# Copy your llm_api_server_4bit.py here
# (via SCP, git clone, or direct paste)

# Start the server on port 8888
nohup python llm_api_server_4bit.py --port 8888 > llm_server.log 2>&1 &

# Verify it's running
curl http://localhost:8888/health
```

### **6. Get Your Endpoint URL**
From RunPod dashboard, copy the HTTP port 8888 endpoint:
```
https://<pod-id>-8888.proxy.runpod.net
```

Example: `https://abc123xyz-8888.proxy.runpod.net`

### **7. Test Publicly**
```bash
# Health check
curl https://<your-pod-id>-8888.proxy.runpod.net/health

# Test generation
curl -X POST https://<your-pod-id>-8888.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is Istanbul?", "max_tokens": 100}'
```

### **8. Update Your Backend**
```bash
# Set environment variable
export LLM_API_URL=https://<your-pod-id>-8888.proxy.runpod.net

# Restart your backend service
sudo systemctl restart backend  # or equivalent
```

### **9. Done!** 🎉

---

## 🔍 **VERIFICATION CHECKLIST**

Use this to verify everything is correct:

### **RunPod Pod Configuration**
- [ ] Container image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- [ ] Container disk is 30GB
- [ ] Volume disk is 50GB (mounted to `/workspace`)
- [ ] **Port 8888 is exposed** (NOT 8000!)
- [ ] `HF_TOKEN` is set as a Secret (not plain env var)
- [ ] GPU is RTX 4090 or RTX A6000
- [ ] Region is EUR-IS-1 or EUR-NO-1

### **Server Running**
- [ ] Dependencies installed: `pip list | grep transformers`
- [ ] HuggingFace login successful: `huggingface-cli whoami`
- [ ] Server process running: `ps aux | grep llm_api_server`
- [ ] GPU detected: `nvidia-smi`
- [ ] Local health check: `curl http://localhost:8888/health` returns `{"status": "ok"}`

### **Public Endpoint**
- [ ] Endpoint URL format: `https://<pod-id>-8888.proxy.runpod.net`
- [ ] Public health check works: `curl https://<pod-id>-8888.proxy.runpod.net/health`
- [ ] Test generation works: `curl -X POST ... /generate`
- [ ] Response time < 2 seconds

### **Backend Integration**
- [ ] `LLM_API_URL` environment variable set correctly
- [ ] URL includes port 8888 (not 8000)
- [ ] Backend can connect to RunPod endpoint
- [ ] Error handling implemented (timeout, retry)
- [ ] Fallback message for LLM failures

### **Frontend Integration**
- [ ] End-to-end test: User query → Backend → RunPod → Response
- [ ] UI displays LLM responses correctly
- [ ] Loading states work
- [ ] Error messages shown gracefully

---

## 💰 **COST ESTIMATE (CORRECT NUMBERS)**

| Component | Spec | Price | Monthly |
|-----------|------|-------|---------|
| **RunPod GPU** | RTX 4090 Spot | $0.20-0.40/hr | $144-288 (24/7) |
| | | | $48-96 (8hr/day) |
| **AWS EC2** | t3.small | $0.023/hr | $17 |
| **AWS RDS** | db.t3.micro | $0.018/hr | $13 |
| **AWS S3** | 100GB | $0.023/GB | $2 |
| **Data Transfer** | 100GB/mo | $0.09/GB | $9 |
| **TOTAL (8hr/day)** | | | **$89-137/mo** |
| **TOTAL (24/7)** | | | **$185-329/mo** |

**Savings vs AWS Batch:** 60-70% 🎉

---

## 🛡️ **SECURITY BEST PRACTICES (VERIFIED)**

### **RunPod Pod**
- ✅ Use Secrets for `HF_TOKEN` (not plain environment variables)
- ✅ Secrets are encrypted and not exposed in logs
- ✅ HTTPS endpoints by default
- ✅ SSH key authentication (no password)

### **Backend**
- ✅ Add API key validation for LLM endpoint
- ✅ Don't expose RunPod URL publicly (proxy via backend)
- ✅ Implement rate limiting
- ✅ Log all LLM requests for debugging

### **Network**
- ✅ RunPod proxy handles TLS/SSL
- ✅ Backend → RunPod: HTTPS only
- ✅ Frontend → Backend: HTTPS only (Vercel)

---

## 🆘 **TROUBLESHOOTING (COMMON ISSUES)**

### **"Connection refused" when accessing endpoint**
**Cause:** Server not running on port 8888  
**Fix:**
```bash
# Check if server is running
ps aux | grep llm_api_server

# Check which port it's using
netstat -tulpn | grep python

# Restart on correct port
python llm_api_server_4bit.py --port 8888 &
```

### **"404 Not Found" on public endpoint**
**Cause:** Wrong port in URL (using 8000 instead of 8888)  
**Fix:** Update URL to use port 8888:
```bash
# Correct
https://<pod-id>-8888.proxy.runpod.net

# Incorrect
https://<pod-id>-8000.proxy.runpod.net
```

### **Backend can't connect to LLM**
**Cause:** Wrong `LLM_API_URL` or port mismatch  
**Fix:**
```bash
# Verify environment variable
echo $LLM_API_URL

# Should be (note the 8888):
https://<pod-id>-8888.proxy.runpod.net

# Test from backend server
curl https://<pod-id>-8888.proxy.runpod.net/health
```

### **Model not persisting across restarts**
**Cause:** Saving to container disk instead of volume  
**Fix:** Save everything to `/workspace`:
```python
# Set cache directory to volume
os.environ["TRANSFORMERS_CACHE"] = "/workspace/models"
os.environ["HF_HOME"] = "/workspace/huggingface"
```

### **Out of Memory (OOM) errors**
**Cause:** Not using 4-bit quantization or max_tokens too high  
**Fix:**
```python
# Ensure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Reduce max_tokens
max_tokens = 150  # Instead of 250
```

---

## 📚 **NEXT STEPS**

1. **Read the full guide:** [RUNPOD_DEPLOYMENT_GUIDE.md](./RUNPOD_DEPLOYMENT_GUIDE.md)
2. **Deploy your pod:** Follow the 5-minute steps above
3. **Update backend:** Set `LLM_API_URL` with port 8888
4. **Test end-to-end:** Frontend → Backend → RunPod → Response
5. **Monitor costs:** RunPod dashboard + AWS Budgets

---

## ✅ **VERIFICATION: All Documentation Updated**

I've verified that **all** of these files now use the **correct** configuration:

- Port: 8888 (not 8000)
- Container image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Storage: 30GB container + 50GB volume at `/workspace`
- Security: Use RunPod Secrets for `HF_TOKEN`

**You can now confidently deploy using any of the documentation files!**

---

## 📞 **Support & Resources**

- **RunPod Docs:** https://docs.runpod.io/
- **HuggingFace Llama 3.1:** https://huggingface.co/meta-llama/Llama-3.1-8B
- **Troubleshooting:** See section above or check pod logs
- **Community:** RunPod Discord for real-time help

---

**Last Verification:** January 2025  
**Status:** ✅ All documentation verified and corrected  
**Ready to Deploy:** YES! 🚀
