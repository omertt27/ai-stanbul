# RunPod Deployment Quick Reference

**AI Istanbul LLM - RunPod Configuration**  
**Last Updated:** January 2025

---

## 🎯 **Essential Configuration**

### **Container Image**
```
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```
- **Source:** Docker Hub (public, no login required)
- **PyTorch:** 2.8.0
- **CUDA:** 12.8.1
- **Ubuntu:** 24.04
- **Python:** 3.11

### **GPU Selection**
- **Recommended:** RTX 4090 (24GB VRAM)
- **Alternative:** RTX A6000 (48GB VRAM)
- **Pricing:** Spot ~$0.20-0.40/hr, On-Demand ~$0.60/hr

### **Storage Configuration**
```yaml
Container Disk: 30GB (temporary, cleared on restart)
Volume Disk: 50GB (persistent, mounted to /workspace)
```

**Important:** Always save model cache, logs, and data to `/workspace` for persistence!

### **Port Configuration**
```yaml
HTTP Port: 8888  # FastAPI server
SSH Port: 22     # SSH access
```

**Endpoint URL Pattern:**
```
https://<pod-id>-8888.proxy.runpod.net
```

### **Environment Variables**

**Set via RunPod Secrets (secure):**
```bash
HF_TOKEN=hf_xxxxxxxxxxxxx  # HuggingFace token
```

**Optional (can be plain environment variables):**
```bash
MODEL_NAME=meta-llama/Llama-3.1-8B
MAX_TOKENS=250
TEMPERATURE=0.7
```

---

## 🚀 **Deployment Commands**

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

### **1. Create RunPod Pod**
- Go to RunPod → Create Pod
- Select RTX 4090 (EUR-IS-1)
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Container Disk: 30GB, Volume: 50GB
- Expose HTTP port: 8888
- Add Secret: `HF_TOKEN` = `hf_xxxxx`

### **2. SSH into Pod**
```bash
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519

# Get connection details from RunPod Console → Your Pod → Connect → Direct TCP Ports
# Example: ssh root@213.144.200.242 -p 13910 -i ~/.ssh/id_ed25519

# Or use Web Terminal:
# RunPod Console → Your Pod → Connect → Start Web Terminal
```

### **3. Install Dependencies**
```bash
pip install transformers accelerate bitsandbytes fastapi uvicorn
```

### **4. Login to HuggingFace**
```bash
huggingface-cli login --token $HF_TOKEN
```

### **5. Start LLM Server**
```bash
# Copy your llm_api_server_4bit.py to /workspace
cd /workspace
nohup python llm_api_server_4bit.py --port 8888 > llm_server.log 2>&1 &
```

### **6. Test Health Check**
```bash
curl http://localhost:8888/health
```

### **7. Get Public Endpoint**
From RunPod dashboard, copy the endpoint URL:
```
https://<pod-id>-8888.proxy.runpod.net
```

---

## 🔧 **Backend Integration**

### **Environment Variable**
```bash
LLM_API_URL=https://<pod-id>-8888.proxy.runpod.net
```

### **Python Example**
```python
import requests
import os

LLM_API_URL = os.getenv("LLM_API_URL")

def query_llm(prompt: str, max_tokens: int = 250) -> str:
    try:
        response = requests.post(
            f"{LLM_API_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["generated_text"]
    except requests.exceptions.RequestException as e:
        print(f"LLM API error: {e}")
        return "Sorry, I couldn't generate a response."
```

### **Node.js Example**
```javascript
const LLM_API_URL = process.env.LLM_API_URL;

async function queryLLM(prompt, maxTokens = 250) {
  try {
    const response = await fetch(`${LLM_API_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, max_tokens: maxTokens }),
      timeout: 30000
    });
    const data = await response.json();
    return data.generated_text;
  } catch (error) {
    console.error('LLM API error:', error);
    return "Sorry, I couldn't generate a response.";
  }
}
```

---

## 📊 **Cost Estimates**

| Usage Pattern | GPU Hours/Day | Daily Cost | Monthly Cost |
|--------------|---------------|------------|--------------|
| **Light (8h/day)** | 8 | $1.60-3.20 | $48-96 |
| **Moderate (12h/day)** | 12 | $2.40-4.80 | $72-144 |
| **Heavy (24/7)** | 24 | $4.80-9.60 | $144-288 |

*Based on RTX 4090 Spot pricing ($0.20-0.40/hr)*

**Total Stack (RunPod + AWS):**
- Light: ~$87-127/month
- Heavy: ~$183-329/month

**Savings vs AWS Batch:** 60-70% reduction! 🎉

---

## 🛡️ **Security Best Practices**

1. **Use RunPod Secrets for sensitive data:**
   - HF_TOKEN
   - API keys
   - Database credentials

2. **Backend authentication:**
   - Add API key validation in your backend
   - Don't expose RunPod endpoint publicly

3. **Network security:**
   - RunPod endpoints are HTTPS by default
   - SSH access restricted to your IP

4. **Monitoring:**
   - Set up CloudWatch alarms
   - Monitor GPU usage and costs
   - Track API response times

---

## 🧪 **Testing Checklist**

### **RunPod Pod Health**
- [ ] Pod status: Running
- [ ] GPU detected: `nvidia-smi`
- [ ] Dependencies installed: `pip list`
- [ ] HuggingFace login: `huggingface-cli whoami`
- [ ] Server running: `curl http://localhost:8888/health`
- [ ] Public endpoint accessible: `curl https://<pod-id>-8888.proxy.runpod.net/health`

### **LLM Generation**
- [ ] Simple prompt test:
  ```bash
  curl -X POST https://<pod-id>-8888.proxy.runpod.net/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is Istanbul?", "max_tokens": 100}'
  ```
- [ ] Response time: <2 seconds ✅
- [ ] Output quality: Coherent and relevant ✅

### **Backend Integration**
- [ ] Environment variable set: `echo $LLM_API_URL`
- [ ] Backend can connect: Test from backend server
- [ ] Error handling: Test with invalid prompts
- [ ] Timeout handling: Test with very long prompts

### **Frontend Integration**
- [ ] End-to-end test: User query → Backend → LLM → Response
- [ ] UI displays LLM responses correctly
- [ ] Loading states work properly
- [ ] Error messages shown gracefully

---

## 📚 **Documentation Links**

- **Full Deployment Guide:** [RUNPOD_DEPLOYMENT_GUIDE.md](./RUNPOD_DEPLOYMENT_GUIDE.md)
- **Migration Guide:** [TRANSITION_TO_RUNPOD.md](./TRANSITION_TO_RUNPOD.md)
- **Deployment Status:** [DEPLOYMENT_STATUS.md](./DEPLOYMENT_STATUS.md)
- **Configuration Summary:** [RUNPOD_CONFIGURATION_SUMMARY.md](./RUNPOD_CONFIGURATION_SUMMARY.md)

---

## 🆘 **Troubleshooting**

### **Pod won't start**
- Check GPU availability (try different region: EUR-IS-1, EUR-NO-1)
- Verify payment method
- Try different GPU type (RTX 4090 → A6000)

### **Model download fails**
- Verify HF_TOKEN is valid: `huggingface-cli whoami`
- Check internet connectivity: `ping huggingface.co`
- Ensure model access granted: https://huggingface.co/meta-llama/Llama-3.1-8B

### **Server crashes (OOM)**
- Check GPU memory: `nvidia-smi`
- Verify 4-bit quantization is enabled
- Reduce `max_tokens` parameter (250 → 150)

### **Slow response times**
- Check network latency: `ping <pod-id>-8888.proxy.runpod.net`
- Reduce `max_tokens` (default: 250 → 150)
- Enable response caching for common queries
- Consider using a closer region

### **Backend can't connect**
- Verify endpoint URL is correct (check port 8888)
- Test with curl from backend server
- Check firewall rules (allow outbound HTTPS)
- Add timeout and retry logic (30s timeout recommended)

---

## 🎯 **Quick Start (5 Minutes)**

1. **Deploy Pod:**
   - Go to RunPod → Create Pod
   - Select RTX 4090 (EUR-IS-1)
   - Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
   - Container Disk: 30GB, Volume: 50GB
   - Expose HTTP port: 8888
   - Add Secret: `HF_TOKEN` = `hf_xxxxx`

2. **Install & Start:**
   ```bash
   pip install transformers accelerate bitsandbytes fastapi uvicorn
   huggingface-cli login --token $HF_TOKEN
   cd /workspace
   # Copy llm_api_server_4bit.py here
   python llm_api_server_4bit.py --port 8888 &
   ```

3. **Test:**
   ```bash
   curl https://<pod-id>-8888.proxy.runpod.net/health
   ```

4. **Update Backend:**
   ```bash
   export LLM_API_URL=https://<pod-id>-8888.proxy.runpod.net
   # Restart your backend service
   ```

5. **Done!** 🎉

---

**Questions?** Check the full documentation or contact support.
