# 🤗 Hugging Face Quick Reference for AI Istanbul

## 🚀 2-Minute Setup

### Option 1: Hugging Face Inference API (Easiest)

```bash
# 1. Get your token from: https://huggingface.co/settings/tokens

# 2. Run the setup script
./setup_huggingface.sh

# 3. Or manually add to backend/.env:
echo 'LLM_API_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct' >> backend/.env
echo 'HUGGING_FACE_API_KEY=hf_YOUR_TOKEN_HERE' >> backend/.env

# 4. Test
cd backend && python test_runpod_connection.py
```

### Option 2: Deploy on RunPod (Best Performance)

```bash
# 1. SSH into your RunPod
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519

# 2. Deploy with Text Generation Inference
docker run -d --gpus all -p 8000:80 \
  -v $HOME/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct

# 3. Get your RunPod proxy URL from dashboard

# 4. Configure AI Istanbul
echo 'LLM_API_URL=https://YOUR-POD-ID-8000.proxy.runpod.net' >> backend/.env
```

---

## 📊 Recommended Models

### For Istanbul Tourism (Best to Worst)

1. **Llama 3.1 8B Instruct** ⭐ Best Balance
   ```
   meta-llama/Llama-3.1-8B-Instruct
   ```
   - Size: 8B params
   - VRAM: ~10GB
   - Speed: 40-60 tokens/s
   - Quality: Excellent

2. **Mistral 7B Instruct** ⭐ Best Turkish Support
   ```
   mistralai/Mistral-7B-Instruct-v0.3
   ```
   - Size: 7B params
   - VRAM: ~9GB
   - Speed: 50-70 tokens/s
   - Multilingual: Excellent

3. **Phi-3 Mini** ⭐ Fastest
   ```
   microsoft/Phi-3-mini-4k-instruct
   ```
   - Size: 3.8B params
   - VRAM: ~6GB
   - Speed: 80-100 tokens/s
   - Quality: Good

4. **Llama 3.1 70B** (Requires A100)
   ```
   meta-llama/Llama-3.1-70B-Instruct
   ```
   - Size: 70B params
   - VRAM: ~140GB
   - Speed: 10-20 tokens/s
   - Quality: Best (too large for A5000)

---

## 🔧 Configuration Examples

### Hugging Face Inference API

```bash
# backend/.env
LLM_API_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct
HUGGING_FACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_TIMEOUT=30
LLM_MAX_TOKENS=250
```

### RunPod with TGI

```bash
# backend/.env
LLM_API_URL=https://fgkqzve33ssbea-8000.proxy.runpod.net
LLM_TIMEOUT=60
LLM_MAX_TOKENS=250
```

### RunPod with vLLM

```bash
# backend/.env
LLM_API_URL=https://fgkqzve33ssbea-8000.proxy.runpod.net/v1
LLM_TIMEOUT=60
LLM_MAX_TOKENS=250
```

---

## 🧪 Testing Commands

```bash
# Test Hugging Face API directly
curl https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct \
  -X POST \
  -H "Authorization: Bearer $HUGGING_FACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is Istanbul?", "parameters": {"max_new_tokens": 100}}'

# Test AI Istanbul connection
cd backend
python test_runpod_connection.py

# Test backend API
python api_server.py &
curl http://localhost:8001/api/v1/llm/health
```

---

## 💰 Cost Comparison

| Method | Setup Time | Cost | Best For |
|--------|-----------|------|----------|
| HF Inference API | 2 min | ~$0.001/request | Low volume, testing |
| RunPod Spot | 10 min | ~$0.10/hour | Development |
| RunPod On-Demand | 10 min | ~$0.34/hour | Production |

**Break-even:** ~2,500 requests/day

---

## 🎯 Production Deployment (RunPod + TGI)

### Full Deployment Script

```bash
# SSH into RunPod
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519

# Install Docker (if needed)
apt-get update && apt-get install -y docker.io

# Deploy TGI with Llama 3.1 8B
docker run -d \
  --name tgi-llama \
  --gpus all \
  --restart unless-stopped \
  -p 8000:80 \
  -v $HOME/data:/data \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --num-shard 1 \
  --max-total-tokens 4096 \
  --max-input-length 3072

# Check logs
docker logs -f tgi-llama

# Test locally (on RunPod)
curl http://localhost:8000/health
```

### Monitor GPU

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 🐛 Troubleshooting

### Issue: "Model loading failed" (Gated model)

**Solution:** Some models require HF Hub access

```bash
# On RunPod, before deploying
export HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN

# Or with Docker
docker run ... -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN ...
```

### Issue: "Out of memory"

**Solutions:**
1. Use smaller model (Phi-3 Mini instead of Llama 8B)
2. Reduce max tokens: `--max-total-tokens 2048`
3. Use quantization (4-bit): `--quantize bitsandbytes-nf4`

### Issue: "Slow first request"

**Cause:** Model loading/warming up

**Solution:** Wait 2-5 minutes after deployment, or send test request

---

## 📚 Additional Resources

- **Get HF Token:** https://huggingface.co/settings/tokens
- **Browse Models:** https://huggingface.co/models?pipeline_tag=text-generation
- **TGI Docs:** https://huggingface.co/docs/text-generation-inference
- **vLLM Docs:** https://docs.vllm.ai/

---

## ✅ Quick Checklist

- [ ] Got Hugging Face token OR deployed on RunPod
- [ ] Updated `backend/.env` with LLM_API_URL
- [ ] Added API key (if using HF Inference API)
- [ ] Ran `python test_runpod_connection.py`
- [ ] Started backend: `python api_server.py`
- [ ] Tested health: `curl http://localhost:8001/api/v1/llm/health`
- [ ] Checked admin dashboard: `http://localhost:3000/admin`
- [ ] Metrics showing in dashboard

---

**Your AI Istanbul is ready with Hugging Face! 🤗🇹🇷**
