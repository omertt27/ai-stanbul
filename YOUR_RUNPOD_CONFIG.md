# 🔧 Your RunPod Configuration

**Created:** January 2025  
**Status:** ✅ CONFIGURED

---

## 🎯 Your RunPod LLM Server

### RunPod Pod Details
- **Pod ID:** `ytc61lal7ag5sy`
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Status:** ✅ Active

### RunPod Proxy URL
```
https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/
```

### LLM API URL (for Render backend)
```
https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

**Note:** The `/v1` suffix is required for the API endpoint.

### SSH Access

**Standard SSH (via RunPod proxy):**
```bash
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Direct TCP SSH (supports SCP & SFTP):**
```bash
ssh root@194.68.245.173 -p 22001 -i ~/.ssh/id_ed25519
```

**Quick SSH Helper:**
```bash
# Create alias for easy access
alias runpod-ssh='ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519'

# Then simply run:
runpod-ssh
```

---

## ⚙️ Configuration Files Updated

The following files have been updated with your RunPod URL:

1. ✅ `RENDER_ENV_VARS.txt` - Backend environment variables
2. ✅ `PHASE_1_QUICK_START.md` - Quick start guide
3. ✅ `phase1_environment_setup.md` - Environment setup guide

---

## 🚀 Next Steps

### 1. Configure Render (Backend)

Go to https://dashboard.render.com and add this environment variable:

```bash
LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

### 2. Test Your LLM Server

```bash
# Test health endpoint
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health

# Expected response:
# {"status":"healthy","model":"meta-llama/Meta-Llama-3.1-8B-Instruct"}
```

### 3. Test with API v1 endpoint

```bash
# Test v1 endpoint
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/health
```

### 4. Set Environment Variable for Testing

```bash
export LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
export BACKEND_URL=https://api.aistanbul.net
export FRONTEND_URL=https://aistanbul.net
```

### 5. Run Phase 1 Tests

```bash
# Verify environment
python3 verify_env.py

# Run health checks
python3 phase1_health_check.py

# Run multi-language tests
python3 phase1_multilang_tests.py

# Or run everything at once
./phase1_quick_start.sh
```

---

## 📋 Full Render Environment Configuration

Copy-paste this into your Render environment variables:

```bash
# LLM Server (YOUR RUNPOD URL)
LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1

# CORS Configuration
ALLOWED_ORIGINS=["https://aistanbul.net","https://www.aistanbul.net","https://api.aistanbul.net","http://localhost:3000","http://localhost:5173"]

# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Database (update with your credentials)
DATABASE_URL=postgresql://user:password@host:5432/ai_stanbul_prod

# Redis (optional but recommended)
REDIS_URL=redis://your-redis-host:6379

# Security (generate new keys!)
SECRET_KEY=your_super_secret_key_here_CHANGE_THIS
JWT_SECRET_KEY=your_jwt_secret_key_here_CHANGE_THIS

# Rate Limiting
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Feature Flags
USE_NEURAL_RANKING=True
ADVANCED_UNDERSTANDING_ENABLED=True
```

---

## 🧪 Testing Commands

### Test LLM Server Directly
```bash
# Health check
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health

# Test generation (if your server supports it)
curl -X POST https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### Test Backend Integration
```bash
# After configuring Render, test the chat endpoint
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Where can I eat in Istanbul?",
    "language": "en",
    "session_id": "test-123"
  }'
```

---

## ⚠️ Important Notes

1. **Port:** Your RunPod URL uses port 19123 (not the standard 8888)
2. **Path:** Your URL includes a path `/2feph6uogs25wg1sc0i37280ah5ajfmm/`
3. **Keep `/v1` suffix:** Always add `/v1` for the API endpoint
4. **Security:** This URL is public - ensure your LLM server has proper authentication if needed

---

## 🔐 Security Checklist

- [ ] Verify RunPod pod is secured (if authentication is available)
- [ ] Only expose necessary endpoints
- [ ] Monitor usage to detect abuse
- [ ] Set up rate limiting in backend
- [ ] Consider adding API key authentication

---

## 📊 Expected Performance

With your RunPod GPU setup:

- **Response Time:** 2-4 seconds
- **Max Tokens:** 250 tokens
- **Model:** Llama 3.1 8B Instruct (4-bit quantized)
- **Concurrent Requests:** Depends on GPU memory

---

## 🎯 Quick Start Command

All-in-one test command:

```bash
# Set your environment
export LLM_API_URL=https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
export BACKEND_URL=https://api.aistanbul.net
export FRONTEND_URL=https://aistanbul.net

# Run complete Phase 1 tests
./phase1_quick_start.sh
```

---

## ✅ Checklist

- [ ] RunPod URL verified and tested
- [ ] Render environment variable `LLM_API_URL` configured
- [ ] Backend redeployed with new URL
- [ ] Health check passes
- [ ] Chat endpoint works
- [ ] Multi-language tests pass
- [ ] Production ready!

---

**Your RunPod LLM server is configured and ready to use!** 🚀

**Next:** Configure Render and run Phase 1 tests.
