# Quick Start: RunPod LLM Integration

## 🚀 Start Backend (No Local LLM)

```bash
# From workspace root
npm run dev
```

Expected startup (no LLM loading):
```
✅ RunPod LLM Client loaded
   Endpoint: https://4vq1b984pitw8s-8888.proxy.runpod.net
   Model: Llama 3.1 8B (4-bit)
✅ Backend startup complete
```

**Should NOT see:**
```
❌ Loading checkpoint shards...
❌ Initializing LLM Service: LLaMA 3.1 8B
❌ Model path: /models/llama-3.1-8b
```

## 🧪 Test RunPod LLM

```bash
# Test direct connection
python3 test_runpod_llm.py
```

Expected output:
```
🧪 Testing RunPod LLM Integration
✅ Health check passed
✅ Generation test passed  
✅ Istanbul query test passed
```

## 🔍 Test Backend Endpoints

### Health Check
```bash
curl http://localhost:8001/api/llm/health
```

Expected:
```json
{
  "status": "healthy",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "endpoint": "https://4vq1b984pitw8s-8888.proxy.runpod.net"
}
```

### Generate Response
```bash
curl -X POST http://localhost:8001/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the top 3 attractions in Istanbul?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Istanbul Query
```bash
curl -X POST http://localhost:8001/api/llm/istanbul-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about Hagia Sophia"
  }'
```

### Chat Endpoint (with RunPod LLM fallback)
```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the best time to visit Istanbul?",
    "session_id": "test-session-001",
    "user_location": {"lat": 41.0082, "lng": 28.9784}
  }'
```

## ⚙️ Environment Variables

Check your `.env` file has:
```bash
# ML Service - DISABLED
ML_SERVICE_ENABLED=false

# RunPod LLM - PRIMARY
LLM_API_URL=https://4vq1b984pitw8s-8888.proxy.runpod.net
```

## 🐛 Troubleshooting

### Issue: Local LLM still loads
```bash
# Check ml_api_service.py line 85
grep -n "enable_llm" ml_api_service.py

# Should show:
# 85: ml_service = await create_ml_service(enable_llm=False)
```

### Issue: RunPod LLM not responding
```bash
# Test direct connection
curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health

# Should return:
# {"status": "healthy", "model": "meta-llama/Llama-3.1-8B-Instruct"}
```

### Issue: Backend won't start
```bash
# Check ports
lsof -ti:8001

# Kill if needed
kill -9 $(lsof -ti:8001)

# Restart
npm run dev
```

## 📊 Performance Metrics

| Operation | Expected Time |
|-----------|---------------|
| Backend Startup | < 10 seconds |
| Health Check | < 1 second |
| Simple Generation | 1-3 seconds |
| Complex Query | 3-5 seconds |

## ✅ Success Indicators

- [x] Backend starts in < 10 seconds
- [x] No "Loading checkpoint shards" message
- [x] `test_runpod_llm.py` passes all tests
- [x] Chat endpoint returns responses
- [x] Memory usage < 500 MB (backend only)

## 📚 Documentation

- **Integration Guide:** `RUNPOD_LLM_INTEGRATION_COMPLETE.md`
- **Fix Details:** `RUNPOD_LLM_FINAL_FIX.md`
- **Deployment:** `RUNPOD_DEPLOYMENT_GUIDE.md`

## 🎯 Next Steps

1. ✅ Start backend: `npm run dev`
2. ✅ Run tests: `python3 test_runpod_llm.py`
3. ✅ Test chat UI: Open browser to `http://localhost:8001`
4. ✅ Monitor logs for any errors
5. ✅ Deploy to production when ready

## 🆘 Getting Help

If issues persist:
1. Check `RUNPOD_LLM_FINAL_FIX.md` for detailed troubleshooting
2. Verify RunPod server is running: `curl https://4vq1b984pitw8s-8888.proxy.runpod.net/health`
3. Check backend logs: Look for RunPod LLM client initialization messages
4. Verify `.env` configuration is correct
