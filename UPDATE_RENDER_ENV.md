# Update Render Environment Variable for vLLM

## ⚠️ IMPORTANT: Backend is currently using the OLD endpoint (port 8888)

You need to update the Render environment variable to point to the new vLLM endpoint on port 8000.

## Quick Steps:

### 1. Go to Render Dashboard
Visit: https://dashboard.render.com

### 2. Select Your Backend Service
- Look for your backend service (probably named something like `ai-istanbul-backend` or similar)
- Click on it

### 3. Update Environment Variable
- Go to the **"Environment"** tab
- Find the variable: `LLM_API_URL`
- Update its value to:
  ```
  https://vezuyrr1tltd23-8000.proxy.runpod.net/v1
  ```
  
**Important Notes:**
- The pod ID is: `vezuyrr1tltd23`
- The port is now: `8000` (NOT 8888)
- Make sure to include `/v1` at the end

### 4. Save and Redeploy
- Click **"Save Changes"**
- Render will automatically redeploy your backend
- Wait 2-3 minutes for deployment to complete

### 5. Verify It Works
Once deployment is complete, test it:

```bash
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Merhaba, Istanbul hakkında kısaca bilgi ver", "session_id": "test-123"}' \
  | jq
```

### ✅ Success Indicators:
- You should get a real AI-generated response (not a fallback message)
- The response should be relevant to Istanbul
- Response time should be 2-10 seconds

### ❌ If It Still Doesn't Work:
1. Check RunPod: Is vLLM still running?
   ```bash
   ps aux | grep vllm
   ```

2. Test RunPod endpoint directly:
   ```bash
   curl -X POST https://vezuyrr1tltd23-8000.proxy.runpod.net/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "prompt": "Hello", "max_tokens": 10}'
   ```

3. Check Render logs for errors:
   - Go to your service on Render
   - Click "Logs" tab
   - Look for LLM-related errors

## Current Status:
✅ vLLM is running on RunPod (port 8000)
✅ vLLM is accessible via proxy URL
✅ Local `.env` is updated
❌ Render environment variable needs update (still points to port 8888)

## After Update:
Once you update the Render environment variable and redeploy, the full system will be working:
- Backend → vLLM (RunPod) → AI responses
- All Istanbul services integrated (weather, events, restaurants, maps, etc.)
- Full production system operational
