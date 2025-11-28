# 🌍 PUBLIC INTERNET DEPLOYMENT GUIDE

## 📋 Current Setup
- ✅ vLLM running on RunPod (port 8000)
- ✅ Backend running locally (port 8001)
- ✅ Frontend running locally (port 5173)
- ❌ Only accessible from your computer

## 🎯 Goal
Make your chatbot accessible to anyone on the internet!

---

## 🚀 OPTION 1: Ngrok (Fastest - 5 Minutes Setup)

### Why Ngrok?
- ✅ Free tier available
- ✅ HTTPS automatically
- ✅ Setup in minutes
- ✅ Great for testing/demos
- ⚠️ URL changes on free tier (e.g., `https://abc123.ngrok.io`)

### Step 1: Install Ngrok

```bash
# Install via Homebrew
brew install ngrok/ngrok/ngrok

# Or download from https://ngrok.com/download
```

### Step 2: Sign Up & Get Auth Token

1. Go to https://dashboard.ngrok.com/signup
2. Copy your auth token
3. Run: `ngrok config add-authtoken YOUR_AUTH_TOKEN`

### Step 3: Expose Your Backend

**In a new terminal:**

```bash
ngrok http 8001
```

**You'll see output like:**
```
Forwarding   https://abc123xyz.ngrok.io -> http://localhost:8001
```

**Copy that URL!** (e.g., `https://abc123xyz.ngrok.io`)

### Step 4: Update Frontend Configuration

```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
```

Find your API configuration file (usually `src/config.js` or similar) and update:

```javascript
// Change from:
const API_URL = 'http://localhost:8001';

// To your Ngrok URL:
const API_URL = 'https://abc123xyz.ngrok.io';
```

**If you can't find config file, check these locations:**
- `frontend/src/config.js`
- `frontend/src/constants/api.js`
- `frontend/.env`
- `frontend/.env.local`

### Step 5: Update Backend CORS

```bash
cd /Users/omer/Desktop/ai-stanbul/backend
```

Edit `main.py` or wherever CORS is configured:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://abc123xyz.ngrok.io",  # Add your Ngrok URL
        "*"  # Or use "*" to allow all (for testing only!)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Step 6: Restart Backend

```bash
# Stop backend (Ctrl+C)
# Start again
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Step 7: Expose Your Frontend

**In ANOTHER new terminal:**

```bash
ngrok http 5173
```

**You'll see:**
```
Forwarding   https://xyz789def.ngrok.io -> http://localhost:5173
```

### Step 8: Share Your Chatbot! 🎉

**Your public chatbot URL:** `https://xyz789def.ngrok.io`

Share this with anyone in the world!

---

## 🚀 OPTION 2: Cloudflare Tunnel (Free, Permanent URL)

### Why Cloudflare?
- ✅ 100% Free forever
- ✅ Your own domain (if you have one)
- ✅ URL doesn't change
- ✅ Better performance (CDN)
- ⚠️ Slightly more setup

### Step 1: Install Cloudflared

```bash
brew install cloudflare/cloudflare/cloudflared
```

### Step 2: Login to Cloudflare

```bash
cloudflared tunnel login
```

This opens a browser - select your domain or create a free one.

### Step 3: Create a Tunnel

```bash
# Create tunnel
cloudflared tunnel create istanbul-ai

# You'll get a tunnel ID - save it!
# Example: abc123-def456-ghi789
```

### Step 4: Configure the Tunnel

Create config file:

```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Add this configuration:

```yaml
tunnel: YOUR_TUNNEL_ID_HERE
credentials-file: /Users/omer/.cloudflared/YOUR_TUNNEL_ID_HERE.json

ingress:
  # Frontend
  - hostname: istanbul-ai.yourname.com
    service: http://localhost:5173
  
  # Backend API
  - hostname: api.istanbul-ai.yourname.com
    service: http://localhost:8001
  
  # Catch-all
  - service: http_status:404
```

### Step 5: Create DNS Records

```bash
# Frontend
cloudflared tunnel route dns YOUR_TUNNEL_ID istanbul-ai.yourname.com

# Backend
cloudflared tunnel route dns YOUR_TUNNEL_ID api.istanbul-ai.yourname.com
```

### Step 6: Update Frontend Config

```javascript
const API_URL = 'https://api.istanbul-ai.yourname.com';
```

### Step 7: Update Backend CORS

```python
allow_origins=[
    "https://istanbul-ai.yourname.com",
    "http://localhost:5173",
]
```

### Step 8: Start the Tunnel

```bash
cloudflared tunnel run istanbul-ai
```

### Step 9: Access Your Site! 🎉

**Your permanent public URL:** `https://istanbul-ai.yourname.com`

---

## 🚀 OPTION 3: Deploy to Cloud (Production-Ready)

### For Production Use:

1. **Frontend:** Deploy to Vercel/Netlify (free)
2. **Backend:** Deploy to Railway/Render/Fly.io (free tier available)
3. **vLLM:** Keep on RunPod or move to dedicated GPU cloud

### Quick Deploy to Vercel (Frontend)

```bash
cd /Users/omer/Desktop/ai-stanbul/frontend

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow prompts, then get your URL like:
# https://istanbul-ai.vercel.app
```

### Quick Deploy to Railway (Backend)

1. Go to https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Connect your repo
4. Set environment variables:
   - `LLM_API_URL=http://YOUR_RUNPOD_POD_IP:8000/v1`
5. Deploy!

---

## 📊 Comparison

| Option | Setup Time | Cost | URL Type | Best For |
|--------|-----------|------|----------|----------|
| **Ngrok** | 5 min | Free | Random (changes) | Quick demos |
| **Cloudflare** | 15 min | Free | Your domain | Personal projects |
| **Cloud Deploy** | 30-60 min | Free tier | Custom | Production |

---

## 🔒 SECURITY RECOMMENDATIONS

### Before Going Public:

1. **Add Rate Limiting** (prevent abuse)
2. **Add Authentication** (optional, for private chatbot)
3. **Monitor Usage** (set up alerts)
4. **Add API Key** (for backend access)
5. **Enable HTTPS** (Ngrok/Cloudflare do this automatically)

### Quick Rate Limiting (Backend)

```bash
pip install slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def chat(request: Request):
    # your code
```

---

## 🎯 RECOMMENDED: Start with Ngrok

**For your first public deployment, I recommend Ngrok because:**

1. ✅ Fastest setup (5 minutes)
2. ✅ Test if everything works publicly
3. ✅ Can switch to Cloudflare/Cloud later
4. ✅ No risk of breaking current setup

---

## 🚀 QUICK START (Ngrok)

**Run these commands in order:**

```bash
# Terminal 1: Ensure vLLM running on RunPod
# (Use RunPod web terminal)
ps aux | grep vllm | grep -v grep

# Terminal 2: SSH Tunnel (on your Mac)
pkill -f "ssh.*pvj233wwhiu6j3.*8000"
ssh -f -N -L 8000:localhost:8000 pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519

# Terminal 3: Start Backend
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001

# Terminal 4: Expose Backend (NEW!)
brew install ngrok/ngrok/ngrok
ngrok http 8001
# Copy the https://xyz.ngrok.io URL

# Terminal 5: Start Frontend
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev

# Terminal 6: Expose Frontend (NEW!)
ngrok http 5173
# Copy the https://abc.ngrok.io URL
```

**Share the frontend URL with the world!** 🌍

---

## 🐛 Troubleshooting Public Access

### Issue: "CORS Error" in Browser Console

**Fix:** Update backend CORS to allow your public URL

```python
allow_origins=["*"]  # Allow all (for testing)
```

### Issue: "502 Bad Gateway"

**Fix:** Backend not running or Ngrok pointing to wrong port

```bash
# Check backend is running
curl http://localhost:8001/health
```

### Issue: Slow Responses

**Normal!** Llama 3.1 8B takes 5-10 seconds per response.

Consider adding:
- Loading indicator in frontend
- "AI is thinking..." message
- Response streaming (advanced)

### Issue: Ngrok URL Changes After Restart

**Fix:** Upgrade to Ngrok paid ($8/month) for static domain, or use Cloudflare Tunnel (free, permanent).

---

## 📱 Next Steps

1. **Choose deployment method** (I recommend Ngrok for now)
2. **Follow the quick start above**
3. **Test from your phone** (use mobile data, not WiFi)
4. **Share with friends!** 🎉
5. **Monitor usage** (check backend logs)
6. **Optional:** Deploy to cloud for production

---

## ✅ Final Checklist for Public Access

- [ ] vLLM running on RunPod
- [ ] SSH tunnel active (`curl http://localhost:8000/health`)
- [ ] Backend running locally (`curl http://localhost:8001/health`)
- [ ] Frontend running locally (`curl http://localhost:5173`)
- [ ] Ngrok installed (`ngrok version`)
- [ ] Ngrok auth token configured
- [ ] Backend exposed via Ngrok (`ngrok http 8001`)
- [ ] Frontend exposed via Ngrok (`ngrok http 5173`)
- [ ] CORS updated in backend
- [ ] API URL updated in frontend (if needed)
- [ ] Test from mobile phone browser ✅

---

## 🎉 You're Ready to Go Public!

Need help with any step? Let me know which option you want to try first!

**Fastest path:** Follow the "QUICK START (Ngrok)" section above. 🚀
