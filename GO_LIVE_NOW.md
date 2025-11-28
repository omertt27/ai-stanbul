# 🚀 QUICK PUBLIC DEPLOYMENT - Step by Step

## ✅ Prerequisites Check

Run these commands to verify everything is ready:

```bash
# 1. Check SSH tunnel to vLLM
lsof -i :8000 | grep ssh
# Should show: ssh ... localhost:8000

# 2. Check backend
lsof -i :8001 | grep LISTEN
# Should show: Python ... *:8001

# 3. Check frontend  
lsof -i :5173 | grep LISTEN
# Should show: node ... *:5173
```

If any are missing, start them first (see COMPLETE_SETUP_GUIDE.md)

---

## 🌍 DEPLOY IN 3 MINUTES

### Terminal 1: Expose Backend

```bash
ngrok http 8001
```

**You'll see:**
```
Forwarding   https://abc123xyz.ngrok.app -> http://localhost:8001
```

**📋 COPY THAT URL!** (e.g., `https://abc123xyz.ngrok.app`)

**KEEP THIS TERMINAL OPEN!**

---

### Terminal 2: Update Frontend Config

```bash
cd /Users/omer/Desktop/ai-stanbul/frontend

# Create .env.local file
cat > .env.local << 'EOF'
VITE_API_URL=https://YOUR_BACKEND_URL_HERE
EOF
```

**⚠️ IMPORTANT:** Replace `YOUR_BACKEND_URL_HERE` with the URL from Terminal 1!

**Example:**
```bash
cat > .env.local << 'EOF'
VITE_API_URL=https://abc123xyz.ngrok.app
EOF
```

---

### Terminal 3: Restart Frontend

```bash
cd /Users/omer/Desktop/ai-stanbul/frontend

# If frontend is already running, stop it first (Ctrl+C in its terminal)
# Then start fresh:
npm run dev
```

**You'll see:**
```
Local: http://localhost:5173
```

**KEEP THIS RUNNING!**

---

### Terminal 4: Expose Frontend

```bash
ngrok http 5173
```

**You'll see:**
```
Forwarding   https://xyz789def.ngrok.app -> http://localhost:5173
```

**🎉 THIS IS YOUR PUBLIC URL!** Share it with anyone!

**KEEP THIS TERMINAL OPEN TOO!**

---

## ✅ VERIFY IT WORKS

### Test from your phone (use mobile data, not WiFi):

1. Open browser on your phone
2. Go to: `https://xyz789def.ngrok.app` (your frontend URL)
3. Send a message: "Tell me about Istanbul"
4. Wait 5-10 seconds for response

**If it works → SUCCESS! 🎉**

---

## 🐛 Quick Troubleshooting

### Issue: "403 Forbidden" or "Invalid Host Header"

**Fix backend CORS:**
```bash
cd /Users/omer/Desktop/ai-stanbul/backend
```

Edit `main.py` and find the CORS section, add:
```python
allow_origins=["*"],  # Allow all origins for testing
```

Then restart backend (Ctrl+C and run `uvicorn main:app --reload --host 0.0.0.0 --port 8001`)

---

### Issue: Frontend shows "Network Error"

**Check frontend can reach backend:**
```bash
# Open browser console (F12)
# Check Network tab for failed requests
# URL should be: https://abc123xyz.ngrok.app/api/chat
```

**Verify .env.local:**
```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
cat .env.local
```

Should show your ngrok backend URL.

---

### Issue: Chat loads but no response

**Check backend logs** in the terminal running uvicorn. You'll see the actual error.

**Test backend directly:**
```bash
curl https://YOUR_BACKEND_URL.ngrok.app/health
```

Should return status info.

---

## 📊 What's Running

You should have **6 terminals open:**

1. **RunPod Web Terminal**: vLLM running (keep browser tab open)
2. **Mac Terminal 1**: SSH tunnel (`ssh -f -N -L 8000:localhost:8000 ...`)
3. **Mac Terminal 2**: Backend (`uvicorn main:app ...`)
4. **Mac Terminal 3**: Backend Ngrok (`ngrok http 8001`)
5. **Mac Terminal 4**: Frontend (`npm run dev`)
6. **Mac Terminal 5**: Frontend Ngrok (`ngrok http 5173`)

---

## 🎯 QUICK REFERENCE

### Your URLs:

```
Backend API:  https://YOUR-BACKEND-ID.ngrok.app
Frontend:     https://YOUR-FRONTEND-ID.ngrok.app
              👆 SHARE THIS ONE!
```

### Stop Everything:

```bash
# Stop Ngrok tunnels
pkill ngrok

# Stop frontend (Ctrl+C in frontend terminal)
# Stop backend (Ctrl+C in backend terminal)
```

### Restart Everything:

Follow the 4 steps above again!

---

## 💰 COST WARNING

**Your RunPod vLLM is running 24/7 until you stop it!**

- Check costs: https://www.runpod.io/console/billing
- Stop pod when not in use
- Can resume later (model stays loaded)

---

## 🔒 SECURITY TIP

Since you're public now:

1. **Monitor backend logs** for suspicious activity
2. **Check Ngrok dashboard** for traffic: https://dashboard.ngrok.com
3. **Set up rate limiting** (see PUBLIC_INTERNET_DEPLOYMENT.md)
4. **Don't share backend URL** - only share frontend!

---

## 📱 SHARE YOUR CHATBOT

Send this message to friends:

> "Check out my AI chatbot about Istanbul! 🇹🇷
> Built with Llama 3.1 8B running on GPU.
> Try it: https://YOUR-FRONTEND-ID.ngrok.app"

---

## ✅ Final Checklist

- [ ] SSH tunnel active (check: `lsof -i :8000 | grep ssh`)
- [ ] Backend running locally (check: `lsof -i :8001`)
- [ ] Backend exposed via Ngrok (check terminal 3)
- [ ] Frontend .env.local updated with backend URL
- [ ] Frontend running (check: `lsof -i :5173`)
- [ ] Frontend exposed via Ngrok (check terminal 5)
- [ ] Tested from mobile phone ✅

**When all checked → YOU'RE LIVE! 🌍**

---

## 🎉 CONGRATULATIONS!

Your Istanbul AI chatbot is now accessible from anywhere in the world!

Next steps:
- Share with friends
- Get feedback
- Monitor usage
- Consider deploying to production (Vercel/Railway) for permanent URLs

Need help? Check PUBLIC_INTERNET_DEPLOYMENT.md for more options!
