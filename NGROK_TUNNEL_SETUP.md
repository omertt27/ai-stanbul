# 🌐 Ngrok Tunnel Setup for vLLM

## Strategy: Expose vLLM via Ngrok

Instead of SSH tunnel, we'll use Ngrok to expose vLLM from RunPod to the internet!

---

## 🚀 Step 1: Install Ngrok on RunPod

You're already SSH'd into RunPod, so run:

```bash
# Download and install Ngrok
cd /tmp
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
mv ngrok /usr/local/bin/
chmod +x /usr/local/bin/ngrok
```

---

## 🔑 Step 2: Setup Ngrok Auth Token

You need an Ngrok account and auth token:

1. Go to: https://dashboard.ngrok.com/signup (if you don't have account)
2. Or login: https://dashboard.ngrok.com/
3. Get your auth token: https://dashboard.ngrok.com/get-started/your-authtoken

**Then run on RunPod:**

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```

---

## 🌐 Step 3: Start Ngrok Tunnel (on RunPod)

```bash
# Run Ngrok in background to expose port 8000
nohup ngrok http 8000 > /root/ngrok.log 2>&1 &

# Wait a moment
sleep 5

# Get the public URL
curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | grep https | cut -d'"' -f4
```

**Copy the HTTPS URL** that appears (e.g., `https://abc123.ngrok-free.app`)

---

## 📝 Step 4: Update Backend .env (on your Mac)

Exit RunPod:
```bash
exit
```

Then on your Mac, update the backend configuration:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Update .env with your Ngrok URL
# Replace YOUR_NGROK_URL with the URL from Step 3
echo 'LLM_API_URL=https://YOUR_NGROK_URL/v1' > .env.ngrok
cat .env | grep -v LLM_API_URL >> .env.ngrok
mv .env.ngrok .env
```

Or manually edit `/Users/omer/Desktop/ai-stanbul/backend/.env`:
```
LLM_API_URL=https://YOUR_NGROK_URL/v1
```

---

## ✅ Step 5: Test the Ngrok Connection (from Mac)

```bash
# Replace with your actual Ngrok URL
curl https://YOUR_NGROK_URL/v1/models
```

Should see the same JSON response!

---

## 🎯 Step 6: Start Backend (on Mac)

```bash
cd /Users/omer/Desktop/ai-stanbul/backend
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

---

## 🎨 Step 7: Start Frontend (on Mac)

New terminal:
```bash
cd /Users/omer/Desktop/ai-stanbul/frontend
npm run dev
```

---

## 🧪 Step 8: Test!

Open: http://localhost:5173

Try: "Merhaba! Istanbul hakkında bilgi ver."

---

## 🌐 Optional: Expose Frontend with Ngrok Too

If you want public access:

```bash
# On your Mac
ngrok http 5173
```

Share the frontend URL!

---

## 📊 Monitoring

**Check Ngrok on RunPod:**
```bash
# SSH back into RunPod
ssh oln8fcw6x2t614-64410d62@ssh.runpod.io -i ~/.ssh/id_ed25519

# Check Ngrok status
curl http://localhost:4040/api/tunnels

# Check Ngrok logs
tail -f /root/ngrok.log
```

---

## 🐛 Troubleshooting

### Ngrok tunnel not working?
- Check if Ngrok is running: `ps aux | grep ngrok`
- Restart: `pkill ngrok; ngrok http 8000 &`
- Check logs: `tail -f /root/ngrok.log`

### Backend can't connect?
- Verify Ngrok URL in backend `.env`
- Test with curl from Mac
- Check for Ngrok free tier limits

---

**Start with Step 1 on RunPod!** 🚀
