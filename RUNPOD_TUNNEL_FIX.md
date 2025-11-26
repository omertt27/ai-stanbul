# 🔧 SSH TUNNEL FIX - Connection Refused

## Problem
Direct TCP connection refused on port 22124.

## ✅ Solution: Use RunPod Proxy SSH with Tunnel

**Try this command instead:**

```bash
ssh -L 8888:localhost:8888 vn290bqt32835t-64410fd1@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**This should work!** Keep this terminal open.

---

## 🧪 Test in New Terminal

Once connected, open a NEW terminal on your Mac:

```bash
# Test models
curl http://localhost:8888/v1/models

# Test completion
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50
  }' | python3 -m json.tool
```

---

## 🔍 If SSH Key Doesn't Work

Try without the SSH key:

```bash
ssh -L 8888:localhost:8888 vn290bqt32835t-64410fd1@ssh.runpod.io
```

Enter password when prompted.

---

## 🌐 Alternative: Check RunPod Dashboard for Correct Connection Info

1. Go to https://www.runpod.io/console/pods
2. Click on your running pod
3. Click **"Connect"** button
4. Look for SSH connection string
5. Copy the exact command shown there
6. Add `-L 8888:localhost:8888` before the hostname

**Example:**
If dashboard shows:
```
ssh user@host.runpod.io
```

Use:
```
ssh -L 8888:localhost:8888 user@host.runpod.io
```

---

## 🎯 Why Direct TCP Failed

The TCP port (22124) might be:
1. Changed by RunPod (they reassign ports)
2. Behind firewall
3. Not exposed in current pod configuration

**The RunPod proxy SSH is more reliable!**

---

## ⚡ Quick Alternative: Use ngrok Directly on RunPod

If SSH tunnel keeps failing, you can run ngrok inside RunPod:

### In your RunPod SSH terminal:

```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Run ngrok (no account needed for basic use)
./ngrok http 8888
```

You'll see:
```
Forwarding   https://xxxx-yyyy.ngrok-free.app -> http://localhost:8888
```

**Copy that URL!** Use it as your LLM endpoint:

```
RUNPOD_LLM_ENDPOINT=https://xxxx-yyyy.ngrok-free.app
```

---

## 📝 Next Steps

1. **Try SSH proxy with tunnel** (recommended)
2. **If that fails, check RunPod dashboard** for current connection info
3. **If still fails, use ngrok inside RunPod** (works always)

**Try the first command and let me know what happens!**
