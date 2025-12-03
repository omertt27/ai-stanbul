# 🚀 RESTART LLM SERVER - Quick Steps

## The server stopped running. Here's how to restart it:

---

## Option 1: Automated Script (Easiest!)

Run this single command:
```bash
./quick_start_server.sh
```

This will:
1. ✅ Check SSH connection
2. ✅ Verify files exist
3. ✅ Start the server
4. ✅ Wait for it to load
5. ✅ Test that it works

---

## Option 2: Manual Steps

### Step 1: Connect to RunPod
```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153
```

### Step 2: Start the Server
```bash
cd /workspace
./start_llm_server_runpod.sh
```

### Step 3: Wait 30 Seconds
The model needs time to load into GPU memory.

### Step 4: Test It
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### Step 5: Exit SSH
```bash
exit
```

---

## Option 3: Remote Start (No SSH login needed)

```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153 'cd /workspace && ./start_llm_server_runpod.sh'
```

Then wait 30 seconds and test:
```bash
./test_llm_server.sh
```

---

## After Server Starts

### Start SSH Tunnel (Terminal 1):
```bash
./start_tunnel.sh
```
**Keep this terminal open!**

### Test Server (Terminal 2):
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}'
```

---

## Why Did the Server Stop?

RunPod pods can be restarted or hibernated. Common reasons:
- Pod was stopped/restarted
- RunPod maintenance
- Inactivity timeout
- Manual stop

---

## Keep Server Running

The server should stay running as long as the pod is active. To check if it's still running:
```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153 'ps aux | grep llm_server.py'
```

---

## Note: SSH Port Changed!

The SSH port changed from **22077** to **22003**. I've updated all scripts:
- ✅ `test_llm_server.sh`
- ✅ `upload_to_runpod.sh`
- ✅ `start_tunnel.sh`
- ✅ `quick_start_server.sh`

---

## 🎯 Recommended: Use the Automated Script

```bash
./quick_start_server.sh
```

It handles everything for you! 🚀
