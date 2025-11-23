# 🔧 RunPod LLM Server Troubleshooting Guide

**For Pod:** `ytc61lal7ag5sy`  
**Last Updated:** January 2025

---

## 🚀 Quick Start

### Option 1: Use Helper Script (Recommended)
```bash
chmod +x runpod_ssh_helper.sh
./runpod_ssh_helper.sh
```

This interactive menu lets you:
- Connect via SSH
- Check server status
- View logs
- Restart server

### Option 2: Manual SSH
```bash
# Standard SSH (recommended)
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519

# Or direct TCP (for file transfers)
ssh root@194.68.245.173 -p 22001 -i ~/.ssh/id_ed25519
```

---

## 🔍 Common Issues & Solutions

### Issue 1: "Connection refused" when testing LLM endpoint

**Symptoms:**
```bash
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health
# Returns: Connection refused or timeout
```

**Solutions:**

1. **Check if server is running:**
   ```bash
   ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
   ps aux | grep python
   ```
   
   Look for: `python llm_api_server_4bit.py`

2. **If not running, start it:**
   ```bash
   cd /workspace
   python llm_api_server_4bit.py > server.log 2>&1 &
   ```

3. **Wait 15-20 seconds** for model to load, then test:
   ```bash
   curl http://localhost:8888/health
   ```

4. **Check logs for errors:**
   ```bash
   tail -50 /workspace/server.log
   ```

---

### Issue 2: Pod is sleeping or stopped

**Symptoms:**
- SSH connection fails
- All endpoints timeout
- RunPod console shows "Stopped"

**Solutions:**

1. **Go to RunPod Console:**
   ```
   https://www.runpod.io/console/pods
   ```

2. **Find your pod:** `ytc61lal7ag5sy`

3. **Click "Start"** if stopped

4. **Wait 2-3 minutes** for pod to boot

5. **SSH in and restart server:**
   ```bash
   ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
   cd /workspace
   python llm_api_server_4bit.py > server.log 2>&1 &
   ```

---

### Issue 3: Server crashes or GPU out of memory

**Symptoms:**
```bash
# In logs:
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Check GPU memory:**
   ```bash
   nvidia-smi
   ```

2. **Clear GPU cache and restart:**
   ```bash
   # Kill existing server
   pkill -f "python.*llm_api_server"
   
   # Clear Python cache
   python3 -c "import torch; torch.cuda.empty_cache()"
   
   # Restart server
   cd /workspace
   python llm_api_server_4bit.py > server.log 2>&1 &
   ```

3. **If still failing, use smaller max_tokens:**
   Edit `llm_api_server_4bit.py`:
   ```python
   max_tokens: int = 100  # Reduce from 250
   ```

---

### Issue 4: Slow response times (> 10 seconds)

**Symptoms:**
- First request is fast
- Subsequent requests are slow
- High GPU utilization

**Solutions:**

1. **Check GPU temperature:**
   ```bash
   nvidia-smi
   ```
   If temp > 85°C, GPU may be throttling

2. **Check other processes using GPU:**
   ```bash
   nvidia-smi
   ```
   Look for other Python processes

3. **Restart with lower temperature:**
   ```bash
   pkill -f "python.*llm_api_server"
   sleep 5
   cd /workspace
   python llm_api_server_4bit.py > server.log 2>&1 &
   ```

---

### Issue 5: SSH key permission denied

**Symptoms:**
```bash
Permission denied (publickey)
```

**Solutions:**

1. **Check key permissions:**
   ```bash
   ls -la ~/.ssh/id_ed25519
   ```
   Should be: `-rw-------` (600)

2. **Fix permissions if needed:**
   ```bash
   chmod 600 ~/.ssh/id_ed25519
   ```

3. **Verify key is added to RunPod:**
   - Go to https://www.runpod.io/console/user/settings
   - Check "SSH Public Keys" section
   - Add your public key if missing:
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```

---

## 📊 Health Check Commands

### Basic Health Check
```bash
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health
```

### Detailed Status Check
```bash
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519 << 'EOF'
echo "=== Server Status ==="
ps aux | grep python | grep -v grep

echo ""
echo "=== Local Health ==="
curl -s http://localhost:8888/health | jq || curl -s http://localhost:8888/health

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "=== Disk Usage ==="
df -h /workspace

echo ""
echo "=== Recent Logs ==="
tail -20 /workspace/server.log
EOF
```

### Test Generation
```bash
curl -X POST https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 20
  }'
```

---

## 🔄 Server Restart Procedures

### Standard Restart
```bash
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
pkill -f "python.*llm_api_server"
cd /workspace
python llm_api_server_4bit.py > server.log 2>&1 &
```

### Full Reset (if server is misbehaving)
```bash
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519 << 'EOF'
# Kill all Python processes
pkill -f python
sleep 3

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Restart server
cd /workspace
nohup python llm_api_server_4bit.py > server.log 2>&1 &

# Wait and test
sleep 15
curl http://localhost:8888/health
EOF
```

---

## 📁 Important Files & Paths

### On RunPod Pod
```
/workspace/llm_api_server_4bit.py    # Main server file
/workspace/server.log                 # Server logs
/workspace/models/                    # Model files
~/.cache/huggingface/                 # HuggingFace cache
```

### View/Edit Files
```bash
# View server code
cat /workspace/llm_api_server_4bit.py

# View logs
tail -f /workspace/server.log

# Edit server (if needed)
nano /workspace/llm_api_server_4bit.py
```

---

## 🎯 Performance Optimization

### Check Current Performance
```bash
# Response time test
time curl -s https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health > /dev/null
```

### GPU Memory Optimization
```bash
ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
nvidia-smi

# If using > 90% GPU memory, reduce batch size or max_tokens
```

### Reduce Cold Start Time
Keep pod warm by pinging it regularly:
```bash
# Add to cron (every 5 minutes)
*/5 * * * * curl -s https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health > /dev/null 2>&1
```

---

## 🚨 Emergency Procedures

### Complete System Failure
1. **Stop pod in RunPod console**
2. **Wait 1 minute**
3. **Start pod**
4. **Wait 2-3 minutes for boot**
5. **SSH in and restart server:**
   ```bash
   ssh ytc61lal7ag5sy-64410fe8@ssh.runpod.io -i ~/.ssh/id_ed25519
   cd /workspace
   python llm_api_server_4bit.py > server.log 2>&1 &
   ```

### Pod Won't Start
1. **Check RunPod balance/credits**
2. **Check for maintenance:** https://status.runpod.io
3. **Contact RunPod support**

---

## 📞 Support Resources

- **RunPod Console:** https://www.runpod.io/console/pods
- **RunPod Status:** https://status.runpod.io
- **RunPod Discord:** https://discord.gg/runpod
- **Helper Script:** `./runpod_ssh_helper.sh`

---

## ✅ Verification Checklist

After any troubleshooting, verify:

- [ ] SSH connection works
- [ ] Server process is running (`ps aux | grep python`)
- [ ] Local health check works (`curl http://localhost:8888/health`)
- [ ] External health check works (`curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/health`)
- [ ] Generation test succeeds (see test command above)
- [ ] Response time < 5 seconds
- [ ] GPU memory < 90% (`nvidia-smi`)
- [ ] No errors in logs (`tail /workspace/server.log`)

---

**Last Updated:** January 2025  
**Status:** Ready for production
