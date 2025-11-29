# 🔑 Add SSH Key to RunPod

## Issue: Permission Denied (publickey)

Your SSH key needs to be added to RunPod before you can connect.

## Solution

### Step 1: Get Your Public Key

```bash
cat ~/.ssh/id_ed25519.pub
```

This will output something like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... your@email.com
```

**Copy the entire output** (select and Cmd+C)

### Step 2: Add Key to RunPod

1. Go to RunPod dashboard: https://www.runpod.io/console/user/settings
2. Click on **"SSH Public Keys"** in the left sidebar
3. Click **"Add SSH Key"**
4. Paste your public key
5. Give it a name (e.g., "Mac SSH Key")
6. Click **"Add Key"**

### Step 3: Restart Your Pod (Important!)

For the SSH key to take effect:
1. Go to your pod: https://www.runpod.io/console/pods
2. Click **"Stop"** on your pod
3. Wait for it to fully stop
4. Click **"Start"**
5. Wait for it to fully start (green status)

**Note:** When you restart the pod:
- The SSH port might change (check the pod page)
- The IP address might change
- vLLM will need to be restarted

### Step 4: Get New Connection Details

After restarting, check your pod page for the SSH connection command.

It should show something like:
```
ssh root@<IP_ADDRESS> -p <PORT>
```

**Or** use the RunPod proxy:
```
ssh <POD_ID>@ssh.runpod.io
```

### Step 5: Test Connection

```bash
# Try RunPod proxy
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519

# Or try direct (with updated IP/port from pod page)
ssh -p <PORT> root@<IP_ADDRESS> -i ~/.ssh/id_ed25519
```

If successful, you'll be logged into your RunPod instance!

## Alternative: Use RunPod Web Terminal

If SSH still doesn't work, you can use the web terminal:

1. Go to your pod page
2. Click **"Connect"** → **"Start Web Terminal"**
3. A browser-based terminal will open

You can then:
- Check if vLLM is running: `ps aux | grep vllm`
- Restart vLLM if needed
- Get the model path: `find /root/.cache -name "Llama*"`

## After SSH Access is Working

Once you can SSH in, restart vLLM:

```bash
# Kill old vLLM process
pkill -f vllm

# Start vLLM with optimized settings
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 1024 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  > /root/vllm.log 2>&1 &

# Wait a few seconds
sleep 10

# Test locally
curl http://localhost:8000/v1/models
```

## Then Create SSH Tunnel

From your Mac:

```bash
# Using RunPod proxy
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3

# Test tunnel
curl http://localhost:8000/v1/models
```

## Quick Reference

### Get Public Key
```bash
cat ~/.ssh/id_ed25519.pub
```

### Test SSH Connection
```bash
# Via proxy
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519

# Via direct (update IP/port)
ssh -p <PORT> root@<IP> -i ~/.ssh/id_ed25519
```

### Create Tunnel
```bash
ssh -f -N -L 8000:localhost:8000 \
  pvj233wwhiu6j3-64411542@ssh.runpod.io \
  -i ~/.ssh/id_ed25519
```

### Test Tunnel
```bash
curl http://localhost:8000/v1/models
```

---

**Need Help?** Check the RunPod documentation: https://docs.runpod.io/docs/ssh
