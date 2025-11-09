# Google Cloud VM Setup - Quick Reference
## LLM API Server Deployment

**VM Instance:** llm-api-server  
**External IP:** 35.210.251.24  
**Zone:** europe-west1-b (Belgium)  
**Machine Type:** n4-standard-8 (8 vCPUs, 32 GB RAM)  
**Mode:** CPU-only (no GPU)

---

## 🚀 Quick Start (5 Steps)

### 1. SSH into VM

```bash
gcloud compute ssh llm-api-server --zone=europe-west1-b
```

### 2. Upload Required Files

From your **local machine**, upload the server files:

```bash
# Navigate to project directory
cd /Users/omer/Desktop/ai-stanbul

# Upload server file
gcloud compute scp llm_api_server.py llm-api-server:~/ai-istanbul/ --zone=europe-west1-b

# Upload startup script
gcloud compute scp start_llm_server.sh llm-api-server:~/ai-istanbul/ --zone=europe-west1-b

# Upload dependencies file (if you have requirements.txt)
gcloud compute scp requirements.txt llm-api-server:~/ai-istanbul/ --zone=europe-west1-b
```

### 3. Run Startup Script

Back on the **VM** (after SSH):

```bash
cd ~/ai-istanbul
chmod +x start_llm_server.sh
./start_llm_server.sh
```

### 4. Monitor Startup

```bash
# Watch the logs
tail -f /tmp/llm_api_server.log

# Check CPU usage
htop
# Or: top -u $(whoami)

# Check memory
free -h
```

### 5. Verify Server

From **local machine**:

```bash
curl http://35.210.251.24:8000/health
```

---

## 📁 File Upload Commands

### Upload Individual Files

```bash
# From local machine (macOS)
cd /Users/omer/Desktop/ai-stanbul

# Server file
gcloud compute scp llm_api_server.py llm-api-server:~/ai-istanbul/ --zone=europe-west1-b

# Startup script
gcloud compute scp start_llm_server.sh llm-api-server:~/ai-istanbul/ --zone=europe-west1-b

# Config files (if needed)
gcloud compute scp enhanced_llm_config.py llm-api-server:~/ai-istanbul/ --zone=europe-west1-b
gcloud compute scp llm_config.py llm-api-server:~/ai-istanbul/ --zone=europe-west1-b
```

### Upload Multiple Files at Once

```bash
gcloud compute scp llm_api_server.py start_llm_server.sh enhanced_llm_config.py llm-api-server:~/ai-istanbul/ --zone=europe-west1-b
```

### Upload Entire Directory

```bash
gcloud compute scp --recurse /Users/omer/Desktop/ai-stanbul llm-api-server:~/ --zone=europe-west1-b
```

---

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually without the startup script:

### 1. Create Virtual Environment

```bash
cd ~/ai-istanbul
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install transformers torch fastapi uvicorn psutil pydantic
```

### 3. Start Server

```bash
python llm_api_server.py
```

---

## 🎯 Verification Commands

### From Local Machine

```bash
# Health check
curl http://35.210.251.24:8000/health

# Root endpoint
curl http://35.210.251.24:8000/

# Metrics
curl http://35.210.251.24:8000/metrics

# Test generation
curl -X POST http://35.210.251.24:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the best places to visit in Istanbul?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### From VM (after SSH)

```bash
# Health check (local)
curl http://localhost:8000/health

# Check if server is running
ps aux | grep llm_api_server

# Check port
lsof -i :8000

# View logs
tail -f /tmp/llm_api_server.log

# Monitor resources
htop
# Or: top -u $(whoami)
```

---

## 📊 Monitoring Commands

### CPU Monitoring

```bash
# Real-time CPU usage
top -p $(pgrep -f llm_api_server)

# CPU info
lscpu

# Current CPU usage
mpstat 1 5
```

### Memory Monitoring

```bash
# Memory usage
free -h

# Detailed memory info
vmstat 1 5

# Per-process memory
ps aux --sort=-%mem | head
```

### Disk Space

```bash
# Disk usage
df -h

# Model cache size
du -sh ~/.cache/huggingface/
```

### Network

```bash
# Check if port is open
netstat -tuln | grep 8000

# Check connections
ss -tuln | grep 8000
```

---

## 🛠️ Troubleshooting Commands

### Server Won't Start

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Check if Python is available
which python3
python3 --version

# Check if venv is activated
which python  # Should show venv path

# Check dependencies
pip list | grep -E "transformers|torch|fastapi"
```

### Model Loading Issues

```bash
# Check Hugging Face cache
ls -lah ~/.cache/huggingface/hub/

# Check available space
df -h

# Clear cache (if needed)
rm -rf ~/.cache/huggingface/hub/models--meta-llama*

# Check internet connection
ping -c 3 huggingface.co
```

### Memory Issues

```bash
# Check memory usage
free -h

# Check swap
swapon --show

# Kill memory-heavy processes
ps aux --sort=-%mem | head
```

### Performance Issues

```bash
# Check CPU throttling
cat /proc/cpuinfo | grep MHz

# Check load average
uptime

# Check I/O wait
iostat -x 1 5
```

---

## 🔄 Server Management

### Start Server

```bash
# Using startup script
cd ~/ai-istanbul
./start_llm_server.sh

# Or manually
source venv/bin/activate
python llm_api_server.py
```

### Stop Server

```bash
# If running in foreground
# Press Ctrl+C

# If running in background
kill $(cat /tmp/llm_api_server.pid)

# Or force kill
pkill -f llm_api_server
```

### Restart Server

```bash
# Stop
pkill -f llm_api_server

# Wait a moment
sleep 2

# Start
./start_llm_server.sh
```

### View Logs

```bash
# Tail logs
tail -f /tmp/llm_api_server.log

# View last 100 lines
tail -n 100 /tmp/llm_api_server.log

# Search logs
grep -i error /tmp/llm_api_server.log
```

---

## 🔒 Firewall Configuration

### Check Firewall Rules

```bash
# From local machine
gcloud compute firewall-rules list | grep llm
```

### Add Firewall Rule (if needed)

```bash
gcloud compute firewall-rules create allow-llm-api \
  --allow tcp:8000 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow LLM API access"
```

### Check VM Network Tags

```bash
gcloud compute instances describe llm-api-server --zone=europe-west1-b --format="get(tags.items)"
```

---

## 📦 Dependencies

### Required Python Packages

```txt
transformers>=4.36.0
torch>=2.1.0
fastapi>=0.104.0
uvicorn>=0.24.0
psutil>=5.9.0
pydantic>=2.0.0
```

### Create requirements.txt

```bash
# On VM
cat > requirements.txt << EOF
transformers>=4.36.0
torch>=2.1.0
fastapi>=0.104.0
uvicorn>=0.24.0
psutil>=5.9.0
pydantic>=2.0.0
EOF

# Install from file
pip install -r requirements.txt
```

---

## 🚨 Common Issues

### Issue 1: "Port already in use"

```bash
# Solution
lsof -ti:8000 | xargs kill -9
```

### Issue 2: "Model download failed"

```bash
# Check internet
ping -c 3 huggingface.co

# Check disk space
df -h

# Manual download (if needed)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')"
```

### Issue 3: "Out of memory"

```bash
# Check available RAM
free -h

# Restart VM to clear memory
sudo reboot
```

### Issue 4: "Cannot connect from local machine"

```bash
# Check if server is running on VM
curl http://localhost:8000/health

# If works locally but not remotely, check firewall
gcloud compute firewall-rules list
```

---

## 📞 Quick Reference

| Command | Description |
|---------|-------------|
| `gcloud compute ssh llm-api-server --zone=europe-west1-b` | SSH into VM |
| `./start_llm_server.sh` | Start server |
| `tail -f /tmp/llm_api_server.log` | View logs |
| `curl http://35.210.251.24:8000/health` | Health check |
| `pkill -f llm_api_server` | Stop server |
| `htop` | Monitor resources |
| `free -h` | Check memory |
| `df -h` | Check disk space |

---

## 🎓 Next Steps

1. ✅ Upload files to VM
2. ✅ Run startup script
3. ✅ Verify server health
4. ✅ Run test suite (from local machine)
5. ✅ Monitor performance
6. ✅ Document results

**Ready? Start with Step 1!**
