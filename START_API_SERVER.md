# 🚀 Start LLM API Server

## Manual Start (for testing)

```bash
# SSH into VM
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b

# Navigate to directory
cd ~/ai-stanbul

# Activate virtual environment
source venv/bin/activate

# Start the server
python llm_api_server.py
```

## Background Start (keeps running)

```bash
# SSH into VM
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b

# Navigate and activate
cd ~/ai-stanbul && source venv/bin/activate

# Run in background
nohup python llm_api_server.py > llm_api.log 2>&1 &

# Get process ID
echo $!

# Exit SSH (server keeps running)
exit

# Check logs from your Mac
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b --command="tail -50 ~/ai-stanbul/llm_api.log"
```

## Test from Your Mac

```bash
# Health check
curl http://35.210.251.24:8000/health

# Test chat
curl -X POST http://35.210.251.24:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the top 3 places to visit in Istanbul?", "max_tokens": 100}'
```

## Stop Server

```bash
# Find process
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b --command="ps aux | grep llm_api_server"

# Kill process (replace PID)
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b --command="pkill -f llm_api_server"
```
