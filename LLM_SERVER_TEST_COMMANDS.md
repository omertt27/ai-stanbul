# LLM Server Quick Test Commands

After starting the server, test these endpoints:

## 1. Health Check
```bash
curl http://localhost:8000/health | jq
```

## 2. Simple Completion Test
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "List the top 3 tourist attractions in Istanbul:",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq
```

## 3. Chat Completion Test (OpenAI Compatible)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful Istanbul travel guide."},
      {"role": "user", "content": "What are the must-visit places in Istanbul?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }' | jq
```

## 4. Multi-Intent Detection Test
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are an intent classifier. Analyze the user query and return a JSON object with detected intents. Format: {\"intents\": [{\"type\": \"restaurant\", \"confidence\": 0.9, \"details\": {...}}]}"
      },
      {
        "role": "user",
        "content": "I want to find a Turkish restaurant near Sultanahmet and then visit the Hagia Sophia"
      }
    ],
    "max_tokens": 200,
    "temperature": 0.3
  }' | jq
```

## 5. Monitor Logs
```bash
tail -f /workspace/logs/llm_server.log
```

## 6. Check GPU Usage (if applicable)
```bash
nvidia-smi
```

## 7. Check Server Process
```bash
ps aux | grep llm_server
```

## 8. Stop Server
```bash
# Using PID file
kill $(cat /workspace/llm_server.pid)

# Or kill by name
pkill -f llm_server.py
```

## Expected Response Format

### Health Check Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "memory_gb": 5.5,
  "uptime_seconds": 120.5
}
```

### Completion Response:
```json
{
  "text": "1. Hagia Sophia - A magnificent Byzantine church...",
  "tokens_generated": 85,
  "generation_time": 2.3
}
```

### Chat Completion Response:
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Istanbul offers incredible attractions..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 120,
    "total_tokens": 165
  }
}
```

## Troubleshooting

### Server not responding?
```bash
# Check logs
tail -n 50 /workspace/logs/llm_server.log

# Check if process is running
ps aux | grep llm_server

# Check port
netstat -tlnp | grep 8000
```

### Out of memory?
```bash
# Check GPU memory
nvidia-smi

# Restart server with smaller batch size
# (Edit llm_server.py to reduce max_tokens)
```

### Slow responses?
- First request is always slower (model warmup)
- Check GPU utilization with nvidia-smi
- Reduce max_tokens for faster responses
- Lower temperature for more deterministic output
