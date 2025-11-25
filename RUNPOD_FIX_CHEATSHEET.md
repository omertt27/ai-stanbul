# 🎯 RunPod LLM Server - Quick Cheatsheet

> **🚨 PreTrainedTokenizerFast Bug:** Add `use_fast=False` to fix the `'dict' object has no attribute 'model_type'` error!

## The Fix (Critical Line)
```python
# ❌ WRONG (causes error):
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config)

# ❌ ALSO WRONG (if using PreTrainedTokenizerFast):
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ✅ CORRECT (works for all tokenizers):
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True, 
    local_files_only=True,
    use_fast=False  # 🔥 Critical for PreTrainedTokenizerFast
)
```

## Create Server (All Steps)

1. **Navigate:**
   ```bash
   cd /workspace && rm -f llm_server.py
   ```

2. **Create file** using the 11 `cat` commands from `RUNPOD_TOKENIZER_FIX.md` (Steps 2-10)

3. **Start server:**
   ```bash
   python3 /workspace/llm_server.py
   ```

4. **Test:**
   ```bash
   curl http://YOUR_IP:8000/health
   ```

## Or Use The Shell Script

```bash
cd /workspace
bash RUNPOD_FIX_SERVER.sh
python3 llm_server.py
```

## Key Points

- **No config needed** for tokenizer
- **Separate loading:** Tokenizer doesn't need model config
- **Trust remote code:** Required for Llama models
- **Local files only:** Prevents accidental downloads

## Endpoints

- `GET /` - Status check
- `GET /health` - Health check
- `POST /chat` - Chat with LLM
  ```json
  {
    "message": "Hello!",
    "max_tokens": 512,
    "temperature": 0.7
  }
  ```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| EOF errors | Use single quotes: `'EOF1'` not `EOF1` |
| File exists | `rm -f /workspace/llm_server.py` |
| Can't connect | Check port 8000 is exposed in RunPod |
| Server crashes | Check: `nvidia-smi` and `ls /workspace/models/` |

## File Locations

- **Model:** `/workspace/models/Meta-Llama-3.1-8B-Instruct`
- **Server:** `/workspace/llm_server.py`
- **Full docs:** `RUNPOD_TOKENIZER_FIX.md`
- **Setup guide:** `RUNPOD_FIRST_LOGIN_COMMANDS.md`

---

**For detailed step-by-step instructions, see:** `RUNPOD_TOKENIZER_FIX.md`
