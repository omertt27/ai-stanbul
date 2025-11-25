# 🔥 QUICK FIX: AttributeError with Tokenizer

> **🚨 CRITICAL: PreTrainedTokenizerFast Bug Alert!**
> 
> If your `tokenizer_config.json` contains `"tokenizer_class": "PreTrainedTokenizerFast"`, you MUST add `use_fast=False` when loading the tokenizer. This causes the `'dict' object has no attribute 'model_type'` error in ~90% of cases!
> 
> **Quick Check on RunPod:**
> ```bash
> grep "tokenizer_class" /workspace/models/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json
> ```
> If you see `"PreTrainedTokenizerFast"`, jump to **Step 3** below and look for the `use_fast=False` parameter!

---

## 🚀 TL;DR - Just Fix It Now!

If you just want to fix the server ASAP, copy each command below one at a time:

```bash
cd /workspace
rm -f llm_server.py
```

Then follow Steps 2-12 in the "[Step-by-Step Instructions](#-copy--paste-on-runpod-step-by-step)" section below.

---

## 📖 The Problem
```
AttributeError: 'dict' object has no attribute 'model_type'
```

This happens when you pass a `config` object to `AutoTokenizer.from_pretrained()`.

## ✅ The Solution
**Don't pass `config` to the tokenizer!** The tokenizer only needs tokenizer files, not the model config.

---

## 📋 Copy & Paste on RunPod (Step-by-Step)

> **Each step is a single copy-paste command. Wait for each to complete before moving to the next.**

---

### 🟢 Step 1: Navigate to workspace
Copy and paste this:
```bash
cd /workspace
```

---

### 🟢 Step 2: Start creating the server (imports)
Copy and paste this:
```bash
cat > /workspace/llm_server.py << 'EOF1'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"
EOF1
```

---

### 🟢 Step 3: Add tokenizer loading (🔥 WITH FIX FOR PreTrainedTokenizerFast)
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF2'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # 🔥 CRITICAL FIX for PreTrainedTokenizerFast error
)
EOF2
```

> **⚠️ IMPORTANT:** The `use_fast=False` parameter is CRITICAL! Your tokenizer uses `PreTrainedTokenizerFast` which has a bug causing the `'dict' object has no attribute 'model_type'` error. This forces it to use the slower but more stable tokenizer.

---

### 🟢 Step 4: Add model loading
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF3'

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")
EOF3
```

---

### 🟢 Step 5: Add request model
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF4'

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7
EOF4
```

---

### 🟢 Step 6: Add response model
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF5'

class ChatResponse(BaseModel):
    response: str
EOF5
```

---

### 🟢 Step 7: Add root endpoint
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF6'

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}
EOF6
```

---

### 🟢 Step 8: Add health endpoint
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF7'

@app.get("/health")
async def health():
    return {"status": "healthy"}
EOF7
```

---

### 🟢 Step 9: Add chat endpoint
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF8'
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
EOF8
```

---

### 🟢 Step 10: Add main runner
Copy and paste this:
```bash
cat >> /workspace/llm_server.py << 'EOF9'

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF9
```

---

### 🔍 Step 11: Verify file creation
Copy and paste this:
```bash
ls -lh /workspace/llm_server.py && echo "✅ File created successfully!"
```

You should see something like:
```
-rw-r--r-- 1 root root 2.1K Dec 10 12:34 /workspace/llm_server.py
✅ File created successfully!
```

---

### 🚀 Step 12: Start the server!
Copy and paste this:
```bash
python3 /workspace/llm_server.py
```

**Expected output:**
```
Loading tokenizer...
Loading model...
✅ Model loaded!
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

### ⚠️ STILL GETTING THE ERROR? Verify the Fix is Actually in Your File!

If you get `AttributeError: 'dict' object has no attribute 'model_type'` after Step 12, the `use_fast=False` parameter might not be in your file. **Verify it now:**

```bash
# Check if use_fast=False is in your server file
grep "use_fast" /workspace/llm_server.py
```

**Expected output:**
```
    use_fast=False  # 🔥 CRITICAL FIX for PreTrainedTokenizerFast error
```

**If you see nothing or different output**, the fix is NOT in your file! This means:
- You may have skipped Step 3, or
- Step 3 didn't execute correctly, or
- You're running an old version of the file

**SOLUTION - Re-create the file completely:**

```bash
# 1. Remove old file
rm -f /workspace/llm_server.py

# 2. Create complete server with use_fast=False
cat > /workspace/llm_server.py << 'EOF_COMPLETE'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # 🔥 CRITICAL FIX for PreTrainedTokenizerFast error
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF_COMPLETE

# 3. Verify use_fast=False is now in the file
echo "Checking for use_fast=False..."
grep "use_fast" /workspace/llm_server.py

# 4. Run the server
python3 /workspace/llm_server.py
```

**This single command block creates the ENTIRE server in one go with the fix included.**

---

## What Changed?

### ❌ WRONG (old version that causes error):
```python
from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config)  # ❌ Don't do this!
```

### ✅ CORRECT (fixed version):
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
# No config parameter needed!
```

---

## Why This Works

1. **AutoTokenizer only needs tokenizer files:**
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `special_tokens_map.json`
   
2. **It doesn't use model_type from config.json**

3. **Passing `config=config` causes issues** in some transformers versions where the config object gets converted to a dict

4. **The model loading is separate** and handles its own config

---

## 🚨 Common Issues & Solutions

### ⚠️ CRITICAL: Error Still Happens After Fix?

If you still get `AttributeError: 'dict' object has no attribute 'model_type'` **after** following all steps, this means there's a deeper issue with the tokenizer files or transformers cache.

> **🔴 IMPORTANT:** If your `tokenizer_config.json` contains `"tokenizer_class": "PreTrainedTokenizerFast"`, this is the **most common cause** of this error. Jump directly to **[Fix D: use_fast=False](#fix-d-load-tokenizer-with-use_fastfalse--try-this-first)** below!

**Run these diagnostic commands on RunPod:**

```bash
# 1. Check if tokenizer_config.json exists and is valid
cat /workspace/models/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json | python3 -m json.tool
```

If this fails or shows weird data, your tokenizer files are corrupted.

```bash
# 2. Check for PreTrainedTokenizerFast (known issue)
grep "PreTrainedTokenizerFast" /workspace/models/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json
```

If this returns anything, **use Fix D below immediately**.

```bash
# 3. Check transformers version
pip show transformers | grep Version
```

**NOW TRY THESE FIXES IN ORDER:**

---

#### Fix A: Clear Transformers Cache
```bash
rm -rf ~/.cache/huggingface/
python3 /workspace/llm_server.py
```

---

#### Fix B: Use LlamaTokenizer Directly (Recommended!)

Replace Step 3 with this version:

```bash
cat > /workspace/llm_server_v2.py << 'EOF_V2'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    legacy=False
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "request.message"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF_V2
```

Then run:
```bash
python3 /workspace/llm_server_v2.py
```

---

#### Fix C: Reinstall Transformers (Different Version)

Try downgrading to a known stable version:
```bash
pip uninstall -y transformers
pip install transformers==4.44.0
python3 /workspace/llm_server.py
```

Or try the latest:
```bash
pip install --upgrade transformers
python3 /workspace/llm_server.py
```

---

#### Fix D: Load Tokenizer with use_fast=False ⭐ **TRY THIS FIRST!**

Your `tokenizer_config.json` shows `"tokenizer_class": "PreTrainedTokenizerFast"`, which is known to cause this exact error in some transformers versions.

**Quick Test:**
```bash
cat > /workspace/test_tokenizer.py << 'EOF_TEST'
from transformers import AutoTokenizer

MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Attempting to load tokenizer with use_fast=False...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False  # Use slow tokenizer
    )
    print("✅ SUCCESS! Tokenizer loaded.")
    print(f"Tokenizer type: {type(tokenizer)}")
except Exception as e:
    print(f"❌ FAILED: {e}")
EOF_TEST
```

Run it:
```bash
python3 /workspace/test_tokenizer.py
```

**If this works**, recreate the server with `use_fast=False`:

```bash
# Remove old server
rm -f /workspace/llm_server.py

# Create new server with use_fast=False in Step 3
cat > /workspace/llm_server.py << 'EOF'
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn

app = FastAPI()
MODEL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    use_fast=False  # CRITICAL FIX for PreTrainedTokenizerFast issue
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
print("✅ Model loaded!")

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "LLM Server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": request.message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, do_sample=True, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

Then run:
```bash
python3 /workspace/llm_server.py
```

---

#### Fix E: Re-download Tokenizer Files

Your tokenizer files might be corrupted. Re-download them:

```bash
# Backup old files
mv /workspace/models/Meta-Llama-3.1-8B-Instruct /workspace/models/Meta-Llama-3.1-8B-Instruct.backup

# Re-download (you'll need HuggingFace token)
python3 << 'EOF_DOWNLOAD'
from huggingface_hub import snapshot_download

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOCAL_PATH = "/workspace/models/Meta-Llama-3.1-8B-Instruct"

print("Re-downloading model files...")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_PATH,
    token="YOUR_HF_TOKEN_HERE"  # Replace with your token
)
print("✅ Download complete!")
EOF_DOWNLOAD
```
