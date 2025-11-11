# HuggingFace Model Setup Guide for RunPod

**Complete guide to selecting, accessing, and deploying HuggingFace models on RunPod**

---

## 🎯 **Quick Start (5 Minutes)**

```bash
# 1. Go to HuggingFace and request model access
https://huggingface.co/meta-llama/Llama-3.1-8B
# Click "Request Access" → Accept license → Wait for approval

# 2. Get your token
https://huggingface.co/settings/tokens
# Create token → Copy (starts with hf_...)

# 3. Use in RunPod
# Environment Variable: HF_TOKEN=hf_xxxxxxxxxxxxx
# Model Name: meta-llama/Llama-3.1-8B
# Or full URL: https://huggingface.co/meta-llama/Llama-3.1-8B
```

---

## 📚 **Step 1: Request Model Access**

### **For Gated Models (Llama, Mistral, etc.)**

Some models require accepting a license before you can download them.

#### **Llama 3.1 8B (Recommended for AI Istanbul)**

1. **Go to model page:**
   - URL: https://huggingface.co/meta-llama/Llama-3.1-8B
   - Or search: "Llama 3.1 8B" on HuggingFace

2. **Request Access:**
   - Click the **"Request Access"** button (usually at the top of the page)
   - Read and accept **Meta's license agreement**
   - Fill out the form (usually just name and email)
   - Submit

3. **Wait for Approval:**
   - Llama 3.1: Usually **instant** approval ✅
   - Check your email for confirmation
   - Or refresh the model page - "Request Access" button will disappear

4. **Verify Access:**
   - Go back to the model page
   - You should see **"You have been granted access to this model"** ✅
   - Now you can download/use the model

---

## 🔑 **Step 2: Create HuggingFace Token**

### **Why Do You Need a Token?**

- Required to download **gated models** (like Llama)
- Used for authentication in RunPod
- Allows programmatic access to HuggingFace Hub

### **How to Create a Token:**

1. **Go to Token Settings:**
   - URL: https://huggingface.co/settings/tokens
   - Or: Profile → Settings → Access Tokens

2. **Create New Token:**
   - Click **"New token"** or **"Create new token"**
   - **Name:** `RunPod LLM Access` (or any descriptive name)
   - **Type:** Select **"Read"** (sufficient for downloading models)
   - Click **"Generate token"**

3. **Copy Your Token:**
   - Token format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - **⚠️ IMPORTANT:** Copy it now - you won't see it again!
   - Store it securely (password manager recommended)

4. **Test Your Token (Optional):**
   ```bash
   # Install HuggingFace CLI (if not already installed)
   pip install huggingface_hub
   
   # Login with your token
   huggingface-cli login --token hf_xxxxxxxxxxxxx
   
   # Verify
   huggingface-cli whoami
   # Should show your username
   ```

---

## 🔍 **Step 3: Choose Your Model**

### **Recommended Models for AI Istanbul**

| Model | Size | VRAM | Best For | HuggingFace Name |
|-------|------|------|----------|------------------|
| **Llama 3.1 8B** | 8B | ~10GB (4-bit) | General chat, Q&A | `meta-llama/Llama-3.1-8B` |
| **Llama 3.1 8B Instruct** | 8B | ~10GB (4-bit) | Instruction following | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral 7B | 7B | ~8GB (4-bit) | Fast inference | `mistralai/Mistral-7B-v0.1` |
| Mistral 7B Instruct | 7B | ~8GB (4-bit) | Chat, instructions | `mistralai/Mistral-7B-Instruct-v0.1` |
| Phi-3 Mini | 3.8B | ~5GB (4-bit) | Low cost, fast | `microsoft/Phi-3-mini-4k-instruct` |
| CodeLlama 7B | 7B | ~8GB (4-bit) | Code generation | `codellama/CodeLlama-7b-hf` |

**For AI Istanbul, we recommend: `meta-llama/Llama-3.1-8B-Instruct`** ✅

### **How to Find Models:**

1. **Browse HuggingFace:**
   - Go to: https://huggingface.co/models
   - Filter by:
     - Task: Text Generation
     - License: Open Source
     - Size: 7B-8B parameters

2. **Model Page Components:**
   - **Model Card:** Description, capabilities, limitations
   - **Model Name:** Top of page (e.g., `meta-llama/Llama-3.1-8B`)
   - **Files:** Model weights, configs
   - **Community:** Discussions, issues

3. **Check Model Requirements:**
   - **Size:** Larger = better quality but more VRAM
   - **License:** Ensure it allows commercial use (if needed)
   - **Quantization:** Check if 4-bit/8-bit versions available

---

## 📝 **Step 4: Model Name Formats**

### **All These Work! (RunPod is Smart)**

#### **Option 1: Short Name (Recommended) ✅**
```bash
MODEL_NAME=meta-llama/Llama-3.1-8B
```
- Clean and concise
- Easy to read in code
- Works everywhere

#### **Option 2: Full URL**
```bash
MODEL_NAME=https://huggingface.co/meta-llama/Llama-3.1-8B
```
- Copy-paste from browser
- RunPod automatically strips the domain
- Same result as Option 1

#### **Option 3: Just Copy From HuggingFace**
1. Go to any model page
2. Copy the URL from your browser
3. Paste into RunPod environment variable
4. RunPod handles the rest! ✅

**Example Transformation:**
```
Input:  https://huggingface.co/meta-llama/Llama-3.1-8B
Output: meta-llama/Llama-3.1-8B (automatically)
```

---

## 🚀 **Step 5: Use in RunPod**

### **Method 1: Environment Variables (Recommended)**

When deploying your RunPod pod:

1. **In RunPod Console:**
   - Go to: Deploy Pod → Environment Variables
   
2. **Add These Variables:**
   ```bash
   # Required: Your HuggingFace token
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
   # Optional: Model name (or specify in code)
   MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
   
   # Optional: Model configuration
   QUANTIZATION=4bit
   MAX_TOKENS=250
   ```

3. **Security Note:**
   - Use **RunPod Secrets** for `HF_TOKEN` (not plain environment variables)
   - Secrets are encrypted and not exposed in logs

### **Method 2: In Your Python Code**

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get model name from environment or use default
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# Handle full URLs (if user pastes from browser)
if MODEL_NAME.startswith("https://huggingface.co/"):
    MODEL_NAME = MODEL_NAME.replace("https://huggingface.co/", "")

print(f"Loading model: {MODEL_NAME}")

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)
```

---

## 🔧 **Step 6: Verify Setup**

### **Test HuggingFace Login**

```bash
# SSH into your RunPod pod
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519

# Login to HuggingFace
huggingface-cli login --token $HF_TOKEN

# Verify login
huggingface-cli whoami
# Output: Your HuggingFace username

# Check model access
huggingface-cli repo info meta-llama/Llama-3.1-8B-Instruct
# Should show model details, not "access denied"
```

### **Test Model Download**

```bash
# Try downloading model (will be cached for future use)
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('✅ Model access verified!')
"
```

---

## 🛠️ **Troubleshooting**

### **Issue 1: "Repository Not Found" or 403 Error**

**Cause:** Model access not granted or token invalid

**Fix:**
```bash
# 1. Verify you have access
# Go to: https://huggingface.co/meta-llama/Llama-3.1-8B
# Should show "You have been granted access"

# 2. Verify token is correct
huggingface-cli whoami
# Should show your username, not error

# 3. Re-request access if needed
# Visit model page → Request Access → Accept license

# 4. Try logging in again
huggingface-cli logout
huggingface-cli login --token $HF_TOKEN
```

---

### **Issue 2: "Token Not Found" in RunPod**

**Cause:** Environment variable not set or typo

**Fix:**
```bash
# Check if token is set
echo $HF_TOKEN
# Should show: hf_xxxxx

# If empty, set it manually
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Or add to your shell profile
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

---

### **Issue 3: "Model Too Large" / Out of Memory**

**Cause:** Model doesn't fit in GPU VRAM

**Fix:**
```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Or try a smaller model:**
- Llama 3.1 8B → Mistral 7B
- Mistral 7B → Phi-3 Mini (3.8B)

---

### **Issue 4: "Slow Download Speed"**

**Cause:** Large model files (8B models are ~15GB)

**Fix:**
```bash
# First download: ~5-15 minutes (normal)
# Subsequent runs: Instant (cached in /workspace)

# Check download progress
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', 
                  cache_dir='/workspace/models')
"

# Verify cache location
ls -lh /workspace/models/
```

---

## 📋 **Quick Reference**

| Action | Command/URL |
|--------|-------------|
| **Browse Models** | https://huggingface.co/models |
| **Request Access** | Model page → "Request Access" button |
| **Get Token** | https://huggingface.co/settings/tokens |
| **Login CLI** | `huggingface-cli login --token $HF_TOKEN` |
| **Check Login** | `huggingface-cli whoami` |
| **Model Info** | `huggingface-cli repo info <model-name>` |
| **Download Model** | `snapshot_download('<model-name>')` |

---

## 🎯 **Recommended Setup for AI Istanbul**

```yaml
Model: meta-llama/Llama-3.1-8B-Instruct
Token: hf_xxxxxxxxxxxxx (from HuggingFace settings)
Quantization: 4-bit NF4
GPU: RTX 4090 (24GB) or RTX A6000 (48GB)
VRAM Usage: ~10GB (with 4-bit quantization)
Cache Location: /workspace/models (persistent)
```

**Why Llama 3.1 8B Instruct?**
- ✅ Excellent at instruction following
- ✅ Good at Q&A (perfect for AI Istanbul chatbot)
- ✅ Fits in 24GB GPU with 4-bit quantization
- ✅ Fast inference (~1-2 seconds per response)
- ✅ Free and open source (Meta's license)
- ✅ Active community support

---

## 🔐 **Security Best Practices**

1. **Never Commit Tokens to Git:**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo "*.token" >> .gitignore
   ```

2. **Use RunPod Secrets:**
   - RunPod Console → Secrets → Add Secret
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxx`
   - ✅ Encrypted and not shown in logs

3. **Rotate Tokens Regularly:**
   - Delete old tokens: https://huggingface.co/settings/tokens
   - Create new ones every 3-6 months

4. **Use Read-Only Tokens:**
   - For model downloads, "Read" access is sufficient
   - "Write" access only needed for uploading models

---

## ✅ **Setup Checklist**

- [ ] HuggingFace account created
- [ ] Model access requested (e.g., Llama 3.1)
- [ ] Access granted (check model page)
- [ ] HuggingFace token created (Read access)
- [ ] Token saved securely
- [ ] Token added to RunPod Secrets
- [ ] Model name decided (`meta-llama/Llama-3.1-8B-Instruct`)
- [ ] RunPod pod deployed with HF_TOKEN
- [ ] `huggingface-cli whoami` works
- [ ] Model downloads successfully

---

## 📚 **Additional Resources**

- **HuggingFace Documentation:** https://huggingface.co/docs
- **Transformers Library:** https://huggingface.co/docs/transformers
- **Model Hub:** https://huggingface.co/models
- **License Info:** Check each model's license (usually Apache 2.0 or MIT)

---

**🎉 You're now ready to use any HuggingFace model on RunPod!**

For deployment instructions, see: [RUNPOD_DEPLOYMENT_GUIDE.md](./RUNPOD_DEPLOYMENT_GUIDE.md)
