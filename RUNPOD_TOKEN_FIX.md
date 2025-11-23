# 🚨 QUICK FIX - Token Error

You're seeing **"Invalid user token"** because you need to use YOUR actual Hugging Face token.

## ✅ **What To Do RIGHT NOW:**

### **Step 1: Get Your Token**
1. Open this link: **https://huggingface.co/settings/tokens**
2. Click **"New token"** button
3. Name it: `runpod`
4. Click **"Generate"**
5. **COPY THE TOKEN** (looks like: `hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890`)

### **Step 2: Login in RunPod Terminal**

**DON'T copy this literally - replace the token part with YOUR token:**

```bash
huggingface-cli login --token hf_YOUR_ACTUAL_TOKEN_FROM_STEP_1
```

**Example with a REAL token (yours will be different):**
```bash
huggingface-cli login --token hf_BwGkLmNoPqRsTuVwXyZ1234567890EXAMPLE
```

**✅ You should see:**
```
Token is valid (permission: read).
Login successful
```

---

## 📸 **Visual Guide:**

### What your Hugging Face token page looks like:
```
┌─────────────────────────────────────────┐
│  Hugging Face - Access Tokens          │
├─────────────────────────────────────────┤
│  [+ New token]                          │
│                                         │
│  Name: runpod                           │
│  Token: hf_BwGk...890 [Copy]           │  ← COPY THIS!
│  Created: Nov 23, 2025                  │
└─────────────────────────────────────────┘
```

### What to paste in RunPod terminal:
```bash
huggingface-cli login --token hf_BwGkLmNoPqRsTuVwXyZ1234567890EXAMPLE
#                              ↑
#                              YOUR TOKEN HERE (not literally "YOUR_HF_TOKEN_HERE")
```

---

## 🐛 **Still Getting "Invalid user token"?**

### Check these:
1. ✅ Did you replace `YOUR_HF_TOKEN_HERE` with your actual token?
2. ✅ Does your token start with `hf_`?
3. ✅ Did you copy the ENTIRE token (usually 40+ characters)?
4. ✅ No extra spaces before/after the token?

### Try this alternative method:
```bash
# Use the new command (ignore if it doesn't work)
huggingface-cli whoami
```

Or manually create the token file:
```bash
mkdir -p ~/.cache/huggingface
echo "hf_YOUR_ACTUAL_TOKEN" > ~/.cache/huggingface/token
```

---

## ✅ **Once Login Works, Continue:**

After successful login, run these commands:

```bash
# Update system
apt-get update -y && apt-get upgrade -y

# Install Python tools
apt-get install -y python3-pip git wget curl nano

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ML libraries
pip3 install transformers accelerate bitsandbytes scipy fastapi uvicorn pydantic

# Check GPU
nvidia-smi
```

**Then continue with Step 5 in `RUNPOD_5_MINUTE_SETUP.md`**

---

## 🆘 **Still Stuck?**

1. Share what you see after running the login command
2. Check your token is active: https://huggingface.co/settings/tokens
3. Try generating a NEW token and use that one

**Need the token for Llama 3.1 8B because it's a gated model that requires authentication!**
