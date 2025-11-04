# 🦙 LLaMA 3.1 8B Q4 - Quick Reference

## 🏠 100% LOCAL LLM - Runs on YOUR Machine

✅ **No API calls** - Everything runs locally  
✅ **No internet needed** - After initial download  
✅ **No API costs** - Completely free  
✅ **Privacy** - Your data never leaves your machine  
✅ **Offline** - Works without internet connection  

**Metal M2 Pro:** Runs locally on your Mac's GPU  
**T4 GPU:** Runs locally on your cloud VM's GPU  
**No OpenAI, no Claude, no external services!**

---

## ⚡ One-Line Setup

```bash
python3 scripts/download_llama_models.py && python3 scripts/test_llm_metal.py
```

---

## 📋 Checklist

### Prerequisites (Only for Initial Download):
- [ ] HuggingFace account: https://huggingface.co/join
- [ ] LLAMA access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- [ ] Access token: https://huggingface.co/settings/tokens

**Note:** Token only needed ONCE to download model. After that, 100% local!

### Installation:

#### For Metal M2 Pro (Mac):
```bash
# 1. Verify PyTorch with Metal support
python3 -c "import torch; print('Metal MPS:', torch.backends.mps.is_available())"

# 2. Download model
python3 scripts/download_llama_models.py
# → Select option 1 (LLaMA 3.1 8B) or option 2 (LLaMA 3.2 3B)
# → Enter HuggingFace token

# 3. Test
python3 scripts/test_llm_metal.py

# 4. Start services
python3 ml_api_service.py         # Terminal 1
cd backend && python3 main.py     # Terminal 2
```

#### For T4 GPU (CUDA):
```bash
# 1. Install quantization dependencies
pip install bitsandbytes accelerate transformers torch

# 2. Verify CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. Download model
python3 scripts/download_llama_models.py
# → Select option 1 (LLaMA 3.1 8B Q4)
# → Enter HuggingFace token

# 4. Test
python3 scripts/test_llm_metal.py

# 5. Start services
python3 ml_api_service.py         # Terminal 1
cd backend && python3 main.py     # Terminal 2
```

**Note on Quantization:**
- **T4 GPU (CUDA):** Q4 quantization via `bitsandbytes` reduces 16GB → 4-5GB
- **Metal M2 Pro:** Uses FP32/FP16 (no Q4 in PyTorch/MPS, use full models)
- **Alternative:** Use `llama.cpp` with GGUF/GGML for native quantized inference on both

---

## 💡 Key Features

### LLaMA 3.1 8B Q4 - LOCAL MODEL:
- **🏠 Runs locally** on your Metal M2 Pro or T4 GPU
- **💰 Zero API costs** - No OpenAI/Claude fees
- **🔒 100% Private** - Data never leaves your machine
- **📡 Offline capable** - No internet needed after download
- **Size:** 4-5GB (70% smaller than full 16GB)
- **Quality:** ⭐⭐⭐⭐⭐ (98% of full precision)
- **Speed:** 5-8s (Metal M2 Pro), 3-5s (T4 GPU)
- **Memory:** Works with 8GB RAM/VRAM

### KAM Personality:
- Friendly Istanbul local guide 🏙️
- Bilingual (English & Turkish) 🌍
- Natural emojis 😊
- Marks favorites with **⭐ KAM Pick**
- Avoids politics/religion 🚫
- Istanbul-focused only 🇹🇷

---

## 🔥 Quick Commands

```bash
# Check devices
python3 -c "import torch; print('Metal:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"

# Check model
ls -lh models/llama-3.1-8b-q4/

# Download model
python3 scripts/download_llama_models.py

# Test LLM
python3 scripts/test_llm_metal.py

# Start ML service
python3 ml_api_service.py

# Test endpoint
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query":"Hi KAM!"}'
```

---

## 📊 Model Comparison

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **LLaMA 3.1 8B Q4** ⭐ | 5GB | 5-8s | ⭐⭐⭐⭐⭐ | **Production** |
| LLaMA 3.2 3B | 6GB | 3-6s | ⭐⭐⭐⭐ | Development |
| LLaMA 3.2 1B | 2.5GB | 2-4s | ⭐⭐⭐ | Testing |
| TinyLlama | 2GB | 2-3s | ⭐⭐ | Fallback |

---

## 🎯 Environment Detection

### Metal M2 Pro (Development):
```
Device: mps (Metal (Apple Silicon))
Memory: 6-16GB unified RAM (depending on model)
Speed: 5-8s per response (LLaMA 3.1 8B FP16)
Quantization: FP16/FP32 (PyTorch/MPS doesn't support Q4)
Note: Use llama.cpp/GGML for quantized inference on Metal
```

### T4 GPU (Production):
```
Device: cuda (NVIDIA GPU)
Memory: 4-5GB VRAM (with Q4), 16GB VRAM (full FP16)
Speed: 3-5s per response (Q4), 2-4s (FP16)
Quantization: ✅ Q4 (4-bit via bitsandbytes)
```

### CPU (Fallback):
```
Device: cpu
Memory: 6-8GB RAM
Speed: 20-40s per response
Quantization: FP32
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | `python3 scripts/download_llama_models.py` |
| Out of memory | Close apps, use smaller model, or restart |
| Access denied | Request access at huggingface.co/meta-llama |
| Quantization error | `pip install bitsandbytes accelerate` |
| Slow responses | Check device: Metal (5-8s), CPU (20-40s) normal |

---

## 📚 Documentation

- **Setup Guide:** `LLAMA_3_8B_Q4_SETUP_GUIDE.md`
- **Integration:** `LLM_METAL_INTEGRATION_COMPLETE.md`
- **Summary:** `LLAMA_INTEGRATION_SUMMARY.md`
- **Deployment:** `DEPLOYMENT_GUIDE_METAL_T4.md`

---

## 🎊 Example Conversation

**User:** "Hi KAM!"

**KAM:** "Hey there! 👋 How's your day going? Planning to explore anywhere in Istanbul today?"

**User:** "Best restaurants in Beşiktaş?"

**KAM:** "🍽️ Here are some great spots in Beşiktaş:
- ⭐ KAM Pick: Karadeniz Döner — iconic taste, big portions!
- Vogue Restaurant — rooftop with Bosphorus view
- Kayra Meyhane — live music and meze vibes
Want something local or modern?"

---

## ✅ Verification

```bash
# All checks passed?
✓ Metal/CUDA detected
✓ Model downloaded (models/llama-3.1-8b-q4/)
✓ Test script passed
✓ ML service starts
✓ Chat responses working

# You're ready! 🚀
```

---

**🦙 LLaMA 3.1 8B Q4 + Metal M2 Pro = Perfect Development Setup**
**🦙 LLaMA 3.1 8B Q4 + T4 GPU = Perfect Production Setup**
