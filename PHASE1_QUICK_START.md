# 🚀 Quick Start Guide - Phase 1 (Local Development)

## ✅ What We Just Created

All the necessary files for Phase 1 (local development) have been created:

```
ai-stanbul/
├── ml_systems/
│   ├── local_llm_generator.py       ← LLM (works on CPU & GPU)
│   └── semantic_search_engine.py    ← Semantic search
├── scripts/
│   ├── download_models_local.py     ← Download models
│   ├── index_database.py            ← Index database
│   └── test_local_system.py         ← Test everything
├── data/                             ← Will store indexes
├── models/                           ← Will store models
└── setup_local_ml.sh                ← Setup script
```

---

## 🎯 Step-by-Step Instructions

### Step 1: Run Setup Script

```bash
cd /Users/omer/Desktop/ai-stanbul
chmod +x setup_local_ml.sh
./setup_local_ml.sh
```

This will:
- Create virtual environment `venv_ml`
- Install all dependencies (~5-10 minutes)
- Set up directories

### Step 2: Activate Virtual Environment

```bash
source venv_ml/bin/activate
```

You should see `(venv_ml)` in your terminal prompt.

### Step 3: Download Models

```bash
python scripts/download_models_local.py
```

This downloads:
- **TinyLlama** (1.1B params, ~2.2GB) - For local testing
- **Semantic Search Model** (~1GB) - For similarity search

**Time:** 2-3 minutes depending on internet speed

### Step 4: Index Your Database

```bash
python scripts/index_database.py
```

This will:
- Load all restaurants and attractions from `istanbul_ai.db`
- Create semantic embeddings
- Build FAISS index for fast search

**Time:** 1-2 minutes

### Step 5: Test the System

```bash
python scripts/test_local_system.py
```

This runs two tests:

**Test 1: Semantic Search** (Fast, <1s per query)
- Tests similarity search
- Should work perfectly on CPU

**Test 2: LLM Generation** (Slow, 30-90s on CPU)
- Tests text generation with TinyLlama
- ⚠️ Will be MUCH faster on T4 GPU (2-4s)

---

## 🧪 What to Expect

### ✅ Semantic Search (Test 1)

```
Query: 'romantic restaurant with bosphorus view'
  1. Mikla - Beyoğlu (score: 0.856)
  2. Sunset Grill & Bar - Beşiktaş (score: 0.823)
  3. Vogue Restaurant - Beşiktaş (score: 0.801)
  Time: 0.234s
```

### ⏳ LLM Generation (Test 2) - SLOW ON CPU

```
Query: 'What are the best restaurants in Beyoğlu?'
  Generating response...
  ⏳ Please wait 30-90 seconds...
  
  Response (generated in 45.2s):
  Based on the information provided, here are some excellent 
  restaurants in Beyoğlu:
  
  1. Mikla offers modern Turkish cuisine with stunning city views...
  2. Çiya Sofrası serves authentic Anatolian dishes...
  3. ...
```

---

## 📊 Performance Comparison

| Metric | CPU (Local) | T4 GPU (Phase 2) |
|--------|-------------|------------------|
| Semantic Search | <1s | <0.5s |
| LLM Generation | 30-90s | 2-4s |
| Model Size | 1.1B params | 3B params |
| Quality | Good | Excellent |
| Cost | $0 | ~$0.35/hour |

---

## ✅ Success Criteria

After running tests, you should see:

```
🎉 Phase 1 (Local Development) is working!

📌 Next steps:
  1. ✅ Semantic search works perfectly on CPU
  2. ✅ LLM generation works (slow but functional)
  3. 🚀 When ready, deploy to T4 GPU for 10-20x faster LLM
```

---

## 🐛 Troubleshooting

### Problem: "Model not found"
**Solution:** Run `python scripts/download_models_local.py`

### Problem: "Index file not found"
**Solution:** Run `python scripts/index_database.py`

### Problem: "Database not found"
**Solution:** Make sure you're in the `ai-stanbul` directory and `istanbul_ai.db` exists

### Problem: LLM generation is very slow
**This is normal!** CPU is 10-20x slower than GPU. This will be fast on T4.

### Problem: Import errors
**Solution:** Make sure virtual environment is activated:
```bash
source venv_ml/bin/activate
```

---

## 🚀 Next: Phase 2 (T4 GPU Deployment)

Once everything works locally, you can deploy to Google Cloud with T4 GPU:

1. **Create T4 instance** on Google Cloud
2. **Upload code and indexed data**
3. **Download production models** (Llama 3.2 3B)
4. **Start ML server** on T4
5. **Connect your backend** to ML service

See `LOCAL_THEN_T4_DEPLOYMENT_GUIDE.md` for full instructions.

---

## 💰 Cost Summary

**Phase 1 (Local):** $0 - Everything runs on your Mac
**Phase 2 (T4 GPU):** ~$0.35/hour when running
- Development/testing: ~$1-5 (start/stop as needed)
- Production 24/7: ~$252/month

**Recommendation:** Develop locally (Phase 1), deploy to T4 only when ready for production.

---

## 🎉 You're Ready!

Run the commands above and let me know if you encounter any issues!

```bash
# Quick start (copy-paste this):
cd /Users/omer/Desktop/ai-stanbul
chmod +x setup_local_ml.sh
./setup_local_ml.sh
source venv_ml/bin/activate
python scripts/download_models_local.py
python scripts/index_database.py
python scripts/test_local_system.py
```
