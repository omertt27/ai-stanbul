#!/usr/bin/env python3
"""
Download smaller, CPU-friendly models for local testing
Later we'll use larger models on T4 GPU
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

def download_local_models():
    print("📥 Downloading CPU-friendly models for local testing...")
    print("=" * 60)
    
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    # 1. Small LLM for testing (1.1B params, works on CPU)
    print("\n1️⃣ Downloading TinyLlama (1.1B parameters)...")
    print("   This will take 2-3 minutes (~2.2GB download)")
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained("./models/tinyllama")
        model.save_pretrained("./models/tinyllama")
        print("   ✅ TinyLlama saved to ./models/tinyllama")
    except Exception as e:
        print(f"   ❌ Error downloading TinyLlama: {e}")
        print("   You can try again later or skip LLM testing")
    
    # 2. Semantic search model (same for CPU and GPU)
    print("\n2️⃣ Downloading semantic search model...")
    print("   This will take 1-2 minutes (~1GB download)")
    try:
        semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        semantic_model.save("./models/semantic-search")
        print("   ✅ Semantic search model saved to ./models/semantic-search")
    except Exception as e:
        print(f"   ❌ Error downloading semantic search model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All models downloaded for local development!")
    print("\n📦 Models saved to:")
    print("   - ./models/tinyllama (1.1B params, ~2.2GB)")
    print("   - ./models/semantic-search (~420MB)")
    print("\n📌 Next step: Run 'python scripts/index_database.py'")
    return True

if __name__ == "__main__":
    download_local_models()
