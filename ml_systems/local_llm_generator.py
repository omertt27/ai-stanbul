"""
LLM Generator with Q4 Quantization Support
Supports LLaMA 3.1 8B Q4 for Metal M2 Pro and T    def _detect_best_model(self):
        """Auto-detect best available LLAMA model"""
        import os
        
        # Priority order: 8B Q4 > 3B > 1B > 8B full > tinyllama (fallback)
        models = [
            ("./models/llama-3.1-8b-q4", "LLaMA 3.1 8B Q4 (RECOMMENDED - Quantized)"),
            ("./models/llama-3.2-3b", "LLaMA 3.2 3B"),
            ("./models/llama-3.2-1b", "LLaMA 3.2 1B (Fast)"),
            ("./models/llama-3.1-8b", "LLaMA 3.1 8B (Full Precision)"),
            ("./models/tinyllama", "TinyLlama (Fallback)")
        ]
        
        for path, name in models:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                print(f"✅ Found: {name}")
                return path
        
        print("⚠️  No LLAMA model found!")
        print("   Please run: python3 scripts/download_llama_models.py")
        print("   Recommended: Download LLaMA 3.1 8B Q4 for best quality")
        raise FileNotFoundError("No LLAMA model available. Run download_llama_models.py first.") detects and uses best available device
Supports: CPU, Metal (MPS), CUDA (T4/A100)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

class LocalLLMGenerator:
    def __init__(self, model_path=None, use_gpu=None):
        """
        Initialize LLAMA LLM Generator
        
        Args:
            model_path: Path to LLAMA model (default: auto-detect best available)
            use_gpu: Force GPU usage (True/False/None for auto)
        """
        print("🔄 Loading LLAMA LLM...")
        
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = self._detect_best_model()
        
        self.model_path = model_path
        print(f"📦 Model: {model_path}")
        
        # Auto-detect best available device
        if use_gpu is None:
            # Priority: CUDA (T4 GPU) > MPS (Metal/M2) > CPU
            if torch.cuda.is_available():
                self.device = "cuda"
                self.device_type = "CUDA (NVIDIA GPU)"
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.device_type = "Metal (Apple Silicon)"
            else:
                self.device = "cpu"
                self.device_type = "CPU"
        else:
            # Manual override
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                self.device_type = "CUDA (NVIDIA GPU)"
            elif use_gpu and torch.backends.mps.is_available():
                self.device = "mps"
                self.device_type = "Metal (Apple Silicon)"
            else:
                self.device = "cpu"
                self.device_type = "CPU"
        
        print(f"📍 Using device: {self.device} ({self.device_type})")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine optimal dtype based on device
        if self.device == "cuda":
            # T4 GPU: Use float16 for faster inference
            dtype = torch.float16
        elif self.device == "mps":
            # Metal: Use float32 (MPS has better float32 support)
            dtype = torch.float32
        else:
            # CPU: Use float32
            dtype = torch.float32
        
        print(f"📊 Loading model with dtype: {dtype}")
        
        # Check if Q4 quantization should be used (CUDA only, for 8B models)
        use_quantization = (
            self.device == "cuda" and 
            "8b" in model_path.lower() and 
            ("q4" in model_path.lower() or self._should_quantize_8b())
        )
        
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig
                print("📊 Using 4-bit quantization (reduces memory by 75%)")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                print("✅ Model loaded with 4-bit quantization")
                
            except ImportError:
                print("⚠️  bitsandbytes not available, loading without quantization")
                print("   Install: pip install bitsandbytes")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
        else:
            # Standard loading (Metal, CPU, or smaller models)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
        
        self.model.eval()
        
        print(f"✅ LLM loaded on {self.device} ({self.device_type})")
        
        # Show memory usage
        if self.device == "cuda":
            print(f"📊 VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
            print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        elif self.device == "mps":
            print(f"📊 Using Apple Silicon GPU acceleration")
        else:
            print(f"📊 Using CPU (consider GPU for faster inference)")
    
    def _should_quantize_8b(self):
        """Check if 8B model should be quantized"""
        # Auto-enable quantization for 8B models on CUDA to save memory
        return self.device == "cuda"
    
    def _detect_best_model(self):
        """Auto-detect best available LLAMA model"""
        import os
        
        # Priority order: 3B > 1B > 8B (8B is slower) > tinyllama (fallback)
        models = [
            ("./models/llama-3.2-3b", "Llama-3.2-3B (Recommended)"),
            ("./models/llama-3.2-1b", "Llama-3.2-1B (Fast)"),
            ("./models/llama-3.1-8b", "Llama-3.1-8B (Best Quality)"),
            ("./models/tinyllama", "TinyLlama (Fallback)")
        ]
        
        for path, name in models:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                print(f"✅ Found: {name}")
                return path
        
        print("⚠️  No LLAMA model found! Please run: python3 scripts/download_llama_models.py")
        raise FileNotFoundError("No LLAMA model available. Run download_llama_models.py first.")
    
    def generate(self, query, context_data, max_tokens=300):
        """Generate response (works on CPU or GPU)"""
        
        # Format prompt
        prompt = self._create_prompt(query, context_data)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        elif "Assistant:" in response:
            answer = response.split("Assistant:")[-1].strip()
        else:
            # Remove the prompt from response
            answer = response[len(prompt):].strip()
        
        return answer
    
    def _create_prompt(self, query, context_data):
        """Create structured prompt with KAM personality"""
        context_str = self._format_context(context_data)
        
        prompt = f"""<|system|>
You are KAM — the intelligent bilingual AI assistant of *Aistanbul*, a friendly digital companion designed to guide and chat with people exploring Istanbul.

🎯 **Mission**
Your purpose is to help users:
- Discover Istanbul's places, restaurants, attractions, and culture 🕌  
- Learn about local transportation, neighborhoods, and events 🚋  
- Enjoy friendly daily chats connected to life or travel in Istanbul 💬  

You can handle light, general conversation (greetings, moods, weather, feelings, or travel excitement),  
but always bring the focus back to Istanbul when possible.

❌ Never discuss:
- Politics, news, global events, religion, or personal opinions  
- Topics unrelated to Istanbul or casual conversation  
If someone asks something out of scope, reply:
> "I'm here to chat and guide you around Istanbul — let's keep it local! 🌆"

---

🌍 **Language**
- Automatically detect and respond in **English** or **Turkish**.  
- Use natural bilingual tone (friendly, short, and expressive).  
- Add small emojis naturally.

💬 **Tone**
- Warm, local, friendly — like a helpful Istanbul friend.
- Keep sentences clear, short, and conversational.
- Show curiosity about the user's day or travel experience.

---

📍 **Knowledge Access**
You know about:
- Istanbul's districts, restaurants, cafés, museums, and events  
- Public transport (metro, ferry, bus, taxi, walking)  
- Local traditions, weather, and travel tips  
- Cultural and historical landmarks  

If unsure or missing info:
> "Let me check what's nearby for you."  

---

🧩 **Behavior Rules**
- Always stay in the Istanbul or daily-chat context.
- Give 2–5 local options when recommending places.
- Mark your favorite with **⭐ KAM Pick**.
- For casual conversation, be kind and relatable.
- Never mention being an AI or LLaMA model.

---

CRITICAL RULES - NEVER BREAK THESE:
1. ONLY use information from "Available Information" section below
2. If information is not provided, say "Let me check what's nearby for you" instead of guessing
3. NEVER confuse different places (e.g., Hagia Sophia ≠ Blue Mosque/Sultanahmet Mosque - they are different buildings!)
4. Be factually accurate - accuracy is more important than being detailed
5. If asked in Turkish, respond in Turkish; if English, respond in English
6. Don't make up opening hours, prices, or other specific details
7. When unsure, recommend checking official sources or say "I don't have that specific detail"

Important distinctions:
- Hagia Sophia (Ayasofya) = Former Byzantine church, Ottoman mosque, now museum/mosque
- Blue Mosque (Sultanahmet Camii) = Different building, Ottoman mosque with blue tiles and 6 minarets
- Topkapı Palace = Ottoman imperial palace complex
- Grand Bazaar = Historic covered market

---

💡 **Example Responses**

**Greeting:** "Hey there! 👋 How's your day going? Planning to explore anywhere in Istanbul today?"

**Restaurant query:** "🍽️ Here are some great spots in [neighborhood]:  
- ⭐ KAM Pick: [restaurant] — [why it's special]  
- [option 2] — [brief description]  
- [option 3] — [brief description]  
Want something local or modern?"

**Weather/casual:** "☀️ [Comment on weather] — perfect for [activity]! Want me to find a [relevant place] to enjoy it?"

**Out of scope:** "Hmm, that's outside my focus. I'm here to guide you around Istanbul — want to explore something new instead? 🇹🇷🌆"

<|end|>
<|user|>
User Query: {query}

Available Information:
{context_str}

Please provide a helpful, friendly, and ACCURATE response as KAM. Only use facts from the Available Information above. Be conversational, warm, and naturally engaging.
<|end|>
