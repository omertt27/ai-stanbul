"""
LLM Generator that works on CPU, Metal (M2 Pro), and CUDA (T4 GPU)
Automatically detects and uses best available device
Supports: CPU, Metal (MPS), CUDA (T4/A100)
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    
    def _detect_best_model(self):
        """Auto-detect best available LLAMA model"""
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
<|assistant|>
"""
        return prompt
    
    def _format_context(self, context_data):
        """Format context data"""
        if not context_data:
            return "No specific data available."
        
        if isinstance(context_data, list):
            items = []
            for i, item in enumerate(context_data[:5], 1):
                if isinstance(item, dict):
                    name = item.get('name', 'Unknown')
                    desc = item.get('description', '')
                    location = item.get('location', '')
                    
                    # Special formatting for different types
                    item_type = item.get('type', '')
                    
                    if item_type == 'transportation' or item_type == 'metro' or item_type == 'tram' or item_type == 'ferry' or item_type == 'route':
                        # Transportation: include route and tips
                        route = item.get('route', '')
                        tips = item.get('tips', '')
                        duration = item.get('duration', '')
                        cost = item.get('cost', item.get('fare', ''))
                        
                        parts = [f"{i}. {name}"]
                        if route:
                            parts.append(f"Route: {route}")
                        if desc:
                            parts.append(f"Details: {desc}")
                        if tips:
                            parts.append(f"Tips: {tips}")
                        if duration:
                            parts.append(f"Duration: {duration}")
                        if cost:
                            parts.append(f"Cost: {cost}")
                        items.append(" | ".join(parts))
                    
                    elif item_type == 'event':
                        # Events: include date, time, venue
                        date = item.get('date', '')
                        time_str = item.get('time', '')
                        price = item.get('price', '')
                        
                        parts = [f"{i}. {name}"]
                        if location:
                            parts.append(f"Venue: {location}")
                        if date:
                            parts.append(f"Date: {date}")
                        if time_str:
                            parts.append(f"Time: {time_str}")
                        if desc:
                            parts.append(f"Details: {desc}")
                        if price:
                            parts.append(f"Price: {price}")
                        items.append(" | ".join(parts))
                    
                    else:
                        # Default format (restaurants, attractions, etc.)
                        if location and desc:
                            items.append(f"{i}. {name} ({location}): {desc}")
                        elif desc:
                            items.append(f"{i}. {name}: {desc}")
                        elif location:
                            items.append(f"{i}. {name} ({location})")
                        else:
                            items.append(f"{i}. {name}")
                else:
                    items.append(f"{i}. {item}")
            return "\n".join(items)
        return str(context_data)
