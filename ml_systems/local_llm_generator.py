"""
LLM Generator that works on both CPU (local) and GPU (T4)
Automatically detects and uses available device
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalLLMGenerator:
    def __init__(self, model_path="./models/tinyllama", use_gpu=None):
        print("🔄 Loading LLM...")
        
        # Auto-detect device or use specified
        if use_gpu is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"📍 Using device: {self.device}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ LLM loaded on {self.device}")
        if self.device == "cuda":
            print(f"📊 VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
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
        """Create structured prompt"""
        context_str = self._format_context(context_data)
        
        prompt = f"""<|system|>
You are a helpful AI assistant for tourists in Istanbul. Provide accurate, friendly, and concise responses.
<|end|>
<|user|>
User Query: {query}

Available Information:
{context_str}

Please provide a helpful response based on this information.
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
                    items.append(f"{i}. {name} ({location}): {desc}")
                else:
                    items.append(f"{i}. {item}")
            return "\n".join(items)
        return str(context_data)
