"""
Model-agnostic LLM service wrapper
Supports TinyLlama, LLaMA 3.1 8B (production), and LLaMA 3.2

This wrapper allows seamless switching between models:
- Production (Google Cloud CPU): LLaMA 3.1 8B on n2-standard-8 (16 vCPUs, 32.5GB RAM)
- Development (Local): TinyLlama on MPS/CPU (fast, lower accuracy, 2GB VRAM)
- Optional: 4-bit quantization for memory-constrained environments

Deployment Strategy:
- n2-standard-8 (16 vCPUs, 32.5GB RAM): FP32 Llama 3.1 8B with multi-threading (~3-8s latency)
- Local MPS: TinyLlama (development/testing)
- Auto-detection based on available hardware

Performance Expectations (CPU):
- Response time: 3-8 seconds (acceptable for chatbot)
- Throughput: 1-2 concurrent requests
- Cost: ~$0.40/hour (~$290/month 24/7) or ~$100/month with scheduled stop/start

Upgrade: November 8, 2025 - Production-ready for Google Cloud n2-standard-8 CPU deployment
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Import prompt generator
try:
    from .google_maps_style_prompts import get_prompt_generator
    PROMPTS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ google_maps_style_prompts not available")
    PROMPTS_AVAILABLE = False


class LLMServiceWrapper:
    """
    Model-agnostic LLM service wrapper
    
    Automatically detects:
    - Best available device (MPS, CUDA, CPU)
    - Available models (LLaMA 3.2 3B, LLaMA 3.2 1B, TinyLlama)
    
    Usage:
        # Development (automatic - uses TinyLlama)
        llm = LLMServiceWrapper()
        
        # Production (set environment variable)
        # export LLM_MODEL_PATH=./models/llama-3.2-3b
        llm = LLMServiceWrapper()
        
        # Generate response
        response = llm.generate("What's the best way to visit Istanbul?", max_tokens=150)
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize LLM service
        
        Args:
            model_path: Path to model (optional, uses env var or auto-detect)
            device: Device to use (optional, auto-detects best device)
        """
        # Use environment variable or default based on device
        if model_path:
            self.model_path = model_path
        elif os.getenv('LLM_MODEL_PATH'):
            self.model_path = os.getenv('LLM_MODEL_PATH')
        else:
            # Smart default selection based on available hardware
            # Production CPU (n2-standard-16): Use Llama 3.1 8B for best accuracy
            # Production GPU (T4): Use Llama 3.1 8B with CUDA acceleration
            # Development (MPS/CPU): Use TinyLlama for speed
            current_file = os.path.abspath(__file__)
            ml_systems_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(ml_systems_dir)
            
            # Detect device first
            temp_device = device or self._get_best_device_quick()
            
            # Check if we're in production mode (env variable set)
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            if is_production or temp_device == "cuda":
                # Production mode: Use Llama 3.1 8B (CPU or GPU)
                self.model_path = os.path.join(project_root, 'models', 'llama-3.1-8b')
                if temp_device == "cuda":
                    logger.info("🚀 Production mode: Using Llama 3.1 8B on CUDA (T4 GPU)")
                else:
                    logger.info("🚀 Production mode: Using Llama 3.1 8B on CPU (n2-standard-8, 16 vCPUs)")
            else:
                # Development mode: Use TinyLlama on MPS/CPU
                self.model_path = os.path.join(project_root, 'models', 'tinyllama')
                logger.info("🔧 Development mode: Using TinyLlama on MPS/CPU")
        
        self.device = device or self._get_best_device()
        self.model = None
        self.tokenizer = None
        self.model_name = self._get_model_name()
        
        logger.info(f"🤖 Initializing LLM Service: {self.model_name}")
        logger.info(f"📁 Model path: {self.model_path}")
        logger.info(f"🖥️  Device: {self.device}")
        
        self._load_model()
    
    def _get_best_device_quick(self):
        """Quick device detection without logging (for model selection)"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_best_device(self):
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ CUDA acceleration available: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            return "cuda"  # NVIDIA GPU (T4, A100, etc.)
        elif torch.backends.mps.is_available():
            logger.info("✅ Metal (MPS) acceleration available")
            return "mps"  # macOS Metal
        else:
            logger.warning("⚠️ No GPU acceleration available, using CPU")
            return "cpu"
    
    def _get_model_name(self):
        """Extract model name from path"""
        if 'llama-3.2-3b' in self.model_path.lower():
            return "LLaMA 3.2 3B"
        elif 'llama-3.2-1b' in self.model_path.lower():
            return "LLaMA 3.2 1B"
        elif 'llama-3.1-8b' in self.model_path.lower():
            return "LLaMA 3.1 8B"
        elif 'tinyllama' in self.model_path.lower():
            return "TinyLlama"
        else:
            return "Unknown Model"
    
    def _load_model(self):
        """Load model (works with any model)"""
        try:
            # Convert to absolute path if relative
            model_path = self.model_path
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
            
            logger.info(f"📂 Absolute model path: {model_path}")
            
            # Load tokenizer
            logger.info("1️⃣ Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # CPU optimization: Set number of threads for optimal performance
            if self.device == "cpu":
                num_threads = int(os.getenv('OMP_NUM_THREADS', '14'))  # Use 14 threads for n2-standard-8 (leave 2 for system)
                torch.set_num_threads(num_threads)
                logger.info(f"🧵 CPU threads set to: {num_threads} (optimal for n2-standard-8, 16 vCPUs)")
            
            # Determine if we should use quantization
            # Production CPU: FP32 (n2-standard-8 has 32.5GB RAM - enough for Llama 3.1 8B)
            # T4 GPU (16GB): No quantization needed for Llama 3.1 8B (~8GB FP16)
            # MPS/Development CPU: Use quantization for large models
            use_quantization = False
            is_large_model = 'llama-3.1-8b' in self.model_name.lower() or 'llama-3.2-7b' in self.model_name.lower()
            is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
            
            if is_large_model and self.device == 'cpu' and not is_production:
                # Development CPU: Use quantization to save memory
                logger.info("🎯 Using 4-bit quantization for development CPU (memory optimization)...")
                use_quantization = True
            elif is_large_model and self.device == 'cpu' and is_production:
                # Production CPU (n2-standard-8): Use FP32 for best accuracy (32.5GB RAM is enough)
                logger.info("🎯 Using FP32 precision for production CPU (optimal accuracy on n2-standard-8)...")
            elif is_large_model and self.device == 'cuda':
                # T4 GPU: Use FP16 for speed
                logger.info("🎯 Using FP16 precision for CUDA (optimal T4 performance)...")
            elif is_large_model and self.device == 'mps':
                # MPS: Try quantization
                logger.info("🎯 Using 4-bit quantization for MPS (memory optimization)...")
                use_quantization = True
            
            # Load model with appropriate configuration
            logger.info("2️⃣ Loading model...")
            
            if use_quantization:
                try:
                    # Try 4-bit quantization (requires bitsandbytes)
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                        trust_remote_code=True,
                        device_map="auto"  # Auto device placement for quantized models
                    )
                    logger.info("✅ Model loaded with 4-bit quantization (memory reduced by ~75%)")
                    
                except Exception as quant_error:
                    logger.warning(f"⚠️ 4-bit quantization failed: {quant_error}")
                    logger.info("Falling back to standard loading...")
                    use_quantization = False
            
            if not use_quantization:
                # Standard loading without quantization
                dtype = torch.float16 if self.device != "cpu" else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Move to device (only for non-quantized models)
                logger.info(f"3️⃣ Moving model to {self.device}...")
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Print memory usage
            if self.device == "cuda":
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"📊 GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            elif self.device == "mps":
                logger.info("📊 Model loaded on Metal (MPS)")
            
            logger.info(f"✅ {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate(self, prompt=None, query=None, context_data=None, max_tokens=200, temperature=0.7, top_p=0.9, stop_sequences=None):
        """
        Generate response (model-agnostic, backward compatible)
        
        Args:
            prompt: Input prompt (direct prompt string)
            query: User query (for backward compatibility)
            context_data: Context data to include (for backward compatibility)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter
            stop_sequences: List of strings to stop generation (optional, for Llama 3)
            
        Returns:
            Generated text (without the prompt)
            
        Note:
            Supports two modes:
            1. Direct prompt: generate(prompt="...")
            2. Legacy mode: generate(query="...", context_data=...)
            
        Model-specific optimizations:
            - Llama 3.x: Uses <|eot_id|> token for clean stops
            - TinyLlama: Standard EOS handling
        """
        # Build prompt from query + context if using legacy interface
        if query is not None and prompt is None:
            prompt = self._build_prompt_from_query(query, context_data)
        
        if prompt is None:
            logger.error("❌ No prompt or query provided to generate()")
            return "I apologize, but I couldn't generate a response. Please try again."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prepare generation config
            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': 1.1,  # Slightly lower for Llama 3
                'pad_token_id': self.tokenizer.eos_token_id
            }
            
            # Add stop sequences for Llama 3.x models
            if 'llama-3' in self.model_name.lower():
                if stop_sequences is None:
                    stop_sequences = ['<|eot_id|>', '<|end_of_text|>']
                
                # Tokenize stop sequences
                stop_token_ids = []
                for seq in stop_sequences:
                    tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                    stop_token_ids.extend(tokens)
                
                if stop_token_ids:
                    gen_kwargs['eos_token_id'] = list(set(stop_token_ids + [self.tokenizer.eos_token_id]))
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from response
            response = full_response[len(prompt):].strip()
            
            # Clean up Llama 3 special tokens if they leaked through
            for seq in (stop_sequences or []):
                response = response.replace(seq, '').strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def _build_prompt_from_query(self, query: str, context_data: Any) -> str:
        """
        Build prompt from query and context (backward compatibility)
        
        Args:
            query: User query
            context_data: Context data (list of dicts or dict)
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are Istanbul AI, a helpful guide for Istanbul, Turkey.",
            "Provide concise, accurate, and friendly responses.",
            f"\nUser Question: {query}",
        ]
        
        # Add context if available
        if context_data:
            if isinstance(context_data, list):
                prompt_parts.append("\nRelevant Information:")
                for i, item in enumerate(context_data[:3], 1):
                    if isinstance(item, dict):
                        name = item.get('name', 'N/A')
                        prompt_parts.append(f"{i}. {name}")
            elif isinstance(context_data, dict):
                prompt_parts.append(f"\nContext: {str(context_data)[:200]}")
        
        prompt_parts.append("\nResponse:")
        
        return "\n".join(prompt_parts)
    
    def get_transportation_advice(
        self,
        from_location: str,
        to_location: str,
        gps_context: Optional[Dict[str, Any]] = None,
        route_data: Optional[Dict[str, Any]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Get brief, GPS-aware transportation advice
        
        Args:
            from_location: Origin location name
            to_location: Destination location name
            gps_context: GPS location context:
                - gps_location: Tuple[float, float] - (lat, lon)
                - current_district: str - Detected district
                - nearby_landmarks: List[str] - Nearby landmarks
                - location_accuracy: float - GPS accuracy in meters
            route_data: Optional route data from OSRM
            weather_data: Optional current weather conditions
            user_preferences: Optional user preferences
            max_tokens: Maximum tokens (default: 100 for 2-3 sentences)
            temperature: Sampling temperature
        
        Returns:
            Brief transportation advice (2-3 sentences)
        """
        if not PROMPTS_AVAILABLE:
            logger.warning("⚠️ Prompt generator not available, using fallback")
            return f"Take the best route from {from_location} to {to_location}. Check the map for details."
        
        try:
            # Build enhanced prompt with GPS context
            prompt_gen = get_prompt_generator(language='en')
            
            prompt = prompt_gen.create_route_prompt(
                origin=from_location,
                destination=to_location,
                route_data=route_data or {},
                weather_data=weather_data,
                gps_context=gps_context,
                user_preferences=user_preferences
            )
            
            # Generate response
            response = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Log for monitoring
            district = gps_context.get('current_district', 'unknown') if gps_context else 'unknown'
            logger.info(f"🗺️ Transportation advice: {from_location} → {to_location} (from {district})")
            
            return response.strip() if response else "Route found. See map for details."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate transportation advice: {e}")
            return "Route found. See map for details."
    
    def get_poi_recommendation(
        self,
        poi_type: str,
        gps_context: Dict[str, Any],
        nearby_pois: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None,
        max_tokens: int = 100,
        temperature: float = 0.8
    ) -> str:
        """
        Get personalized POI recommendation based on GPS location
        
        Args:
            poi_type: Type of POI ("museums", "restaurants", "attractions")
            gps_context: GPS location context:
                - gps_location: Tuple[float, float]
                - current_district: str
                - nearby_landmarks: List[str]
            nearby_pois: List of nearby POIs from database
            user_preferences: Optional user interests and preferences
            max_tokens: Maximum tokens
            temperature: Sampling temperature (higher for creative recommendations)
        
        Returns:
            Personalized POI recommendation (2-3 sentences)
        """
        if not PROMPTS_AVAILABLE:
            logger.warning("⚠️ Prompt generator not available, using fallback")
            return f"Found {len(nearby_pois)} nearby {poi_type}. Check the map for details."
        
        try:
            # Build POI recommendation prompt
            prompt_gen = get_prompt_generator(language='en')
            
            prompt = prompt_gen.create_poi_recommendation_prompt(
                poi_type=poi_type,
                gps_context=gps_context,
                nearby_pois=nearby_pois,
                user_preferences=user_preferences
            )
            
            # Generate response
            response = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Log for monitoring
            district = gps_context.get('current_district', 'unknown')
            logger.info(f"📍 POI recommendation: {poi_type} near {district} ({len(nearby_pois)} options)")
            
            return response.strip() if response else f"Found {len(nearby_pois)} nearby {poi_type}."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate POI recommendation: {e}")
            return f"Found {len(nearby_pois)} nearby {poi_type}. Check the map for details."
    
    def get_info(self):
        """Get model and device information"""
        info = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'is_loaded': self.model is not None
        }
        
        if self.device == "cuda":
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Testing LLM Service Wrapper")
    print("="*80)
    
    # Initialize service
    llm = LLMServiceWrapper()
    
    # Print info
    print("\n📋 Model Information:")
    info = llm.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    prompt = "What is the best way to visit Istanbul?"
    response = llm.generate(prompt, max_tokens=100)
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"🤖 Response: {response}")
    
    print("\n✅ Test complete!")
