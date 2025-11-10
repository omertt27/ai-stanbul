"""
Enhanced LLM API Server with 4-bit Quantization
Optimized for ECS deployment with GPU support
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import os
import re
from typing import List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Istanbul LLM API (4-bit)", version="4.0")

# Global variables
model = None
tokenizer = None
device = None

# System prompt with Istanbul expertise
SYSTEM_PROMPT = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

CAPABILITIES:
🍽️ RESTAURANTS: Cuisine, dietary (vegetarian/vegan/halal/gluten-free), price (₺-₺₺₺₺), location
🏛️ ATTRACTIONS: Museums, mosques, palaces, parks (hours, fees, history)
🏘️ NEIGHBORHOODS: District guides (Beyoğlu, Kadıköy, Fatih, Beşiktaş)
🚇 TRANSPORTATION: Metro, tram, ferry, Marmaray, bus (routes, transfers, schedules)
🗺️ ROUTE PLANNING: Detailed A-to-B directions with all transfer points
💡 DAILY TALKS: Cultural customs, etiquette, tips
💎 HIDDEN GEMS: Local favorites, off-beaten-path spots
🌤️ WEATHER-AWARE: Activity recommendations based on weather
🎭 EVENTS: Concerts, festivals, exhibitions

CRITICAL RULES:
1. When providing routes/locations, ALWAYS include coordinates
2. Format: COORDINATES: [[lat1,lon1], [lat2,lon2], ...]
3. For routes: origin → transfer points → destination (in order)
4. Be concise (max 250 tokens)
5. Consider Istanbul context (Bosphorus crossings, traffic, prayer times)

COORDINATE FORMAT EXAMPLES:
✅ CORRECT: COORDINATES: [[41.0058, 28.9770], [41.0176, 28.9704]]
❌ WRONG: coordinates: 41.0058, 28.9770

EXAMPLE RESPONSE:
User: "How to get from Taksim to Kadıköy?"
Assistant: "Take M2 metro from Taksim to Şişhane (2 min), transfer to F1 funicular to Karaköy (1 min), then ferry to Kadıköy (20 min). Total: ~25 min, ₺17.50 with IstanbulKart.

COORDINATES: [[41.0369, 28.9850], [41.0256, 28.9734], [41.0257, 28.9780], [40.9916, 29.0255]]
- Taksim Square
- Şişhane (metro)
- Karaköy (ferry)
- Kadıköy

💡 Best views from ferry upper deck!"
"""

# Istanbul POI database (major landmarks)
ISTANBUL_COORDINATES = {
    # Historic Peninsula
    "sultanahmet": [41.0058, 28.9770],
    "hagia sophia": [41.0086, 28.9802],
    "blue mosque": [41.0054, 28.9768],
    "topkapi palace": [41.0115, 28.9833],
    "grand bazaar": [41.0108, 28.9680],
    "spice bazaar": [41.0166, 28.9708],
    
    # Beyoğlu District
    "taksim": [41.0369, 28.9850],
    "istiklal street": [41.0335, 28.9785],
    "galata tower": [41.0256, 28.9744],
    "karaköy": [41.0257, 28.9780],
    
    # Asian Side
    "kadıköy": [40.9916, 29.0255],
    "moda": [40.9839, 29.0302],
    "üsküdar": [41.0226, 29.0079],
    
    # Bosphorus
    "ortaköy": [41.0552, 29.0296],
    "bebek": [41.0779, 29.0431],
    "beşiktaş": [41.0422, 29.0094],
    
    # Other
    "eminönü": [41.0176, 28.9704],
    "fatih": [41.0191, 28.9497],
}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 250
    temperature: float = 0.7
    include_system_prompt: bool = True
    
class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    coordinates: Optional[List[List[float]]] = None
    intent: Optional[str] = None
    processing_time: float

def extract_coordinates(text: str) -> List[List[float]]:
    """Extract coordinates from LLM response"""
    pattern = r'COORDINATES:\s*\[\s*(\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\](?:\s*,\s*\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\])*)\s*\]'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        coord_str = match.group(1)
        pairs = re.findall(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', coord_str)
        return [[float(lat), float(lon)] for lat, lon in pairs]
    
    return []

def resolve_location_coordinates(text: str) -> List[List[float]]:
    """Resolve location names to coordinates from text"""
    coords = []
    text_lower = text.lower()
    
    for location, coord in ISTANBUL_COORDINATES.items():
        if location in text_lower:
            coords.append(coord)
    
    return coords

def detect_intent(prompt: str) -> str:
    """Detect user intent"""
    prompt_lower = prompt.lower()
    
    # Route planning (highest priority)
    if any(word in prompt_lower for word in ['how to get', 'how do i get', 'directions', 'route to', 'go from', 'travel from']):
        return 'route_planning'
    
    # Transportation
    if any(word in prompt_lower for word in ['metro', 'tram', 'bus', 'ferry', 'marmaray', 'transport', 'transit']):
        return 'transportation'
    
    # Food & Restaurants
    if any(word in prompt_lower for word in ['restaurant', 'food', 'eat', 'cuisine', 'dining', 'meal', 'breakfast', 'lunch', 'dinner']):
        return 'restaurant'
    
    # Attractions
    if any(word in prompt_lower for word in ['museum', 'mosque', 'palace', 'park', 'attraction', 'visit', 'see', 'monument']):
        return 'attraction'
    
    # Neighborhoods
    if any(word in prompt_lower for word in ['neighborhood', 'district', 'area', 'where to stay', 'which area']):
        return 'neighborhood'
    
    # Weather
    if any(word in prompt_lower for word in ['weather', 'rain', 'hot', 'cold', 'sunny', 'temperature']):
        return 'weather'
    
    # Events
    if any(word in prompt_lower for word in ['event', 'concert', 'festival', 'exhibition', 'show', 'performance']):
        return 'event'
    
    # Tips & Culture
    if any(word in prompt_lower for word in ['tip', 'custom', 'etiquette', 'culture', 'advice', 'should i know']):
        return 'daily_talk'
    
    # Hidden Gems
    if any(word in prompt_lower for word in ['hidden', 'secret', 'local', 'off beaten', 'locals go']):
        return 'hidden_gem'
    
    return 'general'

@app.on_event("startup")
async def load_model():
    """Load Llama 3.1 8B model with 4-bit quantization on startup"""
    global model, tokenizer, device
    
    logger.info("🚀 Loading Llama 3.1 8B model with 4-bit quantization...")
    
    # Get configuration from environment
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
    quantization_bits = int(os.getenv("QUANTIZATION_BITS", "4"))
    use_hf_token = os.getenv("HF_TOKEN") is not None
    
    # Check for GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA version: {torch.version.cuda}")
        logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        logger.info("⚠️ No GPU detected, using CPU (not recommended for production)")
    
    try:
        # Configure 4-bit quantization
        if device == "cuda" and quantization_bits == 4:
            logger.info(f"📦 Configuring {quantization_bits}-bit quantization with bitsandbytes...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization for better compression
                bnb_4bit_quant_type="nf4"  # Normal Float 4-bit
            )
            
            logger.info("🔑 Loading model with HuggingFace authentication..." if use_hf_token else "🔓 Loading model without authentication...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HF_TOKEN") if use_hf_token else None
            )
            
            # Load model with 4-bit quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=os.getenv("HF_TOKEN") if use_hf_token else None
            )
            
            logger.info("✅ Model loaded with 4-bit quantization!")
            
        else:
            # Fallback: Load without quantization
            logger.info("📦 Loading model without quantization...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HF_TOKEN") if use_hf_token else None
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else "cpu",
                low_cpu_mem_usage=True,
                token=os.getenv("HF_TOKEN") if use_hf_token else None
            )
            
            logger.info("✅ Model loaded without quantization!")
        
        model.eval()
        
        # Calculate memory usage
        if device == "cuda":
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            reserved_memory = torch.cuda.memory_reserved() / 1e9
            logger.info(f"💾 GPU Memory - Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB")
        
        logger.info(f"📊 Model: {model_name}")
        logger.info(f"🖥️ Device: {device}")
        logger.info(f"🔢 Quantization: {quantization_bits}-bit" if quantization_bits == 4 and device == "cuda" else "🔢 Quantization: None")
        logger.info(f"🎯 Ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2)
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "model": os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B"),
        "quantization": f"{os.getenv('QUANTIZATION_BITS', '4')}-bit",
        **gpu_info
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text with coordinate extraction"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Detect intent
        intent = detect_intent(request.prompt)
        
        # Build prompt
        if request.include_system_prompt:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {request.prompt}\nAssistant:"
        else:
            full_prompt = request.prompt
        
        logger.info(f"🔄 Intent: {intent} | Prompt: {request.prompt[:60]}...")
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (remove prompt)
        response_text = generated_text[len(full_prompt):].strip()
        
        # Extract coordinates
        coordinates = extract_coordinates(response_text)
        
        # If no coordinates found but route/transport intent, try to resolve from text
        if not coordinates and intent in ["route_planning", "transportation"]:
            coordinates = resolve_location_coordinates(response_text)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Generated in {processing_time:.2f}s | Coords: {len(coordinates)} | Intent: {intent}")
        
        return GenerateResponse(
            response=response_text,
            tokens_generated=len(outputs[0]) - len(inputs["input_ids"][0]),
            coordinates=coordinates if coordinates else None,
            intent=intent,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: GenerateRequest):
    """Chat endpoint (alias for generate)"""
    return await generate(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"🚀 Starting AI Istanbul LLM API (4-bit) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
