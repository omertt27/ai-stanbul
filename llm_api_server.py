#!/usr/bin/env python3
"""
LLM API Server - Google Cloud Llama 3.1 8B
==========================================

FastAPI server for serving Llama 3.1 8B model on Google Cloud VM (n4-standard-8, CPU-only).

Key Features:
- CPU-optimized inference (no GPU required)
- Health monitoring
- Request logging and metrics
- Error handling
- Domain-specific prompt engineering support

Deployment:
    1. SSH into VM: gcloud compute ssh llm-api-server --zone=europe-west1-b
    2. Activate venv: source venv/bin/activate
    3. Run server: python llm_api_server.py
    4. Access: http://35.210.251.24:8000

Monitoring:
    - Health: curl http://35.210.251.24:8000/health
    - CPU: top -p $(pgrep -f llm_api_server)
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time
from datetime import datetime
import psutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/llm_api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="AI Istanbul LLM API",
    description="Llama 3.1 8B API for AI Istanbul System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder
model = None
tokenizer = None
model_name = "meta-llama/Meta-Llama-3.1-8B"


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=150, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt")


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    processing_time: float
    model: str
    tokens_generated: int


# Request metrics
request_count = 0
total_processing_time = 0.0


def load_model():
    """Load Llama 3.1 8B model (CPU-optimized)"""
    global model, tokenizer
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 Loading Llama 3.1 8B Model (CPU Mode)")
        logger.info("=" * 60)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Check system resources
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"💻 System Resources:")
        logger.info(f"   - CPUs: {cpu_count}")
        logger.info(f"   - RAM: {ram_gb:.1f} GB")
        
        # Load tokenizer
        logger.info(f"📥 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model (CPU-only, optimized)
        logger.info(f"📥 Loading model: {model_name}")
        logger.info("   ⚙️  Configuration: CPU-only, float32")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU uses float32
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        # Set to eval mode
        model.eval()
        
        logger.info("=" * 60)
        logger.info("✅ Model loaded successfully!")
        logger.info(f"📊 Model parameters: {model.num_parameters() / 1e9:.2f}B")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}", exc_info=True)
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("🌟 Starting LLM API Server...")
    success = load_model()
    if not success:
        logger.error("⚠️  Model loading failed - server will run but generation will fail")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Istanbul LLM API",
        "model": model_name,
        "status": "operational" if model is not None else "model_not_loaded",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with system metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count(),
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "memory_percent": memory.percent
        },
        "metrics": {
            "total_requests": request_count,
            "avg_processing_time": total_processing_time / request_count if request_count > 0 else 0
        }
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text using Llama 3.1 8B
    
    Args:
        request: Generation request with prompt and parameters
        
    Returns:
        Generated text with metadata
    """
    global request_count, total_processing_time
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    request_count += 1
    
    try:
        logger.info(f"📝 Request #{request_count}")
        logger.info(f"   Prompt length: {len(request.prompt)} chars")
        logger.info(f"   Max tokens: {request.max_tokens}")
        
        # Prepare prompt with optional system prompt
        if request.system_prompt:
            full_prompt = f"<|system|>\n{request.system_prompt}\n<|user|>\n{request.prompt}\n<|assistant|>\n"
        else:
            full_prompt = request.prompt
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        logger.info("🤖 Generating response...")
        import torch
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        tokens_generated = outputs[0].shape[0] - input_length
        
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        
        logger.info(f"✅ Generated {tokens_generated} tokens in {processing_time:.2f}s")
        logger.info(f"   Tokens/sec: {tokens_generated/processing_time:.2f}")
        
        return GenerateResponse(
            generated_text=generated_text.strip(),
            processing_time=processing_time,
            model=model_name,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"❌ Generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get server metrics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "requests": {
            "total": request_count,
            "avg_processing_time": total_processing_time / request_count if request_count > 0 else 0,
            "total_processing_time": total_processing_time
        },
        "system": {
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count(),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2)
        },
        "model": {
            "loaded": model is not None,
            "name": model_name
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("🌐 Starting LLM API Server")
    logger.info(f"📍 Host: 0.0.0.0")
    logger.info(f"🔌 Port: {port}")
    logger.info(f"🤖 Model: {model_name}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
