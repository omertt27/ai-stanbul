#!/usr/bin/env python3
"""
FastAPI LLM Server for Istanbul Travel Chatbot
Serves Llama 3.1 8B Instruct (4-bit quantized)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model and tokenizer
model = None
tokenizer = None
model_load_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global model, tokenizer, model_load_time
    
    logger.info("🚀 Starting LLM Server...")
    start_time = time.time()
    
    try:
        # Configure 4-bit quantization
        logger.info("🔧 Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        cache_dir = "/workspace/models"
        
        logger.info(f"📥 Loading model from cache: {cache_dir}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        
        model_load_time = time.time() - start_time
        memory_gb = model.get_memory_footprint() / 1e9
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"⏱️  Load time: {model_load_time:.2f}s")
        logger.info(f"📊 Memory footprint: {memory_gb:.2f} GB")
        logger.info(f"🎯 Device: {model.device}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down LLM Server...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()
    logger.info("✅ Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title="Istanbul Travel Chatbot LLM Server",
    description="Llama 3.1 8B Instruct for multi-intent detection and response synthesis",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response Models
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None

class CompletionResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    memory_gb: Optional[float]
    uptime_seconds: Optional[float]

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Istanbul Travel Chatbot LLM Server",
        "model": "Llama 3.1 8B Instruct (4-bit)",
        "status": "running" if model is not None else "loading",
        "endpoints": {
            "health": "/health",
            "completion": "/v1/completions",
            "chat": "/v1/chat/completions"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    is_loaded = model is not None and tokenizer is not None
    
    memory_gb = None
    if model is not None:
        try:
            memory_gb = model.get_memory_footprint() / 1e9
        except:
            pass
    
    return HealthResponse(
        status="healthy" if is_loaded else "starting",
        model_loaded=is_loaded,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        memory_gb=memory_gb,
        uptime_seconds=time.time() - (time.time() - model_load_time) if model_load_time else None
    )

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Generate text completion"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        start_time = time.time()
        
        # Tokenize
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        logger.info(f"✅ Generated {tokens_generated} tokens in {generation_time:.2f}s")
        
        return CompletionResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: dict):
    """
    OpenAI-compatible chat completions endpoint
    Request format:
    {
        "messages": [
            {"role": "system", "content": "You are..."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Extract parameters
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        # Format prompt for Llama 3.1 Instruct
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Generate
        start_time = time.time()
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        logger.info(f"✅ Chat completion: {tokens_generated} tokens in {generation_time:.2f}s")
        
        # OpenAI-compatible response format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3.1-8b-instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": tokens_generated,
                "total_tokens": inputs['input_ids'].shape[1] + tokens_generated
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
