#!/usr/bin/env python3
"""
Llama 3.1 8B Inference Server for RunPod

Serves the downloaded Llama 3.1 8B 4-bit model via FastAPI.
Access via RunPod proxy: https://yst9iajrc1rc7w-8000.proxy.runpod.net/

Author: AI Istanbul Team
Date: December 2024
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Llama 3.1 8B Inference API",
    description="RunPod inference server for Llama 3.1 8B 4-bit model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
MODEL_PATH = "/workspace/models/llama-3.1-8b-4bit"
model = None
tokenizer = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    system_prompt: Optional[str] = None


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    response: str
    prompt: str
    model: str
    generation_time: float
    tokens_generated: int


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model, tokenizer
    
    try:
        logger.info("=" * 80)
        logger.info("🦙 Loading Llama 3.1 8B 4-bit model...")
        logger.info(f"📁 Model path: {MODEL_PATH}")
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model in 4-bit
        logger.info("🤖 Loading model (4-bit quantized)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"📊 Device: {model.device}")
        logger.info(f"📊 Dtype: {model.dtype}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "Llama 3.1 8B Instruct (4-bit)",
        "model_path": MODEL_PATH,
        "device": str(model.device) if model else "not loaded",
        "endpoints": {
            "generate": "/generate",
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model and tokenizer else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "model_path": MODEL_PATH,
        "device": str(model.device) if model else None
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text completion from a prompt.
    
    Example:
    ```json
    {
        "prompt": "What are the best places to visit in Istanbul?",
        "max_tokens": 200,
        "temperature": 0.7
    }
    ```
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Prepare prompt with system message if provided
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response_text.startswith(full_prompt):
            response_text = response_text[len(full_prompt):].strip()
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - input_length
        
        logger.info(f"✅ Generated {tokens_generated} tokens in {generation_time:.2f}s")
        
        return GenerationResponse(
            response=response_text,
            prompt=request.prompt,
            model="Llama-3.1-8B-Instruct-4bit",
            generation_time=generation_time,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat completion endpoint (OpenAI-compatible format).
    
    Example:
    ```json
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Istanbul travel guide."},
            {"role": "user", "content": "What should I visit in Sultanahmet?"}
        ],
        "max_tokens": 200
    }
    ```
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Format messages into a single prompt
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()
        
        generation_time = time.time() - start_time
        tokens_generated = outputs.shape[1] - input_length
        
        logger.info(f"✅ Chat response: {tokens_generated} tokens in {generation_time:.2f}s")
        
        return {
            "response": response_text,
            "model": "Llama-3.1-8B-Instruct-4bit",
            "generation_time": generation_time,
            "tokens_generated": tokens_generated
        }
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🦙 Starting Llama 3.1 8B Inference Server")
    print("=" * 80)
    print("📡 Server will be available at:")
    print("   - Local: http://0.0.0.0:8000")
    print("   - RunPod: https://yst9iajrc1rc7w-8000.proxy.runpod.net/")
    print("=" * 80)
    print("\n📚 API Documentation:")
    print("   - Interactive docs: https://yst9iajrc1rc7w-8000.proxy.runpod.net/docs")
    print("   - ReDoc: https://yst9iajrc1rc7w-8000.proxy.runpod.net/redoc")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
