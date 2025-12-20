#!/usr/bin/env python3
"""
Llama 3.1 8B vLLM Inference Server for RunPod

High-performance inference server using vLLM for optimized serving.
vLLM provides PagedAttention, continuous batching, and much faster inference.

Access via RunPod proxy: https://yst9iajrc1rc7w-8000.proxy.runpod.net/

Author: AI Istanbul Team
Date: December 2024
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time
from vllm import LLM, SamplingParams
from vllm.sampling_params import SamplingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Llama 3.1 8B vLLM Inference API",
    description="High-performance RunPod inference server using vLLM",
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

# Global vLLM model
MODEL_PATH = "/workspace/models/llama-3.1-8b-4bit"
llm = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
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
    top_k: Optional[int] = 50


@app.on_event("startup")
async def load_model():
    """Load the vLLM model on startup."""
    global llm
    
    try:
        logger.info("=" * 80)
        logger.info("🦙 Loading Llama 3.1 8B with vLLM...")
        logger.info(f"📁 Model path: {MODEL_PATH}")
        
        # Load model with vLLM
        # vLLM will automatically use quantization if available
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,  # Adjust based on GPU count
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=4096,  # Context length
            quantization="bitsandbytes",  # Use 4-bit quantization
            dtype="float16",
            trust_remote_code=True
        )
        
        logger.info("✅ vLLM model loaded successfully!")
        logger.info(f"📊 Max model length: 4096 tokens")
        logger.info(f"📊 Quantization: 4-bit (bitsandbytes)")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("💡 Tip: Make sure you have vLLM installed: pip install vllm")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "Llama 3.1 8B Instruct (4-bit)",
        "engine": "vLLM",
        "model_path": MODEL_PATH,
        "loaded": llm is not None,
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
        "status": "healthy" if llm else "unhealthy",
        "model_loaded": llm is not None,
        "engine": "vLLM",
        "model_path": MODEL_PATH,
        "max_model_len": 4096,
        "quantization": "4-bit (bitsandbytes)"
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate text completion from a prompt using vLLM.
    
    Example:
    ```json
    {
        "prompt": "What are the best places to visit in Istanbul?",
        "max_tokens": 200,
        "temperature": 0.7
    }
    ```
    """
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Prepare prompt with system message if provided
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}\n\nAssistant:"
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )
        
        # Generate with vLLM
        outputs = llm.generate([full_prompt], sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.1f} tokens/s)")
        
        return GenerationResponse(
            response=generated_text.strip(),
            prompt=request.prompt,
            model="Llama-3.1-8B-Instruct-4bit-vLLM",
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
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Format messages into a single prompt (Llama 3.1 chat format)
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>")
        
        # Add the assistant prompt for generation
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        full_prompt = "".join(prompt_parts)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"]  # Stop tokens for Llama 3.1
        )
        
        # Generate with vLLM
        outputs = llm.generate([full_prompt], sampling_params)
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Chat response: {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.1f} tokens/s)")
        
        return {
            "response": generated_text.strip(),
            "model": "Llama-3.1-8B-Instruct-4bit-vLLM",
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("🦙 Starting Llama 3.1 8B vLLM Inference Server")
    print("=" * 80)
    print("📡 Server will be available at:")
    print("   - Local: http://0.0.0.0:8000")
    print("   - RunPod: https://yst9iajrc1rc7w-8000.proxy.runpod.net/")
    print("=" * 80)
    print("\n📚 API Documentation:")
    print("   - Interactive docs: https://yst9iajrc1rc7w-8000.proxy.runpod.net/docs")
    print("   - ReDoc: https://yst9iajrc1rc7w-8000.proxy.runpod.net/redoc")
    print("=" * 80)
    print("\n⚡ Using vLLM for high-performance inference!")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
