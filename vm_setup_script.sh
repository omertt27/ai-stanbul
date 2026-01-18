#!/bin/bash
# VM Setup Script - Creates API server and requirements on the VM

echo "🚀 Setting up LLM API server on VM..."

# Create requirements.txt
cat > ~/ai-stanbul/requirements.txt << 'REQUIREMENTS_EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.35.0
torch==2.1.0
accelerate==0.24.1
pydantic==2.5.0
requests==2.31.0
python-dotenv==1.0.0
REQUIREMENTS_EOF

echo "✅ Created requirements.txt"

# Create the LLM API server
cat > ~/ai-stanbul/llm_api_server.py << 'SERVER_EOF'
"""
LLM Inference API Server for Google Cloud VM
Serves Llama 3.1 8B model via FastAPI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Istanbul LLM API", version="1.0")

# Global model variables
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7
    
class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    model_info: dict = None

@app.on_event("startup")
async def load_model():
    """Load Llama 3.1 8B model on startup"""
    global model, tokenizer
    
    logger.info("🚀 Starting AI Istanbul LLM API...")
    logger.info("📂 Loading Llama 3.1 8B model...")
    
    model_path = os.getenv("MODEL_PATH", "./models/llama-3.1-8b")
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}")
        logger.info("💡 Please transfer the model to the VM first")
        return
    
    try:
        # Load tokenizer
        logger.info("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model (CPU, FP32)
        logger.info("🧠 Loading model (this takes 2-3 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"📊 Model parameters: {num_params/1e9:.2f}B")
        logger.info(f"💾 Device: CPU")
        logger.info(f"🔧 Dtype: {model.dtype}")
        logger.info("🎯 Ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"💡 Ensure model files exist at: {model_path}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Istanbul LLM API",
        "version": "1.0",
        "model": "Llama 3.1 8B",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/generate",
            "chat": "/chat"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": "cpu",
        "ready": model is not None and tokenizer is not None
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Llama 3.1 8B"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for model to load or check logs."
        )
    
    try:
        logger.info(f"🔄 Generating response for prompt: {request.prompt[:100]}...")
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        input_length = len(inputs.input_ids[0])
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        response_text = generated_text[len(request.prompt):].strip()
        
        tokens_generated = len(outputs[0]) - input_length
        
        logger.info(f"✅ Generated {tokens_generated} tokens, {len(response_text)} characters")
        
        return GenerateResponse(
            response=response_text,
            tokens_generated=tokens_generated,
            model_info={
                "model": "Llama-3.1-8B",
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat")
async def chat(request: GenerateRequest):
    """Chat endpoint with conversation formatting"""
    
    # Format as chat message
    formatted_prompt = f"""You are a helpful AI assistant for Istanbul tourism.
Provide accurate, friendly, and concise information about Istanbul.

User: {request.prompt}
Assistant:"""
    
    # Call generate with formatted prompt
    result = await generate(GenerateRequest(
        prompt=formatted_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    ))
    
    return result

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"🚀 Starting LLM API server on 0.0.0.0:{port}")
    logger.info(f"📍 Access at: http://localhost:{port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
SERVER_EOF

chmod +x ~/ai-stanbul/llm_api_server.py

echo "✅ Created llm_api_server.py"
echo "🎯 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create virtual environment: python3 -m venv venv"
echo "2. Activate: source venv/bin/activate"
echo "3. Install packages: pip install -r requirements.txt"
echo "4. Run server: python llm_api_server.py"
