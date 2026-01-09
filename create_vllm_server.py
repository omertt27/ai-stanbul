#!/usr/bin/env python3
"""
Script to create vllm_server.py on RunPod
Run this on RunPod: python create_vllm_server.py
"""

server_code = '''"""AI Istanbul - vLLM Server for Llama 3.1 8B AWQ 4-bit"""
import os, time, json
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

app = FastAPI(title="AI Istanbul LLM API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=1024, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = False

llm_engine: Optional[AsyncLLMEngine] = None

def format_llama_chat(messages: List[ChatMessage]) -> str:
    formatted = "<|begin_of_text|>"
    for msg in messages:
        if msg.role == "system":
            formatted += f"<|start_header_id|>system<|end_header_id|>\\n\\n{msg.content}<|eot_id|>"
        elif msg.role == "user":
            formatted += f"<|start_header_id|>user<|end_header_id|>\\n\\n{msg.content}<|eot_id|>"
        elif msg.role == "assistant":
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{msg.content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    return formatted

@app.on_event("startup")
async def startup():
    global llm_engine
    model_path = os.getenv("MODEL_PATH", "/workspace/models/llama-3.1-8b-4bit")
    quantization = os.getenv("QUANTIZATION", "awq")
    print(f"🚀 Loading model: {model_path}")
    print(f"📊 Quantization: {quantization}")
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        quantization=quantization,
        dtype="half",
        trust_remote_code=True
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ vLLM engine ready!")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Istanbul LLM",
        "model": "Llama 3.1 8B AWQ 4-bit",
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "ready" if llm_engine else "initializing"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not llm_engine:
        raise HTTPException(503, "Engine not ready")
    prompt = format_llama_chat(request.messages)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    
    if request.stream:
        async def generate():
            full_text = ""
            async for output in llm_engine.generate(prompt, sampling_params, request_id):
                if not output.finished:
                    new_text = output.outputs[0].text[len(full_text):]
                    full_text = output.outputs[0].text
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "llama-3.1-8b-awq",
                        "choices": [{"index": 0, "delta": {"content": new_text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\\n\\n"
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\\n\\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in llm_engine.generate(prompt, sampling_params, request_id):
            final_output = output
        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama-3.1-8b-awq",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final_output.outputs[0].text},
                "finish_reason": "stop"
            }]
        })

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if not llm_engine:
        raise HTTPException(503, "Engine not ready")
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )
    request_id = f"cmpl-{int(time.time() * 1000)}"
    
    if request.stream:
        async def generate():
            full_text = ""
            async for output in llm_engine.generate(request.prompt, sampling_params, request_id):
                if not output.finished:
                    new_text = output.outputs[0].text[len(full_text):]
                    full_text = output.outputs[0].text
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "choices": [{"text": new_text, "index": 0}]
                    }
                    yield f"data: {json.dumps(chunk)}\\n\\n"
            yield "data: [DONE]\\n\\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        final_output = None
        async for output in llm_engine.generate(request.prompt, sampling_params, request_id):
            final_output = output
        return JSONResponse({
            "id": request_id,
            "object": "text_completion",
            "choices": [{"text": final_output.outputs[0].text, "index": 0, "finish_reason": "stop"}]
        })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 19123))
    print(f"""
╔════════════════════════════════════════════════════════════╗
║          AI Istanbul vLLM Server                           ║
║          Llama 3.1 8B AWQ 4-bit                           ║
╚════════════════════════════════════════════════════════════╝
🌐 Port: {port}
📡 RunPod: https://oge3mpj2wjlj2z-19123.proxy.runpod.net/nfi778289w9c67tsdothsgyw84udcpx6/
    """)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
'''

# Write the file
output_path = "/workspace/vllm_server.py"
with open(output_path, "w") as f:
    f.write(server_code)

print(f"✅ Created {output_path}")
print(f"📊 File size: {len(server_code)} bytes")
print("\n🚀 Ready to start server with:")
print("   nohup python /workspace/vllm_server.py > /workspace/vllm.log 2>&1 &")
