#!/usr/bin/env python3
"""
Single Query Test - Test streaming response parsing
"""

import asyncio
import aiohttp
import json
import time

async def test_single_query():
    queries = [
        "What are the metro lines in Istanbul?",
        "How do I get from Sultanahmet to Taksim Square?",
        "Tell me about Hagia Sophia",
        "What can I do in Sultanahmet district?",
        "Best time to visit Istanbul?"
    ]
    
    async with aiohttp.ClientSession() as session:
        for query in queries:
            print(f"\n🔍 Testing: {query}")
            print("-" * 60)
            
            start_time = time.time()
            
            async with session.post(
                "http://localhost:8000/ai/stream",
                json={"user_input": query}
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # Parse the streaming response
                    lines = content.strip().split('\n')
                    full_response = ""
                    
                    for line in lines:
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                if 'content' in data:
                                    full_response += data['content']
                            except json.JSONDecodeError:
                                continue
                    
                    print(f"\nQuery: {query}")
                    print(f"Response Time: {response_time:.2f}s")
                    print(f"Response: {full_response[:200]}...")
                else:
                    print(f"Error: {response.status}")
            
async def test_single_query():
    queries = [
        "What are the metro lines in Istanbul?",
        "How do I get from Sultanahmet to Taksim Square?",
        "Tell me about Hagia Sophia",
        "What can I do in Sultanahmet district?",
        "Best time to visit Istanbul?"
    ]
    
    async with aiohttp.ClientSession() as session:
        for query in queries:
            print(f"\n🔍 Testing: {query}")
            print("-" * 60)
            
            start_time = time.time()
            
            async with session.post(
                "http://localhost:8000/ai/stream",
                json={"user_input": query}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    continue
                
                print("✅ Got streaming response, parsing...")
                
                full_response = ""
                async for line in response.content:
                    line_text = line.decode('utf-8').strip()
                    if line_text.startswith('data: '):
                        chunk = line_text[6:]  # Remove 'data: ' prefix
                        if chunk and chunk != '[DONE]':
                            try:
                                data = json.loads(chunk)
                                if 'delta' in data and 'content' in data['delta']:
                                    content = data['delta']['content']
                                    full_response += content
                                    print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                
                response_time = time.time() - start_time
                
                print(f"\n\n📊 Response Summary:")
                print(f"Response Time: {response_time:.2f}s")
                print(f"Response Length: {len(full_response)} characters")
                print(f"First 150 chars: {full_response[:150]}...")
                
                # Wait a bit between requests
                await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_single_query())
