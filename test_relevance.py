#!/usr/bin/env python3
"""
Multiple Query Test - Test streaming response parsing for different question types
"""

import asyncio
import aiohttp
import json
import time

async def test_multiple_queries():
    queries = [
        "What are the metro lines in Istanbul?",
        "How do I get from Sultanahmet to Taksim Square?",
        "Tell me about Hagia Sophia"
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("-" * 60)
            
            start_time = time.time()
            
            async with session.post(
                "http://localhost:8000/ai/stream",
                json={"user_input": query}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    continue
                
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
                
                print(f"\n\n📊 Test {i} Summary:")
                print(f"Response Time: {response_time:.2f}s")
                print(f"Response Length: {len(full_response)} characters")
                
                # Check if response is relevant
                query_lower = query.lower()
                response_lower = full_response.lower()
                
                if 'metro' in query_lower:
                    relevant = any(word in response_lower for word in ['metro', 'm1', 'm2', 'm3', 'line', 'subway', 'underground'])
                elif 'sultanahmet' in query_lower and 'taksim' in query_lower:
                    relevant = any(word in response_lower for word in ['metro', 'bus', 'transport', 'tram', 'walk'])
                elif 'hagia sophia' in query_lower:
                    relevant = any(word in response_lower for word in ['hagia sophia', 'ayasofya', 'museum', 'mosque', 'byzantine'])
                else:
                    relevant = True
                
                print(f"Relevant to query: {'✅' if relevant else '❌'}")
                
                # Wait between requests
                await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(test_multiple_queries())
