# LLM Prompt Issue - CRITICAL FIX NEEDED 🔴

## Problem Identified

**Root Cause:** The LLM (Llama 3.1 8B) is returning system instructions as its response instead of actually answering the user's question.

### Evidence from Logs

```
2025-12-09 16:24:02,858 - services.llm.core - INFO - 🔍 RAW LLM RESPONSE (FULL): ' Please use the information in the context to answer the question.'
```

**What's happening:**
- User asks: "How do I get from Taksim to Sultan Ahmed by metro?"
- LLM returns: "Please use the information in the context to answer the question."
- This is a **meta-instruction**, not an answer!

---

## Analysis

### Why This Happens

1. **Prompt Confusion**: The system prompt contains instructions like "Use Context First" which the model is parroting back
2. **Lack of Few-Shot Examples**: The prompt doesn't show the model HOW to answer
3. **Instruction Overload**: Too many rules (🚨, ❌, ✅) may be confusing the 8B model
4. **No Clear Separator**: The prompt doesn't clearly separate instructions from the user query

### Current Prompt Structure (from logs)

```
🚨 CRITICAL INSTRUCTION:
... (long system instructions) ...
Context information will be provided below, followed by the user's question.

[RAG CONTEXT HERE]

Current User Question: How do I get from Taksim to Sultan Ahmed by metro?

Your Direct Answer:
```

**Problem:** The model sees "Please use the information in the context" somewhere in the prompt and returns it verbatim instead of generating an answer.

---

## Solution

###  🔴 Option 1: Immediate Fix - Simplify Prompt (RECOMMENDED)

**Action:** Drastically simplify the system prompt for Llama 3.1 8B

**New Prompt Structure:**
```
You are KAM, a friendly Istanbul tour guide. Answer the user's question directly using the context provided.

[CONTEXT]
{rag_context}

[QUESTION]
{user_query}

[YOUR ANSWER - Start immediately, no labels]
```

**Why this works:**
- Clearer structure
- Fewer confusing instructions
- No meta-commands that model might echo

### 🟡 Option 2: Add Few-Shot Examples

**Action:** Show the model 2-3 examples of good responses

**Example:**
```
Example 1:
Question: Where is Hagia Sophia?
Answer: Hagia Sophia is in Sultanahmet, the historic heart of Istanbul! 🕌 It's right next to the Blue Mosque...

Example 2:
Question: How do I get to Taksim from Kadıköy?
Answer: Here are the best routes from Kadıköy to Taksim:

🚇 ROUTE 1 (Recommended - Scenic):
...

Now answer this question:
{user_query}
```

### 🟢 Option 3: Increase max_tokens

**Current Setting:** `max_tokens: 150`

**Problem:** 150 tokens is VERY restrictive (about 100-120 words)

**Fix:**
```python
# In backend/services/runpod_llm_client.py or config
"max_tokens": 500  # Allow longer, more detailed responses
```

---

## Implementation Steps

### Step 1: Update Prompt (IMMEDIATE) 🔴

**File:** `backend/services/llm/prompts.py`

**Find the prompt in** `_default_system_prompts()` and replace with:

```python
def _default_system_prompts(self) -> Dict[str, str]:
    """Simplified, clear prompts for Llama 3.1 8B"""
    
    simplified_prompt = """You are KAM, a knowledgeable and friendly Istanbul tour guide. 

Your job is to answer the user's question directly and helpfully.

IMPORTANT RULES:
1. Start your answer immediately - no labels like "Assistant:" or "Answer:"
2. Use the context information provided below when available
3. Be specific and helpful
4. For transportation: Give step-by-step directions with real metro/tram lines
5. Always respond in the same language as the user's question

---
CONTEXT INFORMATION:
{context}

---
USER QUESTION:
{query}

---
YOUR ANSWER (start immediately):"""

    return {
        'en': simplified_prompt,
        'tr': simplified_prompt,
        'fr': simplified_prompt,
    }
```

### Step 2: Increase max_tokens 🟡

**File:** `backend/services/runpod_llm_client.py` or wherever LLM config is

**Find:**
```python
"max_tokens": 150
```

**Change to:**
```python
"max_tokens": 500  # Allow detailed responses
```

### Step 3: Test After Changes 🟢

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I get from Taksim to Sultan Ahmed by metro?",
    "language": "en"
  }'
```

**Expected:** Actual directions, not "Please use the information..."

---

## Current System Status

### What's Working ✅
- Backend server: Running (port 8001)
- RAG system: Retrieving 3 relevant items
- Response sanitizer: Integrated and ready
- All services: 13/13 active

### What's Broken ❌
- **LLM Response Generation**: Returning meta-instructions
- Prompt structure confusing the model
- max_tokens too restrictive (150)

###Files Involved

**Need to modify:**
1. `backend/services/llm/prompts.py` - Update system prompt
2. `backend/services/runpod_llm_client.py` or config - Increase max_tokens
3. **Optional:** `backend/core/startup.py` - Update LLM config

**Already working (no changes needed):**
- `backend/utils/response_sanitizer.py` - Ready to clean outputs
- `backend/api/chat.py` - Sanitizer integrated
- `backend/services/database_rag_service.py` - RAG working

---

## Testing Plan

### Test 1: Transportation Query
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How to get to Hagia Sophia from Taksim?", "language": "en"}'
```

**Expected Response:**
```
To get from Taksim to Hagia Sophia:

🚇 ROUTE 1:
Step 1: From Taksim → Take F1 Funicular to Kabataş
Step 2: From Kabataş → Take T1 Tram to Sultanahmet
...
```

### Test 2: Restaurant Query
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Best kebab restaurants in Sultanahmet?", "language": "en"}'
```

**Expected:** List of 2-3 specific restaurants with details

### Test 3: General Info
```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the Grand Bazaar", "language": "en"}'
```

**Expected:** Description of Grand Bazaar, not meta-instructions

---

## Priority Actions

### 🔴 URGENT (Do Now)
1. **Simplify the prompt** in `prompts.py`
2. **Increase max_tokens** to 500
3. **Restart backend** server
4. **Test with one query**

### 🟡 Important (Do Today)
5. Add few-shot examples if simple prompt doesn't work
6. Test multiple query types
7. Monitor sanitizer effectiveness

### 🟢 Nice to Have (Do This Week)
8. A/B test different prompt variations
9. Fine-tune max_tokens based on response quality
10. Update RAG data (airport, metro info)

---

## Root Cause Summary

**The Issue:**
- Llama 3.1 8B is an instruction-following model
- Our prompt has TOO MANY instructions with confusing formatting (🚨, ❌, ✅)
- The model is echoing instructions instead of following them
- max_tokens=150 is too restrictive for detailed answers

**The Fix:**
1. Simplified, clearer prompt structure
2. More tokens for complete responses
3. Few-shot examples showing desired format

**Expected Outcome:**
- LLM generates actual answers
- Sanitizer cleans up any remaining artifacts
- Users get helpful, specific responses

---

## Verification Checklist

After applying fixes:

- [ ] LLM returns actual answers (not meta-instructions)
- [ ] Transportation queries get step-by-step directions
- [ ] Restaurant queries get specific recommendations
- [ ] Responses are 200-400 characters (not just 65)
- [ ] Sanitizer removes any remaining artifacts
- [ ] Language consistency maintained
- [ ] Response time under 10 seconds

---

*Generated: December 9, 2025*
*Status: Issue Identified, Fix Required*
*Priority: CRITICAL 🔴*
