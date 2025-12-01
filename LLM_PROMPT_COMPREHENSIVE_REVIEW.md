# LLM Prompt Comprehensive Review & Improvement Plan

**Date**: December 1, 2024  
**Status**: ✅ Complete Analysis  
**Priority**: High - Production Enhancement

---

## 📋 Executive Summary

This document provides a comprehensive review of all LLM prompts across the AI Istanbul system, identifies issues, and proposes concrete improvements. The system currently has **8 major prompt locations** with varying quality and consistency.

### Key Findings:
- ✅ **Strong**: Language enforcement, Istanbul expertise, context integration
- ⚠️ **Needs Improvement**: Consistency across files, duplication, output format control
- ❌ **Issues**: Some prompts lack structure, multi-language instructions vary

---

## 🔍 Prompt Inventory

### 1. **Core LLM Client** (`backend/services/runpod_llm_client.py`)

#### Location 1: `generate_istanbul_response()` - Lines 359-390
**Current Prompt:**
```python
system_context = f"""You are an AI assistant specialized in Istanbul tourism.

LANGUAGE RULE - CRITICAL:
User's query is in: {detected_language}
You MUST respond 100% in: {detected_language}
NEVER mix languages in your response.

Provide helpful, accurate, and friendly information about Istanbul's attractions,
restaurants, neighborhoods, transportation, and local culture.
Keep responses concise and actionable (150-200 words).
Use natural, conversational tone in {detected_language}.
"""
```

**Assessment:**
- ✅ Clear language enforcement
- ✅ Concise and focused
- ⚠️ Word limit might be too restrictive for complex queries
- ⚠️ No explicit output format guidance
- ❌ Doesn't mention using provided context data

**Improvements Needed:**
1. Add explicit instruction to use provided context
2. Make word limit flexible based on query complexity
3. Add format guidelines (emojis, bullet points, structure)
4. Add examples of good vs bad responses

---

#### Location 2: `generate_with_service_context()` - Lines 434-490
**Current Prompt:**
```python
system_prompt = f"""You are an AI assistant specialized in Istanbul tourism with access to real-time data.

CRITICAL LANGUAGE RULES - MANDATORY:
🔴 NEVER EVER mix languages in your response
🔴 Keep the ENTIRE response in ONE language ONLY
🔴 User's query language: {detected_language}
🔴 You MUST respond 100% in: {detected_language}
🔴 Do NOT add translations or explanations in other languages
🔴 Do NOT translate proper nouns (Istanbul, Taksim, Galata, etc.)
🔴 Keep place names, restaurant names, street names in original spelling

SUPPORTED LANGUAGES:
✅ English - for English queries
✅ Turkish (Türkçe) - for Turkish queries  
✅ Arabic (العربية) - for Arabic queries
✅ Russian (Русский) - for Russian queries
✅ French (Français) - for French queries
✅ German (Deutsch) - for German queries

RESPONSE INSTRUCTIONS:
1. Use the provided Context data to give accurate, specific recommendations
2. Include specific names, ratings, locations, and practical details from Context
3. Be concise but informative (200-300 words maximum)
4. Always mention specific options from the Context when available
5. If Context is limited, acknowledge it and provide general guidance
6. Use natural, conversational tone in the detected language
7. Format with emojis and bullet points for readability

REMEMBER: Respond ONLY in {detected_language}. No mixed languages!
"""
```

**Assessment:**
- ✅ Excellent language enforcement (multiple redundant reminders)
- ✅ Explicitly mentions using context data
- ✅ Good formatting instructions (emojis, bullets)
- ✅ Lists all supported languages
- ✅ Proper noun handling is clear
- ⚠️ Very repetitive language rules (5+ repetitions)
- ⚠️ Could be more concise while maintaining clarity

**Improvements Needed:**
1. Consolidate language rules (reduce repetition)
2. Add examples of proper context usage
3. Add fallback behavior when context is insufficient
4. Add quality checks (e.g., "Never say 'I don't know' - provide alternatives")

---

### 2. **Prompt Builder** (`backend/services/llm_handler/prompt_builder.py`)

#### Location 3: Base System Prompt - Lines 46-76
**Current Prompt:**
```python
self.base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

CRITICAL LANGUAGE RULES:
🔴 NEVER mix languages in your response
🔴 Keep the ENTIRE response in ONE language
🔴 Match the language of the user's query
🔴 If query is Turkish, respond 100% in Turkish
🔴 If query is English, respond 100% in English
🔴 Keep place names in original (e.g., "Sultanahmet", "Beyoğlu")
🔴 Do NOT translate proper nouns (restaurant/place names)

Response Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the SAME LANGUAGE as the query (100% consistency)
6. Never make up information - use context only

Format:
- Start with direct answer
- List 3-5 specific recommendations
- Include practical details
- Add a local tip or insight
"""
```

**Assessment:**
- ✅ Clear identity ("AI Istanbul")
- ✅ Good domain coverage with emojis
- ✅ Strong language rules
- ✅ Explicit instruction to use context only (no hallucinations)
- ✅ Format guidelines provided
- ⚠️ Only mentions Turkish and English (missing 4 languages)
- ⚠️ "3-5 recommendations" might not fit all query types
- ❌ No guidance on tone/personality
- ❌ No instruction on handling uncertainty

**Improvements Needed:**
1. List all 6 supported languages
2. Make recommendation count flexible
3. Add personality guidelines (friendly, helpful, enthusiastic but professional)
4. Add uncertainty handling ("I don't have current data, but typically...")
5. Add safety disclaimers for health/legal questions

---

#### Location 4: Intent-Specific Prompts - Lines 77-115
**Current Prompts:**
```python
self.intent_prompts = {
    'restaurant': """Focus on restaurants from the provided database context.
Include: name, location, cuisine, price range, rating.
Mention dietary options if relevant.""",

    'attraction': """Focus on attractions and museums from the provided context.
Include: name, district, description, opening hours, ticket price.
Prioritize based on location and interests.""",

    'transportation': """Provide clear transportation directions.
Include: metro lines, bus numbers, ferry routes.
Mention transfer points and approximate times.
If available, include a map visualization link.""",
    
    # ... etc
}
```

**Assessment:**
- ✅ Intent-specific guidance is helpful
- ✅ Specifies required information fields
- ⚠️ Very brief, lacks examples
- ⚠️ No language-specific adaptations
- ❌ No tone guidance per intent
- ❌ No fallback behavior

**Improvements Needed:**
1. Expand each intent prompt with examples
2. Add tone guidance (e.g., transportation = precise, attractions = enthusiastic)
3. Add fallback strategies when data is limited
4. Add multi-language considerations per intent
5. Add cross-selling opportunities (e.g., restaurants near attractions)

---

#### Location 5: Language Enforcement Box - Lines 205-213
**Current Prompt:**
```python
prompt_parts.append(f"""
╔════════════════════════════════════════════════════╗
║  CRITICAL: LANGUAGE CONSISTENCY RULE               ║
║  ✅ Respond ONLY in {lang_name}                     ║
║  ❌ Do NOT mix languages                            ║
║  ❌ Do NOT use English words in {lang_name} response║
║  ❌ Do NOT translate names (keep original)         ║
║  ✅ Use {lang_name} throughout entire response     ║
╚════════════════════════════════════════════════════╝
""")
```

**Assessment:**
- ✅ Visually striking format (hard to miss)
- ✅ Clear do's and don'ts
- ✅ Covers key language issues
- ⚠️ Redundant with other language rules in the same prompt
- ⚠️ ASCII box might confuse some LLMs

**Recommendation:** Keep this format but consolidate with other language rules to avoid 3-4 separate language instructions in the same prompt.

---

### 3. **Query Validator** (`backend/services/query_validator.py`)

#### Location 6: Validation Prompt - Lines 258-298
**Current Prompt:**
```python
prompt = f"""You are a query validation expert for Istanbul tourism.

Query: "{query}"
Language: {lang_name}

Analyze this query and determine:
1. Is it answerable? (Does it ask something we can help with about Istanbul?)
2. What's the complexity level?
   - SIMPLE: Basic fact (hours, location, price)
   - MEDIUM: Recommendations, comparisons, planning
   - COMPLEX: Multi-step itinerary, detailed analysis
3. Are there any issues or missing information?
4. Estimated response time in seconds

Respond in this JSON format:
{...examples...}
"""
```

**Assessment:**
- ✅ Clear role definition
- ✅ Structured output format (JSON)
- ✅ Excellent examples provided
- ✅ Clear complexity classification
- ✅ Multiple examples for different cases
- ⚠️ Could add edge cases (off-topic, spam, unsafe queries)
- ⚠️ No instruction on what to do with non-Istanbul queries

**Improvements Needed:**
1. Add handling for off-topic queries
2. Add spam/abuse detection guidance
3. Add multi-language query detection (code-switching)
4. Add handling for follow-up questions without context

---

#### Location 7: Clarification Prompt - Lines 436-467
**Current Prompt:**
```python
prompt = f"""You are a helpful AI assistant for Istanbul tourism.

User's Query: "{query}"
Detected Intent: {intent}
Confidence: {confidence:.2f}

This query is ambiguous or unclear. {instruction} to help understand what the user wants.

Guidelines:
- Keep it short and natural (one question)
- Focus on the most important missing information
- Be friendly and helpful
- Don't repeat the user's query

Examples:
Query: "best places"
Clarification: "What kind of places are you interested in? Museums, restaurants, parks, or something else?"
...
"""
```

**Assessment:**
- ✅ Clear purpose (clarification)
- ✅ Good guidelines (short, natural, friendly)
- ✅ Helpful examples
- ✅ Multi-language support
- ⚠️ Could add more diverse examples
- ⚠️ No guidance on offering suggestions alongside clarification

**Improvements Needed:**
1. Add examples for different ambiguity types
2. Include instruction to offer 2-3 options in clarification
3. Add tone variation based on user's language/culture

---

### 4. **Query Explainer** (`backend/services/query_explainer.py`)

#### Location 8: Explanation Prompt - Lines 169-204
**Current Prompt:**
```python
prompt = f"""You are explaining how an AI query understanding system interpreted a user's question.

USER QUERY: "{query}"
DETECTED SIGNALS: {signal_summary}
PRIMARY INTENT: {primary_intent}
CONTEXT: {context_summary}

YOUR TASK: Explain to the user how you understood their question. Be transparent and helpful.

{lang_instruction}

Format your explanation as a JSON object with these fields:
{{
  "summary": "One sentence summary",
  "detected_intents": [...],
  "confidence": "high/medium/low",
  "explanation": "Detailed explanation",
  "signals_breakdown": {...},
  "what_ill_do": "Action based on understanding"
}}

IMPORTANT: Respond with ONLY the JSON object, no additional text.
"""
```

**Assessment:**
- ✅ Clear meta-cognitive task (explaining AI reasoning)
- ✅ Structured JSON output
- ✅ Transparency focus (good for trust)
- ✅ Multi-language support
- ⚠️ Complex output structure might lead to errors
- ⚠️ No examples provided
- ❌ Missing guidance on simplifying technical terms for users

**Improvements Needed:**
1. Add 2-3 complete examples
2. Add instruction to avoid technical jargon
3. Add fallback for when explanation can't be generated
4. Simplify JSON structure (fewer nested fields)

---

### 5. **Query Rewriter** (`backend/services/query_rewriter_simple.py`)

#### Location 9: Enhancement Prompt - Lines 245-280
**Current Prompt:**
```python
prompt_parts = [
    "You are a query enhancement assistant for an Istanbul tourism chatbot.",
    "Your task: Rewrite the user's query to be clearer and more specific.",
    "",
    "Guidelines:",
    "- Expand abbreviations and shorthand",
    "- Add context from conversation if needed",
    "- Make implicit information explicit",
    "- Keep the original intent",
    "- Keep it concise (max 20 words)",
    "- Maintain the same language",
    ...
]
```

**Assessment:**
- ✅ Clear task definition
- ✅ Good guidelines
- ✅ Incorporates conversation context
- ✅ Maintains language consistency
- ⚠️ No examples
- ⚠️ 20-word limit might be too strict
- ❌ No guidance on when NOT to rewrite

**Improvements Needed:**
1. Add examples (before/after pairs)
2. Add instruction to detect already-clear queries
3. Relax word limit for complex queries
4. Add instruction to preserve user's tone/style

---

### 6. **A/B Testing Prompts** (`backend/services/ab_testing/experiment_manager.py`)

#### Location 10: Test Variants - Lines 339-346
**Current Prompts:**
```python
"system_prompt": "You are a helpful Istanbul travel assistant."  # Control
"system_prompt": "You are an enthusiastic Istanbul travel expert!"  # Creative
```

**Assessment:**
- ⚠️ Too simplistic for production use
- ⚠️ Lacks all the rich context of main prompts
- ❌ No language enforcement
- ❌ No Istanbul-specific knowledge
- ❌ Testing only tone, not substance

**Improvements Needed:**
1. Use full production prompts as base
2. Test meaningful variations (context usage, format, specificity)
3. Ensure all variants have language enforcement
4. Test domain-specific variations

---

## 🎯 Priority Improvements

### **Critical (Do Now)**

1. **Consolidate Language Rules** - Reduce redundancy across 3-4 language instructions per prompt
2. **Add Output Format Control** - Ensure consistent JSON output where needed
3. **Standardize Personality** - Define consistent tone across all prompts
4. **Add Comprehensive Examples** - Every prompt should have 2-3 examples

### **High Priority (This Week)**

5. **Expand Intent Prompts** - Add detailed guidance for each intent type
6. **Add Fallback Behaviors** - What to do when context is insufficient
7. **Improve Multi-language Support** - Ensure all 6 languages are explicitly mentioned
8. **Add Safety Guidelines** - Handle sensitive queries (health, legal, safety)

### **Medium Priority (Next Sprint)**

9. **Optimize Prompt Length** - Balance between detail and token efficiency
10. **Add Cross-Intent Linking** - Suggest related services naturally
11. **Enhance Validation** - Better edge case handling
12. **A/B Test Real Variations** - Test meaningful prompt improvements

---

## 📝 Recommended Prompt Template (Standard)

Here's a proposed standard template for all main LLM prompts:

```python
STANDARD_PROMPT_TEMPLATE = """
# ROLE & IDENTITY
You are AI Istanbul, an expert travel assistant specializing in Istanbul, Turkey.

# CORE COMPETENCIES
- 🏛️ Attractions & Culture
- 🍽️ Restaurants & Dining
- 🚇 Transportation & Navigation
- 🏘️ Neighborhoods & Local Tips
- 🎭 Events & Activities
- 💎 Hidden Gems

# LANGUAGE RULES (MANDATORY)
🌍 Supported Languages: English, Turkish, Arabic, Russian, French, German
🔴 Respond 100% in the user's query language: {detected_language}
🔴 NEVER mix languages in your response
✅ Keep place names in original Turkish (Sultanahmet, Beyoğlu, etc.)
✅ Use natural, conversational {detected_language} throughout

# DATA USAGE
📊 ALWAYS use provided Context data for factual information
📊 Include specific names, locations, ratings, prices from Context
📊 If Context is limited, acknowledge it and provide general guidance
🚫 NEVER make up information - use Context or say "I don't have current data"

# RESPONSE FORMAT
1. Start with a direct answer to the user's question
2. Provide 2-5 specific recommendations (adjust based on query)
3. Include practical details (hours, prices, directions, tips)
4. Use emojis and bullet points for readability
5. Keep response length appropriate (150-400 words based on complexity)
6. End with a helpful tip or suggestion

# TONE & PERSONALITY
- Friendly and enthusiastic about Istanbul
- Professional but approachable
- Helpful and informative, never dismissive
- Culturally sensitive and respectful

# SPECIAL CASES
- Unsafe/health/legal questions → Provide disclaimers, suggest professionals
- Off-topic questions → Politely redirect to Istanbul tourism
- Ambiguous questions → Ask clarifying question while offering options
- Follow-up questions → Use conversation context naturally

# CONTEXT PROVIDED
{context_sections}

# USER QUERY
{query}

# YOUR RESPONSE (in {detected_language})
"""
```

---

## 🧪 Testing Recommendations

### **Prompt Quality Tests**

1. **Language Consistency Test**
   - Test each prompt with queries in all 6 languages
   - Verify no language mixing in responses
   - Check proper noun handling

2. **Context Usage Test**
   - Provide rich context, verify it's used
   - Provide minimal context, check fallback behavior
   - Provide no context, ensure appropriate response

3. **Format Compliance Test**
   - Verify JSON output where required
   - Check emoji and bullet point usage
   - Validate response length guidelines

4. **Edge Case Test**
   - Off-topic queries
   - Ambiguous queries
   - Multi-language queries (code-switching)
   - Sensitive topics
   - Follow-up questions

### **A/B Testing Ideas**

1. **Context Emphasis**: Test prompts with/without explicit "use context" instructions
2. **Verbosity**: Test different word limits (150 vs 300 vs flexible)
3. **Examples**: Test prompts with/without examples in the prompt
4. **Structure**: Test different formatting (bullet points vs paragraphs vs mixed)
5. **Personality**: Test different tones (enthusiastic vs professional vs casual)

---

## 🚀 Implementation Plan

### **Phase 1: Immediate Fixes (This Session)**
- [ ] Consolidate language rules across all prompts
- [ ] Add missing language support (Arabic, Russian, French, German) to base prompt
- [ ] Add output format control to main prompts
- [ ] Create standard prompt template

### **Phase 2: Enhancements (Next)**
- [ ] Expand intent-specific prompts with examples
- [ ] Add fallback behavior instructions
- [ ] Improve query validator edge cases
- [ ] Enhance safety guidelines

### **Phase 3: Optimization (Future)**
- [ ] A/B test prompt variations
- [ ] Gather user feedback on response quality
- [ ] Optimize for token efficiency
- [ ] Add dynamic prompt adaptation based on performance

---

## 📊 Success Metrics

Track these metrics to measure prompt improvements:

1. **Language Consistency**: % of responses with no language mixing
2. **Context Usage**: % of responses that reference provided context
3. **Format Compliance**: % of responses following format guidelines
4. **User Satisfaction**: Feedback scores on response quality
5. **Response Relevance**: % of responses that directly address query
6. **Hallucination Rate**: % of responses with made-up information
7. **Clarification Need**: % of queries requiring clarification

---

## 💡 Key Insights

### **What's Working Well:**
✅ Strong language enforcement (multiple reminders work)
✅ Explicit context usage instructions
✅ Clear role definitions
✅ Structured output formats (JSON)
✅ Multi-language support infrastructure

### **What Needs Improvement:**
⚠️ Too much redundancy (3-4 language rules per prompt)
⚠️ Inconsistency across different files
⚠️ Missing examples in many prompts
⚠️ Limited fallback behavior guidance
⚠️ No standardized prompt structure

### **Opportunities:**
💡 Create a prompt library with reusable components
💡 Implement dynamic prompt selection based on query type
💡 Add user feedback loop to improve prompts over time
💡 Develop language-specific prompt variations
💡 Build prompt versioning and A/B testing framework

---

## 📚 References

### **Prompt Engineering Best Practices:**
1. Be specific and explicit
2. Provide examples (few-shot learning)
3. Use structured output formats
4. Give the model a role/persona
5. Break complex tasks into steps
6. Use positive instructions ("do this") over negative ("don't do this")
7. Test edge cases extensively
8. Iterate based on real-world performance

### **Istanbul-Specific Considerations:**
- Multi-language environment (6 languages)
- Rich cultural context (history, religion, customs)
- Practical tourism needs (navigation, safety, currency)
- Real-time data integration (weather, events, traffic)
- Local vs tourist perspective balance

---

**Next Steps**: Ready to implement Phase 1 improvements. Please review and approve, or let me know if you'd like to focus on specific prompts first.
