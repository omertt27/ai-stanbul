# Phase 4.3: Quick Reference Guide

**Status**: ✅ COMPLETE  
**Date**: December 2, 2025

---

## Quick Import

```python
from backend.services.llm import (
    # Detector
    MultiIntentDetector,
    get_multi_intent_detector,
    detect_multi_intent,
    # Orchestrator
    IntentOrchestrator,
    get_intent_orchestrator,
    orchestrate_intents,
    # Synthesizer
    ResponseSynthesizer,
    get_response_synthesizer,
    synthesize_multi_intent_response,
    # Models
    MultiIntentDetection,
    ExecutionPlan,
    MultiIntentResponse,
    DetectedIntent,
    IntentResult
)
```

---

## Usage Patterns

### Pattern 1: Using Singletons (Recommended)

```python
from services.runpod_llm_client import get_llm_client
from backend.services.llm import (
    get_multi_intent_detector,
    get_intent_orchestrator,
    get_response_synthesizer
)

# Get LLM client once
llm_client = get_llm_client()

# Initialize singletons
detector = get_multi_intent_detector(llm_client)
orchestrator = get_intent_orchestrator(llm_client)
synthesizer = get_response_synthesizer(llm_client)

# Use in pipeline
detection = await detector.detect_intents(query, context)
plan = await orchestrator.create_execution_plan(detection)
response = await synthesizer.synthesize_responses(query, results, detection)
```

### Pattern 2: Using Convenience Functions

```python
from backend.services.llm import (
    detect_multi_intent,
    orchestrate_intents,
    synthesize_multi_intent_response
)

# One-liners (uses singletons internally)
detection = await detect_multi_intent(query, context, llm_client)
plan = await orchestrate_intents(detection, context, llm_client)
response = await synthesize_multi_intent_response(query, results, detection, llm_client)
```

### Pattern 3: Direct Instantiation (Not Recommended)

```python
from backend.services.llm import (
    MultiIntentDetector,
    IntentOrchestrator,
    ResponseSynthesizer
)

# Create new instances (less efficient)
detector = MultiIntentDetector(llm_client, config={...})
orchestrator = IntentOrchestrator(llm_client, config={...})
synthesizer = ResponseSynthesizer(llm_client, config={...})
```

---

## Complete Pipeline Example

```python
async def handle_multi_intent_query(query: str, context: dict, llm_client):
    """Handle query with potential multiple intents."""
    
    # Step 1: Detect intents
    detection = await detect_multi_intent(query, context, llm_client)
    
    # Step 2: Check if multi-intent
    if not detection.is_multi_intent:
        # Single intent - use existing pipeline
        return await handle_single_intent(detection.intents[0])
    
    # Step 3: Orchestrate execution
    plan = await orchestrate_intents(detection, context, llm_client)
    
    # Step 4: Execute intents according to plan
    intent_results = []
    for step in plan.steps:
        if step.execution_mode == "parallel":
            # Execute in parallel
            tasks = [
                execute_intent(detection.intents[idx])
                for idx in step.intent_indices
            ]
            step_results = await asyncio.gather(*tasks)
            intent_results.extend(step_results)
        else:
            # Execute sequentially
            for idx in step.intent_indices:
                result = await execute_intent(detection.intents[idx])
                intent_results.append(result)
    
    # Step 5: Synthesize response
    final_response = await synthesize_multi_intent_response(
        query, intent_results, detection, context, llm_client
    )
    
    return final_response.synthesized_response


async def execute_intent(intent: DetectedIntent) -> IntentResult:
    """Execute a single intent (your implementation)."""
    try:
        # Route to appropriate handler based on intent_type
        if intent.intent_type == "route":
            response = await handle_route(intent.parameters)
        elif intent.intent_type == "restaurant":
            response = await handle_restaurant(intent.parameters)
        # ... other handlers ...
        
        return IntentResult(
            intent_index=intent.priority - 1,
            intent_type=intent.intent_type,
            success=True,
            response=response,
            execution_time_ms=100
        )
    except Exception as e:
        return IntentResult(
            intent_index=intent.priority - 1,
            intent_type=intent.intent_type,
            success=False,
            error=str(e),
            execution_time_ms=50
        )
```

---

## Configuration Options

### MultiIntentDetector Config:

```python
detector_config = {
    'timeout_seconds': 5,
    'max_retries': 2,
    'fallback_enabled': True
}
detector = MultiIntentDetector(llm_client, config=detector_config)
```

### IntentOrchestrator Config:

```python
orchestrator_config = {
    'timeout_seconds': 5,
    'max_retries': 2,
    'fallback_enabled': True
}
orchestrator = IntentOrchestrator(llm_client, config=orchestrator_config)
```

### ResponseSynthesizer Config:

```python
synthesizer_config = {
    'timeout_seconds': 5,
    'temperature': 0.7,
    'max_tokens': 1000,
    'fallback_enabled': True
}
synthesizer = ResponseSynthesizer(llm_client, config=synthesizer_config)
```

---

## Model Reference

### MultiIntentDetection

```python
detection = MultiIntentDetection(
    original_query=str,
    intent_count=int,
    intents=List[DetectedIntent],
    relationships=List[IntentRelationship],
    execution_strategy="sequential|parallel|conditional|mixed",
    is_multi_intent=bool,
    confidence=float,
    detection_method="llm|fallback",
    processing_time_ms=float
)

# Methods:
detection.is_simple_query()  # → bool
detection.has_dependencies()  # → bool
detection.has_conditions()  # → bool
```

### ExecutionPlan

```python
plan = ExecutionPlan(
    plan_id=str,
    query=str,
    steps=List[ExecutionStep],
    total_steps=int,
    has_parallel_execution=bool,
    has_conditional_logic=bool,
    fallback_strategy="sequential|skip_failed|stop_on_error",
    estimated_duration_ms=int,
    planning_method="llm|fallback|simple",
    planning_time_ms=float
)
```

### MultiIntentResponse

```python
response = MultiIntentResponse(
    original_query=str,
    intent_results=List[IntentResult],
    synthesized_response=str,  # ← The final answer
    response_structure="narrative|structured|comparison|conditional",
    coherence_score=float,
    completeness=float,
    synthesis_method="llm|template",
    synthesis_time_ms=float,
    total_processing_time_ms=float
)
```

---

## Common Patterns

### Check for Multi-Intent:

```python
detection = await detect_multi_intent(query, context, llm_client)

if detection.is_multi_intent:
    # Handle as multi-intent
    ...
else:
    # Handle as single intent
    ...
```

### Handle Dependencies:

```python
if detection.has_dependencies():
    # Ensure sequential execution for dependent intents
    plan = await orchestrate_intents(detection, context, llm_client)
    # Execute following plan's step order
```

### Handle Conditional Logic:

```python
if detection.has_conditions():
    # Execute first intent, then decide on second
    first_result = await execute_intent(detection.intents[0])
    
    # Choose next intent based on condition
    if evaluate_condition(first_result, detection.intents[1].condition):
        second_result = await execute_intent(detection.intents[1])
```

---

## Error Handling

```python
try:
    detection = await detect_multi_intent(query, context, llm_client)
    
    if detection.detection_method == "fallback":
        logger.warning("Used fallback detection (LLM failed)")
    
    plan = await orchestrate_intents(detection, context, llm_client)
    
    if plan.planning_method == "fallback":
        logger.warning("Used fallback orchestration (LLM failed)")
    
    # Execute...
    
    response = await synthesize_multi_intent_response(
        query, results, detection, llm_client
    )
    
    if response.synthesis_method == "template":
        logger.warning("Used template synthesis (LLM failed)")
    
except Exception as e:
    logger.error(f"Multi-intent pipeline failed: {e}")
    # Fall back to single-intent handling
    return await handle_as_single_intent(query)
```

---

## Testing

```python
import pytest
from backend.services.llm import detect_multi_intent

@pytest.mark.asyncio
async def test_sequential_detection(llm_client):
    query = "Show me route to Hagia Sophia and find restaurants near there"
    detection = await detect_multi_intent(query, {}, llm_client)
    
    assert detection.is_multi_intent is True
    assert detection.intent_count >= 2
    assert "route" in [i.intent_type for i in detection.intents]
    assert "restaurant" in [i.intent_type for i in detection.intents]
```

---

## Performance Tips

1. **Use Singletons**: Initialize once, reuse many times
2. **Enable Caching**: Consider caching detection results for similar queries
3. **Parallel Execution**: Let orchestrator identify parallel opportunities
4. **Monitor Latency**: Track `processing_time_ms` and `total_processing_time_ms`
5. **Fallback Strategy**: Ensure `skip_failed` to handle partial failures gracefully

---

## Integration Checklist

- [ ] Import Phase 4.3 modules
- [ ] Get existing LLM client
- [ ] Initialize singletons (detector, orchestrator, synthesizer)
- [ ] Add to chat pipeline after Phase 4.2 (context resolution)
- [ ] Check `is_multi_intent` before routing
- [ ] Execute intents according to plan
- [ ] Synthesize final response
- [ ] Add logging and metrics
- [ ] Test with production queries

---

## Quick Commands

```bash
# Verify installation
python -c "from backend.services.llm import MultiIntentDetector; print('✅ OK')"

# Run tests
pytest test_phase4_3_multi_intent.py -v

# Check syntax
python -m py_compile backend/services/llm/multi_intent_detector.py
python -m py_compile backend/services/llm/intent_orchestrator.py
python -m py_compile backend/services/llm/response_synthesizer.py
```

---

**Phase 4.3 Ready** ✅  
**For detailed docs**: See `PHASE4_3_MULTI_INTENT_COMPLETE.md`

---

*Istanbul AI Team - December 2, 2025*
