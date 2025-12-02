"""
Phase 4.3: Intent Orchestrator

LLM-powered orchestration of multiple intent execution.
This module gives the LLM 90% control over planning execution order, dependencies,
and parallel vs sequential execution strategy.

The orchestrator creates an execution plan that determines:
1. What order to execute intents
2. Which intents can run in parallel
3. How to handle dependencies between intents
4. Fallback strategy for errors

Author: AI Istanbul Team
Date: December 2, 2025
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional

from .models import (
    MultiIntentDetection,
    ExecutionPlan,
    ExecutionStep
)

logger = logging.getLogger(__name__)


class IntentOrchestrator:
    """
    LLM-powered intent orchestrator.
    
    Creates execution plans for multi-intent queries, determining
    optimal execution order and parallelization strategy.
    
    LLM Responsibility: 90%
    - Execution order planning
    - Dependency resolution
    - Parallel execution identification
    - Conditional logic handling
    - Error handling strategy
    
    Fallback: 10%
    - Simple sequential execution
    - No parallelization
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent orchestrator.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'timeout_seconds': 5,
            'max_retries': 2,
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info(f"IntentOrchestrator initialized")
    
    async def create_execution_plan(
        self,
        detection: MultiIntentDetection,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create execution plan for detected intents using LLM.
        
        Args:
            detection: Multi-intent detection result
            context: Optional conversation context
            
        Returns:
            ExecutionPlan with ordered steps
        """
        start_time = time.time()
        
        # If single intent, create simple plan
        if detection.is_simple_query():
            return self._create_simple_plan(detection, start_time)
        
        try:
            # Prepare intent information for LLM
            intents_info = []
            for i, intent in enumerate(detection.intents):
                info = {
                    "index": i,
                    "type": intent.intent_type,
                    "parameters": intent.parameters,
                    "requires_location": intent.requires_location,
                    "depends_on": intent.depends_on,
                    "condition": intent.condition
                }
                intents_info.append(info)
            
            relationships_info = [
                {
                    "type": rel.relationship_type,
                    "intents": rel.intent_indices,
                    "description": rel.description
                }
                for rel in detection.relationships
            ]
            
            # Build prompt
            prompt = f"""Create execution plan for these intents:

Query: "{detection.original_query}"

Intents:
{json.dumps(intents_info, indent=2)}

Relationships:
{json.dumps(relationships_info, indent=2)}

Strategy: {detection.execution_strategy}

{self._get_orchestration_instructions()}

Return ONLY a valid JSON object with the structure specified above."""
            
            logger.info(f"Creating execution plan for {len(detection.intents)} intents...")
            
            # Call LLM via existing client
            llm_output = await self._call_llm(prompt)
            
            # Parse JSON response
            data = json.loads(llm_output)

            
            # Parse JSON response
            data = json.loads(llm_output)
            
            # Build ExecutionPlan
            steps = [
                ExecutionStep(
                    step_number=step_data["step_number"],
                    intent_indices=step_data["intent_indices"],
                    execution_mode=step_data["execution_mode"],
                    requires_results_from=step_data.get("requires_results_from"),
                    condition=step_data.get("condition"),
                    timeout_ms=step_data.get("timeout_ms", 5000)
                )
                for step_data in data["steps"]
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            plan = ExecutionPlan(
                plan_id=str(uuid.uuid4()),
                query=detection.original_query,
                steps=steps,
                total_steps=data["total_steps"],
                has_parallel_execution=data.get("has_parallel_execution", False),
                has_conditional_logic=data.get("has_conditional_logic", False),
                fallback_strategy=data.get("fallback_strategy", "skip_failed"),
                estimated_duration_ms=data.get("estimated_duration_ms"),
                planning_method="llm",
                planning_time_ms=processing_time
            )
            
            logger.info(
                f"Created execution plan: {plan.total_steps} steps, "
                f"parallel={plan.has_parallel_execution}, "
                f"conditional={plan.has_conditional_logic} "
                f"({processing_time:.0f}ms)"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"LLM orchestration failed: {e}, using fallback")
            return self._fallback_orchestration(detection, start_time)
    
    def _get_orchestration_instructions(self) -> str:
        """Get orchestration instructions for LLM."""
        return """You are an expert at planning efficient execution of multiple intents.

Your task is to create an optimal execution plan that:
1. Determines the correct order to execute intents
2. Identifies which intents can run in parallel
3. Respects dependencies between intents
4. Handles conditional logic
5. Plans for error scenarios

EXECUTION MODES:
- sequential: Execute one intent at a time (when order matters)
- parallel: Execute multiple intents simultaneously (when independent)

DEPENDENCY RULES:
- If Intent B depends on Intent A's result → A must run before B
- If Intent B uses Intent A's location → A must run before B
- If Intent A and B are independent → can run in parallel
- If Intent B is conditional on Intent A → A must complete first

ERROR STRATEGIES:
- sequential: Stop on error, or skip failed and continue
- skip_failed: Skip failed intents and continue
- stop_on_error: Stop entire execution on first error

RESPOND IN JSON:
{
  "steps": [
    {
      "step_number": 1,
      "intent_indices": [0],
      "execution_mode": "sequential|parallel",
      "requires_results_from": null,
      "condition": null,
      "timeout_ms": 5000
    }
  ],
  "total_steps": 2,
  "has_parallel_execution": false,
  "has_conditional_logic": false,
  "fallback_strategy": "skip_failed|stop_on_error|sequential",
  "estimated_duration_ms": 8000,
  "reasoning": "Step 1 gets route, Step 2 uses that location for restaurants"
}

IMPORTANT:
- Keep steps minimal but correct
- Parallelize when safe (no dependencies)
- Respect data flow between intents
- Plan realistic timeouts
- Explain your reasoning"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via existing client.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
        """
        try:
            # Check if client has OpenAI-style interface
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                # OpenAI-style client
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert execution planner."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    timeout=self.config['timeout_seconds']
                )
                return response.choices[0].message.content.strip()
            else:
                # Generic LLM client with generate method
                return await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.3
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _create_simple_plan(
        self,
        detection: MultiIntentDetection,
        start_time: float
    ) -> ExecutionPlan:
        """
        Create simple plan for single-intent query.
        
        Args:
            detection: Detection with single intent
            start_time: Start time for metrics
            
        Returns:
            ExecutionPlan with single step
        """
        processing_time = (time.time() - start_time) * 1000
        
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            query=detection.original_query,
            steps=[
                ExecutionStep(
                    step_number=1,
                    intent_indices=[0],
                    execution_mode="sequential",
                    requires_results_from=None,
                    condition=None,
                    timeout_ms=5000
                )
            ],
            total_steps=1,
            has_parallel_execution=False,
            has_conditional_logic=False,
            fallback_strategy="stop_on_error",
            estimated_duration_ms=3000,
            planning_method="simple",
            planning_time_ms=processing_time
        )
    
    def _fallback_orchestration(
        self,
        detection: MultiIntentDetection,
        start_time: float
    ) -> ExecutionPlan:
        """
        Fallback orchestration using simple sequential execution.
        
        Used when LLM orchestration fails (10% of cases).
        Simply executes all intents in order, one by one.
        
        Args:
            detection: Multi-intent detection
            start_time: Start time for metrics
            
        Returns:
            ExecutionPlan with sequential steps
        """
        # Create one step per intent, all sequential
        steps = []
        for i, intent in enumerate(detection.intents):
            steps.append(
                ExecutionStep(
                    step_number=i + 1,
                    intent_indices=[i],
                    execution_mode="sequential",
                    requires_results_from=[i] if i > 0 else None,
                    condition=intent.condition,
                    timeout_ms=5000
                )
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            query=detection.original_query,
            steps=steps,
            total_steps=len(steps),
            has_parallel_execution=False,
            has_conditional_logic=any(intent.condition for intent in detection.intents),
            fallback_strategy="skip_failed",
            estimated_duration_ms=len(steps) * 3000,  # Rough estimate
            planning_method="fallback",
            planning_time_ms=processing_time
        )
    
    def validate_plan(self, plan: ExecutionPlan, detection: MultiIntentDetection) -> bool:
        """
        Validate that execution plan is consistent with detected intents.
        
        Args:
            plan: Execution plan to validate
            detection: Original intent detection
            
        Returns:
            True if plan is valid
        """
        # Check all intents are covered
        all_indices = set()
        for step in plan.steps:
            all_indices.update(step.intent_indices)
        
        expected_indices = set(range(len(detection.intents)))
        if all_indices != expected_indices:
            logger.warning(
                f"Plan validation failed: missing intents. "
                f"Expected {expected_indices}, got {all_indices}"
            )
            return False
        
        # Check step dependencies are valid
        for step in plan.steps:
            if step.requires_results_from:
                for req_step in step.requires_results_from:
                    if req_step >= step.step_number:
                        logger.warning(
                            f"Plan validation failed: step {step.step_number} "
                            f"requires future step {req_step}"
                        )
                        return False
        
        return True


# Singleton instance
_orchestrator_instance: Optional[IntentOrchestrator] = None


def get_intent_orchestrator(llm_client, config: Optional[Dict[str, Any]] = None) -> IntentOrchestrator:
    """
    Get or create the singleton IntentOrchestrator instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        IntentOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = IntentOrchestrator(llm_client=llm_client, config=config)
    return _orchestrator_instance


async def orchestrate_intents(
    detection: MultiIntentDetection,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> ExecutionPlan:
    """
    Convenience function to orchestrate intents.
    
    Args:
        detection: Multi-intent detection result
        context: Optional conversation context
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        ExecutionPlan with ordered steps
    """
    if llm_client:
        orchestrator = get_intent_orchestrator(llm_client)
    else:
        orchestrator = _orchestrator_instance
        if orchestrator is None:
            raise ValueError("IntentOrchestrator not initialized. Provide llm_client.")
    
    return await orchestrator.create_execution_plan(detection, context)
