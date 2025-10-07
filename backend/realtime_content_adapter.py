#!/usr/bin/env python3
"""
Real-time Content Adaptation System
===================================

Adapts content in real-time based on user behavior, current context,
and dynamic factors like time, location, events, and user engagement patterns.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class AdaptationTrigger(Enum):
    USER_ENGAGEMENT = "user_engagement"
    TIME_CONTEXT = "time_context"
    LOCATION_CONTEXT = "location_context"
    CONVERSATION_FLOW = "conversation_flow"
    CONTENT_PERFORMANCE = "content_performance"
    REAL_TIME_EVENTS = "real_time_events"

@dataclass
class ContentAdaptation:
    """Represents a real-time content adaptation"""
    trigger: AdaptationTrigger
    adaptation_type: str
    original_content: str
    adapted_content: str
    confidence_score: float
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class UserEngagementMetrics:
    """Track user engagement with content"""
    user_id: str
    session_id: str
    response_time: float  # Time to respond to AI
    query_complexity: float  # Complexity of follow-up questions
    satisfaction_indicators: List[str]  # "follow_up", "detailed_query", "thank_you", etc.
    content_preferences: Dict[str, float]  # Preference scores for content types
    attention_span: float  # Estimated attention span based on behavior
    interaction_depth: int  # How deep into conversation they go
    last_updated: datetime

class RealTimeContentAdapter:
    """Main real-time content adaptation engine"""
    
    def __init__(self):
        self.user_engagement_metrics = {}
        self.content_performance_data = {}
        self.adaptation_rules = self._load_adaptation_rules()
        self.context_monitors = {}
        
    def _load_adaptation_rules(self) -> Dict[str, Any]:
        """Load rules for real-time content adaptation"""
        return {
            "engagement_thresholds": {
                "high_engagement": {
                    "response_time": {"min": 0, "max": 30},  # Quick responses
                    "query_complexity": {"min": 0.7, "max": 1.0},  # Complex follow-ups
                    "satisfaction_indicators": 3,  # Multiple positive indicators
                    "adaptations": ["increase_detail", "add_expert_content", "provide_alternatives"]
                },
                "medium_engagement": {
                    "response_time": {"min": 30, "max": 120},
                    "query_complexity": {"min": 0.4, "max": 0.7},
                    "satisfaction_indicators": 1,
                    "adaptations": ["maintain_current", "add_visual_cues", "suggest_followups"]
                },
                "low_engagement": {
                    "response_time": {"min": 120, "max": 300},
                    "query_complexity": {"min": 0, "max": 0.4},
                    "satisfaction_indicators": 0,
                    "adaptations": ["simplify_content", "add_engagement_hooks", "provide_quick_wins"]
                },
                "very_low_engagement": {
                    "response_time": {"min": 300, "max": 999999},
                    "query_complexity": {"min": 0, "max": 0.2},
                    "satisfaction_indicators": 0,
                    "adaptations": ["emergency_simplification", "attention_grabbers", "immediate_value"]
                }
            },
            "time_based_adaptations": {
                "morning": {
                    "hours": [6, 7, 8, 9, 10, 11],
                    "adaptations": ["breakfast_focus", "energy_building", "planning_oriented"]
                },
                "afternoon": {
                    "hours": [12, 13, 14, 15, 16, 17],
                    "adaptations": ["lunch_recommendations", "activity_focused", "practical_immediate"]
                },
                "evening": {
                    "hours": [18, 19, 20, 21],
                    "adaptations": ["dinner_focus", "entertainment_options", "relaxation_oriented"]
                },
                "night": {
                    "hours": [22, 23, 0, 1, 2, 3, 4, 5],
                    "adaptations": ["next_day_planning", "quiet_options", "safety_conscious"]
                }
            },
            "conversation_flow_adaptations": {
                "opening": {
                    "turn_number": 1,
                    "adaptations": ["welcoming_tone", "overview_first", "build_rapport"]
                },
                "building": {
                    "turn_number": [2, 3, 4, 5],
                    "adaptations": ["build_on_previous", "add_depth", "show_expertise"]
                },
                "deep_dive": {
                    "turn_number": [6, 7, 8, 9, 10],
                    "adaptations": ["expert_level", "insider_knowledge", "personalized_insights"]
                },
                "conclusion": {
                    "turn_number": [11, 12, 13, 14, 15],
                    "adaptations": ["summarize_key_points", "action_oriented", "memorable_ending"]
                }
            },
            "content_performance_adaptations": {
                "high_performing": {
                    "user_rating": {"min": 0.8, "max": 1.0},
                    "follow_up_rate": {"min": 0.7, "max": 1.0},
                    "adaptations": ["replicate_pattern", "expand_successful_elements", "maintain_quality"]
                },
                "underperforming": {
                    "user_rating": {"min": 0, "max": 0.6},
                    "follow_up_rate": {"min": 0, "max": 0.3},
                    "adaptations": ["content_restructure", "add_engagement_elements", "simplify_approach"]
                }
            }
        }
    
    def track_user_engagement(self, user_id: str, session_id: str, interaction_data: Dict[str, Any]):
        """Track and update user engagement metrics"""
        if user_id not in self.user_engagement_metrics:
            self.user_engagement_metrics[user_id] = UserEngagementMetrics(
                user_id=user_id,
                session_id=session_id,
                response_time=60.0,  # Default
                query_complexity=0.5,  # Default
                satisfaction_indicators=[],
                content_preferences={},
                attention_span=180.0,  # Default 3 minutes
                interaction_depth=0,
                last_updated=datetime.now()
            )
        
        metrics = self.user_engagement_metrics[user_id]
        
        # Update response time (time between AI response and user's next query)
        if "response_time" in interaction_data:
            # Use exponential moving average to smooth out response times
            alpha = 0.3  # Smoothing factor
            metrics.response_time = (alpha * interaction_data["response_time"] + 
                                   (1 - alpha) * metrics.response_time)
        
        # Update query complexity based on query analysis
        if "query_complexity" in interaction_data:
            metrics.query_complexity = interaction_data["query_complexity"]
        
        # Update satisfaction indicators
        satisfaction_signals = interaction_data.get("satisfaction_signals", [])
        for signal in satisfaction_signals:
            if signal not in metrics.satisfaction_indicators:
                metrics.satisfaction_indicators.append(signal)
        
        # Keep only recent satisfaction indicators (last 10)
        if len(metrics.satisfaction_indicators) > 10:
            metrics.satisfaction_indicators = metrics.satisfaction_indicators[-10:]
        
        # Update content preferences
        content_type = interaction_data.get("content_type", "general")
        user_rating = interaction_data.get("user_rating", 0.5)
        
        if content_type not in metrics.content_preferences:
            metrics.content_preferences[content_type] = 0.5
        
        # Update preference using exponential moving average
        alpha = 0.2
        metrics.content_preferences[content_type] = (
            alpha * user_rating + (1 - alpha) * metrics.content_preferences[content_type]
        )
        
        # Update interaction depth
        metrics.interaction_depth = interaction_data.get("turn_number", metrics.interaction_depth)
        
        # Estimate attention span based on engagement patterns
        if metrics.response_time < 30 and metrics.query_complexity > 0.6:
            metrics.attention_span = min(metrics.attention_span + 30, 600)  # Increase up to 10 min
        elif metrics.response_time > 180:
            metrics.attention_span = max(metrics.attention_span - 30, 60)   # Decrease down to 1 min
        
        metrics.last_updated = datetime.now()
    
    def get_engagement_level(self, user_id: str) -> str:
        """Determine user's current engagement level"""
        if user_id not in self.user_engagement_metrics:
            return "medium_engagement"  # Default
        
        metrics = self.user_engagement_metrics[user_id]
        rules = self.adaptation_rules["engagement_thresholds"]
        
        # Check each engagement level
        for level, criteria in rules.items():
            response_time_ok = (criteria["response_time"]["min"] <= metrics.response_time <= 
                              criteria["response_time"]["max"])
            complexity_ok = (criteria["query_complexity"]["min"] <= metrics.query_complexity <= 
                           criteria["query_complexity"]["max"])
            satisfaction_ok = len(metrics.satisfaction_indicators) >= criteria["satisfaction_indicators"]
            
            if response_time_ok and complexity_ok and satisfaction_ok:
                return level
        
        return "medium_engagement"  # Fallback
    
    async def adapt_content_realtime(self, original_response: str, user_id: str, 
                                   session_id: str, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main method to adapt content in real-time"""
        adaptations = []
        
        # 1. Engagement-based adaptation
        engagement_level = self.get_engagement_level(user_id)
        engagement_adaptation = await self._adapt_for_engagement(
            original_response, engagement_level, user_id, query
        )
        if engagement_adaptation:
            adaptations.append(engagement_adaptation)
        
        # 2. Time-based adaptation
        time_adaptation = await self._adapt_for_time_context(original_response, query)
        if time_adaptation:
            adaptations.append(time_adaptation)
        
        # 3. Conversation flow adaptation
        turn_number = context.get("turn_number", 1) if context else 1
        flow_adaptation = await self._adapt_for_conversation_flow(
            original_response, turn_number, user_id
        )
        if flow_adaptation:
            adaptations.append(flow_adaptation)
        
        # 4. Performance-based adaptation
        performance_adaptation = await self._adapt_for_content_performance(
            original_response, user_id, query
        )
        if performance_adaptation:
            adaptations.append(performance_adaptation)
        
        # 5. Real-time event adaptation
        event_adaptation = await self._adapt_for_real_time_events(
            original_response, query, context
        )
        if event_adaptation:
            adaptations.append(event_adaptation)
        
        # Apply adaptations to create final response
        adapted_response = await self._apply_adaptations(original_response, adaptations)
        
        return {
            "adapted_response": adapted_response,
            "adaptations_applied": len(adaptations),
            "engagement_level": engagement_level,
            "adaptation_details": [
                {
                    "trigger": a.trigger.value,
                    "type": a.adaptation_type,
                    "confidence": a.confidence_score,
                    "reasoning": a.reasoning
                } for a in adaptations
            ],
            "performance_metrics": await self._calculate_adaptation_performance(adaptations)
        }
    
    async def _adapt_for_engagement(self, response: str, engagement_level: str, 
                                  user_id: str, query: str) -> Optional[ContentAdaptation]:
        """Adapt content based on user engagement level"""
        rules = self.adaptation_rules["engagement_thresholds"].get(engagement_level, {})
        adaptations = rules.get("adaptations", [])
        
        if not adaptations:
            return None
        
        # Apply the most relevant adaptation
        primary_adaptation = adaptations[0]
        
        adaptation_strategies = {
            "increase_detail": self._increase_content_detail,
            "add_expert_content": self._add_expert_insights,
            "provide_alternatives": self._add_alternative_options,
            "simplify_content": self._simplify_content_structure,
            "add_engagement_hooks": self._add_engagement_elements,
            "provide_quick_wins": self._add_immediate_value,
            "emergency_simplification": self._emergency_content_simplification,
            "attention_grabbers": self._add_attention_grabbing_elements,
            "immediate_value": self._focus_on_immediate_practical_value
        }
        
        if primary_adaptation in adaptation_strategies:
            adapted_content = await adaptation_strategies[primary_adaptation](response, query, user_id)
            
            return ContentAdaptation(
                trigger=AdaptationTrigger.USER_ENGAGEMENT,
                adaptation_type=primary_adaptation,
                original_content=response,
                adapted_content=adapted_content,
                confidence_score=0.8,
                reasoning=f"User showing {engagement_level} - applied {primary_adaptation}",
                metadata={"engagement_level": engagement_level, "user_id": user_id}
            )
        
        return None
    
    async def _adapt_for_time_context(self, response: str, query: str) -> Optional[ContentAdaptation]:
        """Adapt content based on current time"""
        current_hour = datetime.now().hour
        time_rules = self.adaptation_rules["time_based_adaptations"]
        
        # Find the appropriate time period
        time_period = None
        for period, config in time_rules.items():
            if current_hour in config["hours"]:
                time_period = period
                break
        
        if not time_period:
            return None
        
        adaptations = time_rules[time_period]["adaptations"]
        primary_adaptation = adaptations[0]
        
        # Apply time-based content modifications
        time_strategies = {
            "breakfast_focus": lambda r, q: self._add_morning_content_focus(r, q, "breakfast"),
            "lunch_recommendations": lambda r, q: self._add_meal_time_focus(r, q, "lunch"),
            "dinner_focus": lambda r, q: self._add_evening_content_focus(r, q, "dinner"),
            "next_day_planning": lambda r, q: self._add_planning_focus(r, q, "next_day"),
            "energy_building": lambda r, q: self._add_energetic_tone(r, q),
            "relaxation_oriented": lambda r, q: self._add_relaxed_tone(r, q),
            "safety_conscious": lambda r, q: self._add_safety_focus(r, q, "night")
        }
        
        if primary_adaptation in time_strategies:
            adapted_content = await time_strategies[primary_adaptation](response, query)
            
            return ContentAdaptation(
                trigger=AdaptationTrigger.TIME_CONTEXT,
                adaptation_type=primary_adaptation,
                original_content=response,
                adapted_content=adapted_content,
                confidence_score=0.7,
                reasoning=f"Adapted for {time_period} time context",
                metadata={"time_period": time_period, "hour": current_hour}
            )
        
        return None
    
    async def _adapt_for_conversation_flow(self, response: str, turn_number: int, 
                                         user_id: str) -> Optional[ContentAdaptation]:
        """Adapt content based on conversation flow position"""
        flow_rules = self.adaptation_rules["conversation_flow_adaptations"]
        
        # Determine conversation stage
        stage = None
        for stage_name, config in flow_rules.items():
            if isinstance(config["turn_number"], list):
                if turn_number in config["turn_number"]:
                    stage = stage_name
                    break
            else:
                if turn_number == config["turn_number"]:
                    stage = stage_name
                    break
        
        if not stage:
            return None
        
        adaptations = flow_rules[stage]["adaptations"]
        primary_adaptation = adaptations[0]
        
        flow_strategies = {
            "welcoming_tone": self._add_welcoming_elements,
            "overview_first": self._structure_as_overview,
            "build_on_previous": self._add_continuity_elements,
            "add_depth": self._increase_content_depth,
            "expert_level": self._elevate_to_expert_level,
            "insider_knowledge": self._add_insider_insights,
            "summarize_key_points": self._add_summary_elements,
            "action_oriented": self._make_more_actionable,
            "memorable_ending": self._add_memorable_conclusion
        }
        
        if primary_adaptation in flow_strategies:
            adapted_content = await flow_strategies[primary_adaptation](response, turn_number, user_id)
            
            return ContentAdaptation(
                trigger=AdaptationTrigger.CONVERSATION_FLOW,
                adaptation_type=primary_adaptation,
                original_content=response,
                adapted_content=adapted_content,
                confidence_score=0.75,
                reasoning=f"Adapted for {stage} stage of conversation (turn {turn_number})",
                metadata={"conversation_stage": stage, "turn_number": turn_number}
            )
        
        return None
    
    # Content adaptation strategy implementations
    async def _increase_content_detail(self, response: str, query: str, user_id: str) -> str:
        """Increase detail level for highly engaged users"""
        # Add more specific information, examples, and context
        enhanced_response = response
        
        # Add detailed examples
        if "restaurant" in query.lower():
            enhanced_response += "\n\n🍽️ **Detailed Dining Guide:**\n"
            enhanced_response += "• **Menu Navigation**: Look for 'günün menüsü' (daily menu) for authentic seasonal dishes\n"
            enhanced_response += "• **Ordering Etiquette**: Turkish dining emphasizes sharing - order multiple dishes to share\n"
            enhanced_response += "• **Payment Culture**: Service charge ('servis ücreti') may be included, tip 10-15% additionally\n"
        
        # Add contextual depth
        enhanced_response += "\n\n📚 **Additional Context**: This recommendation is based on local expertise and current visitor feedback."
        
        return enhanced_response
    
    async def _add_expert_insights(self, response: str, query: str, user_id: str) -> str:
        """Add expert-level insights for engaged users"""
        enhanced_response = response
        
        enhanced_response += "\n\n🎓 **Expert Insight**: "
        
        if "museum" in response.lower():
            enhanced_response += "Professional guides recommend visiting during the first or last hour of operation for optimal viewing conditions and photography opportunities."
        elif "district" in response.lower():
            enhanced_response += "Urban planning experts note that this area represents a unique blend of historical preservation and modern development, making it a case study in successful city evolution."
        else:
            enhanced_response += "Local cultural experts emphasize that understanding the historical context enhances the authentic Istanbul experience significantly."
        
        return enhanced_response
    
    async def _simplify_content_structure(self, response: str, query: str, user_id: str) -> str:
        """Simplify content for users with low engagement"""
        # Break down into simple, digestible points
        lines = response.split('\n')
        simplified_lines = []
        
        for line in lines:
            if line.strip():
                # Remove complex phrases and simplify
                simplified_line = line.replace("comprehensive", "complete")
                simplified_line = simplified_line.replace("sophisticated", "good")
                simplified_line = simplified_line.replace("exceptional", "great")
                
                # Keep lines under 80 characters when possible
                if len(simplified_line) > 80:
                    # Try to break at natural points
                    words = simplified_line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 75:
                            simplified_lines.append(current_line)
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        simplified_lines.append(current_line.strip())
                else:
                    simplified_lines.append(simplified_line)
        
        simplified_response = '\n'.join(simplified_lines)
        
        # Add quick action summary
        simplified_response += "\n\n⚡ **Quick Summary**: " + self._extract_key_action(response)
        
        return simplified_response
    
    async def _add_engagement_elements(self, response: str, query: str, user_id: str) -> str:
        """Add elements to increase user engagement"""
        enhanced_response = response
        
        # Add interactive elements
        enhanced_response += "\n\n🎯 **Quick Question**: What aspect of Istanbul interests you most - the food scene, historical sites, or local culture?"
        
        # Add personalization hook
        enhanced_response += "\n\n💡 **Just for You**: Based on your interest, I can provide more specific recommendations!"
        
        return enhanced_response
    
    async def _add_immediate_value(self, response: str, query: str, user_id: str) -> str:
        """Focus on immediate practical value"""
        enhanced_response = response
        
        # Add immediate actionable tips
        enhanced_response += "\n\n⚡ **Right Now You Can**:\n"
        enhanced_response += "1. Save these locations in Google Maps for offline access\n"
        enhanced_response += "2. Download the BiTaksi app for easy transportation\n"
        enhanced_response += "3. Screenshot this response for quick reference while exploring\n"
        
        return enhanced_response
    
    async def _add_morning_content_focus(self, response: str, query: str, focus_type: str) -> str:
        """Add morning-specific content focus"""
        enhanced_response = response
        
        if focus_type == "breakfast":
            enhanced_response += "\n\n🌅 **Perfect Morning Start**: These locations offer excellent Turkish breakfast (kahvaltı) - try the traditional spread with olives, cheese, tomatoes, and fresh bread."
        
        return enhanced_response
    
    async def _add_evening_content_focus(self, response: str, query: str, focus_type: str) -> str:
        """Add evening-specific content focus"""
        enhanced_response = response
        
        if focus_type == "dinner":
            enhanced_response += "\n\n🌆 **Evening Atmosphere**: These venues transform beautifully at sunset - consider booking for golden hour views over the Bosphorus."
        
        return enhanced_response
    
    def _extract_key_action(self, response: str) -> str:
        """Extract the most important actionable item from response"""
        # Simple extraction of action-oriented sentences
        sentences = response.split('.')
        for sentence in sentences:
            if any(action_word in sentence.lower() for action_word in ['visit', 'go to', 'try', 'book', 'take']):
                return sentence.strip()[:100] + "..."
        
        return "Explore the recommended locations and enjoy your Istanbul experience!"
    
    async def _apply_adaptations(self, original_response: str, adaptations: List[ContentAdaptation]) -> str:
        """Apply all adaptations to create final response"""
        if not adaptations:
            return original_response
        
        # Sort adaptations by confidence score
        sorted_adaptations = sorted(adaptations, key=lambda x: x.confidence_score, reverse=True)
        
        # Apply the highest confidence adaptation
        return sorted_adaptations[0].adapted_content
    
    async def _calculate_adaptation_performance(self, adaptations: List[ContentAdaptation]) -> Dict[str, Any]:
        """Calculate performance metrics for adaptations"""
        if not adaptations:
            return {"total_adaptations": 0}
        
        return {
            "total_adaptations": len(adaptations),
            "average_confidence": sum(a.confidence_score for a in adaptations) / len(adaptations),
            "adaptation_types": [a.adaptation_type for a in adaptations],
            "triggers_used": list(set(a.trigger.value for a in adaptations))
        }
    
    # Additional placeholder methods for other adaptation strategies
    async def _add_alternative_options(self, response: str, query: str, user_id: str) -> str:
        return response + "\n\n🔄 **Alternative Options**: [Additional alternatives would be generated here]"
    
    async def _emergency_content_simplification(self, response: str, query: str, user_id: str) -> str:
        # Ultra-simplified version
        key_points = response.split('\n')[:3]  # Take first 3 lines
        return '\n'.join(key_points) + "\n\n⚡ **Quick Tip**: Ask me for specific details about any of these!"
    
    async def _add_attention_grabbing_elements(self, response: str, query: str, user_id: str) -> str:
        return "🎉 **Amazing Discovery Awaits!** " + response
    
    async def _focus_on_immediate_practical_value(self, response: str, query: str, user_id: str) -> str:
        return response + "\n\n🎯 **Take Action Now**: Use these recommendations within the next 2 hours for the best experience!"
    
    # Placeholder methods for conversation flow adaptations
    async def _add_welcoming_elements(self, response: str, turn_number: int, user_id: str) -> str:
        return "👋 **Welcome to Istanbul Discovery!** " + response
    
    async def _structure_as_overview(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n📋 **Overview Complete**: Feel free to ask for details about any specific aspect!"
    
    async def _add_continuity_elements(self, response: str, turn_number: int, user_id: str) -> str:
        return "Building on what we discussed... " + response
    
    async def _increase_content_depth(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n🔍 **Deeper Insight**: [Enhanced depth would be added here]"
    
    async def _elevate_to_expert_level(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n🎓 **Expert Level**: [Professional insights would be added here]"
    
    async def _add_insider_insights(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n🔑 **Insider Knowledge**: [Local secrets would be shared here]"
    
    async def _add_summary_elements(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n📝 **Key Takeaways**: [Summary points would be listed here]"
    
    async def _make_more_actionable(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n✅ **Your Action Plan**: [Specific steps would be outlined here]"
    
    async def _add_memorable_conclusion(self, response: str, turn_number: int, user_id: str) -> str:
        return response + "\n\n🌟 **Remember**: Istanbul's magic lies in its unexpected moments - stay curious!"
    
    # Placeholder methods for other adaptations
    async def _adapt_for_content_performance(self, response: str, user_id: str, query: str) -> Optional[ContentAdaptation]:
        return None  # Implementation would analyze past performance and adapt accordingly
    
    async def _adapt_for_real_time_events(self, response: str, query: str, context: Dict[str, Any] = None) -> Optional[ContentAdaptation]:
        return None  # Implementation would check for real-time events, weather, etc.
    
    async def _add_meal_time_focus(self, response: str, query: str, meal_type: str) -> str:
        return response + f"\n\n🍽️ **{meal_type.title()} Time**: Perfect timing for {meal_type} recommendations!"
    
    async def _add_planning_focus(self, response: str, query: str, planning_type: str) -> str:
        return response + f"\n\n📅 **{planning_type.replace('_', ' ').title()}**: [Planning suggestions would be added here]"
    
    async def _add_energetic_tone(self, response: str, query: str) -> str:
        return response.replace(".", "! 🚀")  # Add energy to the tone
    
    async def _add_relaxed_tone(self, response: str, query: str) -> str:
        return response + "\n\n😌 **Take Your Time**: Enjoy the peaceful evening atmosphere..."
    
    async def _add_safety_focus(self, response: str, query: str, time_context: str) -> str:
        return response + f"\n\n🛡️ **{time_context.title()} Safety**: Stay in well-lit areas and use official taxis or ride-sharing apps."

# Global instance
realtime_content_adapter = RealTimeContentAdapter()

async def adapt_content_realtime(response: str, user_id: str, session_id: str, 
                                query: str, interaction_data: Dict[str, Any] = None, 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main function to adapt content in real-time"""
    # Track user engagement first
    if interaction_data:
        realtime_content_adapter.track_user_engagement(user_id, session_id, interaction_data)
    
    # Adapt content
    return await realtime_content_adapter.adapt_content_realtime(
        response, user_id, session_id, query, context
    )
