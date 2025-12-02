"""
Phase 4.4: Suggestion Presenter

Formats and presents proactive suggestions for display in the chat interface.
This module handles presentation logic with minimal LLM usage (20%).

The presenter handles:
1. Formatting suggestions for different display modes
2. Adding UI metadata and styling hints
3. Tracking suggestion interactions
4. Generating user-friendly displays

Author: AI Istanbul Team
Date: December 3, 2025
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    ProactiveSuggestion,
    ProactiveSuggestionResponse,
    SuggestionInteraction
)

logger = logging.getLogger(__name__)


class SuggestionPresenter:
    """
    Suggestion presenter and formatter.
    
    Formats suggestions for display in chat interface with appropriate
    styling, grouping, and interaction tracking.
    
    LLM Responsibility: 20%
    - Format optimization (optional)
    - Display text enhancement (optional)
    
    Primary Method: Template-based formatting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the suggestion presenter.
        
        Args:
            config: Optional configuration overrides
        """
        # Configuration
        self.config = {
            'default_format': 'inline',  # inline, grouped, pills, dropdown
            'show_icons': True,
            'show_categories': False,
            'max_display_suggestions': 5,
            'track_interactions': True,
            **(config or {})
        }
        
        # Interaction tracking
        self.interactions: List[SuggestionInteraction] = []
        
        logger.info("SuggestionPresenter initialized")
    
    def format_suggestions(
        self,
        response: ProactiveSuggestionResponse,
        format_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format suggestions for display.
        
        Args:
            response: Complete suggestion response
            format_type: Display format (inline, grouped, pills, dropdown)
            
        Returns:
            Formatted display data
        """
        fmt = format_type or self.config['default_format']
        
        if fmt == 'inline':
            return self._format_inline(response)
        elif fmt == 'grouped':
            return self._format_grouped(response)
        elif fmt == 'pills':
            return self._format_pills(response)
        elif fmt == 'dropdown':
            return self._format_dropdown(response)
        else:
            logger.warning(f"Unknown format type: {fmt}, using inline")
            return self._format_inline(response)
    
    def format_for_chat(
        self,
        response: ProactiveSuggestionResponse
    ) -> Dict[str, Any]:
        """
        Format suggestions for chat API response.
        
        This is the main method used by the chat endpoint.
        
        Args:
            response: Complete suggestion response
            
        Returns:
            Chat-ready suggestion data
        """
        suggestions = response.suggestions[:self.config['max_display_suggestions']]
        
        return {
            'suggestions': [
                {
                    'id': s.suggestion_id,
                    'text': s.suggestion_text,
                    'icon': s.icon if self.config['show_icons'] else None,
                    'type': s.suggestion_type if self.config['show_categories'] else None,
                    'action': s.action_type,
                    'entities': s.entities,
                    'intent': s.intent_type
                }
                for s in suggestions
            ],
            'metadata': {
                'generation_method': response.generation_method,
                'generation_time_ms': response.generation_time_ms,
                'confidence': response.confidence,
                'diversity_score': response.diversity_score,
                'timestamp': response.timestamp.isoformat()
            }
        }
    
    def track_interaction(
        self,
        suggestion_id: str,
        action: str,
        session_id: Optional[str] = None,
        query_after: Optional[str] = None,
        rating: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> SuggestionInteraction:
        """
        Track user interaction with a suggestion.
        
        Args:
            suggestion_id: ID of the suggestion
            action: Action taken (clicked, ignored, dismissed, rated)
            session_id: User's session ID
            query_after: Query made after clicking
            rating: User rating (1-5)
            context: Additional context
            
        Returns:
            SuggestionInteraction record
        """
        interaction = SuggestionInteraction(
            suggestion_id=suggestion_id,
            action=action,
            timestamp=datetime.now(),
            session_id=session_id,
            query_after=query_after,
            rating=rating,
            context=context
        )
        
        if self.config['track_interactions']:
            self.interactions.append(interaction)
            logger.info(
                f"Tracked interaction: {suggestion_id} - {action}"
            )
        
        return interaction
    
    def get_interaction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about suggestion interactions.
        
        Returns:
            Statistics dictionary
        """
        if not self.interactions:
            return {
                'total_interactions': 0,
                'acceptance_rate': 0.0,
                'avg_rating': 0.0
            }
        
        total = len(self.interactions)
        clicked = sum(1 for i in self.interactions if i.action == 'clicked')
        ratings = [i.rating for i in self.interactions if i.rating is not None]
        
        return {
            'total_interactions': total,
            'clicked': clicked,
            'ignored': sum(1 for i in self.interactions if i.action == 'ignored'),
            'dismissed': sum(1 for i in self.interactions if i.action == 'dismissed'),
            'acceptance_rate': (clicked / total) if total > 0 else 0.0,
            'avg_rating': (sum(ratings) / len(ratings)) if ratings else 0.0,
            'total_rated': len(ratings)
        }
    
    def _format_inline(
        self,
        response: ProactiveSuggestionResponse
    ) -> Dict[str, Any]:
        """
        Format suggestions for inline display.
        
        Suggestions appear as a list below the main response.
        
        Args:
            response: Suggestion response
            
        Returns:
            Inline format data
        """
        suggestions = response.suggestions[:self.config['max_display_suggestions']]
        
        # Build display text
        display_lines = ["", "💡 **You might also want to:**"]
        for i, sugg in enumerate(suggestions, 1):
            icon = f"{sugg.icon} " if sugg.icon and self.config['show_icons'] else ""
            display_lines.append(f"{i}. {icon}{sugg.suggestion_text}")
        
        return {
            'format': 'inline',
            'display_text': '\n'.join(display_lines),
            'suggestions': [self._format_suggestion(s) for s in suggestions],
            'metadata': self._format_metadata(response)
        }
    
    def _format_grouped(
        self,
        response: ProactiveSuggestionResponse
    ) -> Dict[str, Any]:
        """
        Format suggestions grouped by category.
        
        Args:
            response: Suggestion response
            
        Returns:
            Grouped format data
        """
        suggestions = response.suggestions[:self.config['max_display_suggestions']]
        
        # Group by type
        groups: Dict[str, List[ProactiveSuggestion]] = {}
        for sugg in suggestions:
            if sugg.suggestion_type not in groups:
                groups[sugg.suggestion_type] = []
            groups[sugg.suggestion_type].append(sugg)
        
        # Build display
        display_lines = ["", "💡 **Suggestions:**", ""]
        
        category_labels = {
            'exploration': '🗺️ Explore',
            'practical': '🎯 Practical',
            'cultural': '🎭 Cultural',
            'dining': '🍽️ Dining',
            'refinement': '⚙️ Refine'
        }
        
        for category, items in groups.items():
            label = category_labels.get(category, category.title())
            display_lines.append(f"**{label}:**")
            for item in items:
                icon = f"{item.icon} " if item.icon and self.config['show_icons'] else "• "
                display_lines.append(f"  {icon}{item.suggestion_text}")
            display_lines.append("")
        
        return {
            'format': 'grouped',
            'display_text': '\n'.join(display_lines),
            'groups': {
                category: [self._format_suggestion(s) for s in items]
                for category, items in groups.items()
            },
            'metadata': self._format_metadata(response)
        }
    
    def _format_pills(
        self,
        response: ProactiveSuggestionResponse
    ) -> Dict[str, Any]:
        """
        Format suggestions as clickable pills/chips.
        
        Args:
            response: Suggestion response
            
        Returns:
            Pills format data
        """
        suggestions = response.suggestions[:self.config['max_display_suggestions']]
        
        pills = []
        for sugg in suggestions:
            pill = {
                'id': sugg.suggestion_id,
                'label': sugg.suggestion_text,
                'icon': sugg.icon if self.config['show_icons'] else None,
                'type': sugg.suggestion_type,
                'action': sugg.action_type,
                'data': {
                    'intent': sugg.intent_type,
                    'entities': sugg.entities
                }
            }
            pills.append(pill)
        
        return {
            'format': 'pills',
            'display_text': '\n\n💡 **Quick actions:**',
            'pills': pills,
            'metadata': self._format_metadata(response)
        }
    
    def _format_dropdown(
        self,
        response: ProactiveSuggestionResponse
    ) -> Dict[str, Any]:
        """
        Format suggestions for dropdown menu.
        
        Args:
            response: Suggestion response
            
        Returns:
            Dropdown format data
        """
        suggestions = response.suggestions[:self.config['max_display_suggestions']]
        
        options = []
        for sugg in suggestions:
            icon = f"{sugg.icon} " if sugg.icon and self.config['show_icons'] else ""
            option = {
                'id': sugg.suggestion_id,
                'label': f"{icon}{sugg.suggestion_text}",
                'value': sugg.intent_type,
                'data': {
                    'intent': sugg.intent_type,
                    'entities': sugg.entities
                }
            }
            options.append(option)
        
        return {
            'format': 'dropdown',
            'display_text': '\n\n💡 **What would you like to do next?**',
            'options': options,
            'metadata': self._format_metadata(response)
        }
    
    def _format_suggestion(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """
        Format a single suggestion for display.
        
        Args:
            suggestion: Suggestion to format
            
        Returns:
            Formatted suggestion data
        """
        return {
            'id': suggestion.suggestion_id,
            'text': suggestion.suggestion_text,
            'type': suggestion.suggestion_type,
            'intent': suggestion.intent_type,
            'icon': suggestion.icon,
            'relevance': suggestion.relevance_score,
            'action': suggestion.action_type,
            'entities': suggestion.entities
        }
    
    def _format_metadata(self, response: ProactiveSuggestionResponse) -> Dict[str, Any]:
        """
        Format metadata for display.
        
        Args:
            response: Suggestion response
            
        Returns:
            Formatted metadata
        """
        return {
            'generation_method': response.generation_method,
            'generation_time_ms': round(response.generation_time_ms, 2),
            'confidence': round(response.confidence, 2),
            'diversity_score': round(response.diversity_score, 2) if response.diversity_score else None,
            'total_considered': response.total_suggestions_considered,
            'llm_used': response.llm_used,
            'timestamp': response.timestamp.isoformat()
        }
    
    def create_quick_reply_buttons(
        self,
        suggestions: List[ProactiveSuggestion],
        max_buttons: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Create quick reply buttons for messaging platforms.
        
        Args:
            suggestions: List of suggestions
            max_buttons: Maximum number of buttons
            
        Returns:
            List of button data
        """
        buttons = []
        for sugg in suggestions[:max_buttons]:
            button = {
                'type': 'postback',
                'title': sugg.suggestion_text[:20],  # Truncate for button
                'payload': json.dumps({
                    'suggestion_id': sugg.suggestion_id,
                    'intent': sugg.intent_type,
                    'entities': sugg.entities
                })
            }
            buttons.append(button)
        
        return buttons
    
    def export_interactions(self, filepath: str) -> None:
        """
        Export interaction data to file.
        
        Args:
            filepath: Path to export file
        """
        try:
            data = {
                'interactions': [
                    {
                        'suggestion_id': i.suggestion_id,
                        'action': i.action,
                        'timestamp': i.timestamp.isoformat(),
                        'session_id': i.session_id,
                        'query_after': i.query_after,
                        'rating': i.rating
                    }
                    for i in self.interactions
                ],
                'stats': self.get_interaction_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported {len(self.interactions)} interactions to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export interactions: {e}")


# Singleton instance
_presenter_instance: Optional[SuggestionPresenter] = None


def get_suggestion_presenter(config: Optional[Dict[str, Any]] = None) -> SuggestionPresenter:
    """
    Get or create the singleton SuggestionPresenter instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        SuggestionPresenter instance
    """
    global _presenter_instance
    if _presenter_instance is None:
        _presenter_instance = SuggestionPresenter(config=config)
    return _presenter_instance


def format_suggestions_for_chat(
    response: ProactiveSuggestionResponse
) -> Dict[str, Any]:
    """
    Convenience function to format suggestions for chat.
    
    Args:
        response: Suggestion response
        
    Returns:
        Formatted suggestions
    """
    presenter = get_suggestion_presenter()
    return presenter.format_for_chat(response)
