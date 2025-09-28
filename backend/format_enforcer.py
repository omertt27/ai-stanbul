#!/usr/bin/env python3
"""
Response Format Enforcement Module
================================

This module ensures AI responses follow required format structures
for each category, fixing formatting issues and adding missing structure.
"""

from enum import Enum
from typing import Dict, List

class PromptCategory(Enum):
    """Categories for specialized prompts"""
    DAILY_TALK = "daily_talk"
    RESTAURANT_SPECIFIC = "restaurant_specific"  
    RESTAURANT_GENERAL = "restaurant_general"
    DISTRICT_ADVICE = "district_advice"
    MUSEUM_ADVICE = "museum_advice"
    TRANSPORTATION = "transportation"
    CULTURAL_SITES = "cultural_sites"
    SHOPPING = "shopping"
    NIGHTLIFE = "nightlife"
    SAFETY_PRACTICAL = "safety_practical"
    GENERIC = "generic"

def enforce_response_format(response: str, category: PromptCategory) -> str:
    """
    Post-process AI response to ensure strict format compliance
    """
    # Remove markdown formatting
    response = response.replace('**', '').replace('*', '')
    
    # Define required format structures for each category
    format_templates = {
        PromptCategory.TRANSPORTATION: {
            'sections': [
                'IMMEDIATE BEST ROUTE',
                'STEP-BY-STEP DETAILED DIRECTIONS', 
                'ISTANBULKART COMPREHENSIVE GUIDE',
                'MULTIPLE ROUTE ALTERNATIVES',
                'CULTURAL TRANSPORT ETIQUETTE & SAFETY'
            ],
            'max_words': 300
        },
        PromptCategory.MUSEUM_ADVICE: {
            'sections': [
                'IMMEDIATE PRACTICAL ANSWER',
                'TOP MUSEUMS/CULTURAL SITES',
                'DETAILED ACCESS & LOGISTICS', 
                'CULTURAL CONTEXT & ETIQUETTE',
                'EXPERT VISITING STRATEGIES'
            ],
            'max_words': 350
        },
        PromptCategory.RESTAURANT_SPECIFIC: {
            'sections': [
                'DIRECT RECOMMENDATION',
                'TOP RESTAURANT RECOMMENDATIONS',
                'DETAILED LOCATION ACCESS WITH MAPS INTEGRATION',
                'AUTHENTIC DINING EXPERIENCE',
                'LOCAL FOOD CULTURE INTEGRATION'
            ],
            'max_words': 400
        },
        PromptCategory.DAILY_TALK: {
            'sections': [
                'EMPATHETIC OPENING',
                'IMMEDIATE REASSURANCE', 
                'ACTIONABLE SOLUTIONS',
                'CULTURAL GUIDANCE',
                'CONFIDENCE BUILDERS',
                'BACKUP OPTIONS'
            ],
            'max_words': 350
        }
    }
    
    if category not in format_templates:
        return response
        
    template = format_templates[category]
    required_sections = template['sections']
    
    # Check if response has proper structure
    has_structure = any(section.lower().replace(' ', '') in response.lower().replace(' ', '') for section in required_sections)
    
    if not has_structure:
        # Response doesn't have required structure - add it
        response = add_structure_to_response(response, category, required_sections)
    
    # Ensure word limit
    words = response.split()
    if len(words) > template['max_words']:
        response = ' '.join(words[:template['max_words']]) + '...'
    
    return response

def add_structure_to_response(response: str, category: PromptCategory, required_sections: list) -> str:
    """
    Add proper structure to unstructured response
    """
    lines = response.split('\n')
    content_lines = [line.strip() for line in lines if line.strip()]
    
    if category == PromptCategory.DAILY_TALK:
        return format_daily_talk_response(content_lines)
    elif category == PromptCategory.RESTAURANT_SPECIFIC:
        return format_restaurant_response(content_lines)
    elif category == PromptCategory.TRANSPORTATION:
        return format_transportation_response(content_lines)
    elif category == PromptCategory.MUSEUM_ADVICE:
        return format_museum_response(content_lines)
    
    return response

def format_daily_talk_response(content_lines: list) -> str:
    """Format daily talk response with proper structure"""
    response = ""
    
    # Empathetic opening
    response += "EMPATHETIC OPENING:\n"
    response += f"{content_lines[0] if content_lines else 'I understand your concern about Istanbul.'}\n\n"
    
    # Immediate reassurance
    response += "IMMEDIATE REASSURANCE:\n"
    response += f"{content_lines[1] if len(content_lines) > 1 else 'Let me help you navigate this amazing city step by step.'}\n\n"
    
    # Actionable solutions
    response += "ACTIONABLE SOLUTIONS:\n"
    for i, line in enumerate(content_lines[2:6] if len(content_lines) > 2 else content_lines):
        response += f"{i+1}. {line}\n"
    
    response += "\nCULTURAL GUIDANCE:\n"
    response += "Remember to dress modestly when visiting mosques, try local Turkish tea, and don't hesitate to ask locals for help - they are usually very friendly.\n\n"
    
    response += "CONFIDENCE BUILDERS:\n"
    response += "Take your time, use translation apps, and remember that getting a bit lost is part of the Istanbul adventure.\n\n"
    
    response += "BACKUP OPTIONS:\n"
    response += "If you feel overwhelmed, consider joining a guided tour or staying in well-known tourist areas until you feel more confident."
    
    return response

def format_restaurant_response(content_lines: list) -> str:
    """Format restaurant response with proper structure"""
    response = ""
    
    response += "DIRECT RECOMMENDATION:\n"
    response += f"{content_lines[0] if content_lines else 'Here are excellent dining options in your requested area.'}\n\n"
    
    response += "TOP RESTAURANT RECOMMENDATIONS:\n"
    for i, line in enumerate(content_lines[1:5] if len(content_lines) > 1 else content_lines):
        if line and not line.startswith('DIRECT'):
            response += f"• {line}\n"
    
    response += "\nDETAILED LOCATION ACCESS:\n"
    response += "All recommended restaurants are easily accessible by public transport with walking directions provided.\n\n"
    
    response += "AUTHENTIC DINING EXPERIENCE:\n"
    response += "These locations offer authentic Turkish cuisine with local atmosphere and traditional preparations.\n\n"
    
    response += "LOCAL FOOD CULTURE INTEGRATION:\n"
    response += "Experience Turkish hospitality, try Turkish tea with your meal, and enjoy the social dining culture."
    
    return response

def format_transportation_response(content_lines: list) -> str:
    """Format transportation response with proper structure"""
    response = ""
    
    response += "IMMEDIATE BEST ROUTE:\n"
    response += f"{content_lines[0] if content_lines else 'Take the most efficient public transport route.'}\n\n"
    
    response += "STEP-BY-STEP DETAILED DIRECTIONS:\n"
    for i, line in enumerate(content_lines[1:4] if len(content_lines) > 1 else content_lines):
        response += f"{i+1}. {line}\n"
    
    response += "\nISTANBULKART COMPREHENSIVE GUIDE:\n"
    response += "Use Istanbulkart for all public transport - available at metro stations, most cost-effective option.\n\n"
    
    response += "MULTIPLE ROUTE ALTERNATIVES:\n"
    response += "Alternative routes include taxi, bus, or walking depending on traffic and time preferences.\n\n"
    
    response += "CULTURAL TRANSPORT ETIQUETTE & SAFETY:\n"
    response += "Respect priority seating, keep belongings secure, have Istanbulkart ready, stations are well-lit and safe."
    
    return response

def format_museum_response(content_lines: list) -> str:
    """Format museum response with proper structure"""
    response = ""
    
    response += "IMMEDIATE PRACTICAL ANSWER:\n"
    response += f"{content_lines[0] if content_lines else 'Here is essential information for visiting this cultural site.'}\n\n"
    
    response += "TOP MUSEUMS/CULTURAL SITES:\n"
    for i, line in enumerate(content_lines[1:3] if len(content_lines) > 1 else content_lines):
        response += f"• {line}\n"
    
    response += "\nDETAILED ACCESS & LOGISTICS:\n"
    response += "Accessible by tram/metro with specific walking directions and best visiting times provided.\n\n"
    
    response += "CULTURAL CONTEXT & ETIQUETTE:\n"
    response += "Follow dress codes for religious sites, respect prayer times, no flash photography in certain areas.\n\n"
    
    response += "EXPERT VISITING STRATEGIES:\n"
    response += "Visit early morning or late afternoon, consider museum passes, allow adequate time for exploration."
    
    return response
