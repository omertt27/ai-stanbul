"""
Integration module to enhance the existing AI Istanbul chatbot
with improved context awareness, query understanding, and knowledge scope
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request

# Import our enhanced modules
from enhanced_chatbot import (
    EnhancedContextManager, 
    EnhancedQueryUnderstanding, 
    EnhancedKnowledgeBase,
    ContextAwareResponseGenerator,
    get_museum_detailed_info,
    get_neighborhood_detailed_info,
    generate_day_itinerary
)

# Global instances for enhanced functionality
context_manager = EnhancedContextManager()
query_understander = EnhancedQueryUnderstanding()
knowledge_base = EnhancedKnowledgeBase()
response_generator = ContextAwareResponseGenerator(context_manager, knowledge_base)

logger = logging.getLogger(__name__)

def enhance_query_understanding(user_input: str) -> str:
    """Enhanced version of the existing query understanding function"""
    # Apply our improved corrections and enhancements
    corrected_query = query_understander.correct_and_enhance_query(user_input)
    logger.info(f"Query enhanced: '{user_input}' → '{corrected_query}'")
    return corrected_query

def enhanced_create_ai_response(message: str, db, session_id: str, user_message: str, request: Request = None):
    """Enhanced version that includes context management"""
    from main import save_chat_history  # Import from main to avoid circular imports
    
    try:
        # Get user IP for logging
        user_ip = request.client.host if request and hasattr(request, 'client') and request.client else 'unknown'
        
        # Save to chat history
        save_chat_history(db, session_id, user_message, message, user_ip)
        
        # Update context with the new interaction
        context_manager.update_context(session_id, user_message, message)
        
        return {"message": message, "session_id": session_id}
    
    except Exception as e:
        logger.error(f"Error in enhanced_create_ai_response: {e}")
        return {"message": message, "session_id": session_id}

async def process_enhanced_query(user_input: str, session_id: str, db, request: Request = None) -> Dict[str, Any]:
    """Main enhanced query processing function"""
    
    # Get conversation context
    context = context_manager.get_context(session_id)
    
    # Correct and enhance query
    corrected_query = query_understander.correct_and_enhance_query(user_input)
    
    # Parse query with context awareness
    parsed_query = query_understander.extract_intent_and_entities(corrected_query, context)
    
    logger.info(f"Enhanced processing - Intent: {parsed_query['intent']}, Confidence: {parsed_query['confidence']}")
    
    # Extract user preferences if context exists
    if context:
        context_manager.extract_preferences(corrected_query, context)
    
    # Handle different types of enhanced responses
    
    # 1. Follow-up questions with context
    if parsed_query['intent'] == 'follow_up_question' and context:
        response = response_generator.generate_follow_up_response(corrected_query, context, parsed_query)
        context_manager.update_context(session_id, user_input, response, topic='follow_up')
        return {"message": response, "session_id": session_id}
    
    # 2. Enhanced museum information
    if parsed_query['intent'] == 'museum_inquiry':
        return await handle_enhanced_museum_query(corrected_query, parsed_query, session_id, context)
    
    # 3. Enhanced neighborhood/place information
    if parsed_query['intent'] == 'place_recommendation':
        return await handle_enhanced_place_query(corrected_query, parsed_query, session_id, context)
    
    # 4. Itinerary planning
    if any(word in corrected_query.lower() for word in ['itinerary', 'plan', 'schedule', 'days', 'trip plan']):
        return await handle_itinerary_request(corrected_query, parsed_query, session_id, context)
    
    # 5. Cultural and etiquette questions
    if any(word in corrected_query.lower() for word in ['culture', 'etiquette', 'customs', 'tradition', 'behavior']):
        return await handle_cultural_query(corrected_query, parsed_query, session_id, context)
    
    # 6. Enhanced historical information
    if any(word in corrected_query.lower() for word in ['history', 'historical', 'byzantine', 'ottoman', 'background']):
        return await handle_historical_query(corrected_query, parsed_query, session_id, context)
    
    # If no enhanced handling applies, return None to use original processing
    return None

async def handle_enhanced_museum_query(query: str, parsed_query: Dict, session_id: str, context) -> Dict[str, Any]:
    """Handle museum queries with detailed information"""
    
    museums_mentioned = []
    museum_keywords = {
        'topkapi': ['topkapi', 'topkapı', 'palace'],
        'hagia_sophia': ['hagia sophia', 'ayasofya', 'santa sophia'],
        'istanbul_modern': ['istanbul modern', 'modern art'],
        'pera': ['pera museum', 'pera'],
        'archaeological': ['archaeological', 'archaeology museum']
    }
    
    query_lower = query.lower()
    for museum_key, keywords in museum_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            museums_mentioned.append(museum_key)
    
    if museums_mentioned:
        # Get detailed info for the first mentioned museum
        museum_info = get_museum_detailed_info(museums_mentioned[0])
        
        if museum_info:
            response = f"🏛️ **{museum_info.get('full_name', 'Museum')} - Detailed Information:**\n\n"
            response += f"⏰ **Hours:** {museum_info.get('opening_hours', 'Check official website')}\n"
            response += f"🚫 **Closed:** {museum_info.get('closed', 'Check schedule')}\n"
            response += f"🎫 **Tickets:** {museum_info.get('ticket_price', 'Contact for pricing')}\n"
            response += f"⏱️ **Visit Duration:** {museum_info.get('duration', '1-3 hours')}\n\n"
            
            if museum_info.get('highlights'):
                response += "✨ **Must-See Highlights:**\n"
                for highlight in museum_info['highlights']:
                    response += f"• {highlight}\n"
                response += "\n"
            
            if museum_info.get('tips'):
                response += f"💡 **Visitor Tips:** {museum_info['tips']}\n\n"
            
            # Add context-aware follow-up if there's previous restaurant talk
            if context and context.last_recommendation_type == 'restaurant':
                response += "Since you were asking about restaurants earlier, there are some great dining options near this museum too!"
            
            context_manager.update_context(session_id, query, response, places=museums_mentioned, topic='museum')
            return {"message": response, "session_id": session_id}
    
    # Generic museum response if no specific museum detected
    response = """🏛️ **Top Museums in Istanbul:**

**Historical Museums:**
• **Topkapi Palace** - Ottoman imperial palace (3-4 hours)
• **Hagia Sophia** - Byzantine/Ottoman architecture (1-2 hours)  
• **Istanbul Archaeological Museum** - Ancient artifacts

**Art Museums:**
• **Istanbul Modern** - Contemporary Turkish art
• **Pera Museum** - Orientalist paintings and weights & measures
• **Sakıp Sabancı Museum** - Ottoman calligraphy and paintings

**Specialty Museums:**
• **Rahmi M. Koç Museum** - Industrial and transport history
• **Istanbul Toy Museum** - Vintage toys from around the world

💡 **Tips:** Many museums are closed on Mondays. Consider the Museum Pass Istanbul for multiple visits!

Would you like detailed information about any specific museum?"""
    
    context_manager.update_context(session_id, query, response, topic='museum')
    return {"message": response, "session_id": session_id}

async def handle_enhanced_place_query(query: str, parsed_query: Dict, session_id: str, context) -> Dict[str, Any]:
    """Handle place/neighborhood queries with detailed information"""
    
    # Extract location from entities
    locations = parsed_query['entities'].get('locations', [])
    
    if locations:
        location = locations[0].lower()
        neighborhood_info = get_neighborhood_detailed_info(location)
        
        if neighborhood_info:
            response = f"🏘️ **{location.title()} - Neighborhood Guide:**\n\n"
            response += f"**Character:** {neighborhood_info.get('character', 'Vibrant Istanbul neighborhood')}\n\n"
            
            if neighborhood_info.get('attractions'):
                response += "🎯 **Main Attractions:**\n"
                for attraction in neighborhood_info['attractions']:
                    response += f"• {attraction}\n"
                response += "\n"
            
            if neighborhood_info.get('dining'):
                response += "🍽️ **Dining Scene:**\n"
                for dining in neighborhood_info['dining']:
                    response += f"• {dining}\n"
                response += "\n"
            
            response += f"🌟 **Atmosphere:** {neighborhood_info.get('atmosphere', 'Unique Istanbul vibe')}\n"
            response += f"👥 **Best For:** {neighborhood_info.get('best_for', 'All travelers')}\n"
            response += f"🚇 **Getting There:** {neighborhood_info.get('transportation', 'Public transport available')}\n\n"
            
            # Context-aware addition
            if context and context.last_recommendation_type == 'restaurant':
                response += f"Since you were interested in restaurants, {location.title()} has excellent dining options I mentioned above!"
            
            context_manager.update_context(session_id, query, response, places=[location.title()], topic='place')
            return {"message": response, "session_id": session_id}
    
    # Generic places response
    response = """🗺️ **Istanbul's Best Neighborhoods:**

**Historic Districts:**
• **Sultanahmet** - Byzantine & Ottoman monuments, tourist hub
• **Fatih** - Conservative, traditional markets, local life

**Modern & Trendy:**
• **Beyoğlu** - European feel, nightlife, arts scene
• **Galata** - Bohemian, cafes, Galata Tower
• **Karaköy** - Hipster area, design studios, galleries

**Waterfront Areas:**
• **Beşiktaş** - Upscale, Dolmabahçe Palace, ferry connections
• **Ortaköy** - Bosphorus views, weekend market
• **Kadıköy** - Asian side, local markets, young crowd

**Traditional:**
• **Balat** - Colorful houses, Jewish heritage, Instagram-worthy
• **Fener** - Greek Orthodox heritage, historic churches

Which neighborhood interests you most? I can provide detailed information!"""
    
    context_manager.update_context(session_id, query, response, topic='place')
    return {"message": response, "session_id": session_id}

async def handle_itinerary_request(query: str, parsed_query: Dict, session_id: str, context) -> Dict[str, Any]:
    """Handle itinerary planning requests"""
    
    # Extract duration if mentioned
    duration_match = None
    for i in range(1, 8):  # Support 1-7 days
        if str(i) in query or f"{i} day" in query.lower():
            duration_match = i
            break
    
    duration = duration_match or 3  # Default to 3 days
    
    # Extract interests from context if available
    interests = []
    if context and context.user_preferences:
        if context.user_preferences.get('cuisine'):
            interests.append('food')
        if 'museum' in context.conversation_topics:
            interests.append('culture')
        if 'shopping' in context.conversation_topics:
            interests.append('shopping')
    
    # Default interests if none detected
    if not interests:
        interests = ['history', 'culture', 'food']
    
    # Extract budget from context or query
    budget = 'moderate'
    if context and context.user_preferences.get('budget'):
        budget = context.user_preferences['budget']
    
    itinerary = generate_day_itinerary(duration, interests, budget)
    
    # Add personalized touches based on context
    if context and context.mentioned_places:
        itinerary += f"\n💡 **Personalized Note:** Since you've shown interest in {', '.join(context.mentioned_places)}, I've incorporated similar areas into your itinerary.\n"
    
    itinerary += "\n🔄 **Want to modify this itinerary?** Just let me know your specific interests or constraints!"
    
    context_manager.update_context(session_id, query, itinerary, topic='itinerary')
    return {"message": itinerary, "session_id": session_id}

async def handle_cultural_query(query: str, parsed_query: Dict, session_id: str, context) -> Dict[str, Any]:
    """Handle cultural and etiquette questions"""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['mosque', 'prayer', 'religious']):
        etiquette = knowledge_base.get_cultural_advice('mosque_etiquette')
        response = "🕌 **Mosque Etiquette in Istanbul:**\n\n"
        for rule in etiquette:
            response += f"• {rule}\n"
        response += "\n💡 **Additional Tips:** Most mosques provide plastic bags for shoes and head scarves for women at the entrance."
    
    elif any(word in query_lower for word in ['dining', 'restaurant', 'eating']):
        etiquette = knowledge_base.get_cultural_advice('dining_etiquette')
        response = "🍽️ **Turkish Dining Etiquette:**\n\n"
        for rule in etiquette:
            response += f"• {rule}\n"
    
    elif any(word in query_lower for word in ['general', 'customs', 'behavior']):
        etiquette = knowledge_base.get_cultural_advice('general_customs')
        response = "🇹🇷 **General Turkish Customs:**\n\n"
        for rule in etiquette:
            response += f"• {rule}\n"
    
    else:
        response = """🇹🇷 **Turkish Culture Quick Guide:**

**Religious Considerations:**
• Turkey is 99% Muslim but quite secular in cities
• Ramadan affects restaurant hours and atmosphere
• Call to prayer 5 times daily from mosques

**Social Customs:**
• Turks are very hospitable and helpful to tourists
• Family is extremely important in Turkish culture
• Respect for elders is highly valued

**Language & Communication:**
• Learning basic Turkish phrases is appreciated
• Turks often speak loudly - it's not anger, it's passion!
• Direct eye contact shows honesty and respect

What specific aspect of Turkish culture interests you?"""
    
    context_manager.update_context(session_id, query, response, topic='culture')
    return {"message": response, "session_id": session_id}

async def handle_historical_query(query: str, parsed_query: Dict, session_id: str, context) -> Dict[str, Any]:
    """Handle historical information queries"""
    
    query_lower = query.lower()
    
    # Check for specific historical sites
    if any(word in query_lower for word in ['hagia sophia', 'ayasofya']):
        info = knowledge_base.get_historical_info('hagia_sophia')
        response = "⛪🕌 **Hagia Sophia - Historical Overview:**\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Historical Significance:** {info['historical_significance']}\n\n"
        response += f"**Architecture:** {info['architecture']}\n\n"
        response += f"**Visiting Tips:** {info['visiting_tips']}"
    
    elif any(word in query_lower for word in ['blue mosque', 'sultanahmet mosque']):
        info = knowledge_base.get_historical_info('blue_mosque')
        response = "🕌 **Blue Mosque - Historical Overview:**\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Architecture:** {info['architectural_features']}\n\n"
        response += f"**Best Times:** {info['best_viewing_times']}\n\n"
        response += f"**Visiting Tips:** {info['visiting_tips']}"
    
    elif any(word in query_lower for word in ['topkapi', 'palace']):
        info = knowledge_base.get_historical_info('topkapi_palace')
        response = "🏰 **Topkapi Palace - Historical Overview:**\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Historical Period:** {info['historical_period']}\n\n"
        response += f"**Highlights:** {', '.join(info['highlights'])}\n\n"
        response += f"**Visiting Tips:** {info['visiting_tips']}"
    
    else:
        response = """📚 **Istanbul's Rich History:**

**Byzantine Period (330-1453):**
• Constantinople was the "New Rome"
• Hagia Sophia built as the world's largest cathedral
• Center of Orthodox Christianity for 1000+ years

**Ottoman Period (1453-1922):**
• Mehmed II conquered Constantinople in 1453
• Became capital of Ottoman Empire for 470 years
• Architectural marvels like Blue Mosque and Topkapi Palace

**Modern Period (1923-present):**
• Became part of modern Turkey under Atatürk
• No longer capital (moved to Ankara) but cultural heart
• Bridges Europe and Asia both literally and culturally

**Key Historical Sites:**
• Hagia Sophia - Byzantine/Ottoman legacy
• Topkapi Palace - Ottoman imperial power
• Blue Mosque - Ottoman architectural peak
• Basilica Cistern - Byzantine engineering marvel

Which historical period or site interests you most?"""
    
    context_manager.update_context(session_id, query, response, topic='history')
    return {"message": response, "session_id": session_id}

def get_context_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Get a summary of the conversation context for debugging"""
    context = context_manager.get_context(session_id)
    
    if not context:
        return None
    
    return {
        'session_id': context.session_id,
        'query_count': len(context.previous_queries),
        'mentioned_places': context.mentioned_places,
        'user_preferences': context.user_preferences,
        'last_topic': context.last_recommendation_type,
        'conversation_topics': context.conversation_topics,
        'user_location': context.user_location
    }
