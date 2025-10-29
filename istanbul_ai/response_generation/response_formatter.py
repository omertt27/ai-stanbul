"""
Response Formatter - Format and structure AI responses

This module handles response formatting with:
- Markdown formatting
- List formatting
- Section formatting
- Truncation
- Icons and emojis
- Bilingual support

Week 7-8 Refactoring: Extracted from main_system.py
"""

import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats responses with consistent structure and styling
    
    Provides utilities for:
    - Markdown formatting (headers, lists, emphasis)
    - Structured sections
    - Emoji/icon prefixes
    - List truncation
    - Bilingual formatting
    """
    
    def __init__(self):
        """Initialize response formatter"""
        self.max_list_items = 10
        self.max_description_length = 200
        logger.info("✅ ResponseFormatter initialized")
    
    def format_list_response(
        self,
        items: List[Any],
        header: str,
        language: str = 'en',
        max_items: Optional[int] = None,
        show_count: bool = True
    ) -> str:
        """
        Format a list of items with header and count
        
        Args:
            items: List of items to format
            header: Header text
            language: Language code ('en' or 'tr')
            max_items: Maximum items to show (None for default)
            show_count: Whether to show item count
        
        Returns:
            Formatted list string
        """
        if not items:
            return self._get_no_results_message(language)
        
        max_items = max_items or self.max_list_items
        items_to_show = items[:max_items]
        
        response = f"{header}\n\n"
        
        if show_count:
            count_text = self._get_count_text(len(items_to_show), len(items), language)
            response += f"{count_text}\n\n"
        
        # Format items
        for i, item in enumerate(items_to_show, 1):
            if isinstance(item, str):
                response += f"{i}. {item}\n"
            elif hasattr(item, '__dict__'):
                response += f"{i}. {self._format_object_brief(item)}\n"
            else:
                response += f"{i}. {str(item)}\n"
        
        # Add truncation notice if needed
        if len(items) > max_items:
            response += f"\n{self._get_truncation_message(len(items), max_items, language)}"
        
        return response
    
    def format_detailed_item(
        self,
        item: Any,
        sections: Optional[List[str]] = None,
        language: str = 'en'
    ) -> str:
        """
        Format a detailed item with sections
        
        Args:
            item: Item to format
            sections: List of section names to include
            language: Language code
        
        Returns:
            Formatted detailed string
        """
        if not item:
            return self._get_no_results_message(language)
        
        response = ""
        
        # Add name/title if available
        if hasattr(item, 'name'):
            response += f"🌟 **{item.name}**\n\n"
        
        # Add sections
        if sections:
            for section in sections:
                section_content = self._format_section(item, section, language)
                if section_content:
                    response += section_content + "\n"
        else:
            # Auto-detect sections
            response += self._format_auto_sections(item, language)
        
        return response
    
    def format_with_sections(
        self,
        sections: List[Dict[str, Any]],
        language: str = 'en'
    ) -> str:
        """
        Format response with multiple sections
        
        Args:
            sections: List of section dicts with 'header' and 'content'
            language: Language code
        
        Returns:
            Formatted response with sections
        """
        response = ""
        
        for section in sections:
            header = section.get('header', '')
            content = section.get('content', '')
            icon = section.get('icon', '')
            
            if header:
                response += f"{icon} **{header}**\n\n" if icon else f"**{header}**\n\n"
            
            if content:
                if isinstance(content, list):
                    for item in content:
                        response += f"• {item}\n"
                else:
                    response += f"{content}\n"
            
            response += "\n"
        
        return response.strip()
    
    def truncate_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_ellipsis: bool = True
    ) -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Text to truncate
            max_length: Maximum length (None for default)
            add_ellipsis: Whether to add '...' at end
        
        Returns:
            Truncated text
        """
        max_length = max_length or self.max_description_length
        
        if len(text) <= max_length:
            return text
        
        if add_ellipsis:
            return text[:max_length-3] + "..."
        else:
            return text[:max_length]
    
    def add_helpful_tip(
        self,
        response: str,
        intent: Optional[str] = None,
        language: str = 'en'
    ) -> str:
        """
        Add helpful tip to response based on intent
        
        Args:
            response: Base response
            intent: Intent type for context-specific tips
            language: Language code
        
        Returns:
            Response with helpful tip
        """
        tips = self._get_intent_tips(intent, language)
        
        if tips:
            tip_header = "💡 **Tip:**" if language == 'en' else "💡 **İpucu:**"
            response += f"\n\n{tip_header} {tips[0]}"
        
        return response
    
    def format_number(self, number: float, decimal_places: int = 1) -> str:
        """
        Format number with consistent decimal places
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
        
        Returns:
            Formatted number string
        """
        return f"{number:.{decimal_places}f}"
    
    def format_distance(self, distance_km: float, language: str = 'en') -> str:
        """
        Format distance with appropriate units
        
        Args:
            distance_km: Distance in kilometers
            language: Language code
        
        Returns:
            Formatted distance string
        """
        # Handle invalid or non-numeric values
        try:
            distance_km = float(distance_km)
        except (TypeError, ValueError):
            return ""
        
        if distance_km < 1:
            meters = int(distance_km * 1000)
            return f"{meters}m" if language == 'en' else f"{meters}m"
        else:
            return f"{distance_km:.1f}km" if language == 'en' else f"{distance_km:.1f}km"
    
    def format_price(self, price: float, currency: str = 'TL', language: str = 'en') -> str:
        """
        Format price with currency
        
        Args:
            price: Price amount
            currency: Currency code
            language: Language code
        
        Returns:
            Formatted price string
        """
        # Handle invalid or non-numeric values
        try:
            price = float(price)
        except (TypeError, ValueError):
            return ""
        
        if price == 0:
            return "FREE ✨" if language == 'en' else "ÜCRETSİZ ✨"
        
        return f"{int(price)} {currency}"
    
    def format_rating(self, rating: float, max_rating: float = 5.0) -> str:
        """
        Format rating with stars
        
        Args:
            rating: Rating value
            max_rating: Maximum rating value
        
        Returns:
            Formatted rating string
        """
        stars = "⭐" * int(rating)
        return f"{stars} {rating}/{max_rating}"
    
    def _format_object_brief(self, obj: Any) -> str:
        """Format brief representation of object"""
        if hasattr(obj, 'name'):
            result = obj.name
            if hasattr(obj, 'district'):
                result += f" • {obj.district}"
            return result
        return str(obj)
    
    def _format_section(self, item: Any, section: str, language: str) -> Optional[str]:
        """Format a specific section of an item"""
        content = ""
        
        if section == 'location' and hasattr(item, 'district'):
            content = f"📍 **Location:** {item.district}"
            if hasattr(item, 'address'):
                content += f" • {item.address}"
            content += "\n"
        
        elif section == 'description' and hasattr(item, 'description'):
            desc = self.truncate_text(item.description)
            content = f"📖 **About:**\n{desc}\n"
        
        elif section == 'price' and hasattr(item, 'price_tl'):
            price = self.format_price(item.price_tl, language=language)
            content = f"🎫 **Entry:** {price}\n"
        
        elif section == 'rating' and hasattr(item, 'rating'):
            rating_str = self.format_rating(item.rating)
            if hasattr(item, 'reviews_count'):
                rating_str += f" ({item.reviews_count} reviews)"
            content = f"{rating_str}\n"
        
        elif section == 'highlights' and hasattr(item, 'highlights'):
            if item.highlights:
                content = f"✨ **Highlights:**\n"
                for highlight in item.highlights[:5]:
                    content += f"• {highlight}\n"
        
        return content if content else None
    
    def _format_auto_sections(self, item: Any, language: str) -> str:
        """Auto-format item with detected sections"""
        sections = ['location', 'description', 'price', 'rating', 'highlights']
        response = ""
        
        for section in sections:
            section_content = self._format_section(item, section, language)
            if section_content:
                response += section_content
        
        return response
    
    def _get_count_text(self, shown: int, total: int, language: str) -> str:
        """Get count text based on language"""
        if language == 'tr':
            return f"Sizin için **{shown}** harika sonuç buldum:"
        else:
            return f"I found **{shown}** amazing result{'s' if shown > 1 else ''} for you:"
    
    def _get_truncation_message(self, total: int, shown: int, language: str) -> str:
        """Get truncation message"""
        if language == 'tr':
            return f"📋 *İlk {shown} sonuç gösteriliyor ({total} sonuçtan). Daha fazla detay için bana sorun!*"
        else:
            return f"📋 *Showing top {shown} of {total} results. Ask for more details or use filters!*"
    
    def _get_no_results_message(self, language: str) -> str:
        """Get no results message"""
        if language == 'tr':
            return "😔 Üzgünüm, sonuç bulamadım. Farklı bir şey aramayı deneyelim mi?"
        else:
            return "😔 I'm sorry, I couldn't find any results. Shall we try something different?"
    
    def _get_intent_tips(self, intent: Optional[str], language: str) -> List[str]:
        """Get helpful tips based on intent"""
        if language == 'tr':
            tips_map = {
                'restaurant': ["Restoranlar hakkında daha fazla detay için, özel bir mutfak türü veya bölge belirtebilirsiniz."],
                'attraction': ["İstanbul'un gizli köşelerini keşfetmek için 'gizli yerler' diyebilirsiniz."],
                'transportation': ["Ulaşım rotası için başlangıç ve varış noktalarını belirtebilirsiniz."],
                'hotel': ["Otel önerileri için bütçe tercihinizi belirtebilirsiniz."]
            }
        else:
            tips_map = {
                'restaurant': ["For more details about restaurants, you can specify a cuisine type or district."],
                'attraction': ["To discover Istanbul's hidden gems, just ask about 'hidden places'."],
                'transportation': ["For transport routes, specify your starting point and destination."],
                'hotel': ["For hotel recommendations, let me know your budget preference."]
            }
        
        return tips_map.get(intent, [])
