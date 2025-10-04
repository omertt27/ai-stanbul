"""
Template Engine for AI Istanbul System

This service handles all template-based responses, replacing GPT with deterministic
string formatting and structured templates for consistent, fast responses.
"""

import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, time
import re

class TemplateEngine:
    """
    Advanced template engine for generating natural language responses
    without using GPT. Uses structured templates with variables, conditionals,
    and randomization for variety.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.response_variants = self._load_response_variants()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load all response templates"""
        return {
            "greeting": {
                "variants": [
                    "Merhaba! İstanbul hakkında size nasıl yardımcı olabilirim?",
                    "Hoş geldiniz! İstanbul'da neyi keşfetmek istiyorsunuz?",
                    "Selam! İstanbul rehberiniz olarak buradayım.",
                    "İyi günler! İstanbul'un güzelliklerini keşfetmenize yardımcı olayım."
                ],
                "english_variants": [
                    "Hello! How can I help you explore Istanbul?",
                    "Welcome! What would you like to discover in Istanbul?",
                    "Hi! I'm here as your Istanbul guide.",
                    "Good day! Let me help you discover Istanbul's beauty."
                ]
            },
            
            "attraction_info": {
                "template": "📍 **{name}**\n\n{description}\n\n📍 **Konum:** {location}\n⏰ **Açılış Saatleri:** {hours}\n💰 **Giriş:** {price}\n🚇 **Ulaşım:** {transport}\n\n💡 **İpucu:** {tip}",
                "english_template": "📍 **{name}**\n\n{description}\n\n📍 **Location:** {location}\n⏰ **Hours:** {hours}\n💰 **Admission:** {price}\n🚇 **Getting There:** {transport}\n\n💡 **Tip:** {tip}"
            },
            
            "restaurant_recommendation": {
                "template": "🍽️ **{name}** - {cuisine} Mutfağı\n\n{description}\n\n📍 **Konum:** {location}\n💰 **Fiyat Aralığı:** {price_range}\n⭐ **Öne Çıkan:** {specialties}\n⏰ **Saatler:** {hours}\n\n💡 **Tavsiye:** {recommendation}",
                "english_template": "🍽️ **{name}** - {cuisine} Cuisine\n\n{description}\n\n📍 **Location:** {location}\n💰 **Price Range:** {price_range}\n⭐ **Highlights:** {specialties}\n⏰ **Hours:** {hours}\n\n💡 **Recommendation:** {recommendation}"
            },
            
            "transport_route": {
                "template": "🚇 **{from_location}** → **{to_location}**\n\n**En İyi Rota:**\n{route_steps}\n\n⏱️ **Süre:** {duration}\n💰 **Maliyet:** {cost}\n🕐 **İlk/Son Sefer:** {schedule}\n\n💡 **İpucu:** {tip}",
                "english_template": "🚇 **{from_location}** → **{to_location}**\n\n**Best Route:**\n{route_steps}\n\n⏱️ **Duration:** {duration}\n💰 **Cost:** {cost}\n🕐 **First/Last Service:** {schedule}\n\n💡 **Tip:** {tip}"
            },
            
            "itinerary": {
                "template": "📅 **{day_name} İtinerarı** ({duration})\n\n{activities}\n\n🍽️ **Yemek Önerileri:** {dining}\n🚇 **Ulaşım Notu:** {transport_note}\n💰 **Tahmini Maliyet:** {estimated_cost}",
                "english_template": "📅 **{day_name} Itinerary** ({duration})\n\n{activities}\n\n🍽️ **Dining Suggestions:** {dining}\n🚇 **Transport Note:** {transport_note}\n💰 **Estimated Cost:** {estimated_cost}"
            },
            
            "no_results": {
                "variants": [
                    "Üzgünüm, aradığınız kriterlere uygun bir sonuç bulamadım. Farklı arama terimleri deneyebilirsiniz.",
                    "Bu konuda şu anda bilgim yok. Başka nasıl yardımcı olabilirim?",
                    "Aradığınız bilgiyi bulamadım. Daha spesifik bir soru sorabilir misiniz?"
                ],
                "english_variants": [
                    "Sorry, I couldn't find results matching your criteria. You might try different search terms.",
                    "I don't have information on that topic right now. How else can I help?",
                    "I couldn't find that information. Could you ask a more specific question?"
                ]
            },
            
            "error": {
                "variants": [
                    "Özür dilerim, bir hata oluştu. Lütfen tekrar deneyin.",
                    "Bir sorun yaşandı. Biraz sonra tekrar deneyebilirsiniz.",
                    "Sistemde geçici bir sorun var. Lütfen daha sonra tekrar deneyin."
                ],
                "english_variants": [
                    "Sorry, an error occurred. Please try again.",
                    "There was a problem. You can try again in a moment.",
                    "There's a temporary system issue. Please try again later."
                ]
            }
        }
    
    def _load_response_variants(self) -> Dict[str, List[str]]:
        """Load response variation patterns"""
        return {
            "positive_connectors": [
                "Harika seçim!", "Mükemmel!", "İyi düşünmüşsünüz!", 
                "Kesinlikle tavsiye ederim!", "Great choice!", "Perfect!", "Excellent!"
            ],
            "transition_phrases": [
                "Ayrıca,", "Bunun yanında,", "Also,", "Additionally,", "Furthermore,"
            ],
            "time_expressions": {
                "morning": ["sabah", "morning", "am"],
                "afternoon": ["öğleden sonra", "afternoon", "pm"],
                "evening": ["akşam", "evening"],
                "night": ["gece", "night"]
            }
        }
    
    def generate_response(self, template_type: str, data: Dict[str, Any], 
                         language: str = "turkish") -> str:
        """
        Generate a response using templates and provided data
        
        Args:
            template_type: Type of template to use
            data: Data to fill template variables
            language: Response language (turkish/english)
            
        Returns:
            Formatted response string
        """
        if template_type not in self.templates:
            return self._get_error_response(language)
        
        template_config = self.templates[template_type]
        
        # Handle simple variant templates (greeting, no_results, error)
        if "variants" in template_config:
            variants_key = f"{language}_variants" if language == "english" else "variants"
            variants = template_config.get(variants_key, template_config["variants"])
            return random.choice(variants)
        
        # Handle complex templates with data substitution
        template_key = f"{language}_template" if language == "english" else "template"
        template = template_config.get(template_key, template_config.get("template", ""))
        
        if not template:
            return self._get_error_response(language)
        
        # Process data and apply template
        processed_data = self._process_template_data(data, language)
        
        try:
            response = template.format(**processed_data)
            return self._add_natural_variations(response, language)
        except KeyError as e:
            print(f"Template key error: {e}")
            # Try to format with only available keys
            import string
            available_keys = {k: v for k, v in processed_data.items() 
                            if k in [field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name]}
            try:
                response = template.format(**available_keys)
                return self._add_natural_variations(response, language)
            except:
                return self._get_error_response(language)
    
    def _process_template_data(self, data: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Process and enhance template data"""
        processed = data.copy()
        
        # Add time-based greetings
        current_hour = datetime.now().hour
        if current_hour < 12:
            time_greeting = "Good morning" if language == "english" else "Günaydın"
        elif current_hour < 17:
            time_greeting = "Good afternoon" if language == "english" else "İyi günler"
        else:
            time_greeting = "Good evening" if language == "english" else "İyi akşamlar"
        
        # Add default values for common template keys to prevent KeyError
        defaults = {
            'name': 'Unknown Place',
            'description': 'A wonderful place to visit in Istanbul.',
            'location': 'Istanbul',
            'hours': 'Variable hours',
            'price': 'Contact for pricing',
            'transport': 'Public transport available',
            'tip': 'Visit during off-peak hours for the best experience.',
            'cuisine': 'Turkish',
            'price_range': 'Moderate',
            'specialties': 'Local specialties',
            'recommendation': 'Highly recommended for visitors.',
            'district': 'Istanbul',
            'category': 'Attraction'
        }
        
        # Fill in missing keys with defaults
        for key, default_value in defaults.items():
            if key not in processed:
                processed[key] = default_value
        
        processed["time_greeting"] = time_greeting
        
        # Format lists as bullet points
        for key, value in processed.items():
            if isinstance(value, list):
                if language == "english":
                    processed[key] = "\n".join([f"• {item}" for item in value])
                else:
                    processed[key] = "\n".join([f"• {item}" for item in value])
        
        # Add default values for missing keys
        defaults = {
            "name": "Bilinmeyen" if language == "turkish" else "Unknown",
            "location": "İstanbul" if language == "turkish" else "Istanbul",
            "hours": "Değişken" if language == "turkish" else "Variable",
            "price": "Bilgi yok" if language == "turkish" else "Not available",
            "tip": ""
        }
        
        for key, default_value in defaults.items():
            if key not in processed or not processed[key]:
                processed[key] = default_value
        
        return processed
    
    def _add_natural_variations(self, response: str, language: str) -> str:
        """Add natural language variations to make responses less robotic"""
        
        # Add random positive connectors
        connectors = self.response_variants["positive_connectors"]
        if random.random() < 0.3:  # 30% chance to add connector
            connector = random.choice([c for c in connectors if 
                                     (language == "english" and c in ["Great choice!", "Perfect!", "Excellent!"]) or
                                     (language == "turkish" and c not in ["Great choice!", "Perfect!", "Excellent!"])])
            response = f"{connector} {response}"
        
        return response
    
    def _get_error_response(self, language: str) -> str:
        """Get appropriate error response"""
        return self.generate_response("error", {}, language)
    
    def generate_attraction_response(self, attraction: Dict[str, Any], 
                                   language: str = "turkish") -> str:
        """Generate formatted attraction information"""
        return self.generate_response("attraction_info", attraction, language)
    
    def generate_restaurant_response(self, restaurant: Dict[str, Any], 
                                   language: str = "turkish") -> str:
        """Generate formatted restaurant recommendation"""
        return self.generate_response("restaurant_recommendation", restaurant, language)
    
    def generate_transport_response(self, route_info: Dict[str, Any], 
                                  language: str = "turkish") -> str:
        """Generate formatted transport route information"""
        return self.generate_response("transport_route", route_info, language)
    
    def generate_itinerary_response(self, itinerary: Dict[str, Any], 
                                  language: str = "turkish") -> str:
        """Generate formatted itinerary"""
        return self.generate_response("itinerary", itinerary, language)
    
    def generate_greeting(self, language: str = "turkish") -> str:
        """Generate welcome greeting"""
        return self.generate_response("greeting", {}, language)
    
    def generate_no_results(self, language: str = "turkish") -> str:
        """Generate no results found message"""
        return self.generate_response("no_results", {}, language)
    
    def create_list_response(self, items: List[Dict[str, Any]], 
                           item_type: str, language: str = "turkish") -> str:
        """Create a formatted list response for multiple items"""
        if not items:
            return self.generate_no_results(language)
        
        if item_type == "attractions":
            responses = [self.generate_attraction_response(item, language) for item in items]
        elif item_type == "restaurants":
            responses = [self.generate_restaurant_response(item, language) for item in items]
        else:
            # Generic list formatting
            responses = [f"• {item.get('name', 'Unknown')}" for item in items]
        
        separator = "\n---\n"
        return separator.join(responses)
    
    def format_price(self, price_info: Any, language: str = "turkish") -> str:
        """Format price information consistently"""
        if not price_info:
            return "Ücretsiz" if language == "turkish" else "Free"
        
        if isinstance(price_info, (int, float)):
            if price_info == 0:
                return "Ücretsiz" if language == "turkish" else "Free"
            return f"{price_info} TL"
        
        return str(price_info)
    
    def format_duration(self, minutes: int, language: str = "turkish") -> str:
        """Format duration consistently"""
        if minutes < 60:
            return f"{minutes} dakika" if language == "turkish" else f"{minutes} minutes"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if remaining_minutes == 0:
            return f"{hours} saat" if language == "turkish" else f"{hours} hours"
        
        if language == "turkish":
            return f"{hours} saat {remaining_minutes} dakika"
        else:
            return f"{hours}h {remaining_minutes}min"
    
    def create_safe_list_response(self, items: List[Dict[str, Any]], 
                                 item_type: str, language: str = "turkish") -> str:
        """Create a formatted list response with safe handling of missing keys"""
        if not items:
            return self.generate_no_results(language)
        
        if item_type == "attractions":
            title = "🏛️ **Gezilecek Yerler:**" if language == "turkish" else "🏛️ **Places to Visit:**"
            responses = [title]
            
            for item in items:
                name = item.get('name', 'Bilinmeyen Yer' if language == "turkish" else 'Unknown Place')
                district = item.get('district', item.get('location', ''))
                category = item.get('category', item.get('type', ''))
                
                # Create safe bullet point format
                bullet = f"• **{name}**"
                if district:
                    bullet += f" ({district})"
                if category:
                    bullet += f" - {category}"
                
                # Add any available description or reason
                if 'description' in item:
                    bullet += f": {item['description'][:100]}..."
                elif 'reasons' in item and item['reasons']:
                    bullet += f": {', '.join(item['reasons'][:2])}"
                
                responses.append(bullet)
            
            return "\n".join(responses)
            
        elif item_type == "restaurants":
            # Use existing restaurant response generation
            return self.create_list_response(items, item_type, language)
        else:
            # Generic safe formatting
            responses = []
            for item in items:
                name = item.get('name', 'Unknown')
                bullet = f"• {name}"
                if 'district' in item:
                    bullet += f" ({item['district']})"
                responses.append(bullet)
            
            return "\n".join(responses)
