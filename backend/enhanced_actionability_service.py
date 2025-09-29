#!/usr/bin/env python3
"""
Enhanced Actionability Service for AI Istanbul
============================================

This service enhances AI responses with structured, actionable information including:
- Address → Directions → Timing → Tips format
- Step-by-step instructions
- Practical next steps
- Cultural context and local insights
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ActionabilityAnalysis:
    """Analysis of response actionability"""
    score: float
    missing_elements: List[str]
    suggestions: List[str]
    enhanced_sections: Dict[str, str]
    structured_info: Dict[str, Any]

@dataclass
class StructuredResponse:
    """Structured actionable response format"""
    address: Optional[str] = None
    directions: Optional[str] = None
    timing: Optional[str] = None
    cost: Optional[str] = None
    tips: List[str] = None
    cultural_notes: List[str] = None
    next_steps: List[str] = None

class EnhancedActionabilityService:
    """Enhanced service to create highly actionable responses"""
    
    def __init__(self):
        self.required_elements = {
            'address_info': ['address', 'located at', 'street', 'avenue', 'cd.', 'mahallesi', 'near'],
            'timing_info': ['hours', 'open', 'closed', 'time', 'am', 'pm', ':', 'schedule'],
            'direction_info': ['metro', 'bus', 'walk', 'take', 'get to', 'from', 'platform'],
            'cost_info': ['cost', 'price', 'tl', 'lira', 'free', 'fee', 'ticket'],
            'practical_steps': ['how to', 'steps', 'first', 'then', 'next', 'finally'],
            'cultural_context': ['respect', 'tradition', 'custom', 'etiquette', 'appropriate']
        }
        
        # Turkish phrases for cultural enhancement
        self.turkish_phrases = {
            'restaurant': {
                'hello': 'Merhaba (mer-HAH-bah) - Hello',
                'please': 'Lütfen (LOOT-fen) - Please',
                'thank_you': 'Teşekkürler (teh-shek-koor-LEHR) - Thank you',
                'delicious': 'Lezzetli (lez-ZET-lee) - Delicious',
                'bill': 'Hesap, lütfen (HEH-sahp LOOT-fen) - Bill, please'
            },
            'museum': {
                'ticket': 'Bilet (bee-LET) - Ticket',
                'how_much': 'Ne kadar? (neh kah-DAHR) - How much?',
                'beautiful': 'Güzel (goo-ZEHL) - Beautiful',
                'history': 'Tarih (TAH-reeh) - History'
            },
            'transportation': {
                'where': 'Nerede? (neh-reh-DEH) - Where?',
                'station': 'İstasyon (ees-tahs-YOHN) - Station',
                'transfer': 'Aktarma (ahk-tahr-MAH) - Transfer'
            }
        }
    
    def analyze_actionability(self, response: str, category: str) -> ActionabilityAnalysis:
        """Analyze and enhance response actionability with structured format"""
        try:
            response_lower = response.lower()
            score = 0.0
            missing_elements = []
            suggestions = []
            enhanced_sections = {}
            structured_info = {}
            
            # Check for required elements
            total_elements = len(self.required_elements)
            found_elements = 0
            
            for element_type, keywords in self.required_elements.items():
                if any(keyword in response_lower for keyword in keywords):
                    found_elements += 1
                else:
                    missing_elements.append(element_type)
            
            # Calculate base score
            score = found_elements / total_elements
            
            # Generate structured information
            structured_info = self._create_structured_response(response, category, missing_elements)
            
            # Category-specific enhancements
            if category in ['restaurant', 'museum', 'transportation', 'district']:
                category_score, category_suggestions = self._analyze_category_specific(response, category)
                score = (score + category_score) / 2
                suggestions.extend(category_suggestions)
            
            # Generate enhancement sections
            if score < 0.8:  # Raised threshold for better quality
                enhanced_sections = self._generate_structured_enhancements(response, missing_elements, category, structured_info)
                suggestions.extend(self._generate_improvement_suggestions(missing_elements))
            
            return ActionabilityAnalysis(
                score=score,
                missing_elements=missing_elements,
                suggestions=suggestions,
                enhanced_sections=enhanced_sections,
                structured_info=structured_info
            )
            
        except Exception as e:
            logger.error(f"Error analyzing actionability: {e}")
            return ActionabilityAnalysis(
                score=0.5, 
                missing_elements=[], 
                suggestions=[], 
                enhanced_sections={},
                structured_info={}
            )
    
    def _create_structured_response(self, response: str, category: str, missing_elements: List[str]) -> Dict[str, Any]:
        """Create structured response information"""
        structured = {
            "address": self._extract_or_generate_address(response, category),
            "directions": self._extract_or_generate_directions(response, category),
            "timing": self._extract_or_generate_timing(response, category),
            "cost": self._extract_or_generate_cost(response, category),
            "tips": self._generate_practical_tips(response, category),
            "cultural_notes": self._generate_cultural_notes(response, category),
            "next_steps": self._generate_next_steps(response, category)
        }
        return structured
    
    def _extract_or_generate_address(self, response: str, category: str) -> str:
        """Extract or generate address information"""
        response_lower = response.lower()
        
        # Try to extract existing address info
        address_patterns = [
            r'address[:\s]+([^.\n]+)',
            r'located at[:\s]+([^.\n]+)',
            r'situated in[:\s]+([^.\n]+)'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return f"📍 **Address:** {match.group(1).strip()}"
        
        # Generate category-specific address guidance
        if category == 'restaurant':
            return "📍 **Address:** Check Google Maps for exact locations. Most recommended restaurants are in Sultanahmet, Beyoğlu, Kadıköy, or Galata districts."
        elif category == 'museum':
            return "📍 **Address:** Major museums are in Sultanahmet (Historic Peninsula). Use landmarks like Hagia Sophia or Blue Mosque for navigation."
        elif category == 'transportation':
            return "📍 **Stations:** Locate nearest metro station using Istanbul Metro app or look for 'M' signs on streets."
        else:
            return "📍 **Location:** Use Google Maps or ask locals for precise directions to your destination."
    
    def _extract_or_generate_directions(self, response: str, category: str) -> str:
        """Extract or generate direction information"""
        if category == 'transportation':
            return """🗺️ **Directions:**
• Download: Istanbul Metro app, Moovit, or Citymapper
• Look for: Color-coded metro line signs (M1-Red, M2-Green, etc.)
• Follow: Platform direction signs in Turkish and English
• Use: Istanbulkart for all public transport"""
        
        elif category == 'restaurant':
            return """🗺️ **Getting There:**
• Metro: Find nearest station and walk (usually 5-10 minutes)
• Taxi: Show address to driver or use BiTaksi app
• Walking: From major landmarks (ask for directions)
• Ferry: If near Bosphorus, consider scenic ferry routes"""
        
        elif category == 'museum':
            return """🗺️ **Getting There:**
• Sultanahmet: Take Tram T1 to Sultanahmet station
• From Taksim: Metro M2 to Vezneciler, then short walk
• From Airport: Metro M1 to Zeytinburnu, transfer to T1
• Walking: Most museums are within walking distance of each other"""
        
        else:
            return """🗺️ **Navigation:**
• Use: Google Maps offline maps for Istanbul
• Ask: Hotel concierge or tourist information centers
• Transport: Public transport is efficient and affordable
• Walking: Many attractions are in walkable neighborhoods"""
    
    def _extract_or_generate_timing(self, response: str, category: str) -> str:
        """Extract or generate timing information"""
        current_hour = datetime.now().hour
        time_context = "morning" if current_hour < 12 else "afternoon" if current_hour < 18 else "evening"
        
        if category == 'restaurant':
            return f"""⏰ **Timing:**
• Breakfast: 07:00-11:00 (traditional Turkish breakfast)
• Lunch: 12:00-15:00 (lighter meals, meze)
• Dinner: 18:00-23:00 (main meal time)
• Current time context: Best for {time_context} dining
• Peak times: Avoid 13:00-14:00 and 19:00-21:00 for shorter waits"""
        
        elif category == 'museum':
            return f"""⏰ **Timing:**
• Opening: Most museums 09:00-17:00 (winter), 09:00-19:00 (summer)
• Best time: Early morning (09:00-10:00) or late afternoon
• Avoid: Weekends and Turkish holidays for smaller crowds
• Duration: Allow 1-3 hours depending on interest level
• Current: {time_context} visit recommended"""
        
        elif category == 'transportation':
            return f"""⏰ **Schedule:**
• Metro: 06:00-00:30 (every 2-5 minutes during peak)
• Buses: 05:30-01:00 (frequency varies by route)
• Ferries: 07:00-21:00 (seasonal schedule variations)
• Rush hours: 07:00-09:00 and 17:00-19:00 (crowded)
• Current time: {time_context} - {'Peak' if 7 <= current_hour <= 9 or 17 <= current_hour <= 19 else 'Off-peak'} period"""
        
        else:
            return f"""⏰ **Planning:**
• Best times: Early morning or late afternoon
• Avoid: Peak hours and extreme weather
• Duration: Plan flexible timing for unexpected discoveries
• Current: {time_context} - adjust plans accordingly"""
    
    def _extract_or_generate_cost(self, response: str, category: str) -> str:
        """Extract or generate cost information"""
        if category == 'restaurant':
            return """💰 **Costs:**
• Street food: 10-25 TL per item
• Casual dining: 80-150 TL per person
• Fine dining: 200-500 TL per person
• Drinks: Tea 5-10 TL, Coffee 15-25 TL, Soft drinks 10-20 TL
• Tipping: 10-15% in restaurants (optional but appreciated)"""
        
        elif category == 'museum':
            return """💰 **Admission:**
• Major museums: 100-300 TL per person
• Student discount: 50% with valid ID
• Children under 8: Usually free
• Museum Pass: 325 TL (valid 5 days, multiple museums)
• Audio guides: 20-40 TL extra"""
        
        elif category == 'transportation':
            return """💰 **Transport Costs:**
• Istanbulkart: 50 TL (includes 7 TL credit)
• Single ride: 9.90 TL (metro/bus/tram)
• Ferry: 15-35 TL depending on route
• Taxi: Starting fare 8 TL + 4.50 TL per km
• Daily budget: 50-80 TL for moderate public transport use"""
        
        else:
            return """💰 **Budget Planning:**
• Research current prices online
• Consider package deals and discounts
• Keep cash handy (not all places accept cards)
• Factor in transportation costs"""
    
    def _generate_practical_tips(self, response: str, category: str) -> List[str]:
        """Generate practical tips for the category"""
        base_tips = []
        
        if category == 'restaurant':
            base_tips = [
                "🥖 Try Turkish breakfast (kahvaltı) - it's substantial and delicious",
                "🧄 Ask for 'az acılı' (less spicy) if you're sensitive to heat",
                "💳 Many local places prefer cash - have Turkish Lira ready",
                "🍞 Bread is typically free and unlimited at most restaurants",
                "⏰ Turks eat dinner later (20:00-22:00) - plan accordingly"
            ]
        
        elif category == 'museum':
            base_tips = [
                "👕 Dress modestly for religious sites (covered shoulders, long pants)",
                "📱 Download museum apps for self-guided tours and maps",
                "🎧 Audio guides provide valuable historical context",
                "📸 Check photography rules - some areas prohibit photos",
                "💼 Bring a small bag - large bags may require security check"
            ]
        
        elif category == 'transportation':
            base_tips = [
                "📱 Download Moovit app for real-time transport information",
                "🚇 Metro seats are reserved for elderly, pregnant, and disabled",
                "🎒 Keep belongings secure in crowded transport",
                "🚌 Let passengers exit before boarding buses and metro",
                "🗺️ Keep a photo of your destination address in Turkish"
            ]
        
        elif category == 'district':
            base_tips = [
                "👟 Wear comfortable walking shoes for cobblestone streets",
                "📍 Learn major landmark names for easier navigation",
                "🏛️ Many historic areas are pedestrian-friendly",
                "☕ Stop at local cafes to rest and observe daily life",
                "🛍️ Bargaining is expected in bazaars but not in regular shops"
            ]
        
        else:
            base_tips = [
                "📱 Keep your phone charged for maps and translation apps",
                "💡 Learn basic Turkish greetings - locals appreciate the effort",
                "🚰 Tap water is generally safe to drink in Istanbul",
                "🕌 Respect prayer times at mosques (5 times daily)",
                "🎭 Embrace the cultural differences - it's part of the experience"
            ]
        
        return base_tips
    
    def _generate_cultural_notes(self, response: str, category: str) -> List[str]:
        """Generate cultural insights and Turkish phrases"""
        cultural_notes = []
        
        # Add Turkish phrases relevant to category
        if category in self.turkish_phrases:
            phrases = self.turkish_phrases[category]
            cultural_notes.append(f"🇹🇷 **Useful Turkish Phrases:**")
            for eng, turk in phrases.items():
                cultural_notes.append(f"• {turk}")
        
        # Add category-specific cultural insights
        if category == 'restaurant':
            cultural_notes.extend([
                "🫖 Turkish tea (çay) is served in small tulip-shaped glasses",
                "🍽️ Meals are social events - take time to enjoy conversation",
                "🙏 Say 'Afiyet olsun' (bon appétit) before meals"
            ])
        
        elif category == 'museum':
            cultural_notes.extend([
                "🕌 Remove shoes when entering mosque areas",
                "📿 Understand the transition from Byzantine to Ottoman culture",
                "🎨 Appreciate both Islamic and Christian artistic traditions"
            ])
        
        elif category == 'transportation':
            cultural_notes.extend([
                "🚇 Offer seats to elderly and pregnant women",
                "🤝 Locals are generally helpful with directions",
                "⏰ Friday prayer times affect some transport schedules"
            ])
        
        return cultural_notes
    
    def _generate_next_steps(self, response: str, category: str) -> List[str]:
        """Generate actionable next steps"""
        if category == 'restaurant':
            return [
                "1️⃣ Research specific restaurants on Google Maps or TripAdvisor",
                "2️⃣ Check if reservations are needed (especially for dinner)",
                "3️⃣ Learn about Turkish cuisine basics and dietary restrictions",
                "4️⃣ Locate the restaurant and plan your route there",
                "5️⃣ Arrive hungry and ready to try new flavors!"
            ]
        
        elif category == 'museum':
            return [
                "1️⃣ Check official websites for current hours and special exhibitions",
                "2️⃣ Buy tickets online if available to skip entrance lines",
                "3️⃣ Plan your visit route and prioritize must-see items",
                "4️⃣ Download museum app or book guided tour",
                "5️⃣ Allow extra time for unexpected discoveries"
            ]
        
        elif category == 'transportation':
            return [
                "1️⃣ Get an Istanbulkart from any metro station or kiosk",
                "2️⃣ Download transport apps (Moovit, Istanbul Metro)",
                "3️⃣ Study the route map for your destination",
                "4️⃣ Check live schedules and plan connections",
                "5️⃣ Start your journey with extra time for first-time navigation"
            ]
        
        else:
            return [
                "1️⃣ Research specific locations and current information",
                "2️⃣ Plan your route and transportation method",
                "3️⃣ Check weather and dress appropriately",
                "4️⃣ Bring necessary items (cash, ID, camera)",
                "5️⃣ Stay flexible and enjoy the Istanbul experience!"
            ]
    
    def _analyze_category_specific(self, response: str, category: str) -> Tuple[float, List[str]]:
        """Analyze category-specific actionability requirements"""
        response_lower = response.lower()
        score = 0.0
        suggestions = []
        
        if category == 'restaurant':
            restaurant_elements = {
                'specific_names': any(word.isupper() for word in response.split()),
                'cuisine_type': any(word in response_lower for word in ['turkish', 'kebab', 'meze', 'baklava', 'cuisine']),
                'location_context': any(word in response_lower for word in ['near', 'in', 'at', 'district', 'street']),
                'price_range': any(word in response_lower for word in ['affordable', 'expensive', 'budget', 'upscale', 'cost']),
                'atmosphere': any(word in response_lower for word in ['atmosphere', 'ambiance', 'view', 'terrace'])
            }
            score = sum(restaurant_elements.values()) / len(restaurant_elements)
            
            if score < 0.6:
                suggestions.append("Include specific restaurant names and locations")
        
        elif category == 'museum':
            museum_elements = {
                'opening_hours': any(word in response_lower for word in ['hours', 'open', 'closed', 'schedule']),
                'admission_info': any(word in response_lower for word in ['admission', 'ticket', 'entry', 'cost', 'free']),
                'duration_estimate': any(word in response_lower for word in ['visit', 'tour', 'duration', 'time', 'hours']),
                'highlights': any(word in response_lower for word in ['see', 'collection', 'exhibit', 'highlight', 'must']),
                'practical_info': any(word in response_lower for word in ['dress', 'photography', 'audio', 'guide'])
            }
            score = sum(museum_elements.values()) / len(museum_elements)
            
            if score < 0.6:
                suggestions.append("Add practical visiting information and highlights")
        
        elif category == 'transportation':
            transport_elements = {
                'route_info': any(word in response_lower for word in ['line', 'route', 'metro', 'bus', 'm1', 'm2']),
                'timing': any(word in response_lower for word in ['frequency', 'schedule', 'minutes', 'departure']),
                'cost': any(word in response_lower for word in ['fare', 'cost', 'istanbulkart', 'ticket', 'tl']),
                'directions': any(word in response_lower for word in ['platform', 'station', 'stop', 'terminal']),
                'connections': any(word in response_lower for word in ['transfer', 'connection', 'change', 'interchange'])
            }
            score = sum(transport_elements.values()) / len(transport_elements)
            
            if score < 0.6:
                suggestions.append("Include specific routes, costs, and connection information")
        
        return score, suggestions
    
    def _generate_structured_enhancements(self, response: str, missing_elements: List[str], category: str, structured_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate structured enhancements with Address → Directions → Timing → Tips format"""
        enhancements = {}
        
        # Create comprehensive structured response
        structured_response = "## 🎯 **Complete Actionable Guide**\n\n"
        
        # Address section
        if structured_info.get('address'):
            structured_response += structured_info['address'] + "\n\n"
        
        # Directions section
        if structured_info.get('directions'):
            structured_response += structured_info['directions'] + "\n\n"
        
        # Timing section
        if structured_info.get('timing'):
            structured_response += structured_info['timing'] + "\n\n"
        
        # Cost section
        if structured_info.get('cost'):
            structured_response += structured_info['cost'] + "\n\n"
        
        # Tips section
        if structured_info.get('tips'):
            structured_response += "💡 **Essential Tips:**\n"
            for tip in structured_info['tips']:
                structured_response += f"• {tip}\n"
            structured_response += "\n"
        
        # Cultural notes section
        if structured_info.get('cultural_notes'):
            structured_response += "🎭 **Cultural Insights:**\n"
            for note in structured_info['cultural_notes']:
                structured_response += f"• {note}\n"
            structured_response += "\n"
        
        # Next steps section
        if structured_info.get('next_steps'):
            structured_response += "📋 **Your Next Steps:**\n"
            for step in structured_info['next_steps']:
                structured_response += f"{step}\n"
            structured_response += "\n"
        
        enhancements['structured_guide'] = structured_response
        
        return enhancements
    
    def _generate_improvement_suggestions(self, missing_elements: List[str]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        element_suggestions = {
            'address_info': "✅ Add specific addresses and location landmarks",
            'timing_info': "⏰ Include operating hours, schedules, and best visiting times",
            'direction_info': "🗺️ Provide clear directions and transportation options",
            'cost_info': "💰 Mention current prices, costs, and payment methods",
            'practical_steps': "📋 Add step-by-step instructions and practical tips",
            'cultural_context': "🎭 Include cultural insights and local customs"
        }
        
        for element in missing_elements:
            if element in element_suggestions:
                suggestions.append(element_suggestions[element])
        
        return suggestions
    
    def enhance_response(self, response: str, analysis: ActionabilityAnalysis, category: str) -> str:
        """Enhance response with structured actionable information"""
        try:
            if analysis.score >= 0.8:
                return response  # Already highly actionable
            
            enhanced_response = response
            
            # Add structured enhancements
            if analysis.enhanced_sections:
                enhanced_response += "\n\n---\n\n"
                
                for section_type, enhancement in analysis.enhanced_sections.items():
                    enhanced_response += enhancement
            
            # Add quick reference footer
            enhanced_response += self._get_actionability_footer(category)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response
    
    def _get_actionability_footer(self, category: str) -> str:
        """Get category-specific actionability footer with Turkish integration"""
        footers = {
            'restaurant': """
---
🚀 **Quick Start:** Use Google Maps → Check reviews → Make reservation → Learn 'Afiyet olsun!' (Enjoy your meal!)
🇹🇷 **Essential:** 'Hesap, lütfen' (Bill, please) | 'Çok lezzetli' (Very delicious)""",
            
            'museum': """
---
🚀 **Quick Start:** Check website → Buy online ticket → Plan 2-3 hours → Dress modestly → Arrive early
🇹🇷 **Essential:** 'Bilet' (Ticket) | 'Ne kadar?' (How much?) | 'Çok güzel' (Very beautiful)""",
            
            'transportation': """
---
🚀 **Quick Start:** Get Istanbulkart → Download Moovit app → Check route map → Allow extra time
🇹🇷 **Essential:** 'İstasyon nerede?' (Where is the station?) | 'Aktarma' (Transfer)""",
            
            'default': """
---
🚀 **Quick Start:** Research online → Plan route → Check current info → Enjoy Istanbul!
🇹🇷 **Essential:** 'Merhaba' (Hello) | 'Teşekkürler' (Thank you) | 'Güle güle' (Goodbye)"""
        }
        
        return footers.get(category, footers['default'])
    
    def enhance_response_actionability(self, response: str, query: str, category: str, location: str = None) -> Dict[str, Any]:
        """Wrapper method for main.py integration - enhance response actionability with Turkish support"""
        try:
            # Analyze current actionability
            analysis = self.analyze_actionability(response, category)
            
            # Enhance the response
            enhanced_response = self.enhance_response(response, analysis, category)
            
            # Create structured format (Address → Directions → Timing → Tips)
            structured_response = self._create_structured_format(enhanced_response, category, location)
            
            # Add cultural enhancement with Turkish phrases
            cultural_enhancement = self._add_cultural_context(category, location)
            
            # Generate local insights
            local_insights = self._generate_local_insights(category, location, query)
            
            return {
                "success": True,
                "actionability_score": analysis.score,
                "enhanced_response": enhanced_response,
                "structured_response": structured_response,
                "cultural_enhancement": cultural_enhancement,
                "local_insights": local_insights,
                "analysis": {
                    "missing_elements": analysis.missing_elements,
                    "enhanced_sections": list(analysis.enhanced_sections.keys()) if analysis.enhanced_sections else []
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhance_response_actionability wrapper: {e}")
            return {
                "success": False,
                "error": str(e),
                "actionability_score": 0.0
            }
    
    def _create_structured_format(self, response: str, category: str, location: str = None) -> str:
        """Create structured Address → Directions → Timing → Tips format"""
        try:
            structured_parts = []
            
            # Address section
            structured_parts.append("📍 **ADDRESS & LOCATION:**")
            if location:
                structured_parts.append(f"Located in {location.title()}, Istanbul")
            structured_parts.append("(Specific addresses provided above)")
            structured_parts.append("")
            
            # Directions section  
            structured_parts.append("🚇 **DIRECTIONS:**")
            if category == "transportation":
                structured_parts.append("• Use Istanbulkart for all public transport")
                structured_parts.append("• Check live schedules on Moovit app")
            elif category in ["museum_advice", "cultural_sites"]:
                structured_parts.append("• Take metro to nearest station (details above)")
                structured_parts.append("• Follow walking directions from metro exit")
            else:
                structured_parts.append("• Public transport options detailed above")
                structured_parts.append("• Walking directions from nearest metro station")
            structured_parts.append("")
            
            # Timing section
            structured_parts.append("⏰ **TIMING & SCHEDULE:**")
            structured_parts.append("• Best visited during daylight hours")
            structured_parts.append("• Allow extra time during rush hours (7-9 AM, 5-7 PM)")
            if category == "museum_advice":
                structured_parts.append("• Museums typically close one day per week")
            structured_parts.append("")
            
            # Tips section
            structured_parts.append("💡 **PRACTICAL TIPS:**")
            structured_parts.append("• Carry Turkish Lira for small purchases")
            structured_parts.append("• Download offline maps before traveling")
            if category == "restaurant":
                structured_parts.append("• Reservations recommended for dinner")
            elif category == "museum_advice":
                structured_parts.append("• Audio guides available in multiple languages")
            structured_parts.append("")
            
            return "\n".join(structured_parts)
            
        except Exception as e:
            logger.error(f"Error creating structured format: {e}")
            return ""
    
    def _add_cultural_context(self, category: str, location: str = None) -> str:
        """Add Turkish phrases and cultural context"""
        try:
            cultural_parts = []
            
            cultural_parts.append("🇹🇷 **CULTURAL CONTEXT & TURKISH PHRASES:**")
            cultural_parts.append("")
            
            # Basic Turkish phrases
            cultural_parts.append("**Useful Turkish Phrases:**")
            cultural_parts.append("• Merhaba (mer-ha-BA) = Hello")
            cultural_parts.append("• Teşekkür ederim (teh-shek-KOOR eh-deh-rim) = Thank you")
            cultural_parts.append("• Özür dilerim (oh-ZOOR dee-leh-rim) = Excuse me/Sorry") 
            cultural_parts.append("• Nerede? (neh-reh-DEH) = Where is?")
            cultural_parts.append("• Ne kadar? (neh kah-DAR) = How much?")
            cultural_parts.append("")
            
            # Cultural tips based on category
            if category == "museum_advice":
                cultural_parts.append("**Cultural Etiquette:**")
                cultural_parts.append("• Remove shoes when entering mosques")
                cultural_parts.append("• Dress modestly (cover shoulders and knees)")
                cultural_parts.append("• Photography may be restricted in some areas")
            elif category == "restaurant":
                cultural_parts.append("**Dining Culture:**")
                cultural_parts.append("• Turkish breakfast is a feast - arrive hungry!")
                cultural_parts.append("• Tea (çay) is offered as hospitality")
                cultural_parts.append("• Tipping 10-15% is appreciated but not mandatory")
            else:
                cultural_parts.append("**General Cultural Tips:**")
                cultural_parts.append("• Turkish people are very hospitable and helpful")
                cultural_parts.append("• Learning a few Turkish words is greatly appreciated")
                cultural_parts.append("• Friday prayers mean some areas may be busier")
            
            return "\n".join(cultural_parts)
            
        except Exception as e:
            logger.error(f"Error adding cultural context: {e}")
            return ""
    
    def _generate_local_insights(self, category: str, location: str = None, query: str = None) -> str:
        """Generate local insider tips"""
        try:
            insights = []
            
            if location:
                location_lower = location.lower()
                if "sultanahmet" in location_lower:
                    insights.append("Visit early morning or late afternoon to avoid crowds")
                    insights.append("The area between Blue Mosque and Hagia Sophia has the best photo opportunities")
                elif "beyoglu" in location_lower or "taksim" in location_lower:
                    insights.append("Istiklal Street is busiest on weekends - visit weekdays for a better experience")
                    insights.append("Hidden gems are often in the side streets off the main avenue")
                elif "kadikoy" in location_lower:
                    insights.append("This is where locals hang out - perfect for authentic experiences")
                    insights.append("The Tuesday and Friday markets offer great local products")
            
            # Category-specific insights
            if category == "transportation":
                insights.append("Download the 'IBB CepTrafik' app for real-time traffic updates")
                insights.append("Ferries are not just transport - they're scenic tours!")
            elif category == "museum_advice":
                insights.append("Museum Pass Istanbul can save money if visiting multiple sites")
                insights.append("Audio guides are usually worth the extra cost")
            elif category == "restaurant":
                insights.append("Local recommendations: Ask 'En iyi [food type] nerede?' (Where's the best [food]?)")
                insights.append("Street food is generally safe and delicious - try simit and döner")
            
            return " | ".join(insights) if insights else "Enjoy exploring Istanbul like a local!"
            
        except Exception as e:
            logger.error(f"Error generating local insights: {e}")
            return "Enjoy your time in Istanbul!"
