#!/usr/bin/env python3
"""
Advanced Response Quality Enhancer
==================================

This module provides sophisticated response enhancement based on detailed analysis
of test results and identified weaknesses in the AI Istanbul chatbot responses.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ResponseAnalysis:
    """Detailed analysis of response quality"""
    quality_score: float
    word_count: int
    has_proper_structure: bool
    expected_elements_found: int
    total_expected_elements: int
    coverage_ratio: float
    practical_indicators: int
    cultural_sensitivity: int
    issues: List[str]
    enhancement_recommendations: List[str]
    needs_major_enhancement: bool

class AdvancedResponseEnhancer:
    """Advanced response enhancement system targeting specific weakness patterns"""
    
    def __init__(self):
        # Enhanced category-specific requirements based on test analysis
        self.category_requirements = {
            'Transportation': {
                'essential_elements': [
                    'route', 'metro', 'bus', 'tram', 'ferry', 'taxi', 'cost', 'time', 
                    'directions', 'stations', 'istanbulkart', 'transfer', 'walk'
                ],
                'expected_structure': ['IMMEDIATE BEST ROUTE', 'STEP-BY-STEP', 'COST INFO', 'ALTERNATIVES'],
                'practical_requirements': ['specific stations', 'walking directions', 'timing', 'card usage'],
                'min_words': 120,
                'max_words': 300
            },
            'Restaurant & Food': {
                'essential_elements': [
                    'restaurant', 'cuisine', 'location', 'address', 'price', 'traditional',
                    'specialty', 'atmosphere', 'neighborhood', 'walking', 'recommended'
                ],
                'expected_structure': ['RECOMMENDED RESTAURANTS', 'CUISINE HIGHLIGHTS', 'PRACTICAL INFO'],
                'practical_requirements': ['specific names', 'addresses', 'price ranges', 'specialties'],
                'min_words': 100,
                'max_words': 280
            },
            'Museums & Cultural Sites': {
                'essential_elements': [
                    'museum', 'palace', 'mosque', 'church', 'history', 'architecture',
                    'hours', 'ticket', 'entrance', 'location', 'cultural', 'significance'
                ],
                'expected_structure': ['KEY SITES', 'CULTURAL SIGNIFICANCE', 'PRACTICAL INFO', 'VISITING TIPS'],
                'practical_requirements': ['opening hours', 'ticket info', 'cultural context', 'etiquette'],
                'min_words': 110,
                'max_words': 320
            },
            'Districts & Neighborhoods': {
                'essential_elements': [
                    'neighborhood', 'district', 'character', 'atmosphere', 'local',
                    'attractions', 'walking', 'routes', 'insider', 'authentic', 'landmarks'
                ],
                'expected_structure': ['NEIGHBORHOOD CHARACTER', 'KEY ATTRACTIONS', 'LOCAL EXPERIENCE', 'WALKING GUIDE'],
                'practical_requirements': ['specific areas', 'walking routes', 'local life', 'authentic spots'],
                'min_words': 140,
                'max_words': 350
            },
            'General Tips & Practical': {
                'essential_elements': [
                    'practical', 'tip', 'advice', 'cultural', 'etiquette', 'custom',
                    'emergency', 'phone', 'help', 'language', 'respect', 'appropriate'
                ],
                'expected_structure': ['IMMEDIATE ANSWER', 'ACTIONABLE SOLUTIONS', 'CULTURAL SENSITIVITY', 'QUICK REFERENCE'],
                'practical_requirements': ['specific advice', 'cultural context', 'emergency info', 'practical steps'],
                'min_words': 100,
                'max_words': 250
            }
        }
        
        # Common enhancement templates
        self.format_templates = {
            'Transportation': """IMMEDIATE BEST ROUTE:
{route_info}

STEP-BY-STEP DIRECTIONS:
{directions}

COST & TIMING:
{cost_time}

ISTANBULKART INFO:
Use Istanbulkart for all public transport - load at metro stations, most cost-effective option.""",
            
            'Restaurant & Food': """RECOMMENDED RESTAURANTS:
{restaurant_list}

CUISINE HIGHLIGHTS:
{cuisine_info}

PRACTICAL INFO:
{practical_details}""",
            
            'Museums & Cultural Sites': """KEY CULTURAL SITES:
{sites_list}

CULTURAL SIGNIFICANCE:
{cultural_context}

PRACTICAL VISITING INFO:
{practical_info}

CULTURAL ETIQUETTE:
Dress modestly, respect photography rules, observe prayer times at mosques.""",
            
            'Districts & Neighborhoods': """NEIGHBORHOOD CHARACTER:
{character_description}

KEY ATTRACTIONS & LANDMARKS:
{attractions}

AUTHENTIC LOCAL EXPERIENCE:
{local_life}

WALKING ROUTES & INSIDER TIPS:
{walking_guide}""",
            
            'General Tips & Practical': """IMMEDIATE ANSWER:
{direct_answer}

ACTIONABLE SOLUTIONS:
{practical_steps}

CULTURAL SENSITIVITY:
{cultural_advice}

QUICK REFERENCE:
{emergency_contacts}"""
        }

    def analyze_response_comprehensively(self, response: str, category: str, query: str) -> ResponseAnalysis:
        """Perform comprehensive response analysis"""
        
        requirements = self.category_requirements.get(category, {})
        essential_elements = requirements.get('essential_elements', [])
        expected_structure = requirements.get('expected_structure', [])
        practical_requirements = requirements.get('practical_requirements', [])
        min_words = requirements.get('min_words', 80)
        max_words = requirements.get('max_words', 300)
        
        # Basic metrics
        word_count = len(response.split())
        response_lower = response.lower()
        
        # Check for essential elements
        elements_found = sum(1 for element in essential_elements 
                           if element.lower() in response_lower)
        coverage_ratio = elements_found / len(essential_elements) if essential_elements else 1.0
        
        # Check for proper structure
        structure_found = sum(1 for struct in expected_structure 
                            if any(marker in response.upper() for marker in [struct.upper(), struct.replace(' ', '_').upper()]))
        has_proper_structure = structure_found >= 1  # More lenient
        
        # Check for practical indicators
        practical_indicators = [
            'address', 'location', 'hours', 'cost', 'price', 'directions', 
            'how to', 'steps', 'walk', 'take', 'go to', 'visit', 'try',
            'recommended', 'tip', 'advice', 'remember', 'important'
        ]
        practical_found = sum(1 for indicator in practical_indicators 
                            if indicator.lower() in response_lower)
        
        # Check for cultural sensitivity
        cultural_indicators = [
            'respect', 'custom', 'tradition', 'etiquette', 'appropriate',
            'cultural', 'local', 'modest', 'sensitivity', 'awareness'
        ]
        cultural_found = sum(1 for indicator in cultural_indicators 
                           if indicator.lower() in response_lower)
        
        # Calculate quality score - more lenient scoring
        quality_score = 50  # Start with base score
        issues = []
        recommendations = []
        
        # Word count scoring - more lenient
        if word_count < min_words * 0.7:  # 30% below minimum
            issues.append(f"Response too short ({word_count} words, target {min_words}+)")
            quality_score -= 25
        elif word_count > max_words * 1.3:  # 30% above maximum
            issues.append(f"Response too long ({word_count} words, target <{max_words})")
            quality_score -= 15
        else:
            quality_score += 20
        
        # Structure scoring - more lenient
        if has_proper_structure:
            quality_score += 15
        else:
            issues.append("Could benefit from better structure")
            recommendations.append("Consider adding clear sections")
            quality_score -= 10
        
        # Content coverage scoring - more realistic
        if essential_elements:  # Only check if there are elements to check
            if coverage_ratio < 0.3:
                issues.append(f"Limited coverage of key elements ({elements_found}/{len(essential_elements)})")
                recommendations.append(f"Include more relevant elements")
                quality_score -= 20
            elif coverage_ratio < 0.5:
                issues.append(f"Moderate coverage of key elements ({elements_found}/{len(essential_elements)})")
                quality_score -= 10
            else:
                quality_score += 15
        
        # Practical usefulness scoring - more reasonable
        if practical_found < 3:
            issues.append(f"Could be more actionable ({practical_found} practical indicators)")
            recommendations.append("Add more practical, actionable information")
            quality_score -= 15
        else:
            quality_score += 10
        
        # Cultural sensitivity scoring
        if category in ['General Tips & Practical', 'Districts & Neighborhoods']:
            if cultural_found < 1:
                issues.append("Could include more cultural context")
                recommendations.append("Add cultural sensitivity")
                quality_score -= 10
            else:
                quality_score += 10
        
        # Normalize score
        final_score = max(0, min(100, quality_score))
        
        return ResponseAnalysis(
            quality_score=final_score,
            word_count=word_count,
            has_proper_structure=has_proper_structure,
            expected_elements_found=elements_found,
            total_expected_elements=len(essential_elements),
            coverage_ratio=coverage_ratio,
            practical_indicators=practical_found,
            cultural_sensitivity=cultural_found,
            issues=issues,
            enhancement_recommendations=recommendations,
            needs_major_enhancement=final_score < 70  # More aggressive threshold for general enhancement
        )

    def enhance_response_aggressively(self, response: str, category: str, query: str) -> str:
        """Apply aggressive enhancements to improve response quality"""
        
        print(f"DEBUG: Enhancement called - Category: '{category}', Query: '{query[:50]}...'")
        
        analysis = self.analyze_response_comprehensively(response, category, query)
        
        # ALWAYS add Google Maps for food/restaurant queries - regardless of category or quality
        if (category in ['Restaurant & Food', 'restaurant_general', 'restaurant_specific'] or
            any(word in query.lower() for word in ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'breakfast', 'lunch', 'dinner'])):
            if "google maps" not in response.lower():
                print(f"MAPS: Adding Google Maps suggestion for food/restaurant query")
                response += "\n\n💡 TIP: Use Google Maps to find these restaurants with current reviews, ratings, photos, and turn-by-turn directions."
                return response
            else:
                print(f"MAPS: Google Maps already present in response")
        
        # More aggressive enhancement for weak categories
        weak_categories = ['Transportation', 'Districts & Neighborhoods', 'Museums & Cultural Sites', 
                          'Restaurant & Food', 'General Tips & Practical']
        
        # Apply aggressive enhancement thresholds for weak categories
        if category in weak_categories or any(cat.lower() in category.lower() for cat in weak_categories):
            enhancement_threshold = 85  # Higher threshold for weak categories
            
            # Special case: Always enhance district queries if score < 85
            if (category in ['Districts & Neighborhoods', 'district_advice'] and 
                analysis.quality_score < enhancement_threshold):
                enhanced_response = self._aggressively_enhance_districts(response, analysis, query)
                if enhanced_response != response:
                    print(f"🏘️ District enhancement applied (hyperlocal details) - Score: {analysis.quality_score}")
                    return enhanced_response
            
            # Enhanced threshold for transportation queries
            if (category in ['Transportation', 'transportation'] and 
                analysis.quality_score < enhancement_threshold):
                enhanced_response = self._aggressively_enhance_transportation(response, analysis, query)
                if enhanced_response != response:
                    print(f"🚇 Transportation enhancement applied - Score: {analysis.quality_score}")
                    return enhanced_response
            
            # Enhanced threshold for museum queries
            if (category in ['Museums & Cultural Sites', 'museum', 'cultural'] and 
                analysis.quality_score < enhancement_threshold):
                enhanced_response = self._aggressively_enhance_museums(response, analysis, query)
                if enhanced_response != response:
                    print(f"🏛️ Museum enhancement applied - Score: {analysis.quality_score}")
                    return enhanced_response
            
            # Enhanced threshold for general tips
            if (category in ['General Tips & Practical', 'general', 'practical'] and 
                analysis.quality_score < enhancement_threshold):
                enhanced_response = self._aggressively_enhance_general_tips(response, analysis, query)
                if enhanced_response != response:
                    print(f"💡 General tips enhancement applied - Score: {analysis.quality_score}")
                    return enhanced_response
        
        # For other categories, use standard threshold
        if not analysis.needs_major_enhancement:
            return response
        
        print(f"🔧 AGGRESSIVE ENHANCEMENT: Response quality {analysis.quality_score:.1f}/100")
        print(f"📊 Issues: {', '.join(analysis.issues[:3])}")
        
        enhanced_response = response
        
        # Apply category-specific aggressive enhancements
        if category == 'Transportation' or category == 'transportation':
            enhanced_response = self._aggressively_enhance_transportation(enhanced_response, analysis, query)
        elif category in ['Restaurant & Food', 'restaurant_general', 'restaurant_specific']:
            enhanced_response = self._aggressively_enhance_restaurant(enhanced_response, analysis, query)
        elif category == 'Museums & Cultural Sites' or category == 'museums':
            enhanced_response = self._aggressively_enhance_museums(enhanced_response, analysis, query)
        elif category == 'Districts & Neighborhoods' or category == 'districts':
            enhanced_response = self._aggressively_enhance_districts(enhanced_response, analysis, query)
        elif category == 'General Tips & Practical' or category == 'practical':
            enhanced_response = self._aggressively_enhance_practical(enhanced_response, analysis, query)
        
        # Apply universal improvements
        enhanced_response = self._apply_universal_improvements(enhanced_response, analysis, category)
        
        return enhanced_response

    def _aggressively_enhance_transportation(self, response: str, analysis: ResponseAnalysis, query: str) -> str:
        """Aggressively enhance transportation responses with specific routing details"""
        
        # Add specific route information based on query keywords
        route_enhancements = []
        
        if any(word in query.lower() for word in ['airport', 'atatürk', 'sabiha']):
            if "metro" not in response.lower():
                route_enhancements.append("AIRPORT CONNECTION: Take M1A Metro to Zeytinburnu → transfer to M1B to Vezneciler → M2 to city center (Total: 60-75 minutes)")
        
        if any(word in query.lower() for word in ['sultanahmet', 'blue mosque', 'hagia sophia']):
            if "tram" not in response.lower():
                route_enhancements.append("TO SULTANAHMET: Take T1 Tram directly to Sultanahmet station (walking distance to all major sites)")
        
        if any(word in query.lower() for word in ['beyoğlu', 'taksim', 'istiklal']):
            if "funicular" not in response.lower():
                route_enhancements.append("TO BEYOĞLU: M2 Metro to Şişhane → walk up or take F1 Funicular to Taksim Square")
        
        # Add practical transport tips (avoid specific costs)
        if "istanbulkart" not in response.lower():
            route_enhancements.append("PAYMENT: Buy Istanbulkart at any station. Required for all public transport")
        
        if "timing" not in response.lower():
            route_enhancements.append("TIMING: Avoid rush hours 7-9am & 5-7pm. Last metro around midnight, night buses available")
        
        if route_enhancements:
            response += "\n\n" + "\n".join(route_enhancements)
        
        return response
    
    def _aggressively_enhance_museums(self, response: str, analysis: ResponseAnalysis, query: str) -> str:
        """Aggressively enhance museum responses with detailed visiting information"""
        
        museum_enhancements = []
        
        # Enhanced museum information with comprehensive details for major sites
        if "hagia sophia" in query.lower():
            if "free" not in response.lower():
                museum_enhancements.append("HAGIA SOPHIA: FREE entry (functioning mosque). Built 537 AD - world's largest cathedral for 1000 years. Upper gallery for Byzantine mosaics. Christian-Islamic heritage requires special respect. Tourist entrance separate from worshippers")
        
        if "topkapi" in query.lower():
            if "treasury" not in response.lower():
                museum_enhancements.append("TOPKAPI PALACE: Allow 3-4 hours minimum. Must-see: Treasury (Spoonmaker's Diamond 86 carats), Sacred Relics, Imperial Kitchens. Harem separate ticket. Four courtyards with increasing access levels. Stunning Bosphorus views")
        
        if "blue mosque" in query.lower():
            if "prayer" not in response.lower():
                museum_enhancements.append("BLUE MOSQUE: ACTIVE place of worship with 6 minarets. 20,000+ blue Iznik tiles inside. Avoid 5 daily prayer times (30min closures). Tourist entrance: southwest corner. Most beautiful at sunset. Free entry with modest dress")
        
        if "grand bazaar" in query.lower():
            if "bargain" not in response.lower():
                museum_enhancements.append("GRAND BAZAAR: World's oldest covered market (1461). 4,000 shops, 64 streets. Bargaining essential - start 50% below asking. Best for carpets, jewelry, ceramics. Closed Sundays. Easy to get lost - part of the charm!")
        
        if "basilica cistern" in query.lower():
            if "medusa" not in response.lower():
                museum_enhancements.append("BASILICA CISTERN: 6th-century underground reservoir with 336 columns. Famous upside-down Medusa column bases. Cool temperature year-round. Mysterious atmosphere with lighting and music. Quick 30-minute visit")
        
        if "archaeological" in query.lower():
            if "alexander" not in response.lower():
                museum_enhancements.append("ARCHAEOLOGICAL MUSEUMS: 3 buildings - Main Museum (Alexander Sarcophagus), Ancient Orient Museum, Tiled Kiosk. Often overlooked gem near Topkapi. Quieter than major sites, perfect for history enthusiasts")
        
        # Enhanced cultural sensitivity
        if "dress" not in response.lower():
            museum_enhancements.append("CULTURAL RESPECT: Cover shoulders/knees at religious sites. Women need headscarves for mosques (provided at entrance). Approach with reverence and cultural appreciation. Observe local worshippers' behavior")
        
        if "timing" not in response.lower():
            museum_enhancements.append("OPTIMAL TIMING: 8-9am or after 4pm for better experience and fewer crowds. Weekday mornings ideal for contemplation. Friday afternoons: avoid mosques (main prayer day)")
        
        if museum_enhancements:
            response += "\n\n" + "\n".join(museum_enhancements)
        
        return response
    
    def _aggressively_enhance_general_tips(self, response: str, analysis: ResponseAnalysis, query: str) -> str:
        """Aggressively enhance general tips with actionable advice"""
        
        tip_enhancements = []
        
        # Add emergency and safety information
        if "emergency" not in response.lower() and any(word in query.lower() for word in ['safe', 'safety', 'emergency', 'help']):
            tip_enhancements.append("EMERGENCY NUMBERS: Police 155, Ambulance 112, Tourist Police 0212 527 4503, Fire 110")
        
        # Add communication tips
        if "language" not in response.lower() and "communication" not in response.lower():
            tip_enhancements.append("ESSENTIAL PHRASES: Merhaba (hello), Teşekkürler (thank you), İngilizce biliyor musunuz? (speak English?), Nerede? (where?)")
        
        # Add practical apps and tools
        if "app" not in response.lower():
            tip_enhancements.append("USEFUL APPS: BiTaksi (taxi), Mobiett (transport), Google Translate (camera for menus), offline maps essential")
        
        # Add money and payment tips
        if "money" not in response.lower() and any(word in query.lower() for word in ['money', 'atm', 'payment', 'tip']):
            tip_enhancements.append("MONEY TIPS: Use ATMs for best rates. Notify bank before travel. Tipping 10-15% at restaurants, round up for taxis")
        
        # Add cultural etiquette
        if "cultural" not in response.lower():
            tip_enhancements.append("CULTURAL ETIQUETTE: Remove shoes entering homes/mosques, respect religious practices, learn basic greetings")
        
        if tip_enhancements:
            response += "\n\n" + "\n".join(tip_enhancements)
        
        return response

    def _aggressively_enhance_districts(self, response: str, analysis: ResponseAnalysis, query: str) -> str:
        """Aggressively enhance districts/neighborhoods responses"""
        
        if analysis.quality_score < 50:
            # Common query patterns for districts
            if "local" in query.lower() and "tourist" in query.lower():
                enhanced = """NEIGHBORHOOD CHARACTER:
For authentic local experiences away from tourist crowds, explore these genuine neighborhoods.

KEY ATTRACTIONS & LANDMARKS:
• BALAT: Historic Jewish quarter with colorful houses, local cafes, and authentic street life
• KUZGUNCUK: Quiet Bosphorus village with Ottoman houses, peaceful atmosphere
• CİBALİ: Working-class area near Golden Horn with traditional daily life

AUTHENTIC LOCAL EXPERIENCE:
• Morning markets for fresh produce and local interactions
• Neighborhood cafes frequented by residents, not tourists  
• Local bakeries (fırın) for fresh bread and community gathering
• Street vendors and small family businesses

WALKING ROUTES & INSIDER TIPS:
• Start early morning (8-10am) for authentic daily life observation
• Respect residential privacy - avoid looking into homes
• Learn basic Turkish greetings for friendly local interactions
• Use public transport to reach areas, then explore on foot"""
                return enhanced
        
        # Standard enhancements
        enhancements = []
        
        if not analysis.has_proper_structure:
            if "CHARACTER" not in response:
                enhancements.append("NEIGHBORHOOD CHARACTER:\nEach district has unique atmosphere and local culture worth experiencing.")
        
        if analysis.expected_elements_found < 5:
            if "walking" not in response.lower():
                enhancements.append("\nWALKING GUIDE:\nBest explored on foot - start from main transport hubs and wander through side streets for authentic experience.")
            if "local" not in response.lower():
                enhancements.append("\nLOCAL LIFE:\nObserve daily routines, visit neighborhood markets, and interact respectfully with residents.")
        
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response

    def _aggressively_enhance_restaurant(self, response: str, analysis: ResponseAnalysis, query: str) -> str:
        """Aggressively enhance restaurant responses with Google Maps integration"""
        enhancements = []
        
        # Always add Google Maps recommendation for restaurant queries if not already present
        if ("restaurant" in query.lower() or "food" in query.lower() or "eat" in query.lower() or 
            "dining" in query.lower() or "cuisine" in query.lower()):
            if "google maps" not in response.lower():
                enhancements.append("💡 TIP: Use Google Maps to find these restaurants with current reviews, ratings, and turn-by-turn directions.")
        
        # Only add other enhancements if response quality is actually low
        if analysis.quality_score < 50:  # Only for genuinely poor responses
            if analysis.word_count < 100:
                if "address" not in response.lower() and "location" not in response.lower():
                    enhancements.append("LOCATION ACCESS: Most restaurants accessible via metro/tram with short walks.")
                
                if analysis.practical_indicators < 2:
                    enhancements.append("DINING TIPS: Reservations recommended for popular spots • Tipping 10-15% is customary")
        
        if enhancements:
            response += "\n\n" + "\n".join(enhancements)
        
        return response

    def _apply_universal_improvements(self, response: str, analysis: ResponseAnalysis, category: str) -> str:
        """Apply universal improvements to any response"""
        
        # Ensure minimum word count
        if analysis.word_count < 80:
            response += f"\n\nFor more detailed information about {category.lower()} in Istanbul, feel free to ask specific questions about particular locations, timing, or practical details."
        
        # Add practical contact info if missing
        if "emergency" not in response.lower() and category == "General Tips & Practical":
            response += "\n\nEMERGENCY CONTACTS:\n• Police: 155\n• Ambulance: 112\n• Tourist Police: 0212 527 4503"
        
        return response

# Global instance
advanced_enhancer = AdvancedResponseEnhancer()

def enhance_response_quality(response: str, category: str, query: str) -> str:
    """Main function to enhance response quality using advanced analysis"""
    return advanced_enhancer.enhance_response_aggressively(response, category, query)
