"""
Emergency & Safety Handler for Istanbul AI
Handles: Medical emergencies, police, embassies, safety information, tourist helplines

This handler provides critical safety and emergency information for tourists in Istanbul.

🚨 PRIORITY: High-priority handler (safety first!)
🌐 MULTILINGUAL: LLM automatically detects and responds in user's language
📍 GPS-AWARE: Provides nearest hospital/embassy based on user location
📞 CONTACT-RICH: Includes emergency numbers, addresses, and directions

Created: November 5, 2025
"""

from typing import Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class EmergencySafetyHandler:
    """
    Emergency & Safety Information Handler
    
    Capabilities:
    - Medical emergencies (hospitals, pharmacies, ambulances)
    - Police assistance (tourist police, emergency numbers)
    - Embassy/Consulate information and locations
    - Safety tips and advice for tourists
    - Lost passport/documents procedures
    - Common scams and how to avoid them
    - 🌐 Automatic multilingual support (LLM detects language from query)
    - 📍 GPS-aware recommendations (nearest facilities)
    """
    
    # Emergency keywords (multilingual)
    EMERGENCY_KEYWORDS = [
        # Medical - English
        'hospital', 'doctor', 'ambulance', 'pharmacy', 'medical', 'sick', 'ill',
        'emergency', 'health', 'clinic', 'medicine', 'pain', 'injury', 'accident',
        
        # Medical - Turkish
        'hastane', 'doktor', 'ambulans', 'eczane', 'sağlık', 'hasta', 'acil',
        'ilaç', 'ağrı', 'yaralanma', 'kaza',
        
        # Police & Safety - English
        'police', 'theft', 'stolen', 'lost', 'crime', 'scam', 'danger', 'safe',
        'unsafe', 'security', 'help', 'trouble', 'attack', 'robbery',
        
        # Police & Safety - Turkish
        'polis', 'çalındı', 'kayıp', 'suç', 'dolandırıcılık', 'tehlike', 'güvenli',
        'güvensiz', 'güvenlik', 'yardım', 'sorun', 'saldırı', 'hırsızlık',
        
        # Embassy - English
        'embassy', 'consulate', 'visa', 'passport', 'consul', 'diplomatic',
        
        # Embassy - Turkish
        'büyükelçilik', 'konsolosluk', 'vize', 'pasaport', 'konsolos',
        
        # Emergency numbers/phrases
        '112', '155', '153', 'emergency number', 'acil numara',
    ]
    
    def __init__(
        self,
        llm_service=None,
        gps_location_service=None,
        neural_processor=None  # Add neural_processor parameter (optional, for compatibility)
    ):
        """
        Initialize emergency & safety handler.
        
        Args:
            llm_service: LLM service for natural language responses (REQUIRED)
            gps_location_service: GPS service for location-based recommendations (OPTIONAL)
            neural_processor: Neural processor (optional, not used but accepted for compatibility)
        """
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        self.neural_processor = neural_processor  # Store but don't require
        
        # Service availability flags
        self.has_llm = llm_service is not None
        self.has_gps = gps_location_service is not None
        
        logger.info(
            f"🚨 Emergency & Safety Handler initialized - "
            f"LLM: {self.has_llm} (auto-multilingual), "
            f"GPS: {self.has_gps}"
        )
    
    def can_handle(self, message: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if this handler should process the query.
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            True if this is an emergency/safety query
        """
        message_lower = message.lower()
        
        # Check for emergency keywords
        return any(keyword in message_lower for keyword in self.EMERGENCY_KEYWORDS)
    
    def handle(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile=None,
        context=None,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle emergency & safety queries.
        
        Args:
            message: User's query (in any language - LLM will auto-detect)
            entities: Extracted entities
            user_profile: User profile (optional)
            context: Conversation context (optional)
            return_structured: Whether to return structured data
            
        Returns:
            Emergency/safety information
        """
        try:
            logger.info(f"🚨 Processing emergency/safety query")
            
            # Detect urgency level
            urgency = self._detect_urgency(message)
            logger.info(f"🚨 Urgency level: {urgency}")
            
            # Get user location if available
            user_location = self._get_user_location(context)
            
            # Classify emergency type
            emergency_type = self._classify_emergency_type(message)
            logger.info(f"🚨 Emergency type: {emergency_type}")
            
            # Generate response based on emergency type
            if emergency_type == 'medical':
                response = self._handle_medical_emergency(
                    message, urgency, user_location
                )
            elif emergency_type == 'police':
                response = self._handle_police_safety(
                    message, urgency, user_location
                )
            elif emergency_type == 'embassy':
                response = self._handle_embassy_consulate(
                    message, user_location
                )
            elif emergency_type == 'safety_info':
                response = self._handle_safety_information(message)
            else:
                response = self._handle_general_emergency(
                    message, user_location
                )
            
            if return_structured:
                return {
                    'response': response,
                    'handler': 'emergency_safety',
                    'emergency_type': emergency_type,
                    'urgency': urgency,
                    'user_location': user_location
                }
            
            return response
            
        except Exception as e:
            logger.error(f"🚨 Emergency handler error: {e}", exc_info=True)
            return self._get_fallback_response(message)
    
    def _detect_urgency(self, message: str) -> str:
        """
        Detect urgency level of emergency.
        
        Args:
            message: User's query
            
        Returns:
            'critical', 'high', 'medium', or 'low'
        """
        message_lower = message.lower()
        
        # Critical urgency indicators
        critical_indicators = [
            'emergency', 'urgent', 'help now', 'immediately', 'right now',
            'acil', 'hemen', 'şimdi', 'ambulans', 'ambulance'
        ]
        
        # High urgency indicators
        high_indicators = [
            'need', 'lost', 'stolen', 'attack', 'injury', 'accident',
            'lazım', 'kayıp', 'çalındı', 'saldırı', 'yaralanma'
        ]
        
        if any(indicator in message_lower for indicator in critical_indicators):
            return 'critical'
        elif any(indicator in message_lower for indicator in high_indicators):
            return 'high'
        elif '?' in message:
            return 'low'  # Just asking for information
        else:
            return 'medium'
    
    def _classify_emergency_type(self, message: str) -> str:
        """
        Classify the type of emergency.
        
        Args:
            message: User's query
            
        Returns:
            Emergency type: 'medical', 'police', 'embassy', 'safety_info', 'general'
        """
        message_lower = message.lower()
        
        # Medical keywords
        medical = ['hospital', 'doctor', 'pharmacy', 'sick', 'ambulance', 'medical',
                   'hastane', 'doktor', 'eczane', 'hasta', 'ambulans']
        
        # Police keywords
        police = ['police', 'stolen', 'theft', 'crime', 'attack', 'robbery',
                  'polis', 'çalındı', 'hırsızlık', 'suç', 'saldırı']
        
        # Embassy keywords
        embassy = ['embassy', 'consulate', 'passport', 'visa', 'consul',
                   'büyükelçilik', 'konsolosluk', 'pasaport', 'vize']
        
        # Safety info keywords
        safety = ['safe', 'danger', 'scam', 'avoid', 'tip', 'advice',
                  'güvenli', 'tehlike', 'dolandırıcılık', 'öneri']
        
        if any(kw in message_lower for kw in medical):
            return 'medical'
        elif any(kw in message_lower for kw in police):
            return 'police'
        elif any(kw in message_lower for kw in embassy):
            return 'embassy'
        elif any(kw in message_lower for kw in safety):
            return 'safety_info'
        else:
            return 'general'
    
    def _get_user_location(self, context) -> Optional[Dict[str, Any]]:
        """Extract user location from context if available"""
        if not context or not self.has_gps:
            return None
        
        # Try to get GPS coordinates from context
        if hasattr(context, 'location'):
            return context.location
        
        return None
    
    def _handle_medical_emergency(
        self,
        message: str,
        urgency: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Handle medical emergency queries"""
        
        if self.has_llm:
            prompt = self._create_medical_emergency_prompt(
                message, urgency, user_location
            )
            try:
                response = self.llm_service.generate(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return self._get_medical_fallback(urgency)
        else:
            return self._get_medical_fallback(urgency)
    
    def _create_medical_emergency_prompt(
        self,
        message: str,
        urgency: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for medical emergency"""
        
        prompt = f"""You are KAM, Istanbul's emergency assistance AI. Always respond in the same language as the user's query.

User Question: {message}
Urgency Level: {urgency}
"""
        
        if user_location:
            prompt += f"User Location: {user_location.get('district', 'Istanbul')}\n"
        
        prompt += """
Provide CLEAR, ACTIONABLE medical emergency information:

1. Emergency Number: 112 (ambulance)
2. Nearest Hospitals (English-speaking):
   - American Hospital (Nişantaşı): +90 212 444 3777
   - Acıbadem Taksim Hospital: +90 212 252 4400
   - Memorial Şişli Hospital: +90 212 314 6666

3. 24/7 Pharmacies (ask hotel concierge or look for "Nöbetçi Eczane" signs)

4. If CRITICAL: Call 112 immediately
5. One practical tip for their situation

Keep response concise and calm. Include addresses if urgency is high.

Response:"""
        
        return prompt
    
    def _handle_police_safety(
        self,
        message: str,
        urgency: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Handle police and safety queries"""
        
        if self.has_llm:
            prompt = self._create_police_safety_prompt(
                message, urgency, user_location
            )
            try:
                response = self.llm_service.generate(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return self._get_police_fallback(urgency)
        else:
            return self._get_police_fallback(urgency)
    
    def _create_police_safety_prompt(
        self,
        message: str,
        urgency: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for police/safety query"""
        
        prompt = f"""You are KAM, Istanbul's emergency assistance AI. Always respond in the same language as the user's query.

User Question: {message}
Urgency Level: {urgency}
"""
        
        if user_location:
            prompt += f"User Location: {user_location.get('district', 'Istanbul')}\n"
        
        prompt += """
Provide CLEAR, ACTIONABLE police/safety information:

1. Emergency Numbers:
   - Police: 155
   - Tourist Police: +90 212 527 4503 (Sultanahmet)
   - Tourist Helpline: 153

2. Lost/Stolen Passport:
   - Report to police immediately (get report for insurance)
   - Contact your embassy/consulate
   - Embassy will issue emergency travel document

3. Common scams to avoid:
   - Overpriced taxis (use app-based: BiTaksi, Uber)
   - Restaurant bill scams (check prices first)
   - Fake police officers (real police don't check money)

4. Safety tips:
   - Taksim, Sultanahmet generally safe but watch for pickpockets
   - Avoid dark alleys at night
   - Keep valuables secure

Keep response calm and practical.

Response:"""
        
        return prompt
    
    def _handle_embassy_consulate(
        self,
        message: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Handle embassy/consulate queries"""
        
        if self.has_llm:
            prompt = self._create_embassy_prompt(message, user_location)
            try:
                response = self.llm_service.generate(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return self._get_embassy_fallback()
        else:
            return self._get_embassy_fallback()
    
    def _create_embassy_prompt(
        self,
        message: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for embassy query"""
        
        prompt = f"""You are KAM, Istanbul's assistant AI. Always respond in the same language as the user's query.

User Question: {message}

Provide embassy/consulate information for major countries:

Major Embassies in Istanbul:
- US Consulate: Kaplicalar Mevkii No.2, Istinye (+90 212 335 9000)
- UK Consulate: Meşrutiyet Cad. No.34, Beyoğlu (+90 212 334 6400)
- German Consulate: İnönü Cad. No.10, Gümüşsuyu (+90 212 334 6100)

For other countries: Search "Your Country embassy Istanbul"

If passport lost:
1. Report to police (get police report)
2. Contact your consulate immediately
3. Bring: police report, photos, ID, birth certificate if available
4. They'll issue emergency travel document

Keep response helpful and include phone numbers.

Response:"""
        
        return prompt
    
    def _handle_safety_information(self, message: str) -> str:
        """Handle general safety information queries"""
        
        if self.has_llm:
            prompt = f"""You are KAM, Istanbul's safety advisor. Always respond in the same language as the user's query.

User Question: {message}

Provide practical Istanbul safety tips:

General Safety:
- Istanbul is generally safe for tourists
- Taksim, Sultanahmet, Beyoğlu are safe day and night
- Watch for pickpockets in crowded areas
- Use official yellow taxis or app-based (BiTaksi, Uber)

Common Scams:
- Fake police asking to check money (real police don't do this)
- Overpriced restaurants (check menu prices first)
- Taxi meter tricks (insist on meter: "Taksimetre açar mısınız?")
- Shoe-shine scam (don't feel obligated to pay)

Emergency Numbers:
- Police: 155
- Ambulance: 112
- Tourist Police: +90 212 527 4503
- Tourist Helpline: 153

Keep advice practical and reassuring. Be concise (2-3 sentences).

Response:"""
            
            try:
                response = self.llm_service.generate(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return self._get_safety_info_fallback()
        else:
            return self._get_safety_info_fallback()
    
    def _handle_general_emergency(
        self,
        message: str,
        user_location: Optional[Dict[str, Any]]
    ) -> str:
        """Handle general emergency queries"""
        
        if self.has_llm:
            prompt = f"""You are KAM, Istanbul's emergency assistant. Always respond in the same language as the user's query.

User Question: {message}

Provide appropriate emergency information based on their query.

Key Emergency Contacts:
- Ambulance: 112
- Police: 155
- Fire: 110
- Tourist Police: +90 212 527 4503
- Tourist Helpline: 153 (24/7, multilingual)

Keep response clear and actionable.

Response:"""
            
            try:
                response = self.llm_service.generate(prompt)
                return response
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return self._get_general_emergency_fallback()
        else:
            return self._get_general_emergency_fallback()
    
    # ===== FALLBACK RESPONSES =====
    
    def _get_fallback_response(self, message: str) -> str:
        """Generic fallback response"""
        return """🚨 EMERGENCY CONTACTS:
- Ambulance: 112
- Police: 155
- Tourist Police: +90 212 527 4503
- Tourist Helpline: 153 (24/7)

Stay calm. Help is available."""
    
    def _get_medical_fallback(self, urgency: str) -> str:
        """Medical emergency fallback"""
        if urgency == 'critical':
            return """🚨 CALL 112 IMMEDIATELY (Ambulance)

Nearest Hospitals:
- American Hospital: +90 212 444 3777
- Acıbadem Taksim: +90 212 252 4400

Stay calm. Help is on the way."""
        else:
            return """🏥 MEDICAL ASSISTANCE:

Emergency: 112 (ambulance)

English-speaking hospitals:
- American Hospital (Nişantaşı)
- Acıbadem Taksim Hospital
- Memorial Şişli Hospital

24/7 Pharmacies: Look for "Nöbetçi Eczane" signs"""
    
    def _get_police_fallback(self, urgency: str) -> str:
        """Police/safety fallback"""
        return """👮 POLICE ASSISTANCE:

Emergency: 155
Tourist Police: +90 212 527 4503
Tourist Helpline: 153 (24/7)

Lost passport? Report to police first, then contact your embassy."""
    
    def _get_embassy_fallback(self) -> str:
        """Embassy fallback"""
        return """🏛️ EMBASSY INFORMATION:

Major Consulates in Istanbul:
- US: +90 212 335 9000
- UK: +90 212 334 6400
- Germany: +90 212 334 6100

For others: Search "Your Country embassy Istanbul"

Lost passport: Report to police, contact consulate."""
    
    def _get_safety_info_fallback(self) -> str:
        """Safety info fallback"""
        return """🛡️ ISTANBUL SAFETY TIPS:

✅ Generally safe for tourists
✅ Main areas (Taksim, Sultanahmet) safe day/night
⚠️ Watch for pickpockets in crowds
⚠️ Use official taxis or BiTaksi/Uber

Emergency: 155 (police), 112 (ambulance), 153 (tourist helpline)"""
    
    def _get_general_emergency_fallback(self) -> str:
        """General emergency fallback"""
        return """🚨 EMERGENCY CONTACTS:

Ambulance: 112
Police: 155
Tourist Police: +90 212 527 4503
Tourist Helpline: 153 (24/7, multilingual)

Stay calm. Describe your situation clearly."""


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example: Initialize handler
    handler = EmergencySafetyHandler()
    
    # Test queries
    test_queries = [
        "I need a hospital",
        "Acil hastane nerede?",
        "Lost my passport",
        "Where's the US embassy?",
        "Is Taksim safe at night?",
    ]
    
    print("Emergency & Safety Handler - Test Queries:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        can_handle = handler.can_handle(query, {})
        print(f"Can Handle: {can_handle}")
        if can_handle:
            urgency = handler._detect_urgency(query)
            emergency_type = handler._classify_emergency_type(query)
            print(f"Urgency: {urgency}, Type: {emergency_type}")
