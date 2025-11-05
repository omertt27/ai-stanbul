"""
Emergency & Safety Handler for Istanbul AI
Handles: Emergency services, hospitals, police, embassies, safety info, tourist assistance

This handler provides critical safety and emergency information for tourists in Istanbul,
with LLM-enhanced natural language responses and GPS-aware recommendations.

🚨 EMERGENCY: Quick access to emergency numbers, nearest hospitals, police stations
🏥 HEALTHCARE: Hospital info, pharmacies, medical tourism support
🏛️ EMBASSIES: Embassy/consulate locations and contact info for all countries
🌐 MULTILINGUAL: LLM automatically responds in user's language (50+ languages)

Created: November 5, 2025
"""

from typing import Dict, Optional, List, Any, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmergencyHandler:
    """
    Emergency & Safety Information Handler
    
    Capabilities:
    - Emergency numbers (police, ambulance, fire, tourist police)
    - Nearest hospitals and clinics (GPS-aware)
    - Embassy and consulate information
    - Tourist safety tips and advice
    - Lost passport/documents assistance
    - Pharmacy locations (24/7 pharmacies)
    - LLM-enhanced natural responses
    - Multilingual support (automatic)
    """
    
    # 🚨 EMERGENCY NUMBERS (ALWAYS IN ENGLISH + LOCAL)
    EMERGENCY_NUMBERS = {
        'police': {'number': '155', 'international': '+90 155', 'name': 'Police (Polis)'},
        'ambulance': {'number': '112', 'international': '+90 112', 'name': 'Ambulance (Ambulans)'},
        'fire': {'number': '110', 'international': '+90 110', 'name': 'Fire Brigade (İtfaiye)'},
        'tourist_police': {'number': '527 45 03', 'international': '+90 212 527 45 03', 'name': 'Tourist Police (Turizm Polisi)'},
        'coast_guard': {'number': '158', 'international': '+90 158', 'name': 'Coast Guard (Sahil Güvenlik)'},
        'forest_fire': {'number': '177', 'international': '+90 177', 'name': 'Forest Fire (Orman Yangını)'},
        'emergency_general': {'number': '112', 'international': '+90 112', 'name': 'Emergency Services (Acil Servisler)'}
    }
    
    # 🏥 MAJOR HOSPITALS IN ISTANBUL (GPS coordinates included)
    MAJOR_HOSPITALS = [
        {
            'name': 'American Hospital (Amerikan Hastanesi)',
            'district': 'Nişantaşı',
            'type': 'Private',
            'coordinates': {'lat': 41.0533, 'lng': 28.9936},
            'phone': '+90 212 444 37 77',
            'emergency_phone': '+90 212 444 37 77',
            'languages': ['English', 'Turkish', 'Arabic', 'German'],
            'specialties': ['Emergency', 'International Patient Services', 'All specialties'],
            '24_7': True,
            'insurance_accepted': True
        },
        {
            'name': 'Acıbadem Maslak Hospital',
            'district': 'Maslak',
            'type': 'Private',
            'coordinates': {'lat': 41.1072, 'lng': 29.0250},
            'phone': '+90 212 304 44 44',
            'emergency_phone': '+90 212 304 44 44',
            'languages': ['English', 'Turkish', 'Arabic', 'Russian'],
            'specialties': ['Emergency', 'Cardiology', 'Oncology', 'Orthopedics'],
            '24_7': True,
            'insurance_accepted': True
        },
        {
            'name': 'Memorial Şişli Hospital',
            'district': 'Şişli',
            'type': 'Private',
            'coordinates': {'lat': 41.0608, 'lng': 28.9864},
            'phone': '+90 212 314 66 66',
            'emergency_phone': '+90 212 314 66 66',
            'languages': ['English', 'Turkish', 'Arabic', 'German', 'French'],
            'specialties': ['Emergency', 'International Patient Services', 'All specialties'],
            '24_7': True,
            'insurance_accepted': True
        },
        {
            'name': 'İstanbul University Cerrahpaşa Hospital',
            'district': 'Fatih',
            'type': 'Public',
            'coordinates': {'lat': 41.0045, 'lng': 28.9425},
            'phone': '+90 212 414 30 00',
            'emergency_phone': '+90 212 414 30 00',
            'languages': ['Turkish', 'English'],
            'specialties': ['Emergency', 'All specialties', 'Teaching Hospital'],
            '24_7': True,
            'insurance_accepted': True
        },
        {
            'name': 'Koç University Hospital',
            'district': 'Topkapı',
            'type': 'Private',
            'coordinates': {'lat': 41.0128, 'lng': 28.9339},
            'phone': '+90 444 1 923',
            'emergency_phone': '+90 444 1 923',
            'languages': ['English', 'Turkish', 'Arabic'],
            'specialties': ['Emergency', 'Advanced Medical Care', 'Research Hospital'],
            '24_7': True,
            'insurance_accepted': True
        }
    ]
    
    # 🏛️ EMBASSIES AND CONSULATES (Major countries)
    EMBASSIES = {
        'usa': {
            'name': 'United States Consulate General',
            'district': 'İstinye',
            'address': 'Kaplicalar Mevkii No.2, İstinye Mahallesi, Sarıyer',
            'phone': '+90 212 335 90 00',
            'emergency_phone': '+90 212 335 90 00',
            'coordinates': {'lat': 41.1089, 'lng': 29.0547},
            'services': ['Passports', 'Emergency Assistance', 'Notarial Services']
        },
        'uk': {
            'name': 'British Consulate General',
            'district': 'Beyoğlu',
            'address': 'Meşrutiyet Caddesi No.34, Tepebaşı',
            'phone': '+90 212 334 64 00',
            'emergency_phone': '+90 212 334 64 00',
            'coordinates': {'lat': 41.0339, 'lng': 28.9797},
            'services': ['Passports', 'Emergency Assistance', 'Notarial Services']
        },
        'germany': {
            'name': 'German Consulate General',
            'district': 'Beyoğlu',
            'address': 'İnönü Caddesi No.10, Gümüşsuyu',
            'phone': '+90 212 334 61 00',
            'emergency_phone': '+90 212 334 61 00',
            'coordinates': {'lat': 41.0384, 'lng': 28.9873},
            'services': ['Passports', 'Emergency Assistance', 'Visa Services']
        },
        'france': {
            'name': 'French Consulate General',
            'district': 'Beyoğlu',
            'address': 'İstiklal Caddesi No.8, Taksim',
            'phone': '+90 212 334 87 30',
            'emergency_phone': '+90 212 334 87 30',
            'coordinates': {'lat': 41.0358, 'lng': 28.9784},
            'services': ['Passports', 'Emergency Assistance', 'Visa Services']
        },
        'russia': {
            'name': 'Russian Consulate General',
            'district': 'Beyoğlu',
            'address': 'İstiklal Caddesi No.443, Tünel',
            'phone': '+90 212 292 51 01',
            'emergency_phone': '+90 212 292 51 01',
            'coordinates': {'lat': 41.0325, 'lng': 28.9747},
            'services': ['Passports', 'Emergency Assistance', 'Visa Services']
        },
        'china': {
            'name': 'Chinese Consulate General',
            'district': 'Beşiktaş',
            'address': 'Vişnezade Mahallesi, İnönü Caddesi',
            'phone': '+90 212 299 26 88',
            'emergency_phone': '+90 212 299 26 88',
            'coordinates': {'lat': 41.0471, 'lng': 29.0042},
            'services': ['Passports', 'Emergency Assistance', 'Visa Services']
        },
        'japan': {
            'name': 'Japanese Consulate General',
            'district': 'Teşvikiye',
            'address': 'Tekfen Tower, Büyükdere Caddesi No.209',
            'phone': '+90 212 317 46 00',
            'emergency_phone': '+90 212 317 46 00',
            'coordinates': {'lat': 41.0788, 'lng': 29.0086},
            'services': ['Passports', 'Emergency Assistance', 'Notarial Services']
        },
        'canada': {
            'name': 'Canadian Consulate General',
            'district': 'Beyoğlu',
            'address': 'Esentepe Mahallesi, Büyükdere Caddesi No.209',
            'phone': '+90 212 385 97 00',
            'emergency_phone': '+90 212 385 97 00',
            'coordinates': {'lat': 41.0788, 'lng': 29.0086},
            'services': ['Passports', 'Emergency Assistance', 'Consular Services']
        },
        'australia': {
            'name': 'Australian Consulate General',
            'district': 'Elmadağ',
            'address': 'Askerocağı Caddesi No.15, Elmadağ',
            'phone': '+90 212 243 13 33',
            'emergency_phone': '+90 212 243 13 33',
            'coordinates': {'lat': 41.0442, 'lng': 28.9891},
            'services': ['Passports', 'Emergency Assistance', 'Consular Services']
        }
    }
    
    def __init__(
        self,
        llm_service=None,
        gps_location_service=None,
        rag_service=None
    ):
        """
        Initialize emergency handler with optional services
        
        Args:
            llm_service: LLM service for natural responses
            gps_location_service: GPS service for location-based recommendations
            rag_service: RAG service for knowledge retrieval
        """
        self.llm_service = llm_service
        self.gps_location_service = gps_location_service
        self.rag_service = rag_service
        
        self.has_llm = llm_service is not None
        self.has_gps = gps_location_service is not None
        self.has_rag = rag_service is not None and getattr(rag_service, 'available', False)
        
        logger.info(
            f"🚨 Emergency Handler initialized - "
            f"LLM: {self.has_llm} (auto-multilingual), "
            f"GPS: {self.has_gps}, "
            f"RAG: {self.has_rag}"
        )
    
    def can_handle(self, message: str, entities: Dict[str, Any]) -> bool:
        """
        Determine if this handler can process the emergency/safety query
        
        Args:
            message: User's query
            entities: Extracted entities
            
        Returns:
            True if this is an emergency/safety query
        """
        message_lower = message.lower()
        
        # Emergency keywords (multilingual)
        emergency_keywords = [
            # English
            'emergency', 'hospital', 'doctor', 'ambulance', 'police', 'help',
            'embassy', 'consulate', 'lost passport', 'pharmacy', 'clinic',
            'fire', 'accident', 'injured', 'sick', 'medical', 'health',
            'safety', 'danger', 'urgent', 'crisis', 'tourist police',
            
            # Turkish
            'acil', 'hastane', 'doktor', 'ambulans', 'polis', 'yardım',
            'konsolosluk', 'eczane', 'klinik', 'yangın', 'kaza',
            'hasta', 'sağlık', 'güvenlik', 'tehlike',
            
            # Arabic
            'طوارئ', 'مستشفى', 'طبيب', 'إسعاف', 'شرطة',
            
            # Russian
            'скорая', 'больница', 'полиция',
            
            # Common phrases
            'need help', 'need doctor', 'where is hospital', 'call ambulance',
            'lost my passport', 'where is embassy'
        ]
        
        return any(keyword in message_lower for keyword in emergency_keywords)
    
    def handle(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile=None,
        context=None,
        neural_insights: Optional[Dict] = None,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Main entry point for emergency/safety queries
        
        Args:
            message: User's query (in any language - LLM will auto-detect)
            entities: Extracted entities
            user_profile: User profile (contains GPS location if available)
            context: Conversation context
            neural_insights: ML insights
            return_structured: Whether to return structured response
            
        Returns:
            Emergency/safety response
        """
        try:
            logger.info(f"🚨 Emergency/safety query received (LLM auto-multilingual)")
            
            # Classify emergency type
            query_type = self._classify_emergency_query(message)
            
            # Route to appropriate handler
            if query_type == 'immediate_emergency':
                return self._handle_immediate_emergency(
                    message, user_profile, return_structured
                )
            elif query_type == 'hospital':
                return self._handle_hospital_query(
                    message, entities, user_profile, return_structured
                )
            elif query_type == 'embassy':
                return self._handle_embassy_query(
                    message, entities, user_profile, return_structured
                )
            elif query_type == 'pharmacy':
                return self._handle_pharmacy_query(
                    message, user_profile, return_structured
                )
            else:
                return self._handle_general_safety(
                    message, user_profile, return_structured
                )
                
        except Exception as e:
            logger.error(f"Emergency handler error: {e}", exc_info=True)
            return self._get_emergency_fallback(return_structured)
    
    def _classify_emergency_query(self, message: str) -> str:
        """
        Classify the type of emergency query
        
        Args:
            message: User's query
            
        Returns:
            Query type: 'immediate_emergency', 'hospital', 'embassy', 'pharmacy', 'general'
        """
        message_lower = message.lower()
        
        # Immediate emergency indicators
        immediate_keywords = [
            'call ambulance', 'call police', 'emergency now', 'urgent help',
            'ambulans çağır', 'polis çağır', 'acil yardım'
        ]
        
        if any(keyword in message_lower for keyword in immediate_keywords):
            return 'immediate_emergency'
        
        # Hospital indicators
        if any(word in message_lower for word in ['hospital', 'doctor', 'clinic', 'medical', 'hastane', 'doktor']):
            return 'hospital'
        
        # Embassy indicators
        if any(word in message_lower for word in ['embassy', 'consulate', 'passport', 'konsolosluk']):
            return 'embassy'
        
        # Pharmacy indicators
        if any(word in message_lower for word in ['pharmacy', 'medicine', 'drug', 'eczane', 'ilaç']):
            return 'pharmacy'
        
        return 'general'
    
    def _handle_immediate_emergency(
        self,
        message: str,
        user_profile,
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle immediate emergency situations (PRIORITY)
        
        Args:
            message: User's query
            user_profile: User profile
            return_structured: Whether to return structured response
            
        Returns:
            Immediate emergency response with key numbers
        """
        # Build GPS context
        gps_context = self._build_gps_context(user_profile)
        
        # Create emergency response with LLM
        if self.has_llm:
            prompt = f"""You are KAM, an emergency assistant for Istanbul tourists.

User's Emergency: {message}

CRITICAL INFORMATION:
🚨 Emergency Numbers:
- General Emergency: 112
- Police: 155
- Ambulance: 112
- Fire Brigade: 110
- Tourist Police: +90 212 527 45 03

User Location: {gps_context.get('district', 'Unknown')}

Provide:
1. Which number to call IMMEDIATELY
2. Brief instruction (stay calm, describe location)
3. Nearest hospital if medical emergency

Keep it VERY brief (2-3 sentences), clear, and calming. Use emergency emojis (🚨🚑🏥).
Always respond in the same language as the user's query.

Response:"""
            
            try:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.3  # Low temperature for accuracy
                )
                
                if llm_response:
                    response_text = llm_response
                else:
                    response_text = self._get_emergency_template_response(message)
            except Exception as e:
                logger.error(f"LLM emergency response failed: {e}")
                response_text = self._get_emergency_template_response(message)
        else:
            response_text = self._get_emergency_template_response(message)
        
        if return_structured:
            return {
                'response': response_text,
                'emergency_numbers': self.EMERGENCY_NUMBERS,
                'gps_context': gps_context,
                'priority': 'CRITICAL',
                'handler': 'emergency_handler',
                'success': True
            }
        else:
            return response_text
    
    def _handle_hospital_query(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile,
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle hospital and medical queries
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile (GPS location)
            return_structured: Whether to return structured response
            
        Returns:
            Hospital information with GPS-aware recommendations
        """
        # Build GPS context
        gps_context = self._build_gps_context(user_profile)
        
        # Find nearest hospitals if GPS available
        nearest_hospitals = []
        if gps_context.get('has_gps'):
            nearest_hospitals = self._find_nearest_hospitals(
                gps_context['coordinates'],
                max_results=3
            )
        
        # Get LLM-enhanced response
        if self.has_llm:
            prompt = self._create_hospital_prompt(message, nearest_hospitals, gps_context)
            
            try:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                
                response_text = llm_response if llm_response else self._get_hospital_template_response(nearest_hospitals)
            except Exception as e:
                logger.error(f"LLM hospital response failed: {e}")
                response_text = self._get_hospital_template_response(nearest_hospitals)
        else:
            response_text = self._get_hospital_template_response(nearest_hospitals)
        
        if return_structured:
            return {
                'response': response_text,
                'nearest_hospitals': nearest_hospitals,
                'all_hospitals': self.MAJOR_HOSPITALS,
                'emergency_numbers': self.EMERGENCY_NUMBERS,
                'gps_context': gps_context,
                'handler': 'emergency_handler',
                'success': True
            }
        else:
            return response_text
    
    def _handle_embassy_query(
        self,
        message: str,
        entities: Dict[str, Any],
        user_profile,
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle embassy/consulate queries
        
        Args:
            message: User's query
            entities: Extracted entities
            user_profile: User profile
            return_structured: Whether to return structured response
            
        Returns:
            Embassy information response
        """
        # Detect country from message
        country_code = self._detect_country_from_query(message)
        
        # Get GPS context
        gps_context = self._build_gps_context(user_profile)
        
        # Get relevant embassy
        embassy_info = None
        if country_code:
            embassy_info = self.EMBASSIES.get(country_code)
        
        # Create LLM response
        if self.has_llm:
            prompt = self._create_embassy_prompt(message, embassy_info, country_code, gps_context)
            
            try:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                
                response_text = llm_response if llm_response else self._get_embassy_template_response(embassy_info, country_code)
            except Exception as e:
                logger.error(f"LLM embassy response failed: {e}")
                response_text = self._get_embassy_template_response(embassy_info, country_code)
        else:
            response_text = self._get_embassy_template_response(embassy_info, country_code)
        
        if return_structured:
            return {
                'response': response_text,
                'embassy_info': embassy_info,
                'all_embassies': self.EMBASSIES,
                'gps_context': gps_context,
                'handler': 'emergency_handler',
                'success': True
            }
        else:
            return response_text
    
    def _handle_pharmacy_query(
        self,
        message: str,
        user_profile,
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle pharmacy queries (including 24/7 pharmacies)
        
        Args:
            message: User's query
            user_profile: User profile (GPS location)
            return_structured: Whether to return structured response
            
        Returns:
            Pharmacy information response
        """
        gps_context = self._build_gps_context(user_profile)
        
        # Create LLM response with pharmacy info
        if self.has_llm:
            prompt = f"""You are KAM, helping a tourist find a pharmacy in Istanbul.

User Query: {message}
User Location: {gps_context.get('district', 'Istanbul')}

PHARMACY INFO:
- Call 112 to find nearest 24/7 pharmacy (Nöbetçi Eczane)
- Most pharmacies: 9 AM - 7 PM
- 24/7 pharmacies rotate weekly (Nöbetçi system)
- Look for green cross ✚ signs
- Pharmacists often speak English in tourist areas

Provide:
1. How to find nearest pharmacy
2. 24/7 pharmacy info
3. One practical tip (prescription requirements, payment methods)

Keep it concise (3-4 sentences). Always respond in the same language as the user's query.

Response:"""
            
            try:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                
                response_text = llm_response if llm_response else "Call 112 to find the nearest 24/7 pharmacy (Nöbetçi Eczane). Most pharmacies are open 9 AM-7 PM. Look for the green cross ✚ sign."
            except Exception as e:
                logger.error(f"LLM pharmacy response failed: {e}")
                response_text = "Call 112 to find the nearest 24/7 pharmacy (Nöbetçi Eczane). Most pharmacies are open 9 AM-7 PM. Look for the green cross ✚ sign."
        else:
            response_text = "Call 112 to find the nearest 24/7 pharmacy (Nöbetçi Eczane). Most pharmacies are open 9 AM-7 PM. Look for the green cross ✚ sign."
        
        if return_structured:
            return {
                'response': response_text,
                'emergency_phone': '112',
                'pharmacy_hours': '9 AM - 7 PM (most pharmacies)',
                'system': 'Nöbetçi Eczane (24/7 rotation)',
                'gps_context': gps_context,
                'handler': 'emergency_handler',
                'success': True
            }
        else:
            return response_text
    
    def _handle_general_safety(
        self,
        message: str,
        user_profile,
        return_structured: bool
    ) -> Union[str, Dict[str, Any]]:
        """
        Handle general safety and tourist assistance queries
        
        Args:
            message: User's query
            user_profile: User profile
            return_structured: Whether to return structured response
            
        Returns:
            Safety information response
        """
        gps_context = self._build_gps_context(user_profile)
        
        # Get RAG context if available
        rag_context = ""
        if self.has_rag:
            try:
                rag_context = self.rag_service.get_context_for_llm(
                    query=f"tourist safety istanbul {message}",
                    top_k=3,
                    max_length=500
                )
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Create LLM response
        if self.has_llm:
            prompt = f"""You are KAM, a safety advisor for Istanbul tourists.

User Query: {message}
User Location: {gps_context.get('district', 'Istanbul')}

{rag_context if rag_context else ''}

KEY SAFETY INFO:
- Tourist Police: +90 212 527 45 03 (Sultanahmet)
- General Emergency: 112
- Istanbul is generally safe for tourists
- Common scams: Overpriced restaurants, fake guides
- Safe areas: Sultanahmet, Taksim, Kadıköy, Beşiktaş

Provide helpful, reassuring safety advice (3-4 sentences).
Always respond in the same language as the user's query.

Response:"""
            
            try:
                llm_response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                
                response_text = llm_response if llm_response else "Istanbul is generally safe for tourists. Contact Tourist Police at +90 212 527 45 03 if you need help. Stay in well-known areas like Sultanahmet, Taksim, and Kadıköy."
            except Exception as e:
                logger.error(f"LLM safety response failed: {e}")
                response_text = "Istanbul is generally safe for tourists. Contact Tourist Police at +90 212 527 45 03 if you need help. Stay in well-known areas like Sultanahmet, Taksim, and Kadıköy."
        else:
            response_text = "Istanbul is generally safe for tourists. Contact Tourist Police at +90 212 527 45 03 if you need help. Stay in well-known areas like Sultanahmet, Taksim, and Kadıköy."
        
        if return_structured:
            return {
                'response': response_text,
                'emergency_numbers': self.EMERGENCY_NUMBERS,
                'gps_context': gps_context,
                'handler': 'emergency_handler',
                'success': True
            }
        else:
            return response_text
    
    # ===== HELPER METHODS =====
    
    def _build_gps_context(self, user_profile) -> Dict[str, Any]:
        """Build GPS context from user profile"""
        gps_context = {
            'has_gps': False,
            'coordinates': None,
            'district': None
        }
        
        if not user_profile or not hasattr(user_profile, 'current_location'):
            return gps_context
        
        gps_location = user_profile.current_location
        if gps_location and isinstance(gps_location, tuple) and len(gps_location) == 2:
            gps_context['has_gps'] = True
            gps_context['coordinates'] = {'lat': gps_location[0], 'lng': gps_location[1]}
            
            # Get district if GPS service available
            if self.has_gps:
                try:
                    district = self.gps_location_service.get_district_from_coords(
                        gps_location[0], gps_location[1]
                    )
                    gps_context['district'] = district
                except Exception as e:
                    logger.warning(f"Could not get district: {e}")
        
        return gps_context
    
    def _find_nearest_hospitals(
        self,
        user_coords: Dict[str, float],
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find nearest hospitals to user's GPS location
        
        Args:
            user_coords: {'lat': float, 'lng': float}
            max_results: Maximum number of results
            
        Returns:
            List of nearest hospitals with distance
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points in km"""
            R = 6371  # Earth's radius in km
            
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return R * c
        
        # Calculate distances
        hospitals_with_distance = []
        for hospital in self.MAJOR_HOSPITALS:
            distance = haversine_distance(
                user_coords['lat'],
                user_coords['lng'],
                hospital['coordinates']['lat'],
                hospital['coordinates']['lng']
            )
            
            hospital_copy = hospital.copy()
            hospital_copy['distance_km'] = round(distance, 2)
            hospitals_with_distance.append(hospital_copy)
        
        # Sort by distance
        hospitals_with_distance.sort(key=lambda x: x['distance_km'])
        
        return hospitals_with_distance[:max_results]
    
    def _detect_country_from_query(self, message: str) -> Optional[str]:
        """Detect country code from user's message"""
        message_lower = message.lower()
        
        country_keywords = {
            'usa': ['usa', 'united states', 'american', 'us embassy'],
            'uk': ['uk', 'british', 'england', 'britain'],
            'germany': ['germany', 'german', 'deutsch'],
            'france': ['france', 'french', 'français'],
            'russia': ['russia', 'russian', 'российский'],
            'china': ['china', 'chinese', '中国'],
            'japan': ['japan', 'japanese', '日本'],
            'canada': ['canada', 'canadian'],
            'australia': ['australia', 'australian']
        }
        
        for country_code, keywords in country_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return country_code
        
        return None
    
    def _create_hospital_prompt(
        self,
        message: str,
        nearest_hospitals: List[Dict],
        gps_context: Dict
    ) -> str:
        """Create LLM prompt for hospital query"""
        hospital_list = "\n".join([
            f"- {h['name']} ({h['district']}) - {h['distance_km']}km away\n"
            f"  Phone: {h['emergency_phone']}, Languages: {', '.join(h['languages'][:3])}"
            for h in nearest_hospitals[:3]
        ]) if nearest_hospitals else "No GPS location - showing major hospitals"
        
        prompt = f"""You are KAM, helping a tourist find medical care in Istanbul.

User Query: {message}
User Location: {gps_context.get('district', 'Unknown')}

NEAREST HOSPITALS:
{hospital_list}

EMERGENCY: Call 112 for ambulance

Provide:
1. Which hospital is best for them (consider distance, languages spoken)
2. Emergency number reminder
3. One reassurance/tip (insurance accepted, English-speaking staff)

Keep it concise (3-4 sentences), helpful, and reassuring. Use medical emojis (🏥🚑).
Always respond in the same language as the user's query.

Response:"""
        
        return prompt
    
    def _create_embassy_prompt(
        self,
        message: str,
        embassy_info: Optional[Dict],
        country_code: Optional[str],
        gps_context: Dict
    ) -> str:
        """Create LLM prompt for embassy query"""
        if embassy_info:
            embassy_details = f"""
EMBASSY INFO:
- Name: {embassy_info['name']}
- District: {embassy_info['district']}
- Address: {embassy_info['address']}
- Phone: {embassy_info['phone']}
- Services: {', '.join(embassy_info['services'])}
"""
        else:
            embassy_details = "No specific embassy detected. General embassy info will be provided."
        
        prompt = f"""You are KAM, helping a tourist with embassy/consulate assistance.

User Query: {message}
User Location: {gps_context.get('district', 'Istanbul')}
Country: {country_code.upper() if country_code else 'Not specified'}

{embassy_details}

Provide:
1. Embassy location and contact info
2. Services available (lost passport, emergency assistance)
3. One practical tip (business hours, bring ID, etc.)

Keep it concise (3-4 sentences), helpful. Use embassy emojis (🏛️📞).
Always respond in the same language as the user's query.

Response:"""
        
        return prompt
    
    # ===== TEMPLATE FALLBACK RESPONSES =====
    
    def _get_emergency_template_response(self, message: str) -> str:
        """Template response for immediate emergencies"""
        return """🚨 EMERGENCY NUMBERS:
- General Emergency: 112
- Police: 155
- Ambulance: 112
- Tourist Police: +90 212 527 45 03

Stay calm and call the appropriate number. Describe your location clearly."""
    
    def _get_hospital_template_response(self, nearest_hospitals: List[Dict]) -> str:
        """Template response for hospital queries"""
        if nearest_hospitals:
            hospital = nearest_hospitals[0]
            return f"""🏥 Nearest Hospital: {hospital['name']}
📍 {hospital['district']} ({hospital['distance_km']}km away)
📞 {hospital['emergency_phone']}
🌐 Languages: {', '.join(hospital['languages'][:3])}
🚑 Emergency: Call 112 for ambulance"""
        else:
            return "🏥 Call 112 for emergency medical assistance. Major hospitals: American Hospital (Nişantaşı), Memorial Şişli, Acıbadem Maslak."
    
    def _get_embassy_template_response(self, embassy_info: Optional[Dict], country_code: Optional[str]) -> str:
        """Template response for embassy queries"""
        if embassy_info:
            return f"""🏛️ {embassy_info['name']}
📍 {embassy_info['district']}: {embassy_info['address']}
📞 {embassy_info['phone']}
✅ Services: {', '.join(embassy_info['services'])}"""
        else:
            return "🏛️ Most embassies are in Beyoğlu/Beşiktaş area. Specify your country for exact location. Emergency: Contact your embassy's 24/7 helpline."
    
    def _get_emergency_fallback(self, return_structured: bool) -> Union[str, Dict[str, Any]]:
        """Fallback response for errors"""
        response_text = """🚨 EMERGENCY NUMBERS:
- General Emergency: 112
- Police: 155
- Ambulance: 112
- Tourist Police: +90 212 527 45 03

For immediate help, call these numbers."""
        
        if return_structured:
            return {
                'response': response_text,
                'emergency_numbers': self.EMERGENCY_NUMBERS,
                'handler': 'emergency_handler',
                'success': False
            }
        else:
            return response_text
