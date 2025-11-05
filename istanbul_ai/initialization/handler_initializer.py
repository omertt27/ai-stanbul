"""
Handler Initializer Module
Initializes all ML-Enhanced handlers for the Istanbul AI system.

This module extracts ML handler initialization logic from main_system.py
to improve modularity and maintainability.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class HandlerInitializer:
    """
    Initializes ML-Enhanced handlers for the Istanbul AI system.
    
    Handles the initialization of 12 ML handlers:
    1. ML-Enhanced Event Handler
    2. ML-Enhanced Hidden Gems Handler
    3. ML-Enhanced Weather Handler
    4. ML-Enhanced Route Planning Handler
    5. ML-Enhanced Neighborhood Handler
    6. Nearby Locations Handler (GPS-based)
    7. Transportation Handler (IBB API + GPS + Transfer Maps)
    8. Restaurant Handler
    9. Attraction Handler
    10. ML-Enhanced Local Food Handler
    11. ML-Enhanced Emergency & Safety Handler
    
    Each handler requires:
    - ML Context Builder
    - Neural Processor
    - Response Generator
    - Specific service dependencies
    """
    
    def __init__(self):
        """Initialize the handler initializer"""
        self.handlers = {}
        self.initialization_log = []
        self.initialized_count = 0
        self.total_handlers = 12  # Updated: added local_food_handler + emergency_safety_handler
        
    def initialize_all_handlers(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize all ML-Enhanced handlers
        
        Args:
            services: Dictionary of initialized services from ServiceInitializer
                     Required keys: ml_context_builder, neural_processor, 
                                   response_generator, and handler-specific services
        
        Returns:
            Dictionary of initialized handlers
        """
        logger.info("🤖 Starting ML-Enhanced handlers initialization...")
        
        # Extract common dependencies
        ml_context_builder = services.get('ml_context_builder')
        neural_processor = services.get('neural_processor')
        response_generator = services.get('response_generator')
        
        # Check if base dependencies are available
        if not ml_context_builder:
            logger.warning("⚠️ ML Context Builder not available - skipping all ML handlers")
            self._initialize_all_to_none()
            return self.handlers
            
        if not neural_processor:
            logger.warning("⚠️ Neural Processor not available - skipping all ML handlers")
            self._initialize_all_to_none()
            return self.handlers
            
        if not response_generator:
            logger.warning("⚠️ Response Generator not available - skipping all ML handlers")
            self._initialize_all_to_none()
            return self.handlers
        
        # Initialize each handler
        # PRIORITY 1: Emergency & Safety (ALWAYS first)
        self._initialize_emergency_safety_handler(services, ml_context_builder, neural_processor, response_generator)
        # PRIORITY 2: Local Food (Second priority)
        self._initialize_local_food_handler(services, ml_context_builder, neural_processor, response_generator)
        # PRIORITY 3+: Existing handlers
        self._initialize_restaurant_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_attraction_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_event_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_hidden_gems_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_weather_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_route_planning_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_neighborhood_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_nearby_locations_handler(services, ml_context_builder, neural_processor, response_generator)
        self._initialize_transportation_handler(services, ml_context_builder, neural_processor, response_generator)
        
        # Log summary
        success_rate = (self.initialized_count / self.total_handlers * 100) if self.total_handlers > 0 else 0
        logger.info(f"✅ ML handlers initialization complete: {self.initialized_count}/{self.total_handlers} "
                   f"({success_rate:.1f}%) handlers initialized")
        
        return self.handlers
    
    def _initialize_event_handler(self, services: Dict, ml_context_builder: Any, 
                                  neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Event Handler"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.event_handler import create_ml_enhanced_event_handler
            
            events_service = services.get('events_service')
            
            if events_service and neural_processor and response_generator:
                self.handlers['ml_event_handler'] = create_ml_enhanced_event_handler(
                    events_service=events_service,
                    ml_context_builder=ml_context_builder,
                    ml_processor=neural_processor,
                    response_generator=response_generator
                )
                logger.info("🎭 ML-Enhanced Event Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_event_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Event Handler")
                self.handlers['ml_event_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_event_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Event Handler not available: {e}")
            self.handlers['ml_event_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_event_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Event Handler: {e}")
            self.handlers['ml_event_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_event_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_hidden_gems_handler(self, services: Dict, ml_context_builder: Any,
                                       neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Hidden Gems Handler"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.hidden_gems_handler import create_ml_enhanced_hidden_gems_handler
            
            hidden_gems_service = services.get('hidden_gems_handler')
            
            if hidden_gems_service and neural_processor and response_generator:
                self.handlers['ml_hidden_gems_handler'] = create_ml_enhanced_hidden_gems_handler(
                    hidden_gems_service=hidden_gems_service,
                    ml_context_builder=ml_context_builder,
                    ml_processor=neural_processor,
                    response_generator=response_generator
                )
                logger.info("💎 ML-Enhanced Hidden Gems Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_hidden_gems_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Hidden Gems Handler")
                self.handlers['ml_hidden_gems_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_hidden_gems_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Hidden Gems Handler not available: {e}")
            self.handlers['ml_hidden_gems_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_hidden_gems_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Hidden Gems Handler: {e}")
            self.handlers['ml_hidden_gems_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_hidden_gems_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_weather_handler(self, services: Dict, ml_context_builder: Any,
                                   neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Weather Handler"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.weather_handler import create_ml_enhanced_weather_handler
            
            weather_service = services.get('weather_client')
            weather_recommendations_service = services.get('weather_recommendations')
            
            if weather_service and weather_recommendations_service and neural_processor and response_generator:
                self.handlers['ml_weather_handler'] = create_ml_enhanced_weather_handler(
                    weather_service=weather_service,
                    weather_recommendations_service=weather_recommendations_service,
                    ml_context_builder=ml_context_builder,
                    ml_processor=neural_processor,
                    response_generator=response_generator
                )
                logger.info("🌤️ ML-Enhanced Weather Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_weather_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Weather Handler")
                self.handlers['ml_weather_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_weather_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Weather Handler not available: {e}")
            self.handlers['ml_weather_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_weather_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Weather Handler: {e}")
            self.handlers['ml_weather_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_weather_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_route_planning_handler(self, services: Dict, ml_context_builder: Any,
                                          neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Route Planning Handler"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.route_planning_handler import create_ml_enhanced_route_planning_handler
            
            # Get route service (prefer advanced planner)
            route_service = services.get('advanced_route_planner') or services.get('gps_route_planner')
            transport_service = services.get('transport_processor')
            
            if route_service and transport_service and neural_processor and response_generator:
                self.handlers['ml_route_planning_handler'] = create_ml_enhanced_route_planning_handler(
                    route_planner_service=route_service,
                    transport_service=transport_service,
                    ml_context_builder=ml_context_builder,
                    ml_processor=neural_processor,
                    response_generator=response_generator
                )
                logger.info("🗺️ ML-Enhanced Route Planning Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_route_planning_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Route Planning Handler")
                self.handlers['ml_route_planning_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_route_planning_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Route Planning Handler not available: {e}")
            self.handlers['ml_route_planning_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_route_planning_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Route Planning Handler: {e}")
            self.handlers['ml_route_planning_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_route_planning_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_neighborhood_handler(self, services: Dict, ml_context_builder: Any,
                                        neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Neighborhood Handler"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.neighborhood_handler import create_ml_enhanced_neighborhood_handler
            
            # Use response generator as neighborhood service (it has neighborhood data)
            neighborhood_service = response_generator
            
            if neighborhood_service and neural_processor:
                self.handlers['ml_neighborhood_handler'] = create_ml_enhanced_neighborhood_handler(
                    neighborhood_service=neighborhood_service,
                    ml_context_builder=ml_context_builder,
                    ml_processor=neural_processor,
                    response_generator=response_generator
                )
                logger.info("🏘️ ML-Enhanced Neighborhood Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_neighborhood_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Neighborhood Handler")
                self.handlers['ml_neighborhood_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_neighborhood_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Neighborhood Handler not available: {e}")
            self.handlers['ml_neighborhood_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_neighborhood_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Neighborhood Handler: {e}")
            self.handlers['ml_neighborhood_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_neighborhood_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_nearby_locations_handler(self, services: Dict, ml_context_builder: Any,
                                            neural_processor: Any, response_generator: Any):
        """Initialize Nearby Locations Handler (GPS-based)"""
        try:
            # Import the handler creation function
            from istanbul_ai.handlers.nearby_locations_handler import create_nearby_locations_handler
            
            # Get required services
            gps_route_service = services.get('gps_route_service')
            location_database_service = services.get('location_database_service')
            transport_service = services.get('transport_processor')
            user_manager = services.get('user_manager')
            
            # At least one of these services is required
            if gps_route_service or location_database_service:
                # LLM + GPS integration services
                llm_service = services.get('llm_service')
                gps_location_service = services.get('gps_location_service')
                
                self.handlers['nearby_locations_response_handler'] = create_nearby_locations_handler(
                    gps_route_service=gps_route_service,
                    location_database_service=location_database_service,
                    neural_processor=neural_processor,
                    user_manager=user_manager,
                    transport_service=transport_service,
                    llm_service=llm_service,  # 🤖 LLM integration
                    gps_location_service=gps_location_service  # 📍 GPS location integration
                )
                logger.info(
                    f"📍 Nearby Locations Handler initialized successfully! "
                    f"(LLM: {llm_service is not None}, GPSLocation: {gps_location_service is not None})"
                )
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'nearby_locations_response_handler',
                    'status': 'success',
                    'features': {
                        'llm_service': llm_service is not None,
                        'gps_location_service': gps_location_service is not None
                    }
                })
            else:
                logger.warning("Required GPS/location services not available for Nearby Locations Handler")
                self.handlers['nearby_locations_response_handler'] = None
                self.initialization_log.append({
                    'handler': 'nearby_locations_response_handler',
                    'status': 'skipped',
                    'reason': 'missing_gps_services'
                })
        except ImportError as e:
            logger.warning(f"Nearby Locations Handler not available: {e}")
            self.handlers['nearby_locations_response_handler'] = None
            self.initialization_log.append({
                'handler': 'nearby_locations_response_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize Nearby Locations Handler: {e}")
            self.handlers['nearby_locations_response_handler'] = None
            self.initialization_log.append({
                'handler': 'nearby_locations_response_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_transportation_handler(self, services: Dict, ml_context_builder: Any,
                                          neural_processor: Any, response_generator: Any):
        """Initialize Transportation Handler with IBB API, GPS, Transfer Maps, LLM, RAG, Bilingual, and Multilingual support"""
        try:
            from istanbul_ai.handlers.transportation_handler import TransportationHandler
            
            # Get transportation-related services
            transportation_chat = services.get('transportation_chat')
            transport_processor = services.get('transport_processor')
            gps_route_service = services.get('gps_route_service')
            bilingual_manager = services.get('bilingual_manager')  # 🌐 Bilingual support (EN/TR)
            
            # Initialize MultilingualManager for 9-language support
            multilingual_manager = None
            try:
                from ml_systems.multilingual_manager import MultilingualManager
                multilingual_manager = MultilingualManager()
                logger.info("🌐 MultilingualManager initialized (9 languages: EN/TR/AR/RU/DE/FR/ES/ZH/JA)")
            except Exception as e:
                logger.warning(f"🌐 MultilingualManager not available: {e}")
            
            # LLM + GPS + Weather + Hidden Gems + RAG integration services
            llm_service = services.get('llm_service')
            gps_location_service = services.get('gps_location_service')
            weather_service = services.get('weather_service')
            
            # Initialize Hidden Gems Context Service for transportation
            hidden_gems_context_service = None
            try:
                from ml_systems.hidden_gems_context_service import HiddenGemsContextService
                hidden_gems_context_service = HiddenGemsContextService()
                logger.info("💎 Hidden Gems Context Service initialized for transportation")
            except Exception as e:
                logger.warning(f"💎 Hidden Gems Context Service not available: {e}")
            
            # Initialize RAG Vector Service
            rag_service = None
            try:
                from ml_systems.rag_vector_service import get_rag_service
                rag_service = get_rag_service(rebuild=False)
                if rag_service and rag_service.available:
                    logger.info("🤖 RAG Vector Service initialized for transportation")
                else:
                    logger.warning("🤖 RAG Vector Service not available")
                    rag_service = None
            except Exception as e:
                logger.warning(f"🤖 RAG Vector Service initialization failed: {e}")
                rag_service = None
            
            # Get feature flags
            transfer_map_integration_available = services.get('transfer_map_integration_available', False)
            advanced_transport_available = services.get('advanced_transport_available', False)
            
            # Create the handler (even if some services are None, it will use fallbacks)
            self.handlers['transportation_handler'] = TransportationHandler(
                transportation_chat=transportation_chat,
                transport_processor=transport_processor,
                gps_route_service=gps_route_service,
                bilingual_manager=bilingual_manager,  # 🌐 Pass bilingual manager (EN/TR)
                multilingual_manager=multilingual_manager,  # 🌐 Pass multilingual manager (9 languages)
                transfer_map_integration_available=transfer_map_integration_available,
                advanced_transport_available=advanced_transport_available,
                llm_service=llm_service,  # 🤖 LLM integration
                gps_location_service=gps_location_service,  # 📍 GPS location integration
                weather_service=weather_service,  # 🌤️ Weather integration
                hidden_gems_context_service=hidden_gems_context_service,  # 💎 Hidden gems integration
                rag_service=rag_service  # 🤖 RAG integration
            )
            
            logger.info(
                f"🚇 Transportation Handler initialized successfully! "
                f"(Bilingual: {bilingual_manager is not None}, "
                f"Multilingual: {multilingual_manager is not None}, "
                f"LLM: {llm_service is not None}, "
                f"GPSLocation: {gps_location_service is not None}, "
                f"Weather: {weather_service is not None}, "
                f"HiddenGems: {hidden_gems_context_service is not None}, "
                f"RAG: {rag_service is not None})"
            )
            self.initialized_count += 1
            self.initialization_log.append({
                'handler': 'transportation_handler',
                'status': 'success',
                'features': {
                    'transportation_chat': transportation_chat is not None,
                    'transport_processor': transport_processor is not None,
                    'gps_route_service': gps_route_service is not None,
                    'bilingual_manager': bilingual_manager is not None,  # 🌐 Log bilingual status (EN/TR)
                    'multilingual_manager': multilingual_manager is not None,  # 🌐 Log multilingual status (9 languages)
                    'transfer_maps': transfer_map_integration_available,
                    'advanced_transport': advanced_transport_available,
                    'llm_service': llm_service is not None,  # 🤖 Log LLM status
                    'gps_location_service': gps_location_service is not None,  # 📍 Log GPS status
                    'weather_service': weather_service is not None,  # 🌤️ Log weather status
                    'hidden_gems_context_service': hidden_gems_context_service is not None,  # 💎 Log hidden gems status
                    'rag_service': rag_service is not None  # 🤖 Log RAG status
                }
            })
            
        except ImportError as e:
            logger.warning(f"Transportation Handler not available: {e}")
            self.handlers['transportation_handler'] = None
            self.initialization_log.append({
                'handler': 'transportation_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize Transportation Handler: {e}")
            self.handlers['transportation_handler'] = None
            self.initialization_log.append({
                'handler': 'transportation_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_restaurant_handler(self, services: Dict, ml_context_builder: Any,
                                       neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Restaurant Handler with Bilingual support"""
        try:
            from istanbul_ai.handlers.restaurant_handler import RestaurantHandler
            
            # Get required services
            restaurant_service = services.get('restaurant_service') or response_generator  # Use response_generator as fallback
            bilingual_manager = services.get('bilingual_manager')
            
            if neural_processor and response_generator:
                self.handlers['ml_restaurant_handler'] = RestaurantHandler(
                    neural_processor=neural_processor,
                    restaurant_service=restaurant_service,
                    response_generator=response_generator,
                    bilingual_manager=bilingual_manager
                )
                logger.info(f"🍽️ ML-Enhanced Restaurant Handler initialized successfully! (Bilingual: {bilingual_manager is not None})")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_restaurant_handler',
                    'status': 'success',
                    'bilingual': bilingual_manager is not None
                })
            else:
                logger.warning("Required dependencies not available for ML Restaurant Handler")
                self.handlers['ml_restaurant_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_restaurant_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Restaurant Handler not available: {e}")
            self.handlers['ml_restaurant_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_restaurant_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Restaurant Handler: {e}")
            self.handlers['ml_restaurant_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_restaurant_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_attraction_handler(self, services: Dict, ml_context_builder: Any,
                                      neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Attraction Handler with Bilingual support"""
        try:
            from istanbul_ai.handlers.attraction_handler import AttractionHandler
            
            # Get required services
            attraction_service = services.get('attraction_service') or response_generator  # Use response_generator as fallback
            user_manager = services.get('user_manager')
            weather_service = services.get('weather_client')
            advanced_attraction_service = services.get('advanced_attractions_system')
            transport_service = services.get('transport_processor')
            bilingual_manager = services.get('bilingual_manager')
            
            if neural_processor and response_generator:
                self.handlers['ml_attraction_handler'] = AttractionHandler(
                    neural_processor=neural_processor,
                    user_manager=user_manager or response_generator,  # Fallback to response_generator
                    attraction_service=attraction_service,
                    weather_service=weather_service,
                    advanced_attraction_service=advanced_attraction_service,
                    transport_service=transport_service,
                    bilingual_manager=bilingual_manager
                )
                logger.info(f"🏛️ ML-Enhanced Attraction Handler initialized successfully! (Bilingual: {bilingual_manager is not None})")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_attraction_handler',
                    'status': 'success',
                    'bilingual': bilingual_manager is not None
                })
            else:
                logger.warning("Required dependencies not available for ML Attraction Handler")
                self.handlers['ml_attraction_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_attraction_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Attraction Handler not available: {e}")
            self.handlers['ml_attraction_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_attraction_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Attraction Handler: {e}")
            self.handlers['ml_attraction_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_attraction_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_local_food_handler(self, services: Dict, ml_context_builder: Any,
                                       neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Local Food Handler with Bilingual support"""
        try:
            from istanbul_ai.handlers.local_food_handler import LocalFoodHandler
            
            # Get required services
            local_food_service = services.get('local_food_service') or response_generator  # Use response_generator as fallback
            bilingual_manager = services.get('bilingual_manager')
            
            if neural_processor and response_generator:
                self.handlers['ml_local_food_handler'] = LocalFoodHandler(
                    neural_processor=neural_processor,
                    local_food_service=local_food_service,
                    response_generator=response_generator,
                    bilingual_manager=bilingual_manager
                )
                logger.info(f"🍽️ ML-Enhanced Local Food Handler initialized successfully! (Bilingual: {bilingual_manager is not None})")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_local_food_handler',
                    'status': 'success',
                    'bilingual': bilingual_manager is not None
                })
            else:
                logger.warning("Required dependencies not available for ML Local Food Handler")
                self.handlers['ml_local_food_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_local_food_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Local Food Handler not available: {e}")
            self.handlers['ml_local_food_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_local_food_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Local Food Handler: {e}")
            self.handlers['ml_local_food_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_local_food_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_emergency_safety_handler(self, services: Dict, ml_context_builder: Any,
                                            neural_processor: Any, response_generator: Any):
        """Initialize ML-Enhanced Emergency & Safety Handler"""
        try:
            from istanbul_ai.handlers.emergency_safety_handler import EmergencySafetyHandler
            
            # Get required services
            emergency_service = services.get('emergency_service') or response_generator  # Use response_generator as fallback
            safety_service = services.get('safety_service') or response_generator  # Use response_generator as fallback
            
            if neural_processor and response_generator:
                self.handlers['ml_emergency_safety_handler'] = EmergencySafetyHandler(
                    neural_processor=neural_processor,
                    emergency_service=emergency_service,
                    safety_service=safety_service,
                    response_generator=response_generator
                )
                logger.info(f"🚨 ML-Enhanced Emergency & Safety Handler initialized successfully!")
                self.initialized_count += 1
                self.initialization_log.append({
                    'handler': 'ml_emergency_safety_handler',
                    'status': 'success'
                })
            else:
                logger.warning("Required dependencies not available for ML Emergency & Safety Handler")
                self.handlers['ml_emergency_safety_handler'] = None
                self.initialization_log.append({
                    'handler': 'ml_emergency_safety_handler',
                    'status': 'skipped',
                    'reason': 'missing_dependencies'
                })
        except ImportError as e:
            logger.warning(f"ML Emergency & Safety Handler not available: {e}")
            self.handlers['ml_emergency_safety_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_emergency_safety_handler',
                'status': 'skipped',
                'reason': 'import_error'
            })
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Emergency & Safety Handler: {e}")
            self.handlers['ml_emergency_safety_handler'] = None
            self.initialization_log.append({
                'handler': 'ml_emergency_safety_handler',
                'status': 'failed',
                'error': str(e)
            })
    
    def _initialize_all_to_none(self):
        """Set all handlers to None when base dependencies are missing"""
        handler_names = [
            'emergency_safety_handler',
            'local_food_handler',
            'ml_event_handler',
            'ml_hidden_gems_handler',
            'ml_weather_handler',
            'ml_route_planning_handler',
            'ml_neighborhood_handler',
            'nearby_locations_response_handler',
            'transportation_handler',
            'ml_restaurant_handler',
            'ml_attraction_handler'
        ]
        
        for handler_name in handler_names:
            self.handlers[handler_name] = None
            self.initialization_log.append({
                'handler': handler_name,
                'status': 'skipped',
                'reason': 'missing_base_dependencies'
            })
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """
        Get a detailed report of handler initialization
        
        Returns:
            Dictionary containing:
            - initialized: Number of successfully initialized handlers
            - total_handlers: Total number of handlers
            - success_rate: Percentage string
            - handlers_status: Status of each handler
            - log: Detailed initialization log
        """
        success_rate_pct = (self.initialized_count / self.total_handlers * 100) if self.total_handlers > 0 else 0
        
        return {
            'initialized': self.initialized_count,
            'total_handlers': self.total_handlers,
            'success_rate': f"{success_rate_pct:.1f}%",
            'handlers_status': self.handlers.copy(),
            'log': self.initialization_log.copy()
        }
    
    def get_handler(self, handler_name: str) -> Optional[Any]:
        """
        Get a specific handler by name
        
        Args:
            handler_name: Name of the handler
            
        Returns:
            Handler instance or None if not available
        """
        return self.handlers.get(handler_name)
    
    def is_handler_available(self, handler_name: str) -> bool:
        """
        Check if a specific handler is available and initialized
        
        Args:
            handler_name: Name of the handler
            
        Returns:
            True if handler is available, False otherwise
        """
        return self.handlers.get(handler_name) is not None
