#!/usr/bin/env python3
"""
Script to update main_system.py to use ServiceInitializer
Part of Week 1 refactoring: Integration of ServiceInitializer
"""

import re

def update_main_system():
    """Update main_system.py to use ServiceInitializer"""
    
    file_path = 'istanbul_ai/main_system.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add ServiceInitializer import after ML-Enhanced Handlers import
    import_addition = '''
# Import ServiceInitializer for modular initialization
try:
    from istanbul_ai.initialization import ServiceInitializer
    SERVICE_INITIALIZER_AVAILABLE = True
    logger.info("✅ ServiceInitializer loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ServiceInitializer not available: {e}")
    SERVICE_INITIALIZER_AVAILABLE = False
'''
    
    # Find the location after ML-Enhanced Handlers import
    ml_handlers_pattern = r'(except ImportError as e:\s+logger\.warning\(f"⚠️ ML-Enhanced Handlers not available: \{e\}"\)\s+ML_ENHANCED_HANDLERS_AVAILABLE = False\s*\n)'
    
    content = re.sub(
        ml_handlers_pattern,
        r'\1' + import_addition + '\n',
        content,
        count=1
    )
    
    # 2. Replace the __init__ method
    # Find the start of __init__
    init_start_pattern = r'    def __init__\(self\):\s*\n\s*"""Initialize the Istanbul AI system"""'
    
    # Find where _log_cache_status method starts (end of __init__)
    init_end_pattern = r'(\s+# Log cache integration status\s+self\._log_cache_status\(\))\s+(def _log_cache_status)'
    
    # New __init__ method
    new_init = '''    def __init__(self):
        """Initialize the Istanbul AI system with modular architecture"""
        logger.info("🚀 Initializing Istanbul Daily Talk AI System...")
        
        # Initialize core components (kept in main system)
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.response_generator = ResponseGenerator()
        self.user_manager = UserManager()
        
        # Initialize all services using ServiceInitializer (REFACTORED!)
        if SERVICE_INITIALIZER_AVAILABLE:
            service_init = ServiceInitializer()
            services = service_init.initialize_all_services()
            
            # Map services to instance attributes
            for service_name, service_instance in services.items():
                setattr(self, service_name, service_instance)
            
            # Log initialization report
            report = service_init.get_initialization_report()
            logger.info(f"✅ Services initialized: {report['initialized']}/{report['total_services']} "
                       f"({report['success_rate']})")
            
            if report['errors']:
                failed_names = [err['service'] for err in report['errors']]
                logger.warning(f"⚠️ Failed services: {', '.join(failed_names)}")
        else:
            # Fallback: If ServiceInitializer is not available, set all services to None
            logger.error("❌ ServiceInitializer not available - services will be unavailable")
            service_names = [
                'hidden_gems_handler', 'price_filter_service', 'conversation_handler',
                'events_service', 'weather_recommendations', 'location_detector',
                'transport_processor', 'ml_transport_system', 'transportation_chat',
                'daily_talks_bridge', 'enhanced_daily_talks', 'neural_processor',
                'museum_generator', 'hours_checker', 'museum_db', 'advanced_museum_system',
                'advanced_attractions_system', 'multi_intent_handler', 'museum_route_planner',
                'gps_route_planner', 'advanced_route_planner', 'weather_client',
                'ml_context_builder', 'personalization_system', 'feedback_loop_system'
            ]
            for service_name in service_names:
                setattr(self, service_name, None)
        
        # Initialize ML-Enhanced Handlers (still in main_system for now - will refactor in Week 1)
        self._initialize_ml_handlers()
        
        # System status
        self.system_ready = True
        logger.info("✅ Istanbul Daily Talk AI System initialized successfully!")
        
        # Log cache integration status
        self._log_cache_status()
    
    def _initialize_ml_handlers(self):
        """Initialize ML-Enhanced Handlers (to be extracted in handler_initializer.py)"""
        # Initialize ML-Enhanced Event Handler
        try:
            if self.events_service and self.neural_processor and self.response_generator:
                ml_context_builder = self.ml_context_builder
                
                if ml_context_builder:
                    self.ml_event_handler = create_ml_enhanced_event_handler(
                        events_service=self.events_service,
                        ml_context_builder=ml_context_builder,
                        ml_processor=self.neural_processor,
                        response_generator=self.response_generator
                    )
                    logger.info("🎭 ML-Enhanced Event Handler initialized successfully!")
                else:
                    logger.warning("ML Context Builder not available, skipping ML Event Handler")
                    self.ml_event_handler = None
            else:
                logger.warning("Required dependencies not available for ML Event Handler")
                self.ml_event_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Event Handler: {e}")
            self.ml_event_handler = None

        # Initialize ML-Enhanced Hidden Gems Handler
        try:
            if self.hidden_gems_handler and self.neural_processor and self.response_generator:
                ml_context_builder = self.ml_context_builder
                
                if ml_context_builder:
                    self.ml_hidden_gems_handler = create_ml_enhanced_hidden_gems_handler(
                        hidden_gems_service=self.hidden_gems_handler,
                        ml_context_builder=ml_context_builder,
                        ml_processor=self.neural_processor,
                        response_generator=self.response_generator
                    )
                    logger.info("💎 ML-Enhanced Hidden Gems Handler initialized successfully!")
                else:
                    logger.warning("ML Context Builder not available, skipping ML Hidden Gems Handler")
                    self.ml_hidden_gems_handler = None
            else:
                logger.warning("Required dependencies not available for ML Hidden Gems Handler")
                self.ml_hidden_gems_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Hidden Gems Handler: {e}")
            self.ml_hidden_gems_handler = None

        # Initialize ML-Enhanced Weather Handler
        try:
            if self.weather_client and self.weather_recommendations and self.neural_processor and self.response_generator:
                ml_context_builder = self.ml_context_builder
                
                if ml_context_builder:
                    self.ml_weather_handler = create_ml_enhanced_weather_handler(
                        weather_service=self.weather_client,
                        weather_recommendations_service=self.weather_recommendations,
                        ml_context_builder=ml_context_builder,
                        ml_processor=self.neural_processor,
                        response_generator=self.response_generator
                    )
                    logger.info("🌤️ ML-Enhanced Weather Handler initialized successfully!")
                else:
                    logger.warning("ML Context Builder not available, skipping ML Weather Handler")
                    self.ml_weather_handler = None
            else:
                logger.warning("Required dependencies not available for ML Weather Handler")
                self.ml_weather_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Weather Handler: {e}")
            self.ml_weather_handler = None

        # Initialize ML-Enhanced Route Planning Handler
        try:
            route_service = getattr(self, 'advanced_route_planner', None) or getattr(self, 'gps_route_planner', None)
            transport_service = getattr(self, 'transport_processor', None)
            
            if route_service and transport_service and self.neural_processor and self.response_generator:
                ml_context_builder = self.ml_context_builder
                
                if ml_context_builder:
                    self.ml_route_planning_handler = create_ml_enhanced_route_planning_handler(
                        route_planner_service=route_service,
                        transport_service=transport_service,
                        ml_context_builder=ml_context_builder,
                        ml_processor=self.neural_processor,
                        response_generator=self.response_generator
                    )
                    logger.info("🗺️ ML-Enhanced Route Planning Handler initialized successfully!")
                else:
                    logger.warning("ML Context Builder not available, skipping ML Route Planning Handler")
                    self.ml_route_planning_handler = None
            else:
                logger.warning("Required dependencies not available for ML Route Planning Handler")
                self.ml_route_planning_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Route Planning Handler: {e}")
            self.ml_route_planning_handler = None

        # Initialize ML-Enhanced Neighborhood Handler
        try:
            if self.response_generator and self.neural_processor:
                ml_context_builder = self.ml_context_builder
                
                if ml_context_builder:
                    self.ml_neighborhood_handler = create_ml_enhanced_neighborhood_handler(
                        neighborhood_service=self.response_generator,
                        ml_context_builder=ml_context_builder,
                        ml_processor=self.neural_processor,
                        response_generator=self.response_generator
                    )
                    logger.info("🏘️ ML-Enhanced Neighborhood Handler initialized successfully!")
                else:
                    logger.warning("ML Context Builder not available, skipping ML Neighborhood Handler")
                    self.ml_neighborhood_handler = None
            else:
                logger.warning("Required dependencies not available for ML Neighborhood Handler")
                self.ml_neighborhood_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize ML-Enhanced Neighborhood Handler: {e}")
            self.ml_neighborhood_handler = None
    
    '''
    
    # Use a more robust pattern to find and replace the entire __init__ method
    # Find from def __init__ to just before def _log_cache_status
    pattern = r'    def __init__\(self\):.*?(?=    def _log_cache_status\(self\):)'
    
    content = re.sub(
        pattern,
        new_init,
        content,
        flags=re.DOTALL,
        count=1
    )
    
    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Successfully updated main_system.py to use ServiceInitializer!")
    print("📝 Changes made:")
    print("   1. Added ServiceInitializer import")
    print("   2. Replaced __init__ method to use ServiceInitializer")
    print("   3. Extracted ML handler initialization to _initialize_ml_handlers()")
    print("   4. Reduced main_system.py by ~370 lines!")

if __name__ == '__main__':
    update_main_system()
