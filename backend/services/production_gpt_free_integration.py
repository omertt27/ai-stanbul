"""
Production Integration Guide for GPT-Free AI Istanbul System
Shows how to integrate the enhanced GPT-free system with existing backend
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import existing AI Istanbul components
try:
    from services.enhanced_gpt_free_system import create_gpt_free_system, EnhancedGPTFreeSystem
    from services.query_router import QueryRouter
    from services.template_engine import TemplateEngine
    from services.recommendation_engine import RecommendationEngine
    GPT_FREE_AVAILABLE = True
except ImportError as e:
    GPT_FREE_AVAILABLE = False
    print(f"⚠️ GPT-Free system not available: {e}")

logger = logging.getLogger(__name__)

class ProductionAIOrchestrator:
    """
    Production orchestrator that integrates GPT-free system with existing components
    Provides seamless transition from GPT-dependent to GPT-free operation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize GPT-free system
        if GPT_FREE_AVAILABLE:
            self.gpt_free_system = create_gpt_free_system({
                'cache_dir': self.config.get('cache_dir', 'cache_data'),
                'clustering_dir': self.config.get('clustering_dir', 'clustering_data'),
                'embedding_model': self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            })
            self.gpt_free_enabled = True
            logger.info("✅ GPT-Free system initialized")
        else:
            self.gpt_free_system = None
            self.gpt_free_enabled = False
            logger.warning("⚠️ GPT-Free system not available, using fallback")
        
        # Initialize existing components
        try:
            self.query_router = QueryRouter()
            self.template_engine = TemplateEngine()
            self.recommendation_engine = RecommendationEngine()
            self.existing_components_available = True
            logger.info("✅ Existing AI Istanbul components loaded")
        except Exception as e:
            self.existing_components_available = False
            logger.warning(f"⚠️ Existing components not available: {e}")
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'gpt_free_success': 0,
            'fallback_used': 0,
            'avg_response_time': 0.0,
            'cost_savings_estimated': 0.0,
            'user_satisfaction_avg': 0.0
        }
        
        # Configuration
        self.gpt_free_confidence_threshold = self.config.get('gpt_free_threshold', 0.6)
        self.fallback_to_gpt = self.config.get('enable_gpt_fallback', False)
        
    def process_chat_query(self, query: str, user_id: str = None, 
                          session_context: Dict = None, user_preferences: Dict = None) -> Dict:
        """
        Main chat processing method - production-ready with all fallbacks
        """
        start_time = datetime.now()
        self.performance_metrics['total_requests'] += 1
        
        # Prepare context
        context = dict(session_context or {})
        if user_id:
            context['user_id'] = user_id
        
        try:
            # Primary: GPT-Free System
            if self.gpt_free_enabled:
                gpt_free_result = self.gpt_free_system.process_query(
                    query, context, user_id, user_preferences
                )
                
                # Check if result meets confidence threshold
                if gpt_free_result.confidence >= self.gpt_free_confidence_threshold:
                    self.performance_metrics['gpt_free_success'] += 1
                    
                    # Estimate cost savings (typical GPT-4 call costs ~$0.01-0.02)
                    self.performance_metrics['cost_savings_estimated'] += 0.015
                    
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return {
                        'response': gpt_free_result.response,
                        'source': gpt_free_result.source,
                        'confidence': gpt_free_result.confidence,
                        'metadata': gpt_free_result.metadata,
                        'processing_time_ms': processing_time,
                        'cost_saved': True,
                        'fallback_used': False
                    }
            
            # Fallback 1: Existing Query Router
            if self.existing_components_available:
                try:
                    router_result = self.query_router.route_query(query, context)
                    if router_result and router_result.get('response'):
                        self.performance_metrics['fallback_used'] += 1
                        
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        return {
                            'response': router_result['response'],
                            'source': 'existing_router',
                            'confidence': router_result.get('confidence', 0.5),
                            'metadata': router_result.get('metadata', {}),
                            'processing_time_ms': processing_time,
                            'cost_saved': True,
                            'fallback_used': True
                        }
                except Exception as e:
                    logger.warning(f"Existing router failed: {e}")
            
            # Fallback 2: Template-based response
            template_response = self._generate_template_fallback(query, context)
            if template_response:
                self.performance_metrics['fallback_used'] += 1
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    'response': template_response,
                    'source': 'template_fallback',
                    'confidence': 0.4,
                    'metadata': {'method': 'template_based'},
                    'processing_time_ms': processing_time,
                    'cost_saved': True,
                    'fallback_used': True
                }
            
            # Final Fallback: GPT (if enabled)
            if self.fallback_to_gpt:
                gpt_response = self._call_gpt_fallback(query, context)
                if gpt_response:
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return {
                        'response': gpt_response,
                        'source': 'gpt_fallback',
                        'confidence': 0.8,
                        'metadata': {'method': 'gpt_api'},
                        'processing_time_ms': processing_time,
                        'cost_saved': False,
                        'fallback_used': True
                    }
            
            # Emergency fallback
            emergency_response = self._generate_emergency_response(query)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'response': emergency_response,
                'source': 'emergency_fallback',
                'confidence': 0.2,
                'metadata': {'method': 'emergency'},
                'processing_time_ms': processing_time,
                'cost_saved': True,
                'fallback_used': True
            }
            
        except Exception as e:
            logger.error(f"❌ Critical error in chat processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                'source': 'error',
                'confidence': 0.0,
                'metadata': {'error': str(e)},
                'processing_time_ms': processing_time,
                'cost_saved': True,
                'fallback_used': True
            }
    
    def _generate_template_fallback(self, query: str, context: Dict) -> Optional[str]:
        """Generate response using template engine fallback"""
        try:
            if not self.existing_components_available:
                return None
            
            # Use template engine if available
            template_result = self.template_engine.generate_response(
                query_type=self._detect_query_type(query),
                context=context
            )
            
            return template_result.get('response') if template_result else None
            
        except Exception as e:
            logger.warning(f"Template fallback failed: {e}")
            return None
    
    def _call_gpt_fallback(self, query: str, context: Dict) -> Optional[str]:
        """Call GPT as final fallback (if enabled)"""
        try:
            # This would integrate with your existing GPT calling logic
            # For now, return None to avoid GPT dependency
            logger.info("GPT fallback called but disabled for GPT-free operation")
            return None
            
        except Exception as e:
            logger.error(f"GPT fallback failed: {e}")
            return None
    
    def _generate_emergency_response(self, query: str) -> str:
        """Generate basic emergency response"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['transport', 'get to', 'metro', 'bus']):
            return """🚇 **Istanbul Transport Help:**
            
I can help you get around Istanbul! The main transport options are:
• **Metro**: Fast and efficient for longer distances
• **Tram**: Great for tourist areas (T1 line)
• **Bus**: Extensive network throughout the city
• **Ferry**: Scenic routes across the Bosphorus

Use an İstanbul Card for all public transport. You can buy it at any station.

For specific directions, please try rephrasing your question with your starting point and destination."""
            
        elif any(word in query_lower for word in ['restaurant', 'food', 'eat']):
            return """🍽️ **Istanbul Food Guide:**
            
Istanbul has incredible cuisine! Here are some must-tries:
• **Turkish Breakfast**: Extensive spread with cheese, olives, bread
• **Kebabs**: Try Adana, Urfa, or İskender varieties  
• **Meze**: Small appetizer plates perfect for sharing
• **Baklava**: Sweet pastry with nuts and honey
• **Turkish Coffee**: UNESCO recognized preparation method

Popular dining areas include Sultanahmet (traditional), Karaköy (trendy), and Kadıköy (local favorites).

For specific restaurant recommendations, let me know which area you're visiting!"""
            
        elif any(word in query_lower for word in ['attraction', 'see', 'visit', 'museum']):
            return """🏛️ **Istanbul Attractions:**
            
Istanbul's top attractions include:
• **Hagia Sophia**: Historic Byzantine church/Ottoman mosque
• **Blue Mosque**: Famous for its six minarets and blue tiles
• **Topkapi Palace**: Ottoman palace with amazing views
• **Grand Bazaar**: One of the world's oldest covered markets
• **Galata Tower**: 360° views of the city
• **Bosphorus**: Take a cruise to see both European and Asian sides

Most museums are open 9:00-17:00 and closed on Mondays. Book popular attractions online to avoid queues.

What specific area interests you most?"""
            
        else:
            return """👋 **Welcome to Istanbul!**
            
I'm here to help you explore this amazing city! I can assist with:

🗺️ **Getting Around**: Metro, bus, tram, and ferry information
🏛️ **Attractions**: Opening hours, tickets, and recommendations
🍽️ **Food**: Restaurant suggestions and local specialties
🛍️ **Shopping**: Markets, souvenirs, and local products
📍 **Neighborhoods**: What to see in different districts

What would you like to know about Istanbul? Please be as specific as possible so I can give you the best help!"""
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type for template engine"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['transport', 'get to', 'metro', 'bus']):
            return 'transportation'
        elif any(word in query_lower for word in ['restaurant', 'food', 'eat']):
            return 'food'
        elif any(word in query_lower for word in ['hours', 'open', 'ticket', 'price']):
            return 'practical_info'
        elif any(word in query_lower for word in ['see', 'visit', 'attraction']):
            return 'exploration'
        else:
            return 'general'
    
    def record_user_feedback(self, query: str, response: str, user_satisfaction: float,
                           feedback_metadata: Dict = None):
        """Record user feedback for system improvement"""
        try:
            # Update performance metrics
            if hasattr(self, 'satisfaction_scores'):
                self.satisfaction_scores.append(user_satisfaction)
            else:
                self.satisfaction_scores = [user_satisfaction]
            
            # Calculate running average
            self.performance_metrics['user_satisfaction_avg'] = sum(self.satisfaction_scores) / len(self.satisfaction_scores)
            
            # Pass feedback to GPT-free system for learning
            if self.gpt_free_system:
                self.gpt_free_system.learn_from_feedback(
                    query, response, user_satisfaction, feedback_metadata
                )
            
            logger.info(f"📊 Feedback recorded: {user_satisfaction:.2f} satisfaction")
            
        except Exception as e:
            logger.error(f"❌ Error recording feedback: {e}")
    
    def get_production_statistics(self) -> Dict:
        """Get comprehensive production statistics"""
        base_stats = {
            'production_metrics': self.performance_metrics,
            'gpt_free_enabled': self.gpt_free_enabled,
            'existing_components_available': self.existing_components_available
        }
        
        # Add GPT-free system stats if available
        if self.gpt_free_system:
            gpt_free_stats = self.gpt_free_system.get_system_statistics()
            base_stats['gpt_free_system'] = gpt_free_stats
        
        # Calculate success rates
        total_requests = self.performance_metrics['total_requests']
        if total_requests > 0:
            base_stats['success_rates'] = {
                'gpt_free_success_rate': (self.performance_metrics['gpt_free_success'] / total_requests) * 100,
                'total_success_rate': ((self.performance_metrics['gpt_free_success'] + self.performance_metrics['fallback_used']) / total_requests) * 100,
                'estimated_cost_savings_usd': self.performance_metrics['cost_savings_estimated']
            }
        
        return base_stats
    
    def health_check(self) -> Dict:
        """Perform system health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check GPT-free system
        try:
            if self.gpt_free_system:
                # Quick test query
                test_result = self.gpt_free_system.process_query(
                    "test health check", {}, "health_check_user"
                )
                health_status['components']['gpt_free_system'] = {
                    'status': 'healthy' if test_result.confidence > 0 else 'degraded',
                    'response_time_ms': test_result.processing_time_ms
                }
            else:
                health_status['components']['gpt_free_system'] = {
                    'status': 'unavailable',
                    'response_time_ms': 0
                }
        except Exception as e:
            health_status['components']['gpt_free_system'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check existing components
        try:
            if self.existing_components_available:
                health_status['components']['existing_router'] = {'status': 'healthy'}
            else:
                health_status['components']['existing_router'] = {'status': 'unavailable'}
        except Exception as e:
            health_status['components']['existing_router'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Determine overall status
        component_statuses = [comp.get('status') for comp in health_status['components'].values()]
        if 'error' in component_statuses:
            health_status['overall_status'] = 'error'
        elif 'degraded' in component_statuses:
            health_status['overall_status'] = 'degraded'
        elif all(status in ['healthy', 'unavailable'] for status in component_statuses):
            health_status['overall_status'] = 'healthy'
        
        return health_status
    
    def export_production_report(self, filepath: str = None) -> str:
        """Export comprehensive production report"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"production_report_{timestamp}.json"
        
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'system_configuration': self.config,
            'production_statistics': self.get_production_statistics(),
            'health_check': self.health_check(),
            'performance_summary': {
                'total_queries_processed': self.performance_metrics['total_requests'],
                'gpt_free_coverage_percent': (
                    (self.performance_metrics['gpt_free_success'] / 
                     max(1, self.performance_metrics['total_requests'])) * 100
                ),
                'estimated_monthly_savings_usd': self.performance_metrics['cost_savings_estimated'] * 30,
                'avg_user_satisfaction': self.performance_metrics['user_satisfaction_avg']
            }
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"📊 Production report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error exporting production report: {e}")
            return ""

# Example usage and integration
def setup_production_system(config_overrides: Dict = None) -> ProductionAIOrchestrator:
    """Set up production system with configuration"""
    
    default_config = {
        'cache_dir': os.path.join(os.getcwd(), 'production_cache'),
        'clustering_dir': os.path.join(os.getcwd(), 'production_clustering'),
        'embedding_model': 'all-MiniLM-L6-v2',
        'gpt_free_threshold': 0.6,
        'enable_gpt_fallback': False  # Set to True if you want GPT fallback
    }
    
    if config_overrides:
        default_config.update(config_overrides)
    
    # Create directories
    os.makedirs(default_config['cache_dir'], exist_ok=True)
    os.makedirs(default_config['clustering_dir'], exist_ok=True)
    
    # Initialize system
    orchestrator = ProductionAIOrchestrator(default_config)
    
    logger.info("🚀 Production AI Istanbul system initialized")
    logger.info(f"📊 GPT-Free enabled: {orchestrator.gpt_free_enabled}")
    logger.info(f"🔄 Fallback components: {orchestrator.existing_components_available}")
    
    return orchestrator

# Integration example for existing Flask/FastAPI apps
def integrate_with_flask_app(app, orchestrator: ProductionAIOrchestrator):
    """Example integration with Flask app"""
    
    @app.route('/api/chat', methods=['POST'])
    def chat_endpoint():
        from flask import request, jsonify
        
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('user_id')
        session_context = data.get('context', {})
        user_preferences = data.get('preferences', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Process query
        result = orchestrator.process_chat_query(
            query, user_id, session_context, user_preferences
        )
        
        return jsonify(result)
    
    @app.route('/api/feedback', methods=['POST'])
    def feedback_endpoint():
        from flask import request, jsonify
        
        data = request.get_json()
        query = data.get('query', '')
        response = data.get('response', '')
        satisfaction = data.get('satisfaction', 0.5)
        metadata = data.get('metadata', {})
        
        orchestrator.record_user_feedback(query, response, satisfaction, metadata)
        
        return jsonify({'status': 'feedback_recorded'})
    
    @app.route('/api/system/health', methods=['GET'])
    def health_endpoint():
        from flask import jsonify
        return jsonify(orchestrator.health_check())
    
    @app.route('/api/system/stats', methods=['GET'])
    def stats_endpoint():
        from flask import jsonify
        return jsonify(orchestrator.get_production_statistics())

if __name__ == "__main__":
    # Example standalone usage
    print("🚀 Initializing Production AI Istanbul System...")
    
    # Setup system
    orchestrator = setup_production_system({
        'gpt_free_threshold': 0.7,
        'enable_gpt_fallback': False
    })
    
    # Test queries
    test_queries = [
        "How to get to Hagia Sophia?",
        "Best restaurants in Sultanahmet",
        "Blue Mosque opening hours",
        "What to see in Beyoglu"
    ]
    
    print("\n📝 Testing with sample queries...")
    for query in test_queries:
        result = orchestrator.process_chat_query(query)
        print(f"\nQ: {query}")
        print(f"A: {result['response'][:100]}...")
        print(f"Source: {result['source']}, Confidence: {result['confidence']:.2f}")
    
    # Show statistics
    print("\n📊 System Statistics:")
    stats = orchestrator.get_production_statistics()
    print(f"Total Requests: {stats['production_metrics']['total_requests']}")
    print(f"GPT-Free Success: {stats['production_metrics']['gpt_free_success']}")
    print(f"Estimated Savings: ${stats['production_metrics']['cost_savings_estimated']:.2f}")
    
    # Export report
    report_file = orchestrator.export_production_report()
    print(f"\n📋 Report exported to: {report_file}")
    
    print("\n✅ Production system test completed!")
