"""
Enhanced Production Integration for AI Istanbul GPT-Free System
Full integration of user profiling, enhanced intent classifier, and GPT-free system
Complete production-ready solution with personalization and advanced intent detection
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import all AI Istanbul components
try:
    import sys
    from pathlib import Path
    
    # Add the current directory to Python path for imports
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from enhanced_gpt_free_system import create_gpt_free_system, EnhancedGPTFreeSystem
    from user_profiling_system import UserProfilingSystem, UserProfile
    from enhanced_intent_classifier import EnhancedIntentClassifier, IntentResult
    
    # Try to import existing components (optional)
    try:
        from query_router import QueryRouter
        from template_engine import TemplateEngine
        from recommendation_engine import RecommendationEngine
        EXISTING_COMPONENTS_AVAILABLE = True
    except ImportError:
        EXISTING_COMPONENTS_AVAILABLE = False
        print("ℹ️ Existing AI Istanbul components not available, using enhanced system only")
    
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError as e:
    ENHANCED_SYSTEM_AVAILABLE = False
    EXISTING_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Enhanced system components not available: {e}")

logger = logging.getLogger(__name__)

class EnhancedProductionOrchestrator:
    """
    Enhanced production orchestrator with full personalization and intent classification
    Integrates all AI Istanbul components for maximum GPT-free operation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.initialize_components()
        self.setup_performance_tracking()
        
    def initialize_components(self):
        """Initialize all system components"""
        
        # Core GPT-free system
        if ENHANCED_SYSTEM_AVAILABLE:
            try:
                self.gpt_free_system = create_gpt_free_system({
                    'cache_dir': self.config.get('cache_dir', 'cache_data'),
                    'clustering_dir': self.config.get('clustering_dir', 'clustering_data'),
                    'embedding_model': self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                })
                
                # User profiling system
                profile_db_path = self.config.get('profile_db_path', 'user_profiles.db')
                self.user_profiling = UserProfilingSystem(profile_db_path)
                
                # Enhanced intent classifier
                self.intent_classifier = EnhancedIntentClassifier()
                
                # Load any existing ML models
                if self.config.get('train_intent_model', True):
                    self._initialize_intent_model()
                
                self.enhanced_system_enabled = True
                logger.info("✅ Enhanced GPT-Free system with personalization initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize enhanced system: {e}")
                self.enhanced_system_enabled = False
        else:
            self.enhanced_system_enabled = False
            logger.warning("⚠️ Enhanced system not available")
        
        # Existing AI Istanbul components (fallback)
        if EXISTING_COMPONENTS_AVAILABLE:
            try:
                self.query_router = QueryRouter()
                self.template_engine = TemplateEngine()
                self.recommendation_engine = RecommendationEngine()
                self.existing_components_available = True
                logger.info("✅ Existing AI Istanbul components loaded")
            except Exception as e:
                self.existing_components_available = False
                logger.warning(f"⚠️ Existing components not available: {e}")
        else:
            self.existing_components_available = False
            logger.info("ℹ️ Using enhanced system only")
        
        # Configuration thresholds
        self.gpt_free_confidence_threshold = self.config.get('gpt_free_threshold', 0.7)
        self.intent_confidence_threshold = self.config.get('intent_threshold', 0.6)
        self.personalization_weight = self.config.get('personalization_weight', 0.3)
        self.fallback_to_gpt = self.config.get('enable_gpt_fallback', False)
        
    def setup_performance_tracking(self):
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            # Request metrics
            'total_requests': 0,
            'gpt_free_success': 0,
            'personalized_responses': 0,
            'intent_classified_accurately': 0,
            'fallback_used': 0,
            
            # Performance metrics
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'personalization_improvement': 0.0,
            
            # Cost and satisfaction
            'cost_savings_estimated': 0.0,
            'user_satisfaction_avg': 0.0,
            'user_profile_updates': 0,
            
            # Breakdown by component
            'cache_hits': 0,
            'template_matches': 0,
            'recommendation_served': 0,
            'intent_detection_success': 0
        }
        
        # Session tracking for context
        self.active_sessions = {}
        self.session_cleanup_interval = timedelta(hours=2)
        
    def _initialize_intent_model(self):
        """Initialize and train intent classification model if needed"""
        try:
            # Load training data from existing queries if available
            training_data_path = self.config.get('intent_training_data', 'intent_training.json')
            if Path(training_data_path).exists():
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                self.intent_classifier.train_ml_model(training_data)
                logger.info("✅ Intent classification model trained")
            else:
                logger.info("ℹ️ No training data found, using rule-based intent classification")
        except Exception as e:
            logger.warning(f"⚠️ Failed to train intent model: {e}")
    
    def process_chat_query(self, query: str, user_id: str = None, 
                          session_id: str = None, additional_context: Dict = None) -> Dict:
        """
        Enhanced chat processing with full personalization and context awareness
        """
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        # Prepare session and context
        session_context = self._get_or_create_session(session_id, user_id)
        context = self._build_query_context(query, user_id, session_context, additional_context)
        
        try:
            # Step 1: Enhanced Intent Classification
            intent_result = self._classify_intent_with_context(query, context)
            
            # Step 2: User Profile Integration
            user_profile = self._get_user_profile(user_id)
            personalized_context = self._add_personalization_context(context, user_profile, intent_result)
            
            # Step 3: Enhanced GPT-Free Processing
            if self.enhanced_system_enabled:
                result = self._process_with_enhanced_system(query, personalized_context, intent_result)
                
                if result and result.get('confidence', 0) >= self.gpt_free_confidence_threshold:
                    # Success - update metrics and user profile
                    self._update_success_metrics(result, user_profile, intent_result)
                    self._update_user_profile(user_id, query, result, intent_result)
                    
                    processing_time = (time.time() - start_time) * 1000
                    return self._format_successful_response(result, processing_time, "enhanced_gpt_free")
            
            # Step 4: Existing System Fallback
            if self.existing_components_available:
                result = self._process_with_existing_system(query, personalized_context)
                if result:
                    self.performance_metrics['fallback_used'] += 1
                    processing_time = (time.time() - start_time) * 1000
                    return self._format_successful_response(result, processing_time, "existing_system")
            
            # Step 5: Final fallback
            return self._create_fallback_response(query, context)
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return self._create_error_response(str(e))
    
    def _classify_intent_with_context(self, query: str, context: Dict) -> Optional[IntentResult]:
        """Enhanced intent classification with session context"""
        if not self.enhanced_system_enabled:
            return None
            
        try:
            # Add session context for better classification
            session_history = context.get('session_history', [])
            user_preferences = context.get('user_preferences', {})
            
            intent_result = self.intent_classifier.classify_with_context(
                query, session_history, user_preferences
            )
            
            if intent_result.confidence >= self.intent_confidence_threshold:
                self.performance_metrics['intent_classified_accurately'] += 1
                self.performance_metrics['intent_detection_success'] += 1
                return intent_result
                
            return intent_result  # Still return even if low confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Intent classification failed: {e}")
            return None
    
    def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get or create user profile"""
        if not user_id or not self.enhanced_system_enabled:
            return None
            
        try:
            return self.user_profiling.get_user_profile(user_id)
        except Exception as e:
            logger.warning(f"⚠️ Failed to get user profile: {e}")
            return None
    
    def _add_personalization_context(self, context: Dict, user_profile: Optional[UserProfile], 
                                   intent_result: Optional[IntentResult]) -> Dict:
        """Add personalization context based on user profile and intent"""
        personalized_context = context.copy()
        
        if user_profile:
            # Add user preferences
            personalized_context['user_preferences'] = {
                pref_type: [p.value for p in prefs]
                for pref_type, prefs in user_profile.preferences.items()
            }
            
            # Add behavioral patterns
            personalized_context['visited_locations'] = list(user_profile.visited_locations)
            personalized_context['preferred_times'] = user_profile.preferred_times
            personalized_context['interaction_patterns'] = user_profile.interaction_patterns
            
            # Add contextual suggestions based on intent
            if intent_result:
                suggestions = self.user_profiling.get_personalized_suggestions(
                    user_profile, intent_result.primary_intent.value
                )
                personalized_context['personalized_suggestions'] = suggestions
                
            self.performance_metrics['personalized_responses'] += 1
        
        return personalized_context
    
    def _process_with_enhanced_system(self, query: str, context: Dict, 
                                    intent_result: Optional[IntentResult]) -> Optional[Dict]:
        """Process query with enhanced GPT-free system"""
        try:
            # Add intent information to context
            if intent_result:
                context['detected_intent'] = {
                    'primary': intent_result.primary_intent.value,
                    'secondary': [i.value for i in intent_result.secondary_intents],
                    'confidence': intent_result.confidence,
                    'entities': intent_result.entities
                }
            
            # Process with GPT-free system
            result = self.gpt_free_system.process_query(
                query, 
                context, 
                context.get('user_id'),
                context.get('user_preferences')
            )
            
            # Track component usage
            if hasattr(result, 'source'):
                if result.source == 'cache':
                    self.performance_metrics['cache_hits'] += 1
                elif result.source == 'template':
                    self.performance_metrics['template_matches'] += 1
                elif result.source == 'recommendation':
                    self.performance_metrics['recommendation_served'] += 1
            
            return result.__dict__ if hasattr(result, '__dict__') else result
            
        except Exception as e:
            logger.error(f"❌ Enhanced system processing failed: {e}")
            return None
    
    def _process_with_existing_system(self, query: str, context: Dict) -> Optional[Dict]:
        """Fallback to existing AI Istanbul system"""
        try:
            # Route query through existing system
            routed_result = self.query_router.route_query(query, context)
            
            # Generate response using template engine
            if routed_result:
                response = self.template_engine.generate_response(
                    routed_result['intent'], 
                    routed_result.get('entities', {}),
                    context
                )
                
                # Add recommendations if applicable
                if 'location' in routed_result.get('entities', {}):
                    recommendations = self.recommendation_engine.get_recommendations(
                        routed_result['entities']['location'],
                        context.get('user_preferences', {})
                    )
                    response['recommendations'] = recommendations
                
                return {
                    'response': response,
                    'confidence': routed_result.get('confidence', 0.5),
                    'source': 'existing_system'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Existing system processing failed: {e}")
            return None
    
    def _update_success_metrics(self, result: Dict, user_profile: Optional[UserProfile], 
                              intent_result: Optional[IntentResult]):
        """Update performance metrics on successful processing"""
        self.performance_metrics['gpt_free_success'] += 1
        
        # Estimate cost savings (GPT-4 API call costs ~$0.01-0.02)
        estimated_savings = 0.015
        if user_profile:
            estimated_savings += 0.005  # Additional savings from personalization
        self.performance_metrics['cost_savings_estimated'] += estimated_savings
        
        # Update cache hit rate
        if result.get('source') == 'cache':
            total_cache_attempts = self.performance_metrics['total_requests']
            cache_hits = self.performance_metrics['cache_hits']
            self.performance_metrics['cache_hit_rate'] = cache_hits / total_cache_attempts if total_cache_attempts > 0 else 0
    
    def _update_user_profile(self, user_id: str, query: str, result: Dict, 
                           intent_result: Optional[IntentResult]):
        """Update user profile with interaction data"""
        if not user_id or not self.enhanced_system_enabled:
            return
            
        try:
            # Extract preferences from query and result
            interaction_data = {
                'query': query,
                'response': result.get('response', ''),
                'intent': intent_result.primary_intent.value if intent_result else 'unknown',
                'entities': intent_result.entities if intent_result else {},
                'timestamp': datetime.now(),
                'satisfaction_score': result.get('confidence', 0.5)  # Use confidence as proxy
            }
            
            self.user_profiling.update_user_profile(user_id, interaction_data)
            self.performance_metrics['user_profile_updates'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update user profile: {e}")
    
    def _get_or_create_session(self, session_id: str, user_id: str) -> Dict:
        """Get or create session context"""
        if not session_id:
            session_id = f"session_{user_id}_{int(time.time())}" if user_id else f"anon_{int(time.time())}"
        
        # Clean up old sessions periodically
        current_time = datetime.now()
        self.active_sessions = {
            sid: session for sid, session in self.active_sessions.items()
            if current_time - session.get('created_at', current_time) < self.session_cleanup_interval
        }
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'session_id': session_id,
                'user_id': user_id,
                'created_at': current_time,
                'query_history': [],
                'context_memory': {}
            }
        
        return self.active_sessions[session_id]
    
    def _build_query_context(self, query: str, user_id: str, session_context: Dict, 
                           additional_context: Dict = None) -> Dict:
        """Build comprehensive query context"""
        context = {
            'query': query,
            'user_id': user_id,
            'session_id': session_context.get('session_id'),
            'timestamp': datetime.now().isoformat(),
            'session_history': session_context.get('query_history', [])[-5:],  # Last 5 queries
            'context_memory': session_context.get('context_memory', {})
        }
        
        if additional_context:
            context.update(additional_context)
        
        # Update session history
        session_context['query_history'].append({
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
        
        return context
    
    def _format_successful_response(self, result: Dict, processing_time: float, source: str) -> Dict:
        """Format successful response with metadata"""
        return {
            'success': True,
            'response': result.get('response', result.get('answer', '')),
            'confidence': result.get('confidence', 0.0),
            'source': source,
            'processing_time_ms': round(processing_time, 2),
            'recommendations': result.get('recommendations', []),
            'entities': result.get('entities', {}),
            'intent': result.get('intent', 'unknown'),
            'personalized': result.get('personalized', False),
            'cost_savings': result.get('cost_savings', 0.015),
            'metadata': {
                'cache_hit': result.get('source') == 'cache',
                'template_used': result.get('source') == 'template',
                'personalization_applied': 'user_preferences' in result
            }
        }
    
    def _create_fallback_response(self, query: str, context: Dict) -> Dict:
        """Create fallback response when all systems fail"""
        self.performance_metrics['fallback_used'] += 1
        
        return {
            'success': False,
            'response': "I apologize, but I'm having trouble processing your request right now. Could you please rephrase your question or try again later?",
            'confidence': 0.1,
            'source': 'fallback',
            'error': 'all_systems_unavailable',
            'suggestions': [
                "Try asking about popular Istanbul attractions",
                "Ask for transportation information",
                "Request restaurant recommendations"
            ]
        }
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create error response"""
        return {
            'success': False,
            'response': "I encountered an error while processing your request. Please try again.",
            'confidence': 0.0,
            'source': 'error',
            'error': error_message
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        total_requests = self.performance_metrics['total_requests']
        
        if total_requests == 0:
            return self.performance_metrics
        
        # Calculate derived metrics
        metrics = self.performance_metrics.copy()
        metrics.update({
            'gpt_free_success_rate': metrics['gpt_free_success'] / total_requests,
            'personalization_rate': metrics['personalized_responses'] / total_requests,
            'intent_accuracy_rate': metrics['intent_classified_accurately'] / total_requests,
            'fallback_rate': metrics['fallback_used'] / total_requests,
            'avg_cost_savings_per_request': metrics['cost_savings_estimated'] / total_requests,
            'estimated_monthly_savings': metrics['cost_savings_estimated'] * 30 * 100  # Assuming 100 requests/day
        })
        
        return metrics
    
    def get_system_status(self) -> Dict:
        """Get system status and health check"""
        return {
            'enhanced_system_enabled': self.enhanced_system_enabled,
            'existing_components_available': self.existing_components_available,
            'gpt_free_available': ENHANCED_SYSTEM_AVAILABLE,
            'active_sessions': len(self.active_sessions),
            'configuration': {
                'gpt_free_threshold': self.gpt_free_confidence_threshold,
                'intent_threshold': self.intent_confidence_threshold,
                'personalization_weight': self.personalization_weight,
                'gpt_fallback_enabled': self.fallback_to_gpt
            },
            'performance': self.get_performance_metrics()
        }

# Flask integration example
def create_enhanced_flask_app():
    """Create Flask app with enhanced AI Istanbul integration"""
    try:
        from flask import Flask, request, jsonify, session
        from flask_cors import CORS
    except ImportError:
        print("⚠️ Flask not available for web integration")
        return None
    
    app = Flask(__name__)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    CORS(app)
    
    # Initialize enhanced orchestrator
    config = {
        'cache_dir': os.environ.get('CACHE_DIR', 'cache_data'),
        'profile_db_path': os.environ.get('PROFILE_DB_PATH', 'user_profiles.db'),
        'gpt_free_threshold': float(os.environ.get('GPT_FREE_THRESHOLD', '0.7')),
        'intent_threshold': float(os.environ.get('INTENT_THRESHOLD', '0.6')),
        'train_intent_model': os.environ.get('TRAIN_INTENT_MODEL', 'true').lower() == 'true'
    }
    
    orchestrator = EnhancedProductionOrchestrator(config)
    
    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        query = data.get('query', '').strip()
        user_id = data.get('user_id') or session.get('user_id')
        session_id = data.get('session_id') or session.get('session_id')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"web_session_{int(time.time())}"
            session['session_id'] = session_id
        
        # Process query
        result = orchestrator.process_chat_query(
            query=query,
            user_id=user_id,
            session_id=session_id,
            additional_context=data.get('context', {})
        )
        
        return jsonify(result)
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        return jsonify(orchestrator.get_performance_metrics())
    
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify(orchestrator.get_system_status())
    
    @app.route('/user/<user_id>/profile', methods=['GET'])
    def get_user_profile(user_id):
        if not orchestrator.enhanced_system_enabled:
            return jsonify({'error': 'User profiling not available'}), 503
        
        try:
            profile = orchestrator.user_profiling.get_user_profile(user_id)
            if profile:
                return jsonify({
                    'user_id': profile.user_id,
                    'preferences': {k: [p.__dict__ for p in v] for k, v in profile.preferences.items()},
                    'visited_locations': list(profile.visited_locations),
                    'total_interactions': profile.total_interactions,
                    'satisfaction_avg': sum(profile.satisfaction_scores) / len(profile.satisfaction_scores) if profile.satisfaction_scores else 0
                })
            else:
                return jsonify({'error': 'User profile not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

# Command-line interface
def main():
    """Main CLI for testing enhanced production integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Istanbul Production System')
    parser.add_argument('--mode', choices=['interactive', 'server', 'test'], default='interactive',
                        help='Run mode: interactive chat, server, or test')
    parser.add_argument('--port', type=int, default=5000, help='Server port (server mode only)')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if args.mode == 'server':
        # Run Flask server
        app = create_enhanced_flask_app()
        if app:
            print(f"🚀 Starting Enhanced AI Istanbul server on port {args.port}")
            app.run(host='0.0.0.0', port=args.port, debug=True)
        else:
            print("❌ Flask not available")
    
    elif args.mode == 'test':
        # Run system tests
        print("🧪 Running Enhanced AI Istanbul system tests...")
        orchestrator = EnhancedProductionOrchestrator(config)
        
        test_queries = [
            "What are the best historical sites in Istanbul?",
            "I'm vegetarian, where should I eat in Sultanahmet?",
            "How do I get from Taksim to Galata Tower?",
            "I love photography, what are the best spots?",
            "Plan a day itinerary for Kadıköy"
        ]
        
        test_user_id = "test_user_123"
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Test Query {i+1}: {query} ---")
            result = orchestrator.process_chat_query(query, test_user_id)
            print(f"✅ Success: {result['success']}")
            print(f"📝 Response: {result['response'][:100]}...")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"🔧 Source: {result['source']}")
        
        print(f"\n📊 Final Metrics:")
        metrics = orchestrator.get_performance_metrics()
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"GPT-Free Success Rate: {metrics.get('gpt_free_success_rate', 0):.2%}")
        print(f"Personalization Rate: {metrics.get('personalization_rate', 0):.2%}")
        print(f"Estimated Cost Savings: ${metrics['cost_savings_estimated']:.3f}")
    
    else:
        # Interactive mode
        print("🤖 Enhanced AI Istanbul Interactive Mode")
        print("Type 'exit' to quit, 'metrics' for performance stats, 'status' for system status")
        
        orchestrator = EnhancedProductionOrchestrator(config)
        user_id = input("Enter your user ID (or press Enter for anonymous): ").strip() or None
        
        while True:
            query = input(f"\n{'[' + user_id + ']' if user_id else '[Anonymous]'} Ask about Istanbul: ").strip()
            
            if query.lower() == 'exit':
                break
            elif query.lower() == 'metrics':
                metrics = orchestrator.get_performance_metrics()
                print(json.dumps(metrics, indent=2))
                continue
            elif query.lower() == 'status':
                status = orchestrator.get_system_status()
                print(json.dumps(status, indent=2))
                continue
            elif not query:
                continue
            
            result = orchestrator.process_chat_query(query, user_id)
            
            print(f"\n🤖 {result['response']}")
            if result.get('recommendations'):
                print(f"💡 Recommendations: {', '.join(result['recommendations'][:3])}")
            print(f"ℹ️  Confidence: {result['confidence']:.2f} | Source: {result['source']} | Time: {result.get('processing_time_ms', 0):.1f}ms")

if __name__ == "__main__":
    main()
