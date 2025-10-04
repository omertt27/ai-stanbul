"""
Enhanced System Setup for AI Istanbul GPT-Free Production System
Complete setup, validation, and testing for the enhanced system with personalization
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedSystemSetup:
    """Setup and validation for the enhanced AI Istanbul system"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.services_dir = self.base_dir / "backend" / "services"
        self.setup_results = {
            'dependencies': {'installed': [], 'failed': []},
            'directories': {'created': [], 'failed': []},
            'files': {'validated': [], 'missing': []},
            'databases': {'initialized': [], 'failed': []},
            'tests': {'passed': [], 'failed': []},
            'performance': {}
        }
        
    def run_complete_setup(self) -> Dict:
        """Run complete enhanced system setup"""
        print("🚀 Setting up Enhanced AI Istanbul GPT-Free System...")
        print("=" * 60)
        
        try:
            # Step 1: Check and install dependencies
            print("\n📦 Step 1: Checking Dependencies...")
            self._check_dependencies()
            
            # Step 2: Create required directories
            print("\n📁 Step 2: Creating Directories...")
            self._create_directories()
            
            # Step 3: Validate system files
            print("\n📄 Step 3: Validating System Files...")
            self._validate_system_files()
            
            # Step 4: Initialize databases
            print("\n🗄️ Step 4: Initializing Databases...")
            self._initialize_databases()
            
            # Step 5: Run system tests
            print("\n🧪 Step 5: Running System Tests...")
            self._run_system_tests()
            
            # Step 6: Performance validation
            print("\n⚡ Step 6: Performance Validation...")
            self._validate_performance()
            
            # Step 7: Generate setup report
            print("\n📊 Step 7: Generating Setup Report...")
            report = self._generate_setup_report()
            
            print("\n✅ Enhanced System Setup Complete!")
            return report
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            self.setup_results['setup_error'] = str(e)
            return self.setup_results
    
    def _check_dependencies(self):
        """Check and install required dependencies"""
        required_packages = [
            # Core ML packages
            ('sentence-transformers', 'sentence_transformers'),
            ('scikit-learn', 'sklearn'),
            ('faiss-cpu', 'faiss'),
            
            # Data processing
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            
            # Web framework (optional)
            ('flask', 'flask'),
            ('flask-cors', 'flask_cors'),
            
            # Database
            ('sqlite3', 'sqlite3'),  # Usually built-in
            
            # Text processing
            ('nltk', 'nltk'),
            ('spacy', 'spacy'),
        ]
        
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"✅ {package_name} - Available")
                self.setup_results['dependencies']['installed'].append(package_name)
            except ImportError:
                print(f"⚠️ {package_name} - Missing")
                self.setup_results['dependencies']['failed'].append(package_name)
                
                # Attempt installation
                try:
                    import subprocess
                    print(f"📦 Installing {package_name}...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
                    print(f"✅ {package_name} installed successfully")
                    self.setup_results['dependencies']['installed'].append(package_name)
                except Exception as e:
                    print(f"❌ Failed to install {package_name}: {e}")
        
        # Special handling for NLTK data
        try:
            import nltk
            print("📚 Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("✅ NLTK data downloaded")
        except Exception as e:
            print(f"⚠️ NLTK data download failed: {e}")
    
    def _create_directories(self):
        """Create required directories"""
        required_dirs = [
            'cache_data',
            'cache_data/embeddings',
            'cache_data/faiss_index',
            'clustering_data',
            'clustering_data/clusters',
            'clustering_data/templates',
            'user_data',
            'user_data/profiles',
            'training_data',
            'logs',
            'models'
        ]
        
        for dir_name in required_dirs:
            try:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
                self.setup_results['directories']['created'].append(str(dir_path))
            except Exception as e:
                print(f"❌ Failed to create directory {dir_name}: {e}")
                self.setup_results['directories']['failed'].append(dir_name)
    
    def _validate_system_files(self):
        """Validate that all required system files exist"""
        required_files = [
            'backend/services/enhanced_gpt_free_system.py',
            'backend/services/user_profiling_system.py',
            'backend/services/enhanced_intent_classifier.py',
            'backend/services/ml_semantic_cache.py',
            'backend/services/query_clustering_system.py',
            'backend/services/enhanced_production_integration.py'
        ]
        
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                print(f"✅ Found: {file_path}")
                self.setup_results['files']['validated'].append(str(full_path))
            else:
                print(f"❌ Missing: {file_path}")
                self.setup_results['files']['missing'].append(file_path)
    
    def _initialize_databases(self):
        """Initialize required databases"""
        try:
            # Initialize user profiles database
            db_path = self.base_dir / 'user_profiles.db'
            conn = sqlite3.connect(str(db_path))
            
            # User profiles table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User interactions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    response TEXT,
                    intent TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # User preferences table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_type TEXT,
                    preference_value TEXT,
                    confidence REAL,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print("✅ User profiles database initialized")
            self.setup_results['databases']['initialized'].append('user_profiles.db')
            
            # Initialize system metrics database
            metrics_db_path = self.base_dir / 'system_metrics.db'
            conn = sqlite3.connect(str(metrics_db_path))
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    response_source TEXT,
                    confidence REAL,
                    processing_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print("✅ System metrics database initialized")
            self.setup_results['databases']['initialized'].append('system_metrics.db')
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            self.setup_results['databases']['failed'].append(str(e))
    
    def _run_system_tests(self):
        """Run comprehensive system tests"""
        try:
            # Test 1: Import all modules
            print("🧪 Testing module imports...")
            modules_to_test = [
                'enhanced_gpt_free_system',
                'user_profiling_system',
                'enhanced_intent_classifier',
                'ml_semantic_cache',
                'query_clustering_system'
            ]
            
            # Add services directory to path
            services_path = str(self.base_dir / "backend" / "services")
            if services_path not in sys.path:
                sys.path.insert(0, services_path)
            
            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    print(f"✅ Import test passed: {module_name}")
                    self.setup_results['tests']['passed'].append(f"import_{module_name}")
                except Exception as e:
                    print(f"❌ Import test failed: {module_name} - {e}")
                    self.setup_results['tests']['failed'].append(f"import_{module_name}")
            
            # Test 2: Enhanced production integration
            print("🧪 Testing enhanced production integration...")
            try:
                from enhanced_production_integration import EnhancedProductionOrchestrator
                
                orchestrator = EnhancedProductionOrchestrator({
                    'cache_dir': str(self.base_dir / 'cache_data'),
                    'profile_db_path': str(self.base_dir / 'user_profiles.db')
                })
                
                # Test query processing
                test_query = "What are the best attractions in Istanbul?"
                result = orchestrator.process_chat_query(test_query, "test_user")
                
                if result.get('success'):
                    print("✅ Production integration test passed")
                    self.setup_results['tests']['passed'].append('production_integration')
                else:
                    print(f"⚠️ Production integration test returned: {result}")
                    self.setup_results['tests']['failed'].append('production_integration')
                
            except Exception as e:
                print(f"❌ Production integration test failed: {e}")
                self.setup_results['tests']['failed'].append('production_integration')
            
            # Test 3: User profiling system
            print("🧪 Testing user profiling system...")
            try:
                from user_profiling_system import UserProfilingSystem
                
                profiling = UserProfilingSystem(str(self.base_dir / 'user_profiles.db'))
                
                # Test profile creation and update
                test_interaction = {
                    'query': 'I love historical sites',
                    'response': 'Here are some historical sites...',
                    'intent': 'attractions',
                    'entities': {'attraction_type': 'historical'},
                    'timestamp': datetime.now(),
                    'satisfaction_score': 0.9
                }
                
                profiling.update_user_profile('test_user_profile', test_interaction, 'Test response')
                profile = profiling.get_user_profile('test_user_profile')
                
                if profile and profile.user_id == 'test_user_profile':
                    print("✅ User profiling test passed")
                    self.setup_results['tests']['passed'].append('user_profiling')
                else:
                    print("❌ User profiling test failed")
                    self.setup_results['tests']['failed'].append('user_profiling')
                
            except Exception as e:
                print(f"❌ User profiling test failed: {e}")
                self.setup_results['tests']['failed'].append('user_profiling')
            
            # Test 4: Intent classifier
            print("🧪 Testing enhanced intent classifier...")
            try:
                from enhanced_intent_classifier import EnhancedIntentClassifier
                
                classifier = EnhancedIntentClassifier()
                
                test_queries = [
                    "How do I get to Hagia Sophia?",
                    "Where can I eat Turkish food?",
                    "What are the best museums?"
                ]
                
                for query in test_queries:
                    result = classifier.classify_intent(query)
                    if result and result.confidence > 0:
                        intent_name = result.primary_intent.value if hasattr(result.primary_intent, 'value') else str(result.primary_intent)
                        print(f"✅ Intent classified: {query} -> {intent_name}")
                    else:
                        print(f"⚠️ Intent classification unclear: {query}")
                
                self.setup_results['tests']['passed'].append('intent_classifier')
                
            except Exception as e:
                print(f"❌ Intent classifier test failed: {e}")
                self.setup_results['tests']['failed'].append('intent_classifier')
            
        except Exception as e:
            print(f"❌ System tests failed: {e}")
            self.setup_results['tests']['failed'].append('system_tests_general')
    
    def _validate_performance(self):
        """Validate system performance"""
        try:
            print("⚡ Running performance validation...")
            
            # Performance test with multiple queries
            from enhanced_production_integration import EnhancedProductionOrchestrator
            
            orchestrator = EnhancedProductionOrchestrator({
                'cache_dir': str(self.base_dir / 'cache_data'),
                'profile_db_path': str(self.base_dir / 'user_profiles.db')
            })
            
            test_queries = [
                "What are the top 5 attractions in Istanbul?",
                "How do I get from airport to Sultanahmet?",
                "Where can I find good Turkish breakfast?",
                "What's the best time to visit Blue Mosque?",
                "Are there vegetarian restaurants in Taksim?"
            ]
            
            response_times = []
            success_count = 0
            
            for query in test_queries:
                start_time = time.time()
                result = orchestrator.process_chat_query(query, f"perf_test_user_{len(response_times)}")
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000  # ms
                response_times.append(processing_time)
                
                if result.get('success'):
                    success_count += 1
                
                print(f"⏱️ Query processed in {processing_time:.1f}ms - Success: {result.get('success', False)}")
            
            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            success_rate = success_count / len(test_queries)
            
            self.setup_results['performance'] = {
                'avg_response_time_ms': round(avg_response_time, 2),
                'success_rate': round(success_rate, 2),
                'queries_tested': len(test_queries),
                'total_successes': success_count,
                'response_times': response_times
            }
            
            print(f"📊 Performance Results:")
            print(f"   Average Response Time: {avg_response_time:.1f}ms")
            print(f"   Success Rate: {success_rate:.1%}")
            
            # Get system metrics
            metrics = orchestrator.get_performance_metrics()
            self.setup_results['performance']['system_metrics'] = metrics
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            self.setup_results['performance']['error'] = str(e)
    
    def _generate_setup_report(self) -> Dict:
        """Generate comprehensive setup report"""
        report = {
            'setup_timestamp': datetime.now().isoformat(),
            'setup_successful': True,
            'summary': {},
            'details': self.setup_results,
            'recommendations': [],
            'next_steps': []
        }
        
        # Calculate summary statistics
        total_deps = len(self.setup_results['dependencies']['installed']) + len(self.setup_results['dependencies']['failed'])
        successful_deps = len(self.setup_results['dependencies']['installed'])
        
        total_tests = len(self.setup_results['tests']['passed']) + len(self.setup_results['tests']['failed'])
        successful_tests = len(self.setup_results['tests']['passed'])
        
        report['summary'] = {
            'dependencies_success_rate': successful_deps / total_deps if total_deps > 0 else 0,
            'tests_success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'directories_created': len(self.setup_results['directories']['created']),
            'files_validated': len(self.setup_results['files']['validated']),
            'databases_initialized': len(self.setup_results['databases']['initialized'])
        }
        
        # Check for setup issues
        if self.setup_results['dependencies']['failed']:
            report['setup_successful'] = False
            report['recommendations'].append(
                f"Install missing dependencies: {', '.join(self.setup_results['dependencies']['failed'])}"
            )
        
        if self.setup_results['files']['missing']:
            report['setup_successful'] = False
            report['recommendations'].append(
                f"Create missing files: {', '.join(self.setup_results['files']['missing'])}"
            )
        
        if self.setup_results['tests']['failed']:
            report['recommendations'].append(
                f"Fix failing tests: {', '.join(self.setup_results['tests']['failed'])}"
            )
        
        # Add next steps
        if report['setup_successful']:
            report['next_steps'] = [
                "Run the enhanced production integration in test mode",
                "Configure production settings in environment variables",
                "Set up monitoring and logging for production deployment",
                "Train intent classification model with real data",
                "Implement user feedback collection for continuous improvement"
            ]
        else:
            report['next_steps'] = [
                "Address the issues mentioned in recommendations",
                "Re-run the setup script after fixing problems",
                "Check system logs for detailed error information"
            ]
        
        return report
    
    def save_setup_report(self, report: Dict, filename: str = None):
        """Save setup report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_system_setup_report_{timestamp}.json"
        
        report_path = self.base_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Setup report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"❌ Failed to save setup report: {e}")
            return None

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Istanbul System Setup')
    parser.add_argument('--base-dir', help='Base directory for the project')
    parser.add_argument('--save-report', action='store_true', help='Save detailed setup report')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run setup
    setup = EnhancedSystemSetup(args.base_dir)
    report = setup.run_complete_setup()
    
    # Print summary
    print(f"\n📋 Setup Summary:")
    print(f"✅ Dependencies Success Rate: {report['summary']['dependencies_success_rate']:.1%}")
    print(f"✅ Tests Success Rate: {report['summary']['tests_success_rate']:.1%}")
    print(f"📁 Directories Created: {report['summary']['directories_created']}")
    print(f"📄 Files Validated: {report['summary']['files_validated']}")
    print(f"🗄️ Databases Initialized: {report['summary']['databases_initialized']}")
    
    if 'performance' in report['details'] and 'avg_response_time_ms' in report['details']['performance']:
        perf = report['details']['performance']
        print(f"⚡ Average Response Time: {perf['avg_response_time_ms']}ms")
        print(f"📊 Success Rate: {perf['success_rate']:.1%}")
    
    # Save report if requested
    if args.save_report:
        setup.save_setup_report(report)
    
    # Print recommendations
    if report.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
    
    # Print next steps
    if report.get('next_steps'):
        print(f"\n🚀 Next Steps:")
        for step in report['next_steps']:
            print(f"   • {step}")
    
    if report['setup_successful']:
        print(f"\n🎉 Enhanced AI Istanbul System is ready for production!")
        print(f"Run 'python backend/services/enhanced_production_integration.py --mode interactive' to test")
    else:
        print(f"\n⚠️ Setup completed with issues. Please address the recommendations above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
