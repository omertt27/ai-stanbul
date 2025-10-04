"""
Enhanced Configuration System for AI Istanbul GPT-Free Production
Centralized configuration management with environment variables and defaults
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for semantic cache system"""
    cache_dir: str = "cache_data"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_cache_size: int = 10000
    similarity_threshold: float = 0.85
    cleanup_interval_hours: int = 24
    enable_faiss: bool = True
    enable_tfidf_fallback: bool = True
    batch_size: int = 32

@dataclass
class UserProfilingConfig:
    """Configuration for user profiling system"""
    profile_db_path: str = "user_profiles.db"
    enable_profiling: bool = True
    preference_confidence_threshold: float = 0.6
    max_profile_history: int = 1000
    profile_cleanup_days: int = 90
    enable_behavioral_analysis: bool = True
    personalization_weight: float = 0.3

@dataclass
class IntentClassifierConfig:
    """Configuration for intent classification"""
    confidence_threshold: float = 0.6
    enable_ml_fallback: bool = True
    session_context_window: int = 5
    enable_entity_extraction: bool = True
    train_model_on_startup: bool = False
    model_update_interval_hours: int = 168  # 1 week

@dataclass
class QueryClusteringConfig:
    """Configuration for query clustering system"""
    clustering_dir: str = "clustering_data"
    min_cluster_size: int = 3
    similarity_threshold: float = 0.7
    max_clusters: int = 1000
    enable_auto_clustering: bool = True
    cluster_update_interval_hours: int = 24

@dataclass
class SystemConfig:
    """Main system configuration"""
    # System behavior
    gpt_free_confidence_threshold: float = 0.7
    enable_gpt_fallback: bool = False
    enable_personalization: bool = True
    enable_session_context: bool = True
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    enable_request_caching: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    enable_user_analytics: bool = True
    metrics_db_path: str = "system_metrics.db"
    
    # Security
    rate_limit_per_minute: int = 60
    enable_request_validation: bool = True
    max_query_length: int = 1000
    
    # Component configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    user_profiling: UserProfilingConfig = field(default_factory=UserProfilingConfig)
    intent_classifier: IntentClassifierConfig = field(default_factory=IntentClassifierConfig)
    query_clustering: QueryClusteringConfig = field(default_factory=QueryClusteringConfig)

class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = None, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_file = config_file
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> SystemConfig:
        """Load configuration from file and environment variables"""
        config_data = {}
        
        # Load from file if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"✅ Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config file: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config_data.update(env_config)
        
        # Create configuration objects
        return self._create_system_config(config_data)
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_mapping = {
            # System settings
            'AI_ISTANBUL_GPT_FREE_THRESHOLD': ('gpt_free_confidence_threshold', float),
            'AI_ISTANBUL_ENABLE_GPT_FALLBACK': ('enable_gpt_fallback', bool),
            'AI_ISTANBUL_ENABLE_PERSONALIZATION': ('enable_personalization', bool),
            'AI_ISTANBUL_LOG_LEVEL': ('log_level', str),
            'AI_ISTANBUL_MAX_CONCURRENT_REQUESTS': ('max_concurrent_requests', int),
            'AI_ISTANBUL_REQUEST_TIMEOUT': ('request_timeout_seconds', int),
            
            # Cache settings
            'AI_ISTANBUL_CACHE_DIR': ('cache.cache_dir', str),
            'AI_ISTANBUL_EMBEDDING_MODEL': ('cache.embedding_model', str),
            'AI_ISTANBUL_CACHE_SIZE': ('cache.max_cache_size', int),
            'AI_ISTANBUL_SIMILARITY_THRESHOLD': ('cache.similarity_threshold', float),
            'AI_ISTANBUL_ENABLE_FAISS': ('cache.enable_faiss', bool),
            
            # User profiling settings
            'AI_ISTANBUL_PROFILE_DB_PATH': ('user_profiling.profile_db_path', str),
            'AI_ISTANBUL_ENABLE_PROFILING': ('user_profiling.enable_profiling', bool),
            'AI_ISTANBUL_PREFERENCE_THRESHOLD': ('user_profiling.preference_confidence_threshold', float),
            'AI_ISTANBUL_PERSONALIZATION_WEIGHT': ('user_profiling.personalization_weight', float),
            
            # Intent classifier settings
            'AI_ISTANBUL_INTENT_THRESHOLD': ('intent_classifier.confidence_threshold', float),
            'AI_ISTANBUL_ENABLE_ML_INTENT': ('intent_classifier.enable_ml_fallback', bool),
            'AI_ISTANBUL_SESSION_CONTEXT_WINDOW': ('intent_classifier.session_context_window', int),
            'AI_ISTANBUL_TRAIN_MODEL_ON_STARTUP': ('intent_classifier.train_model_on_startup', bool),
            
            # Clustering settings
            'AI_ISTANBUL_CLUSTERING_DIR': ('query_clustering.clustering_dir', str),
            'AI_ISTANBUL_MIN_CLUSTER_SIZE': ('query_clustering.min_cluster_size', int),
            'AI_ISTANBUL_MAX_CLUSTERS': ('query_clustering.max_clusters', int),
            'AI_ISTANBUL_ENABLE_AUTO_CLUSTERING': ('query_clustering.enable_auto_clustering', bool),
            
            # Security settings
            'AI_ISTANBUL_RATE_LIMIT': ('rate_limit_per_minute', int),
            'AI_ISTANBUL_MAX_QUERY_LENGTH': ('max_query_length', int),
            'AI_ISTANBUL_ENABLE_REQUEST_VALIDATION': ('enable_request_validation', bool)
        }
        
        config = {}
        
        for env_var, (config_key, value_type) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        parsed_value = int(value)
                    elif value_type == float:
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                    
                    # Set nested configuration
                    self._set_nested_config(config, config_key, parsed_value)
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Invalid value for {env_var}: {value} ({e})")
        
        return config
    
    def _set_nested_config(self, config: Dict, key: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _create_system_config(self, config_data: Dict) -> SystemConfig:
        """Create SystemConfig object from configuration data"""
        
        # Cache configuration
        cache_config = CacheConfig(
            cache_dir=config_data.get('cache', {}).get('cache_dir', CacheConfig.cache_dir),
            embedding_model=config_data.get('cache', {}).get('embedding_model', CacheConfig.embedding_model),
            max_cache_size=config_data.get('cache', {}).get('max_cache_size', CacheConfig.max_cache_size),
            similarity_threshold=config_data.get('cache', {}).get('similarity_threshold', CacheConfig.similarity_threshold),
            cleanup_interval_hours=config_data.get('cache', {}).get('cleanup_interval_hours', CacheConfig.cleanup_interval_hours),
            enable_faiss=config_data.get('cache', {}).get('enable_faiss', CacheConfig.enable_faiss),
            enable_tfidf_fallback=config_data.get('cache', {}).get('enable_tfidf_fallback', CacheConfig.enable_tfidf_fallback),
            batch_size=config_data.get('cache', {}).get('batch_size', CacheConfig.batch_size)
        )
        
        # User profiling configuration
        user_profiling_config = UserProfilingConfig(
            profile_db_path=config_data.get('user_profiling', {}).get('profile_db_path', UserProfilingConfig.profile_db_path),
            enable_profiling=config_data.get('user_profiling', {}).get('enable_profiling', UserProfilingConfig.enable_profiling),
            preference_confidence_threshold=config_data.get('user_profiling', {}).get('preference_confidence_threshold', UserProfilingConfig.preference_confidence_threshold),
            max_profile_history=config_data.get('user_profiling', {}).get('max_profile_history', UserProfilingConfig.max_profile_history),
            profile_cleanup_days=config_data.get('user_profiling', {}).get('profile_cleanup_days', UserProfilingConfig.profile_cleanup_days),
            enable_behavioral_analysis=config_data.get('user_profiling', {}).get('enable_behavioral_analysis', UserProfilingConfig.enable_behavioral_analysis),
            personalization_weight=config_data.get('user_profiling', {}).get('personalization_weight', UserProfilingConfig.personalization_weight)
        )
        
        # Intent classifier configuration
        intent_classifier_config = IntentClassifierConfig(
            confidence_threshold=config_data.get('intent_classifier', {}).get('confidence_threshold', IntentClassifierConfig.confidence_threshold),
            enable_ml_fallback=config_data.get('intent_classifier', {}).get('enable_ml_fallback', IntentClassifierConfig.enable_ml_fallback),
            session_context_window=config_data.get('intent_classifier', {}).get('session_context_window', IntentClassifierConfig.session_context_window),
            enable_entity_extraction=config_data.get('intent_classifier', {}).get('enable_entity_extraction', IntentClassifierConfig.enable_entity_extraction),
            train_model_on_startup=config_data.get('intent_classifier', {}).get('train_model_on_startup', IntentClassifierConfig.train_model_on_startup),
            model_update_interval_hours=config_data.get('intent_classifier', {}).get('model_update_interval_hours', IntentClassifierConfig.model_update_interval_hours)
        )
        
        # Query clustering configuration
        query_clustering_config = QueryClusteringConfig(
            clustering_dir=config_data.get('query_clustering', {}).get('clustering_dir', QueryClusteringConfig.clustering_dir),
            min_cluster_size=config_data.get('query_clustering', {}).get('min_cluster_size', QueryClusteringConfig.min_cluster_size),
            similarity_threshold=config_data.get('query_clustering', {}).get('similarity_threshold', QueryClusteringConfig.similarity_threshold),
            max_clusters=config_data.get('query_clustering', {}).get('max_clusters', QueryClusteringConfig.max_clusters),
            enable_auto_clustering=config_data.get('query_clustering', {}).get('enable_auto_clustering', QueryClusteringConfig.enable_auto_clustering),
            cluster_update_interval_hours=config_data.get('query_clustering', {}).get('cluster_update_interval_hours', QueryClusteringConfig.cluster_update_interval_hours)
        )
        
        # Main system configuration
        system_config = SystemConfig(
            gpt_free_confidence_threshold=config_data.get('gpt_free_confidence_threshold', SystemConfig.gpt_free_confidence_threshold),
            enable_gpt_fallback=config_data.get('enable_gpt_fallback', SystemConfig.enable_gpt_fallback),
            enable_personalization=config_data.get('enable_personalization', SystemConfig.enable_personalization),
            enable_session_context=config_data.get('enable_session_context', SystemConfig.enable_session_context),
            max_concurrent_requests=config_data.get('max_concurrent_requests', SystemConfig.max_concurrent_requests),
            request_timeout_seconds=config_data.get('request_timeout_seconds', SystemConfig.request_timeout_seconds),
            enable_request_caching=config_data.get('enable_request_caching', SystemConfig.enable_request_caching),
            log_level=config_data.get('log_level', SystemConfig.log_level),
            enable_performance_monitoring=config_data.get('enable_performance_monitoring', SystemConfig.enable_performance_monitoring),
            enable_user_analytics=config_data.get('enable_user_analytics', SystemConfig.enable_user_analytics),
            metrics_db_path=config_data.get('metrics_db_path', SystemConfig.metrics_db_path),
            rate_limit_per_minute=config_data.get('rate_limit_per_minute', SystemConfig.rate_limit_per_minute),
            enable_request_validation=config_data.get('enable_request_validation', SystemConfig.enable_request_validation),
            max_query_length=config_data.get('max_query_length', SystemConfig.max_query_length),
            cache=cache_config,
            user_profiling=user_profiling_config,
            intent_classifier=intent_classifier_config,
            query_clustering=query_clustering_config
        )
        
        return system_config
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration"""
        return self.config
    
    def save_config(self, filename: str = None) -> str:
        """Save current configuration to file"""
        if not filename:
            filename = str(self.base_dir / "ai_istanbul_config.json")
        
        config_dict = self._config_to_dict(self.config)
        
        try:
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise
    
    def _config_to_dict(self, config: SystemConfig) -> Dict:
        """Convert SystemConfig to dictionary"""
        return {
            'gpt_free_confidence_threshold': config.gpt_free_confidence_threshold,
            'enable_gpt_fallback': config.enable_gpt_fallback,
            'enable_personalization': config.enable_personalization,
            'enable_session_context': config.enable_session_context,
            'max_concurrent_requests': config.max_concurrent_requests,
            'request_timeout_seconds': config.request_timeout_seconds,
            'enable_request_caching': config.enable_request_caching,
            'log_level': config.log_level,
            'enable_performance_monitoring': config.enable_performance_monitoring,
            'enable_user_analytics': config.enable_user_analytics,
            'metrics_db_path': config.metrics_db_path,
            'rate_limit_per_minute': config.rate_limit_per_minute,
            'enable_request_validation': config.enable_request_validation,
            'max_query_length': config.max_query_length,
            'cache': {
                'cache_dir': config.cache.cache_dir,
                'embedding_model': config.cache.embedding_model,
                'max_cache_size': config.cache.max_cache_size,
                'similarity_threshold': config.cache.similarity_threshold,
                'cleanup_interval_hours': config.cache.cleanup_interval_hours,
                'enable_faiss': config.cache.enable_faiss,
                'enable_tfidf_fallback': config.cache.enable_tfidf_fallback,
                'batch_size': config.cache.batch_size
            },
            'user_profiling': {
                'profile_db_path': config.user_profiling.profile_db_path,
                'enable_profiling': config.user_profiling.enable_profiling,
                'preference_confidence_threshold': config.user_profiling.preference_confidence_threshold,
                'max_profile_history': config.user_profiling.max_profile_history,
                'profile_cleanup_days': config.user_profiling.profile_cleanup_days,
                'enable_behavioral_analysis': config.user_profiling.enable_behavioral_analysis,
                'personalization_weight': config.user_profiling.personalization_weight
            },
            'intent_classifier': {
                'confidence_threshold': config.intent_classifier.confidence_threshold,
                'enable_ml_fallback': config.intent_classifier.enable_ml_fallback,
                'session_context_window': config.intent_classifier.session_context_window,
                'enable_entity_extraction': config.intent_classifier.enable_entity_extraction,
                'train_model_on_startup': config.intent_classifier.train_model_on_startup,
                'model_update_interval_hours': config.intent_classifier.model_update_interval_hours
            },
            'query_clustering': {
                'clustering_dir': config.query_clustering.clustering_dir,
                'min_cluster_size': config.query_clustering.min_cluster_size,
                'similarity_threshold': config.query_clustering.similarity_threshold,
                'max_clusters': config.query_clustering.max_clusters,
                'enable_auto_clustering': config.query_clustering.enable_auto_clustering,
                'cluster_update_interval_hours': config.query_clustering.cluster_update_interval_hours
            }
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        config_dict = self._config_to_dict(self.config)
        
        # Apply updates using dot notation
        for key, value in updates.items():
            self._set_nested_config(config_dict, key, value)
        
        # Recreate configuration
        self.config = self._create_system_config(config_dict)
        logger.info(f"✅ Configuration updated with {len(updates)} changes")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate thresholds
        if not (0.0 <= self.config.gpt_free_confidence_threshold <= 1.0):
            issues.append("gpt_free_confidence_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.config.cache.similarity_threshold <= 1.0):
            issues.append("cache.similarity_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.config.user_profiling.preference_confidence_threshold <= 1.0):
            issues.append("user_profiling.preference_confidence_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.config.intent_classifier.confidence_threshold <= 1.0):
            issues.append("intent_classifier.confidence_threshold must be between 0.0 and 1.0")
        
        # Validate positive integers
        if self.config.max_concurrent_requests <= 0:
            issues.append("max_concurrent_requests must be positive")
        
        if self.config.cache.max_cache_size <= 0:
            issues.append("cache.max_cache_size must be positive")
        
        if self.config.query_clustering.min_cluster_size <= 0:
            issues.append("query_clustering.min_cluster_size must be positive")
        
        # Validate paths exist or can be created
        cache_dir = Path(self.config.cache.cache_dir)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create cache directory: {e}")
        
        clustering_dir = Path(self.config.query_clustering.clustering_dir)
        try:
            clustering_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create clustering directory: {e}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level not in valid_log_levels:
            issues.append(f"log_level must be one of: {valid_log_levels}")
        
        return issues
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("🔧 AI Istanbul Enhanced System Configuration")
        print("=" * 50)
        
        print(f"System Settings:")
        print(f"  GPT-Free Threshold: {self.config.gpt_free_confidence_threshold}")
        print(f"  Personalization: {'Enabled' if self.config.enable_personalization else 'Disabled'}")
        print(f"  GPT Fallback: {'Enabled' if self.config.enable_gpt_fallback else 'Disabled'}")
        print(f"  Log Level: {self.config.log_level}")
        
        print(f"\nCache Settings:")
        print(f"  Directory: {self.config.cache.cache_dir}")
        print(f"  Embedding Model: {self.config.cache.embedding_model}")
        print(f"  Max Size: {self.config.cache.max_cache_size}")
        print(f"  Similarity Threshold: {self.config.cache.similarity_threshold}")
        print(f"  FAISS Enabled: {self.config.cache.enable_faiss}")
        
        print(f"\nUser Profiling:")
        print(f"  Database: {self.config.user_profiling.profile_db_path}")
        print(f"  Enabled: {self.config.user_profiling.enable_profiling}")
        print(f"  Confidence Threshold: {self.config.user_profiling.preference_confidence_threshold}")
        print(f"  Personalization Weight: {self.config.user_profiling.personalization_weight}")
        
        print(f"\nIntent Classification:")
        print(f"  Confidence Threshold: {self.config.intent_classifier.confidence_threshold}")
        print(f"  ML Fallback: {self.config.intent_classifier.enable_ml_fallback}")
        print(f"  Session Context Window: {self.config.intent_classifier.session_context_window}")
        
        print(f"\nQuery Clustering:")
        print(f"  Directory: {self.config.query_clustering.clustering_dir}")
        print(f"  Min Cluster Size: {self.config.query_clustering.min_cluster_size}")
        print(f"  Auto Clustering: {self.config.query_clustering.enable_auto_clustering}")
        
        # Validation results
        issues = self.validate_config()
        if issues:
            print(f"\n⚠️ Configuration Issues:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ Configuration is valid")

def create_sample_config():
    """Create a sample configuration file"""
    config_manager = ConfigurationManager()
    sample_config_path = config_manager.save_config("ai_istanbul_config_sample.json")
    
    print(f"📄 Sample configuration created at: {sample_config_path}")
    print("You can customize this file and use it with:")
    print("  export AI_ISTANBUL_CONFIG_FILE=ai_istanbul_config_sample.json")
    
    return sample_config_path

def main():
    """Main configuration CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Istanbul Configuration Manager')
    parser.add_argument('--config-file', help='Configuration file path')
    parser.add_argument('--base-dir', help='Base directory for the project')
    parser.add_argument('--create-sample', action='store_true', help='Create sample configuration file')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--print-summary', action='store_true', help='Print configuration summary')
    parser.add_argument('--save', help='Save configuration to file')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    # Load configuration
    config_manager = ConfigurationManager(args.config_file, args.base_dir)
    
    if args.validate:
        issues = config_manager.validate_config()
        if issues:
            print("❌ Configuration validation failed:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("✅ Configuration is valid")
    
    if args.print_summary:
        config_manager.print_config_summary()
    
    if args.save:
        config_manager.save_config(args.save)

if __name__ == "__main__":
    main()
