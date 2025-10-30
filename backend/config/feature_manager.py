"""
Feature Manager - Centralized feature and module management
Simplifies the complex import logic in main.py
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import importlib
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FeaturePriority(Enum):
    """Feature priority levels"""
    CRITICAL = "critical"  # System fails without it
    REQUIRED = "required"  # Core functionality
    OPTIONAL = "optional"  # Nice to have
    EXPERIMENTAL = "experimental"  # Beta features


@dataclass
class FeatureModule:
    """Feature module configuration"""
    name: str
    module_path: str
    class_name: str
    enabled: bool = True
    priority: FeaturePriority = FeaturePriority.OPTIONAL
    fallback: Any = None
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"


class FeatureManager:
    """
    Centralized feature and module management.
    
    Benefits:
    - Single source of truth for features
    - Lazy loading with proper error handling
    - Clear dependency management
    - Easy to enable/disable features
    - Better error reporting
    
    Usage:
        manager = FeatureManager()
        manager.register_module(FeatureModule(
            name="location_detection",
            module_path="location_intent_detector",
            class_name="LocationIntentDetector",
            priority=FeaturePriority.REQUIRED
        ))
        
        location_detector = manager.get_feature("location_detection")
    """
    
    def __init__(self):
        """Initialize feature manager"""
        self.features: Dict[str, Any] = {}
        self.configs: Dict[str, FeatureModule] = {}
        self.failed_modules: List[str] = []
        self.loaded_modules: List[str] = []
        
        logger.info("🎛️ Feature Manager initialized")
    
    def register_module(self, feature: FeatureModule) -> Optional[Any]:
        """
        Register and lazy-load a module
        
        Args:
            feature: Feature module configuration
            
        Returns:
            Loaded module instance or fallback
            
        Raises:
            RuntimeError: If critical/required module fails to load
        """
        self.configs[feature.name] = feature
        
        # Check if disabled
        if not feature.enabled:
            logger.info(f"⏭️ Feature disabled: {feature.name}")
            return None
        
        # Check dependencies
        for dep in feature.dependencies:
            if dep not in self.loaded_modules:
                logger.warning(f"⚠️ Dependency missing for {feature.name}: {dep}")
                if feature.priority in [FeaturePriority.CRITICAL, FeaturePriority.REQUIRED]:
                    raise RuntimeError(f"Required dependency {dep} not loaded for {feature.name}")
                return feature.fallback
        
        # Try to load module
        try:
            logger.debug(f"📦 Loading module: {feature.name} from {feature.module_path}")
            module = importlib.import_module(feature.module_path)
            cls = getattr(module, feature.class_name)
            instance = cls()
            
            self.features[feature.name] = instance
            self.loaded_modules.append(feature.name)
            
            priority_emoji = {
                FeaturePriority.CRITICAL: "🔴",
                FeaturePriority.REQUIRED: "🟠",
                FeaturePriority.OPTIONAL: "🟢",
                FeaturePriority.EXPERIMENTAL: "🔵"
            }
            emoji = priority_emoji.get(feature.priority, "⚪")
            
            logger.info(f"{emoji} ✅ {feature.name} loaded ({feature.priority.value})")
            if feature.description:
                logger.debug(f"   📝 {feature.description}")
            
            return instance
            
        except Exception as e:
            self.failed_modules.append(feature.name)
            
            error_msg = f"{feature.name} failed to load: {str(e)}"
            
            # Handle based on priority
            if feature.priority == FeaturePriority.CRITICAL:
                logger.error(f"🔴 CRITICAL: {error_msg}")
                raise RuntimeError(f"Critical module {feature.name} failed to load: {e}")
            elif feature.priority == FeaturePriority.REQUIRED:
                logger.error(f"🟠 REQUIRED: {error_msg}")
                raise RuntimeError(f"Required module {feature.name} failed to load: {e}")
            else:
                logger.warning(f"⚠️ Optional module {error_msg}")
                return feature.fallback
    
    def register_many(self, features: List[FeatureModule]) -> Dict[str, Any]:
        """
        Register multiple modules
        
        Args:
            features: List of feature configurations
            
        Returns:
            Dictionary of loaded features
        """
        results = {}
        
        # Sort by priority (critical first)
        priority_order = [
            FeaturePriority.CRITICAL,
            FeaturePriority.REQUIRED,
            FeaturePriority.OPTIONAL,
            FeaturePriority.EXPERIMENTAL
        ]
        
        sorted_features = sorted(
            features,
            key=lambda f: priority_order.index(f.priority)
        )
        
        for feature in sorted_features:
            try:
                instance = self.register_module(feature)
                results[feature.name] = instance
            except Exception as e:
                logger.error(f"Failed to register {feature.name}: {e}")
                if feature.priority in [FeaturePriority.CRITICAL, FeaturePriority.REQUIRED]:
                    raise
        
        return results
    
    def get_feature(self, name: str, default: Any = None) -> Optional[Any]:
        """
        Get loaded feature by name
        
        Args:
            name: Feature name
            default: Default value if not found
            
        Returns:
            Feature instance or default
        """
        return self.features.get(name, default)
    
    def is_available(self, name: str) -> bool:
        """Check if feature is loaded and available"""
        return name in self.features and self.features[name] is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get system status report
        
        Returns:
            Dictionary with feature status
        """
        total = len(self.configs)
        loaded = len(self.loaded_modules)
        failed = len(self.failed_modules)
        
        by_priority = {
            FeaturePriority.CRITICAL: {"loaded": 0, "failed": 0},
            FeaturePriority.REQUIRED: {"loaded": 0, "failed": 0},
            FeaturePriority.OPTIONAL: {"loaded": 0, "failed": 0},
            FeaturePriority.EXPERIMENTAL: {"loaded": 0, "failed": 0}
        }
        
        for name, config in self.configs.items():
            if name in self.loaded_modules:
                by_priority[config.priority]["loaded"] += 1
            elif name in self.failed_modules:
                by_priority[config.priority]["failed"] += 1
        
        return {
            "total_features": total,
            "loaded": loaded,
            "failed": failed,
            "success_rate": f"{(loaded/total*100):.1f}%" if total > 0 else "0%",
            "by_priority": {
                priority.value: stats
                for priority, stats in by_priority.items()
            },
            "loaded_features": self.loaded_modules,
            "failed_features": self.failed_modules
        }
    
    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a feature
        
        Args:
            name: Feature name
            
        Returns:
            Feature information dictionary
        """
        if name not in self.configs:
            return None
        
        config = self.configs[name]
        
        return {
            "name": config.name,
            "module_path": config.module_path,
            "class_name": config.class_name,
            "enabled": config.enabled,
            "priority": config.priority.value,
            "description": config.description,
            "version": config.version,
            "dependencies": config.dependencies,
            "status": "loaded" if name in self.loaded_modules else "failed" if name in self.failed_modules else "not_loaded",
            "available": self.is_available(name)
        }
    
    def print_status(self) -> None:
        """Print formatted status report"""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("🎛️  FEATURE MANAGER STATUS")
        print("="*60)
        print(f"Total Features: {status['total_features']}")
        print(f"✅ Loaded: {status['loaded']}")
        print(f"❌ Failed: {status['failed']}")
        print(f"Success Rate: {status['success_rate']}")
        print("\n📊 By Priority:")
        
        for priority, stats in status['by_priority'].items():
            if stats['loaded'] > 0 or stats['failed'] > 0:
                print(f"  {priority.upper()}: {stats['loaded']} loaded, {stats['failed']} failed")
        
        if status['loaded_features']:
            print(f"\n✅ Loaded Features:")
            for feature in status['loaded_features']:
                print(f"  • {feature}")
        
        if status['failed_features']:
            print(f"\n❌ Failed Features:")
            for feature in status['failed_features']:
                print(f"  • {feature}")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing FeatureManager...")
    
    manager = FeatureManager()
    
    # Test 1: Register critical feature (should work with json)
    print("\n📝 Test 1: Register built-in module")
    try:
        manager.register_module(FeatureModule(
            name="json_parser",
            module_path="json",
            class_name="JSONEncoder",
            priority=FeaturePriority.CRITICAL,
            description="JSON encoding/decoding"
        ))
        assert manager.is_available("json_parser")
        print("✅ Test 1 passed")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Register optional feature that doesn't exist
    print("\n📝 Test 2: Register non-existent optional module")
    try:
        manager.register_module(FeatureModule(
            name="fake_module",
            module_path="nonexistent_module",
            class_name="FakeClass",
            priority=FeaturePriority.OPTIONAL,
            fallback=None
        ))
        assert not manager.is_available("fake_module")
        print("✅ Test 2 passed (gracefully handled)")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Get status
    print("\n📝 Test 3: Get status")
    manager.print_status()
    status = manager.get_status()
    assert status['total_features'] == 2
    assert status['loaded'] == 1
    assert status['failed'] == 1
    print("✅ Test 3 passed")
    
    # Test 4: Feature info
    print("\n📝 Test 4: Get feature info")
    info = manager.get_feature_info("json_parser")
    assert info is not None
    assert info['status'] == 'loaded'
    print(f"Feature info: {info}")
    print("✅ Test 4 passed")
    
    print("\n🎉 All tests passed!")
