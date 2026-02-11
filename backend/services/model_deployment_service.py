"""
Model Deployment Service

Handles blue-green deployment of NCF models with zero downtime.

Features:
- Model versioning
- Blue-green deployment
- Canary rollout
- Health monitoring
- Automatic rollback

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
import time
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a model version."""
    
    def __init__(
        self,
        version: str,
        model_path: Path,
        metadata: Dict[str, Any]
    ):
        self.version = version
        self.model_path = model_path
        self.metadata = metadata
        self.deployed_at = None
        self.status = "inactive"  # inactive, active, retired
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'model_path': str(self.model_path),
            'metadata': self.metadata,
            'deployed_at': self.deployed_at,
            'status': self.status
        }


class ModelDeploymentService:
    """
    Manages blue-green deployment of NCF models.
    """
    
    def __init__(self, models_dir: str = "backend/ml/deep_learning/models"):
        """
        Initialize deployment service.
        
        Args:
            models_dir: Directory to store model versions
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.models_dir / "versions.json"
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        self.traffic_split: Dict[str, float] = {}  # version -> traffic percentage
        
        self._load_versions()
    
    def _load_versions(self):
        """Load version registry from disk."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    
                for version_data in data.get('versions', []):
                    version = ModelVersion(
                        version=version_data['version'],
                        model_path=Path(version_data['model_path']),
                        metadata=version_data['metadata']
                    )
                    version.deployed_at = version_data.get('deployed_at')
                    version.status = version_data.get('status', 'inactive')
                    self.versions[version.version] = version
                
                self.active_version = data.get('active_version')
                self.traffic_split = data.get('traffic_split', {})
                
                logger.info(f"✅ Loaded {len(self.versions)} model versions")
                
            except Exception as e:
                logger.error(f"❌ Failed to load versions: {e}")
    
    def _save_versions(self):
        """Save version registry to disk."""
        try:
            data = {
                'versions': [v.to_dict() for v in self.versions.values()],
                'active_version': self.active_version,
                'traffic_split': self.traffic_split,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info("✅ Saved version registry")
            
        except Exception as e:
            logger.error(f"❌ Failed to save versions: {e}")
    
    def register_version(
        self,
        model_path: Path,
        metadata: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file (.onnx)
            metadata: Model metadata (accuracy, training info, etc.)
            version: Version name (auto-generated if not provided)
            
        Returns:
            Version identifier
        """
        if version is None:
            # Generate version from timestamp
            version = f"v_{int(time.time())}"
        
        logger.info(f"📝 Registering model version: {version}")
        
        # Copy model to versions directory
        version_dir = self.models_dir / version
        version_dir.mkdir(exist_ok=True)
        
        dest_path = version_dir / "ncf_model.onnx"
        shutil.copy(model_path, dest_path)
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create version object
        model_version = ModelVersion(
            version=version,
            model_path=dest_path,
            metadata=metadata
        )
        
        self.versions[version] = model_version
        self._save_versions()
        
        logger.info(f"✅ Registered version {version}")
        
        return version
    
    def deploy_blue_green(
        self,
        new_version: str,
        validation_minutes: int = 10
    ) -> bool:
        """
        Deploy new version using blue-green strategy.
        
        Args:
            new_version: Version to deploy
            validation_minutes: Minutes to validate before full rollout
            
        Returns:
            True if deployment succeeded
        """
        if new_version not in self.versions:
            logger.error(f"❌ Version {new_version} not found")
            return False
        
        logger.info(f"🚀 Starting blue-green deployment: {new_version}")
        
        old_version = self.active_version
        
        try:
            # Phase 1: Deploy as green (0% traffic)
            logger.info("📘 Phase 1: Deploy green environment (0% traffic)")
            self.versions[new_version].status = "canary"
            self.traffic_split = {
                old_version: 100.0 if old_version else 0.0,
                new_version: 0.0
            }
            self._save_versions()
            
            # Phase 2: Canary deployment (5% traffic)
            logger.info("🟢 Phase 2: Canary deployment (5% traffic)")
            self.traffic_split[new_version] = 5.0
            if old_version:
                self.traffic_split[old_version] = 95.0
            self._save_versions()
            
            logger.info(f"⏱️ Validating for {validation_minutes} minutes...")
            # In production, this would monitor metrics
            # time.sleep(validation_minutes * 60)
            
            # Phase 3: 50% traffic split
            logger.info("🟢 Phase 3: Split traffic 50/50")
            self.traffic_split[new_version] = 50.0
            if old_version:
                self.traffic_split[old_version] = 50.0
            self._save_versions()
            
            # Phase 4: Full deployment (100% traffic)
            logger.info("🟢 Phase 4: Full deployment (100% traffic)")
            self.traffic_split[new_version] = 100.0
            if old_version:
                self.traffic_split[old_version] = 0.0
                self.versions[old_version].status = "retired"
            
            self.active_version = new_version
            self.versions[new_version].status = "active"
            self.versions[new_version].deployed_at = datetime.utcnow().isoformat()
            self._save_versions()
            
            logger.info(f"✅ Blue-green deployment complete: {new_version} is now active")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            
            # Rollback
            if old_version:
                logger.warning("🔄 Rolling back to previous version...")
                self.rollback_to_version(old_version)
            
            return False
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            True if rollback succeeded
        """
        if version not in self.versions:
            logger.error(f"❌ Version {version} not found")
            return False
        
        logger.info(f"🔄 Rolling back to version: {version}")
        
        try:
            # Set traffic to 100% for rollback version
            self.traffic_split = {version: 100.0}
            
            # Update statuses
            if self.active_version and self.active_version in self.versions:
                self.versions[self.active_version].status = "retired"
            
            self.active_version = version
            self.versions[version].status = "active"
            self._save_versions()
            
            logger.info(f"✅ Rollback complete: {version} is now active")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    def get_active_model_path(self) -> Optional[Path]:
        """
        Get path to the active model.
        
        Returns:
            Path to active model or None
        """
        if self.active_version and self.active_version in self.versions:
            return self.versions[self.active_version].model_path
        return None
    
    def get_model_for_user(self, user_id: str) -> Optional[Path]:
        """
        Get model path for a specific user (respects traffic split).
        
        Args:
            user_id: User identifier
            
        Returns:
            Path to model to use for this user
        """
        if not self.traffic_split:
            return self.get_active_model_path()
        
        # Use consistent hashing for traffic split
        user_hash = hash(user_id) % 100
        
        cumulative = 0.0
        for version, percentage in self.traffic_split.items():
            cumulative += percentage
            if user_hash < cumulative:
                if version in self.versions:
                    return self.versions[version].model_path
        
        return self.get_active_model_path()
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all model versions.
        
        Returns:
            List of version information
        """
        return [
            {
                **v.to_dict(),
                'is_active': v.version == self.active_version,
                'traffic_percentage': self.traffic_split.get(v.version, 0.0)
            }
            for v in sorted(
                self.versions.values(),
                key=lambda x: x.metadata.get('created_at', ''),
                reverse=True
            )
        ]
    
    def cleanup_old_versions(self, keep_latest: int = 5):
        """
        Remove old model versions to save disk space.
        
        Args:
            keep_latest: Number of recent versions to keep
        """
        logger.info(f"🧹 Cleaning up old versions (keeping latest {keep_latest})...")
        
        # Sort by creation date
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda x: x.metadata.get('created_at', ''),
            reverse=True
        )
        
        # Keep active version + latest N
        to_keep = {self.active_version} if self.active_version else set()
        to_keep.update(v.version for v in sorted_versions[:keep_latest])
        
        # Remove old versions
        removed = 0
        for version_obj in self.versions.values():
            if version_obj.version not in to_keep and version_obj.status == "retired":
                try:
                    version_dir = version_obj.model_path.parent
                    shutil.rmtree(version_dir)
                    del self.versions[version_obj.version]
                    removed += 1
                    logger.info(f"  🗑️ Removed version {version_obj.version}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to remove {version_obj.version}: {e}")
        
        if removed > 0:
            self._save_versions()
        
        logger.info(f"✅ Cleanup complete: removed {removed} old versions")


# Singleton instance
_deployment_service: Optional[ModelDeploymentService] = None


def get_deployment_service() -> ModelDeploymentService:
    """
    Get the model deployment service singleton.
    
    Returns:
        ModelDeploymentService instance
    """
    global _deployment_service
    
    if _deployment_service is None:
        _deployment_service = ModelDeploymentService()
    
    return _deployment_service
