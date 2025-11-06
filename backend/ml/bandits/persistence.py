"""
Contextual Bandit State Persistence
Handles saving and loading bandit state to/from Redis
"""

import json
import logging
from typing import Optional, Dict, Any
import redis
from .contextual_thompson_sampling import ContextualThompsonSampling

logger = logging.getLogger(__name__)


class BanditStateManager:
    """
    Manages persistence of contextual bandit state
    
    Features:
    - Save/load bandit state to Redis
    - Periodic auto-save
    - State versioning
    - Backup and recovery
    """
    
    def __init__(self, redis_url: str, key_prefix: str = "bandit:"):
        """
        Initialize state manager
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis for bandit persistence")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _get_key(self, name: str) -> str:
        """Get full Redis key"""
        return f"{self.key_prefix}{name}"
    
    def save_bandit(
        self, 
        bandit: ContextualThompsonSampling, 
        name: str = "contextual_thompson"
    ) -> bool:
        """
        Save bandit state to Redis
        
        Args:
            bandit: Bandit instance to save
            name: Name for this bandit (allows multiple bandits)
        
        Returns:
            bool: True if successful
        """
        if not self.redis_client:
            logger.warning("Redis not available, cannot save bandit state")
            return False
        
        try:
            # Get bandit state
            state = bandit.save_state()
            
            # Save to Redis
            key = self._get_key(name)
            self.redis_client.set(key, json.dumps(state))
            
            # Also save metadata
            meta_key = self._get_key(f"{name}:meta")
            metadata = {
                'n_arms': state['n_arms'],
                'context_dim': state['context_dim'],
                'total_pulls': state['total_pulls'],
                'timestamp': state['timestamp']
            }
            self.redis_client.set(meta_key, json.dumps(metadata))
            
            logger.info(
                f"✅ Saved bandit '{name}' to Redis: "
                f"total_pulls={state['total_pulls']}, "
                f"timestamp={state['timestamp']}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save bandit state: {e}")
            return False
    
    def load_bandit(self, name: str = "contextual_thompson") -> Optional[ContextualThompsonSampling]:
        """
        Load bandit state from Redis
        
        Args:
            name: Name of the bandit to load
        
        Returns:
            ContextualThompsonSampling instance or None if not found
        """
        if not self.redis_client:
            logger.warning("Redis not available, cannot load bandit state")
            return None
        
        try:
            # Load from Redis
            key = self._get_key(name)
            state_json = self.redis_client.get(key)
            
            if not state_json:
                logger.info(f"No saved state found for bandit '{name}'")
                return None
            
            # Parse and restore
            state = json.loads(state_json)
            bandit = ContextualThompsonSampling.load_state(state)
            
            logger.info(
                f"✅ Loaded bandit '{name}' from Redis: "
                f"total_pulls={bandit.total_pulls}"
            )
            
            return bandit
            
        except Exception as e:
            logger.error(f"❌ Failed to load bandit state: {e}")
            return None
    
    def delete_bandit(self, name: str = "contextual_thompson") -> bool:
        """
        Delete bandit state from Redis
        
        Args:
            name: Name of the bandit to delete
        
        Returns:
            bool: True if successful
        """
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(name)
            meta_key = self._get_key(f"{name}:meta")
            
            self.redis_client.delete(key)
            self.redis_client.delete(meta_key)
            
            logger.info(f"✅ Deleted bandit '{name}' from Redis")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete bandit state: {e}")
            return False
    
    def list_bandits(self) -> list:
        """
        List all saved bandits
        
        Returns:
            List of bandit names
        """
        if not self.redis_client:
            return []
        
        try:
            pattern = f"{self.key_prefix}*:meta"
            keys = self.redis_client.keys(pattern)
            
            bandits = []
            for key in keys:
                # Extract name from key
                name = key.decode('utf-8').replace(self.key_prefix, '').replace(':meta', '')
                
                # Get metadata
                meta_json = self.redis_client.get(key)
                if meta_json:
                    metadata = json.loads(meta_json)
                    bandits.append({
                        'name': name,
                        'n_arms': metadata['n_arms'],
                        'context_dim': metadata['context_dim'],
                        'total_pulls': metadata['total_pulls'],
                        'timestamp': metadata['timestamp']
                    })
            
            return bandits
            
        except Exception as e:
            logger.error(f"❌ Failed to list bandits: {e}")
            return []
    
    def backup_bandit(self, name: str = "contextual_thompson") -> Optional[str]:
        """
        Create a backup of bandit state
        
        Args:
            name: Name of the bandit to backup
        
        Returns:
            Backup key name or None if failed
        """
        if not self.redis_client:
            return None
        
        try:
            from datetime import datetime
            
            # Get current state
            key = self._get_key(name)
            state_json = self.redis_client.get(key)
            
            if not state_json:
                logger.warning(f"No state found for bandit '{name}'")
                return None
            
            # Create backup key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key = self._get_key(f"{name}:backup:{timestamp}")
            
            # Save backup
            self.redis_client.set(backup_key, state_json)
            
            logger.info(f"✅ Created backup: {backup_key}")
            return backup_key
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return None
    
    def restore_from_backup(self, backup_key: str, name: str = "contextual_thompson") -> bool:
        """
        Restore bandit from backup
        
        Args:
            backup_key: Backup key to restore from
            name: Name to restore to
        
        Returns:
            bool: True if successful
        """
        if not self.redis_client:
            return False
        
        try:
            # Get backup
            backup_json = self.redis_client.get(backup_key)
            
            if not backup_json:
                logger.warning(f"Backup not found: {backup_key}")
                return False
            
            # Restore to main key
            key = self._get_key(name)
            self.redis_client.set(key, backup_json)
            
            # Update metadata
            state = json.loads(backup_json)
            meta_key = self._get_key(f"{name}:meta")
            metadata = {
                'n_arms': state['n_arms'],
                'context_dim': state['context_dim'],
                'total_pulls': state['total_pulls'],
                'timestamp': state['timestamp']
            }
            self.redis_client.set(meta_key, json.dumps(metadata))
            
            logger.info(f"✅ Restored from backup: {backup_key} → {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore from backup: {e}")
            return False
    
    def get_bandit_info(self, name: str = "contextual_thompson") -> Optional[Dict[str, Any]]:
        """
        Get information about a saved bandit
        
        Args:
            name: Name of the bandit
        
        Returns:
            Dictionary with bandit info or None
        """
        if not self.redis_client:
            return None
        
        try:
            meta_key = self._get_key(f"{name}:meta")
            meta_json = self.redis_client.get(meta_key)
            
            if not meta_json:
                return None
            
            return json.loads(meta_json)
            
        except Exception as e:
            logger.error(f"❌ Failed to get bandit info: {e}")
            return None


class PeriodicSaver:
    """
    Handles periodic saving of bandit state
    
    Usage:
        saver = PeriodicSaver(state_manager, bandit)
        await saver.start(interval=300)  # Save every 5 minutes
    """
    
    def __init__(
        self, 
        state_manager: BanditStateManager,
        bandit: ContextualThompsonSampling,
        name: str = "contextual_thompson"
    ):
        """
        Initialize periodic saver
        
        Args:
            state_manager: State manager instance
            bandit: Bandit to save
            name: Name for the bandit
        """
        self.state_manager = state_manager
        self.bandit = bandit
        self.name = name
        self.is_running = False
    
    async def start(self, interval: int = 300):
        """
        Start periodic saving
        
        Args:
            interval: Save interval in seconds (default: 5 minutes)
        """
        import asyncio
        
        self.is_running = True
        logger.info(f"🔄 Started periodic bandit save (interval={interval}s)")
        
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                if self.is_running:
                    success = self.state_manager.save_bandit(self.bandit, self.name)
                    if success:
                        logger.debug(f"✅ Periodic save completed for '{self.name}'")
                    
            except asyncio.CancelledError:
                logger.info("Periodic saver cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in periodic save: {e}")
    
    def stop(self):
        """Stop periodic saving"""
        self.is_running = False
        logger.info("🛑 Stopped periodic bandit save")


# Example usage
if __name__ == "__main__":
    import os
    from contextual_thompson_sampling import ContextualThompsonSampling, BanditContext
    import numpy as np
    
    # Initialize
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    state_manager = BanditStateManager(redis_url)
    
    # Create a bandit
    bandit = ContextualThompsonSampling(n_arms=10, context_dim=20)
    
    # Do some updates
    for i in range(10):
        context = BanditContext(
            user_features=np.random.randn(10),
            item_features=np.random.randn(5),
            temporal_features=np.random.randn(5)
        )
        arm = bandit.select_arm(context)
        bandit.update(arm, context, reward=np.random.rand())
    
    # Save
    state_manager.save_bandit(bandit, name="test_bandit")
    
    # Load
    loaded_bandit = state_manager.load_bandit(name="test_bandit")
    if loaded_bandit:
        print(f"Loaded bandit with {loaded_bandit.total_pulls} pulls")
    
    # List all bandits
    bandits = state_manager.list_bandits()
    print(f"Found {len(bandits)} saved bandits:")
    for b in bandits:
        print(f"  - {b['name']}: {b['total_pulls']} pulls")
    
    # Create backup
    backup_key = state_manager.backup_bandit(name="test_bandit")
    print(f"Created backup: {backup_key}")
    
    # Clean up
    state_manager.delete_bandit(name="test_bandit")
