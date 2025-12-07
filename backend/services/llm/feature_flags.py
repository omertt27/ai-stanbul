"""
feature_flags.py - Feature Flag Management System

Gradual feature rollout and testing with:
- Percentage-based rollouts
- User targeting (whitelist/blacklist)
- Context-based rules
- Redis-backed caching
- Real-time flag updates

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RuleOperator(Enum):
    """Operators for context-based rules."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    enabled: bool
    rollout_percentage: int = 100
    description: str = ""
    whitelist: List[str] = None
    blacklist: List[str] = None
    rules: List[Dict[str, Any]] = None
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.whitelist is None:
            self.whitelist = []
        if self.blacklist is None:
            self.blacklist = []
        if self.rules is None:
            self.rules = []
        
        import time
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()


class FeatureFlagManager:
    """
    Manage feature flags for gradual rollout and testing.
    
    Features:
    - Global enable/disable
    - Percentage-based rollout
    - User whitelist/blacklist
    - Context-based rules
    - Redis caching
    - Real-time updates
    
    Example:
        >>> manager = FeatureFlagManager(redis_client)
        >>> await manager.create_flag(
        ...     name="multi_pass_detection",
        ...     description="Enable multi-pass signal detection",
        ...     rollout_percentage=10  # 10% of users
        ... )
        >>> 
        >>> # Check if enabled for user
        >>> enabled = await manager.is_enabled(
        ...     "multi_pass_detection",
        ...     user_id="user_123"
        ... )
    """
    
    def __init__(self, redis_client=None, db_connection=None):
        """
        Initialize feature flag manager.
        
        Args:
            redis_client: Redis client for caching (optional)
            db_connection: Database connection (optional)
        """
        self.redis = redis_client
        self.db = db_connection
        
        # In-memory storage (fallback if no Redis)
        self.flags: Dict[str, FeatureFlag] = {}
        
        logger.info("✅ FeatureFlagManager initialized")
    
    async def create_flag(
        self,
        name: str,
        description: str = "",
        enabled: bool = True,
        rollout_percentage: int = 100,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
        rules: Optional[List[Dict[str, Any]]] = None
    ) -> FeatureFlag:
        """
        Create a new feature flag.
        
        Args:
            name: Flag name (unique identifier)
            description: Human-readable description
            enabled: Whether flag is globally enabled
            rollout_percentage: Percentage of users to enable (0-100)
            whitelist: List of user IDs to always enable
            blacklist: List of user IDs to always disable
            rules: Context-based rules for conditional enabling
        
        Returns:
            Created FeatureFlag
        """
        flag = FeatureFlag(
            name=name,
            description=description,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            whitelist=whitelist or [],
            blacklist=blacklist or [],
            rules=rules or []
        )
        
        # Store in memory
        self.flags[name] = flag
        
        # Persist to Redis if available
        if self.redis:
            await self._save_flag_to_redis(flag)
        
        # Persist to database if available
        if self.db:
            await self._save_flag_to_db(flag)
        
        logger.info(f"✅ Created feature flag: {name} (rollout: {rollout_percentage}%)")
        return flag
    
    async def update_flag(
        self,
        name: str,
        **kwargs
    ) -> Optional[FeatureFlag]:
        """
        Update an existing feature flag.
        
        Args:
            name: Flag name
            **kwargs: Fields to update
        
        Returns:
            Updated FeatureFlag or None if not found
        """
        flag = await self._get_flag(name)
        if not flag:
            logger.warning(f"Feature flag not found: {name}")
            return None
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        import time
        flag.updated_at = time.time()
        
        # Update storage
        self.flags[name] = flag
        
        if self.redis:
            await self._save_flag_to_redis(flag)
        
        if self.db:
            await self._update_flag_in_db(flag)
        
        logger.info(f"✅ Updated feature flag: {name}")
        return flag
    
    async def delete_flag(self, name: str) -> bool:
        """
        Delete a feature flag.
        
        Args:
            name: Flag name
        
        Returns:
            True if deleted, False if not found
        """
        if name not in self.flags:
            return False
        
        # Remove from memory
        del self.flags[name]
        
        # Remove from Redis
        if self.redis:
            try:
                await self.redis.delete(f"feature_flag:{name}")
            except Exception as e:
                logger.warning(f"Redis error: {e}")
        
        # Remove from database
        if self.db:
            try:
                await self.db.execute(
                    "DELETE FROM feature_flags WHERE name = ?",
                    (name,)
                )
            except Exception as e:
                logger.warning(f"Database error: {e}")
        
        logger.info(f"🗑️ Deleted feature flag: {name}")
        return True
    
    async def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature is enabled for user/context.
        
        Evaluation order:
        1. Global enable/disable check
        2. Blacklist check (if user_id provided)
        3. Whitelist check (if user_id provided)
        4. Percentage rollout (if user_id provided)
        5. Context rules (if context provided)
        
        Args:
            flag_name: Name of feature flag
            user_id: User identifier (optional)
            context: Context dict for rule evaluation (optional)
        
        Returns:
            True if enabled, False otherwise
        """
        flag = await self._get_flag(flag_name)
        if not flag:
            # Flag doesn't exist, default to disabled
            return False
        
        # 1. Global enable/disable
        if not flag.enabled:
            return False
        
        # 2. Check blacklist first
        if user_id and flag.blacklist:
            if user_id in flag.blacklist:
                logger.debug(f"User {user_id} in blacklist for flag {flag_name}")
                return False
        
        # 3. Check whitelist
        if user_id and flag.whitelist:
            if user_id in flag.whitelist:
                logger.debug(f"User {user_id} in whitelist for flag {flag_name}")
                return True
        
        # 4. Percentage rollout
        if flag.rollout_percentage < 100:
            if user_id:
                # Use consistent hashing for stable rollout
                hash_input = f"{flag_name}:{user_id}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100)
                
                if user_percentage >= flag.rollout_percentage:
                    logger.debug(
                        f"User {user_id} not in rollout percentage for {flag_name} "
                        f"({user_percentage} >= {flag.rollout_percentage})"
                    )
                    return False
            else:
                # No user_id, can't do percentage rollout
                return False
        
        # 5. Context-based rules
        if flag.rules and context:
            rules_result = self._evaluate_rules(flag.rules, context)
            if not rules_result:
                logger.debug(f"Context rules failed for flag {flag_name}")
                return False
        
        return True
    
    def _evaluate_rules(
        self,
        rules: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate context-based rules.
        
        Rules use AND logic (all must pass).
        
        Args:
            rules: List of rule dicts with {field, operator, value}
            context: Context dict to evaluate against
        
        Returns:
            True if all rules pass, False otherwise
        """
        for rule in rules:
            field = rule.get('field')
            operator = rule.get('operator')
            expected_value = rule.get('value')
            
            if not field or not operator:
                continue
            
            # Get actual value from context
            actual_value = context.get(field)
            if actual_value is None:
                # Field not in context, rule fails
                return False
            
            # Evaluate based on operator
            if operator == RuleOperator.EQUALS.value:
                if actual_value != expected_value:
                    return False
            
            elif operator == RuleOperator.NOT_EQUALS.value:
                if actual_value == expected_value:
                    return False
            
            elif operator == RuleOperator.IN.value:
                if actual_value not in expected_value:
                    return False
            
            elif operator == RuleOperator.NOT_IN.value:
                if actual_value in expected_value:
                    return False
            
            elif operator == RuleOperator.GREATER_THAN.value:
                try:
                    if float(actual_value) <= float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            
            elif operator == RuleOperator.LESS_THAN.value:
                try:
                    if float(actual_value) >= float(expected_value):
                        return False
                except (ValueError, TypeError):
                    return False
            
            elif operator == RuleOperator.CONTAINS.value:
                if expected_value not in str(actual_value):
                    return False
            
            elif operator == RuleOperator.STARTS_WITH.value:
                if not str(actual_value).startswith(str(expected_value)):
                    return False
            
            elif operator == RuleOperator.ENDS_WITH.value:
                if not str(actual_value).endswith(str(expected_value)):
                    return False
        
        # All rules passed
        return True
    
    async def get_flag(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get feature flag configuration.
        
        Args:
            name: Flag name
        
        Returns:
            Flag dict or None if not found
        """
        flag = await self._get_flag(name)
        if not flag:
            return None
        
        return asdict(flag)
    
    async def list_flags(
        self,
        enabled_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all feature flags.
        
        Args:
            enabled_only: Only return enabled flags
        
        Returns:
            List of flag dicts
        """
        flags = []
        
        for name, flag in self.flags.items():
            if enabled_only and not flag.enabled:
                continue
            flags.append(asdict(flag))
        
        return flags
    
    async def _get_flag(self, name: str) -> Optional[FeatureFlag]:
        """
        Get flag from cache/storage.
        
        Priority:
        1. In-memory cache
        2. Redis cache
        3. Database
        """
        # Check in-memory cache
        if name in self.flags:
            return self.flags[name]
        
        # Check Redis
        if self.redis:
            try:
                flag_json = await self.redis.get(f"feature_flag:{name}")
                if flag_json:
                    flag_dict = json.loads(flag_json.decode('utf-8'))
                    flag = FeatureFlag(**flag_dict)
                    self.flags[name] = flag
                    return flag
            except Exception as e:
                logger.warning(f"Redis error: {e}")
        
        # Check database
        if self.db:
            try:
                row = await self.db.fetch_one(
                    "SELECT * FROM feature_flags WHERE name = ?",
                    (name,)
                )
                if row:
                    flag = FeatureFlag(
                        name=row['name'],
                        description=row['description'],
                        enabled=bool(row['enabled']),
                        rollout_percentage=row['rollout_percentage'],
                        whitelist=json.loads(row['whitelist']) if row['whitelist'] else [],
                        blacklist=json.loads(row['blacklist']) if row['blacklist'] else [],
                        rules=json.loads(row['rules']) if row['rules'] else [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    self.flags[name] = flag
                    return flag
            except Exception as e:
                logger.warning(f"Database error: {e}")
        
        return None
    
    async def _save_flag_to_redis(self, flag: FeatureFlag):
        """Save flag to Redis."""
        if not self.redis:
            return
        
        try:
            flag_json = json.dumps(asdict(flag))
            await self.redis.set(
                f"feature_flag:{flag.name}",
                flag_json,
                ex=86400  # 24 hours
            )
        except Exception as e:
            logger.warning(f"Redis error: {e}")
    
    async def _save_flag_to_db(self, flag: FeatureFlag):
        """Save flag to database."""
        if not self.db:
            return
        
        try:
            await self.db.execute(
                """
                INSERT INTO feature_flags 
                (name, description, enabled, rollout_percentage, whitelist, blacklist, rules, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    flag.name,
                    flag.description,
                    int(flag.enabled),
                    flag.rollout_percentage,
                    json.dumps(flag.whitelist),
                    json.dumps(flag.blacklist),
                    json.dumps(flag.rules),
                    flag.created_at,
                    flag.updated_at
                )
            )
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    async def _update_flag_in_db(self, flag: FeatureFlag):
        """Update flag in database."""
        if not self.db:
            return
        
        try:
            await self.db.execute(
                """
                UPDATE feature_flags 
                SET description = ?, enabled = ?, rollout_percentage = ?, 
                    whitelist = ?, blacklist = ?, rules = ?, updated_at = ?
                WHERE name = ?
                """,
                (
                    flag.description,
                    int(flag.enabled),
                    flag.rollout_percentage,
                    json.dumps(flag.whitelist),
                    json.dumps(flag.blacklist),
                    json.dumps(flag.rules),
                    flag.updated_at,
                    flag.name
                )
            )
        except Exception as e:
            logger.error(f"Database error: {e}")


# Global singleton instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager(
    redis_client=None,
    db_connection=None
) -> FeatureFlagManager:
    """Get or create global feature flag manager instance."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager(redis_client, db_connection)
    return _feature_flag_manager
