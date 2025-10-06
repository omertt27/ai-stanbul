#!/usr/bin/env python3
"""
Redis Session Manager for AI Istanbul
====================================

Fast session and context management using Redis for sub-5ms access times.
Handles session state, conversation context, and entity tracking.

Features:
- Session state management in Redis
- Context persistence across requests  
- Entity and intent tracking
- User preference caching
- Automatic session expiration
"""

import json
import redis
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from models import UserSession, ConversationContext, UserPreference
import logging

logger = logging.getLogger(__name__)

@dataclass
class SessionContext:
    """Session context structure for Redis storage"""
    session_id: str
    last_queries: List[str]
    entities: Dict[str, List[str]]  # {type: [entities]}
    user_preferences: Dict[str, Any]
    current_intent: str
    conversation_stage: str
    places_mentioned: List[str]
    topics_discussed: List[str]
    last_activity: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionContext':
        """Create from dictionary loaded from Redis"""
        return cls(**data)

class RedisSessionManager:
    """Fast session manager using Redis for context and PostgreSQL for persistence"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", db_session: Session = None):
        """Initialize Redis connection and database session"""
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.db = db_session
        self.session_ttl = 2 * 60 * 60  # 2 hours in seconds
        self.context_ttl = 24 * 60 * 60  # 24 hours for context data
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def get_or_create_session(self, session_id: Optional[str] = None, 
                            user_ip: Optional[str] = None,
                            user_agent: Optional[str] = None) -> Tuple[str, SessionContext]:
        """Get existing session or create new one with Redis caching"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Try to get from Redis first (fast)
        redis_key = f"session:{session_id}"
        cached_context = self.redis_client.get(redis_key)
        
        if cached_context:
            # Found in Redis - update last activity and return
            context_data = json.loads(cached_context)
            context = SessionContext.from_dict(context_data)
            context.last_activity = datetime.now().isoformat()
            
            # Update Redis cache
            self.redis_client.setex(
                redis_key, 
                self.session_ttl, 
                json.dumps(context.to_dict())
            )
            
            logger.debug(f"🚀 Session retrieved from Redis cache: {session_id}")
            return session_id, context
        
        # Not in Redis - check database and create context
        if self.db:
            db_session = self.db.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            
            if not db_session:
                # Create new session in database
                db_session = UserSession(
                    session_id=session_id,
                    user_ip=user_ip,
                    user_agent=user_agent,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    is_active=True
                )
                self.db.add(db_session)
                
                # Create default preferences
                self._create_default_preferences(session_id)
                self.db.commit()
                logger.info(f"🆕 New session created in database: {session_id}")
            else:
                # Update last activity
                db_session.last_activity = datetime.utcnow()
                self.db.commit()
        
        # Create context and cache in Redis
        context = SessionContext(
            session_id=session_id,
            last_queries=[],
            entities={"locations": [], "restaurants": [], "museums": [], "activities": []},
            user_preferences=self._get_user_preferences(session_id),
            current_intent="initial",
            conversation_stage="greeting",
            places_mentioned=[],
            topics_discussed=[],
            last_activity=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        
        # Cache in Redis
        self.redis_client.setex(
            redis_key, 
            self.session_ttl, 
            json.dumps(context.to_dict())
        )
        
        logger.info(f"✅ Session context created and cached: {session_id}")
        return session_id, context
    
    def update_session_context(self, session_id: str, 
                             query: str,
                             intent: str,
                             entities: Dict[str, List[str]],
                             ai_response: str = "") -> bool:
        """Update session context with new query and entities"""
        
        redis_key = f"session:{session_id}"
        
        try:
            # Get current context
            cached_context = self.redis_client.get(redis_key)
            if not cached_context:
                logger.warning(f"⚠️ Session not found in Redis: {session_id}")
                return False
            
            context = SessionContext.from_dict(json.loads(cached_context))
            
            # Update context with new information
            context.last_queries.append(query)
            if len(context.last_queries) > 10:  # Keep last 10 queries
                context.last_queries = context.last_queries[-10:]
            
            # Update entities
            for entity_type, entity_list in entities.items():
                if entity_type not in context.entities:
                    context.entities[entity_type] = []
                
                # Add new entities, avoid duplicates
                for entity in entity_list:
                    if entity not in context.entities[entity_type]:
                        context.entities[entity_type].append(entity)
                
                # Keep only recent entities (max 20 per type)
                if len(context.entities[entity_type]) > 20:
                    context.entities[entity_type] = context.entities[entity_type][-20:]
            
            # Update intent and conversation stage
            context.current_intent = intent
            context.last_activity = datetime.now().isoformat()
            
            # Update places mentioned (from location entities)
            if "locations" in entities:
                for location in entities["locations"]:
                    if location not in context.places_mentioned:
                        context.places_mentioned.append(location)
                
                if len(context.places_mentioned) > 15:
                    context.places_mentioned = context.places_mentioned[-15:]
            
            # Update conversation stage based on intent
            stage_mapping = {
                "greeting": "greeting",
                "general_info": "exploring", 
                "restaurant_recommendation": "dining_planning",
                "attraction_info": "attraction_planning",
                "transportation": "logistics_planning",
                "closing": "wrapping_up"
            }
            context.conversation_stage = stage_mapping.get(intent, "exploring")
            
            # Cache updated context
            self.redis_client.setex(
                redis_key,
                self.session_ttl,
                json.dumps(context.to_dict())
            )
            
            # Store in database for persistence (async/background)
            if self.db:
                self._persist_context_to_db(session_id, context, query, ai_response, intent, entities)
            
            logger.debug(f"📝 Session context updated: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating session context: {e}")
            return False
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get session context from Redis (fast lookup)"""
        redis_key = f"session:{session_id}"
        
        try:
            cached_context = self.redis_client.get(redis_key)
            if cached_context:
                context = SessionContext.from_dict(json.loads(cached_context))
                logger.debug(f"📖 Context retrieved from Redis: {session_id}")
                return context
            
            logger.warning(f"⚠️ Session context not found: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting session context: {e}")
            return None
    
    def resolve_cross_query_references(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Resolve references like 'which is closest?', 'the first one', 'that restaurant'"""
        
        context = self.get_session_context(session_id)
        if not context:
            return {"resolved": False, "context": {}}
        
        current_query_lower = current_query.lower()
        resolution_result = {
            "resolved": False,
            "context": {},
            "suggested_entities": [],
            "query_type": "new"
        }
        
        # Reference patterns
        reference_patterns = {
            "proximity": ["closest", "nearest", "close to", "near", "nearby"],
            "selection": ["first one", "second one", "that one", "this one", "the one"],
            "previous": ["that restaurant", "that place", "that museum", "it", "there"],
            "comparison": ["better", "cheaper", "more expensive", "alternative"]
        }
        
        # Check for reference patterns
        for ref_type, patterns in reference_patterns.items():
            if any(pattern in current_query_lower for pattern in patterns):
                resolution_result["resolved"] = True
                resolution_result["query_type"] = ref_type
                
                if ref_type == "proximity" and context.places_mentioned:
                    # User asking about proximity - provide last mentioned places
                    resolution_result["context"]["reference_locations"] = context.places_mentioned[-3:]
                    resolution_result["suggested_entities"] = context.places_mentioned[-3:]
                
                elif ref_type in ["selection", "previous"] and context.last_queries:
                    # User referring to previous results
                    last_query = context.last_queries[-1] if context.last_queries else ""
                    resolution_result["context"]["previous_query"] = last_query
                    resolution_result["context"]["previous_entities"] = dict(context.entities)
                    
                    # Suggest entities from last query context
                    all_entities = []
                    for entity_list in context.entities.values():
                        all_entities.extend(entity_list[-3:])  # Last 3 of each type
                    resolution_result["suggested_entities"] = all_entities
                
                break
        
        logger.debug(f"🔗 Cross-query reference resolution: {resolution_result['resolved']}")
        return resolution_result
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of session state for debugging/monitoring"""
        context = self.get_session_context(session_id)
        
        if not context:
            return {"exists": False}
        
        return {
            "exists": True,
            "session_id": session_id,
            "query_count": len(context.last_queries),
            "entities_count": {k: len(v) for k, v in context.entities.items()},
            "places_mentioned": context.places_mentioned,
            "current_intent": context.current_intent,
            "conversation_stage": context.conversation_stage,
            "last_activity": context.last_activity,
            "session_age_minutes": self._get_session_age_minutes(context.created_at)
        }
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from Redis"""
        try:
            # Redis TTL handles automatic cleanup, but we can manually clean if needed
            pattern = "session:*"
            keys = self.redis_client.keys(pattern)
            cleaned = 0
            
            for key in keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    cleaned += 1
            
            logger.info(f"🧹 Cleaned up {cleaned} expired sessions")
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up sessions: {e}")
            return 0
    
    def _create_default_preferences(self, session_id: str):
        """Create default preferences in database"""
        if not self.db:
            return
        
        preferences = UserPreference(
            session_id=session_id,
            preferred_cuisines=[],
            avoided_cuisines=[],
            budget_level="any",
            interests=[],
            travel_style="solo",
            preferred_time_of_day=["any"],
            preferred_districts=[],
            transportation_preference="mixed",
            language="en",
            confidence_score=0.0,
            total_interactions=0
        )
        self.db.add(preferences)
    
    def _get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """Get user preferences from database"""
        if not self.db:
            return {"budget": "medium", "distance_limit_km": 2}
        
        try:
            prefs = self.db.query(UserPreference).filter(
                UserPreference.session_id == session_id
            ).first()
            
            if prefs:
                return {
                    "budget": prefs.budget_level or "medium",
                    "distance_limit_km": 2,
                    "cuisines": prefs.preferred_cuisines or [],
                    "interests": prefs.interests or [],
                    "travel_style": prefs.travel_style or "solo"
                }
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
        
        return {"budget": "medium", "distance_limit_km": 2}
    
    def _persist_context_to_db(self, session_id: str, context: SessionContext, 
                             query: str, ai_response: str, intent: str, 
                             entities: Dict[str, List[str]]):
        """Persist context updates to database (for long-term storage)"""
        if not self.db:
            return
        
        try:
            # Update or create conversation context
            db_context = self.db.query(ConversationContext).filter(
                ConversationContext.session_id == session_id
            ).first()
            
            if not db_context:
                db_context = ConversationContext(
                    session_id=session_id,
                    current_topic=intent,
                    topics_discussed=context.topics_discussed,
                    places_mentioned=context.places_mentioned,
                    travel_stage=context.conversation_stage,
                    current_need=intent,
                    last_location_discussed=entities.get("locations", [""])[0] if entities.get("locations") else "",
                    conversation_mood="active"
                )
                self.db.add(db_context)
            else:
                db_context.current_topic = intent
                db_context.topics_discussed = context.topics_discussed
                db_context.places_mentioned = context.places_mentioned
                db_context.travel_stage = context.conversation_stage
                db_context.current_need = intent
                db_context.updated_at = datetime.utcnow()
                if entities.get("locations"):
                    db_context.last_location_discussed = entities["locations"][0]
            
            # Update session last activity
            self.db.query(UserSession).filter(
                UserSession.session_id == session_id
            ).update({"last_activity": datetime.utcnow()})
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error persisting context to DB: {e}")
            self.db.rollback()
    
    def _get_session_age_minutes(self, created_at: str) -> int:
        """Calculate session age in minutes"""
        try:
            created = datetime.fromisoformat(created_at)
            age = datetime.now() - created
            return int(age.total_seconds() / 60)
        except:
            return 0

# Global instance (will be initialized in main.py)
redis_session_manager = None

def get_redis_session_manager() -> Optional[RedisSessionManager]:
    """Get the global Redis session manager instance"""
    return redis_session_manager
