"""
Transit Alert LLM Service - Step 5.2
Integrates İBB Open Data with LLM for intelligent transit alerts

This service:
1. Fetches real-time transit data from İBB CKAN API
2. Analyzes alerts using context-aware prompts
3. Generates user-friendly, LLM-powered advice
4. Provides route-specific recommendations

Created: November 5, 2025
Status: STEP 5.2 IMPLEMENTATION
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.ibb_open_data_client import IBBOpenDataClient
from ml_systems.llm_service_wrapper import LLMServiceWrapper
from ml_systems.context_aware_prompts import ContextAwarePromptEngine

logger = logging.getLogger(__name__)


class TransitAlertLLMService:
    """
    Service for generating LLM-powered transit alerts and advice
    
    Combines:
    - İBB Open Data (traffic, alerts, disruptions)
    - Weather context
    - GPS location
    - LLM intelligence
    """
    
    def __init__(self):
        """Initialize transit alert LLM service"""
        self.ibb_client = IBBOpenDataClient()
        self.llm_service = LLMServiceWrapper()
        self.prompt_engine = ContextAwarePromptEngine()
        
        logger.info("✅ Transit Alert LLM Service initialized")
    
    def get_transit_advice(
        self,
        from_location: str,
        to_location: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get LLM-powered transit advice with real-time alerts
        
        Args:
            from_location: Starting location
            to_location: Destination
            context: Additional context (weather, time, user prefs)
            
        Returns:
            Dict with transit advice and alerts
        """
        logger.info(f"🚦 Getting transit advice: {from_location} → {to_location}")
        
        # Fetch transit alerts
        alerts = self.ibb_client.get_transit_alerts()
        traffic_index = self.ibb_client.get_traffic_index()
        
        # Build context for LLM
        full_context = {
            'from': from_location,
            'to': to_location,
            'alerts': alerts,
            'traffic_index': traffic_index,
            'timestamp': datetime.now().isoformat(),
            **(context or {})
        }
        
        # Generate context-aware prompt
        prompt = self._create_transit_advice_prompt(full_context)
        
        # Get LLM response
        try:
            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.3,  # More deterministic for transit advice
                max_tokens=300
            )
            
            # Handle both dict and string responses
            if isinstance(response, dict):
                advice = response.get('text', 'Unable to generate advice at this time.')
            elif isinstance(response, str):
                advice = response
            else:
                advice = str(response)
            
            return {
                'success': True,
                'advice': advice,
                'alerts': alerts,
                'traffic_index': traffic_index,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating transit advice: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'fallback_advice': self._generate_fallback_advice(full_context)
            }
    
    def analyze_route_disruptions(
        self,
        route_type: str,
        route_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze disruptions for a specific route type or route
        
        Args:
            route_type: Type of route ('metro', 'bus', 'tram', 'ferry')
            route_name: Specific route name (e.g., 'M2', '500T')
            
        Returns:
            Dict with disruption analysis and LLM recommendations
        """
        logger.info(f"🔍 Analyzing disruptions for {route_type} {route_name or ''}")
        
        # Get relevant alerts
        alerts = self.ibb_client.get_transit_alerts(route_type=route_type)
        
        # Filter for specific route if provided
        if route_name:
            alerts = [a for a in alerts if route_name.lower() in str(a).lower()]
        
        if not alerts:
            return {
                'success': True,
                'disruptions_found': False,
                'message': f'No disruptions found for {route_type} {route_name or "lines"}',
                'advice': 'Normal service expected.'
            }
        
        # Generate LLM analysis
        prompt = self._create_disruption_analysis_prompt(route_type, route_name, alerts)
        
        try:
            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=250
            )
            
            # Handle both dict and string responses
            if isinstance(response, dict):
                analysis = response.get('text', 'Unable to analyze disruptions.')
            elif isinstance(response, str):
                analysis = response
            else:
                analysis = str(response)
            
            return {
                'success': True,
                'disruptions_found': True,
                'alert_count': len(alerts),
                'analysis': analysis,
                'raw_alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing disruptions: {e}")
            return {
                'success': False,
                'error': str(e),
                'alert_count': len(alerts),
                'raw_alerts': alerts
            }
    
    def get_traffic_summary(
        self,
        location: Optional[str] = None,
        weather: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get LLM-powered traffic summary with recommendations
        
        Args:
            location: Specific location to focus on
            weather: Current weather data
            
        Returns:
            Dict with traffic summary and advice
        """
        logger.info(f"📊 Getting traffic summary for {location or 'Istanbul'}")
        
        # Get traffic data
        traffic_index = self.ibb_client.get_traffic_index()
        announcements = self.ibb_client.get_traffic_announcements(limit=5)
        
        # Build context
        context = {
            'location': location,
            'traffic_index': traffic_index,
            'announcements': announcements,
            'weather': weather,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate prompt
        prompt = self._create_traffic_summary_prompt(context)
        
        try:
            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=250
            )
            
            # Handle both dict and string responses
            if isinstance(response, dict):
                summary = response.get('text', 'Unable to generate traffic summary.')
            elif isinstance(response, str):
                summary = response
            else:
                summary = str(response)
            
            return {
                'success': True,
                'summary': summary,
                'traffic_index': traffic_index,
                'announcements': announcements,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating traffic summary: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_summary': self._generate_fallback_traffic_summary(context)
            }
    
    # ===== PROMPT GENERATION =====
    
    def _create_transit_advice_prompt(self, context: Dict[str, Any]) -> str:
        """Create context-aware prompt for transit advice"""
        alerts = context.get('alerts', [])
        traffic = context.get('traffic_index', {})
        weather = context.get('weather', {})
        
        prompt = f"""You are an Istanbul transit expert. Provide concise, practical advice for this journey.

Route: {context['from']} to {context['to']}
Time: {datetime.now().strftime('%H:%M')}

"""
        
        if alerts:
            prompt += f"Current Alerts ({len(alerts)}):\n"
            for alert in alerts[:3]:
                alert_type = alert.get('type', 'general')
                message = alert.get('message', str(alert.get('data', ''))[:100])
                prompt += f"- {alert_type.upper()}: {message}\n"
            prompt += "\n"
        
        if traffic and isinstance(traffic, dict) and not traffic.get('mock_data'):
            try:
                # Handle both lowercase and title case keys
                avg_traffic = (
                    float(traffic.get('average_traffic_index') or 
                          traffic.get('Average Traffic Index') or 0)
                )
                if avg_traffic > 0:
                    prompt += f"Traffic Index: {avg_traffic:.1f}/10 "
                    if avg_traffic > 7.5:
                        prompt += "(Heavy traffic)\n\n"
                    elif avg_traffic > 5.5:
                        prompt += "(Moderate traffic)\n\n"
                    else:
                        prompt += "(Light traffic)\n\n"
            except (ValueError, TypeError):
                pass  # Skip if traffic data is invalid
        
        if weather:
            temp = weather.get('temperature')
            condition = weather.get('condition', '')
            if temp:
                prompt += f"Weather: {temp}°C, {condition}\n\n"
        
        prompt += """Provide:
1. Best transportation option (metro/tram/bus/ferry)
2. Key considerations based on current conditions
3. Estimated travel time

Keep response under 150 words, be specific and practical."""
        
        return prompt
    
    def _create_disruption_analysis_prompt(
        self,
        route_type: str,
        route_name: Optional[str],
        alerts: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for disruption analysis"""
        route_desc = f"{route_type} line {route_name}" if route_name else f"{route_type} lines"
        
        prompt = f"""Analyze these transit disruptions for Istanbul {route_desc}:

Alerts ({len(alerts)}):
"""
        
        for i, alert in enumerate(alerts[:5], 1):
            alert_data = str(alert.get('data', alert))[:150]
            prompt += f"{i}. {alert_data}\n"
        
        prompt += f"""
Provide:
1. Impact severity (low/moderate/high)
2. Affected areas/routes
3. Alternative options
4. Estimated duration (if determinable)

Keep response concise (under 120 words)."""
        
        return prompt
    
    def _create_traffic_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for traffic summary"""
        traffic = context.get('traffic_index', {})
        announcements = context.get('announcements', [])
        location = context.get('location', 'Istanbul')
        
        prompt = f"""Summarize current traffic conditions for {location}:

"""
        
        if traffic and isinstance(traffic, dict) and not traffic.get('mock_data'):
            try:
                # Handle both lowercase and title case keys
                avg = float(traffic.get('average_traffic_index') or 
                           traffic.get('Average Traffic Index') or 0)
                min_idx = float(traffic.get('minimum_traffic_index') or 
                               traffic.get('Minimum Traffic Index') or 0)
                max_idx = float(traffic.get('maximum_traffic_index') or 
                               traffic.get('Maximum Traffic Index') or 0)
                if avg > 0:
                    prompt += f"Traffic Index: {avg:.1f}/10 (Range: {min_idx:.1f}-{max_idx:.1f})\n\n"
            except (ValueError, TypeError):
                pass  # Skip if traffic data is invalid
        
        if announcements:
            prompt += f"Recent Announcements ({len(announcements)}):\n"
            for ann in announcements[:3]:
                loc = ann.get('LOKASYON', 'Unknown')
                msg = ann.get('DUYURU', '')[:80]
                prompt += f"- {loc}: {msg}\n"
            prompt += "\n"
        
        prompt += """Provide:
1. Overall traffic situation
2. Problem areas to avoid
3. Best travel advice for now

Keep it brief (under 100 words) and practical."""
        
        return prompt
    
    # ===== FALLBACK METHODS =====
    
    def _generate_fallback_advice(self, context: Dict[str, Any]) -> str:
        """Generate fallback advice when LLM is unavailable"""
        alerts = context.get('alerts', [])
        traffic = context.get('traffic_index', {})
        
        advice = f"Route: {context['from']} to {context['to']}\n\n"
        
        if alerts:
            advice += f"⚠️ {len(alerts)} active alerts. "
        
        avg_traffic = 5.0  # default
        if isinstance(traffic, dict):
            try:
                # Handle both key formats
                avg_traffic = float(
                    traffic.get('average_traffic_index') or 
                    traffic.get('Average Traffic Index') or 5.0
                )
            except (ValueError, TypeError):
                pass
        
        if avg_traffic > 7.0:
            advice += "Heavy traffic conditions. Consider metro or ferry if available. "
        else:
            advice += "Normal traffic conditions. All transit options available. "
        
        advice += "\n\nFor detailed route planning, please try again."
        
        return advice
    
    def _generate_fallback_traffic_summary(self, context: Dict[str, Any]) -> str:
        """Generate fallback traffic summary"""
        traffic = context.get('traffic_index', {})
        
        if isinstance(traffic, dict) and not traffic.get('mock_data'):
            try:
                # Handle both key formats
                avg = float(
                    traffic.get('average_traffic_index') or 
                    traffic.get('Average Traffic Index') or 0
                )
                if avg > 0:
                    return f"Current traffic index: {avg:.1f}/10. Check specific routes for details."
            except (ValueError, TypeError):
                pass
        
        return "Traffic data temporarily unavailable. Please try again shortly."
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all integrated services"""
        llm_status = 'available' if self.llm_service.model is not None else 'unavailable'
        return {
            'ibb_api': self.ibb_client.get_service_status(),
            'llm_service': {'status': llm_status, 'model': self.llm_service.model_name},
            'timestamp': datetime.now().isoformat()
        }


# Example usage / testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚦 Transit Alert LLM Service - Step 5.2")
    print("=" * 70)
    
    service = TransitAlertLLMService()
    
    print("\n📊 Service Status:")
    status = service.get_service_status()
    print(f"   İBB API: {status['ibb_api']['ckan_available']}")
    print(f"   LLM: {status['llm_service']['status']}")
    
    print("\n🚇 Test 1: Transit Advice with Alerts")
    advice = service.get_transit_advice(
        from_location="Taksim",
        to_location="Kadıköy",
        context={'weather': {'temperature': 18, 'condition': 'Partly Cloudy'}}
    )
    print(f"   Success: {advice['success']}")
    print(f"   Alerts found: {len(advice.get('alerts', []))}")
    print(f"\n   Advice:\n   {advice.get('advice', advice.get('fallback_advice', 'N/A'))}")
    
    print("\n🔍 Test 2: Route Disruption Analysis")
    disruption = service.analyze_route_disruptions(route_type='metro', route_name='M2')
    print(f"   Success: {disruption['success']}")
    print(f"   Disruptions: {disruption.get('disruptions_found', False)}")
    if disruption.get('analysis'):
        print(f"\n   Analysis:\n   {disruption['analysis']}")
    
    print("\n📈 Test 3: Traffic Summary")
    traffic = service.get_traffic_summary(location="Istanbul City Center")
    print(f"   Success: {traffic['success']}")
    if traffic.get('summary'):
        print(f"\n   Summary:\n   {traffic['summary']}")
    
    print("\n" + "=" * 70)
    print("✅ Transit Alert LLM Service (Step 5.2) - Ready!")
