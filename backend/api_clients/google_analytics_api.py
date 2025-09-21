"""
Google Analytics 4 API Integration for Real Analytics Data
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        RunReportRequest,
        DateRange,
        Dimension,
        Metric,
        RunRealtimeReportRequest
    )
    from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        RunReportRequest,
        DateRange,
        Dimension,
        Metric,
        RunRealtimeReportRequest
    )
    from google.oauth2 import service_account
    GA_AVAILABLE = True
except ImportError:
    logger.warning("Google Analytics Data API not available. Install with: pip install google-analytics-data")
    GA_AVAILABLE = False

class GoogleAnalyticsService:
    """Service for fetching real Google Analytics 4 data"""
    
    def __init__(self):
        self.property_id: Optional[str] = None
        self.client: Optional['BetaAnalyticsDataClient'] = None
        self.enabled = False
        
        if GA_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Analytics client with service account"""
        if not GA_AVAILABLE:
            logger.warning("Google Analytics API not available")
            self.enabled = False
            return
            
        try:
            # Get credentials from environment or service account file
            service_account_path = os.getenv('GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH')
            property_id = os.getenv('GOOGLE_ANALYTICS_PROPERTY_ID')
            
            if service_account_path and os.path.exists(service_account_path):
                # Use service account file
                if GA_AVAILABLE:
                    from google.oauth2 import service_account as sa
                    credentials = sa.Credentials.from_service_account_file(
                        service_account_path,
                        scopes=['https://www.googleapis.com/auth/analytics.readonly']
                    )
                    self.client = BetaAnalyticsDataClient(credentials=credentials)
                    self.property_id = property_id
                    self.enabled = True
                    logger.info("✅ Google Analytics API initialized successfully")
                else:
                    logger.warning("⚠️ Google Analytics dependencies not available")
                    self.enabled = False
                
            elif os.getenv('GOOGLE_ANALYTICS_SERVICE_ACCOUNT_JSON'):
                # Use service account JSON from environment variable
                service_account_json = os.getenv('GOOGLE_ANALYTICS_SERVICE_ACCOUNT_JSON')
                if service_account_json and GA_AVAILABLE:
                    from google.oauth2 import service_account as sa
                    service_account_info = json.loads(service_account_json)
                    credentials = sa.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=['https://www.googleapis.com/auth/analytics.readonly']
                    )
                    self.client = BetaAnalyticsDataClient(credentials=credentials)
                    self.property_id = property_id
                    self.enabled = True
                    logger.info("✅ Google Analytics API initialized with JSON credentials")
                else:
                    logger.warning("⚠️ GOOGLE_ANALYTICS_SERVICE_ACCOUNT_JSON is empty or GA not available")
                    self.enabled = False
            else:
                logger.warning("⚠️ Google Analytics credentials not found. Set GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH or GOOGLE_ANALYTICS_SERVICE_ACCOUNT_JSON")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Analytics client: {e}")
            self.enabled = False
    
    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time analytics data from GA4"""
        if not self.enabled or not GA_AVAILABLE:
            return self._get_mock_realtime_data()
        
        try:
            request = RunRealtimeReportRequest(
                property=f"properties/{self.property_id}",
                dimensions=[
                    Dimension(name="country"),
                    Dimension(name="city"),
                    Dimension(name="deviceCategory"),
                ],
                metrics=[
                    Metric(name="activeUsers"),
                    Metric(name="screenPageViews"),
                    Metric(name="eventCount"),
                ]
            )
            
            response = self.client.run_realtime_report(request)  # type: ignore
            
            active_users = 0
            page_views = 0
            event_count = 0
            device_breakdown = {}
            location_breakdown = {}
            
            for row in response.rows:
                country = row.dimension_values[0].value
                city = row.dimension_values[1].value
                device = row.dimension_values[2].value
                
                users = int(row.metric_values[0].value) if row.metric_values[0].value else 0
                views = int(row.metric_values[1].value) if row.metric_values[1].value else 0
                events = int(row.metric_values[2].value) if row.metric_values[2].value else 0
                
                active_users += users
                page_views += views
                event_count += events
                
                # Device breakdown
                device_breakdown[device] = device_breakdown.get(device, 0) + users
                
                # Location breakdown
                location_key = f"{city}, {country}"
                location_breakdown[location_key] = location_breakdown.get(location_key, 0) + users
            
            # Calculate average engagement based on events per user
            avg_engagement = f"{event_count // max(active_users, 1)} events/user" if active_users > 0 else "0 events/user"
            
            return {
                "current_active_readers": active_users,
                "posts_read_today": page_views,
                "average_engagement": avg_engagement,
                "device_breakdown": device_breakdown,
                "location_breakdown": dict(sorted(location_breakdown.items(), key=lambda x: x[1], reverse=True)[:5]),
                "data_source": "Google Analytics 4 Real-time",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time GA data: {e}")
            return self._get_mock_realtime_data()
    
    async def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance analytics from GA4"""
        if not self.enabled or not GA_AVAILABLE:
            return self._get_mock_performance_data()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            request = RunReportRequest(
                property=f"properties/{self.property_id}",
                date_ranges=[DateRange(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )],
                dimensions=[
                    Dimension(name="pagePath"),
                    Dimension(name="pageTitle"),
                    Dimension(name="country"),
                    Dimension(name="deviceCategory"),
                    Dimension(name="sessionSourceMedium"),
                ],
                metrics=[
                    Metric(name="screenPageViews"),
                    Metric(name="sessions"),
                    Metric(name="activeUsers"),
                    Metric(name="userEngagementDuration"),
                    Metric(name="bounceRate"),
                    Metric(name="engagementRate"),
                ]
            )
            
            response = self.client.run_report(request)  # type: ignore
            
            # Process the data
            top_pages = {}
            traffic_sources = {}
            device_stats = {}
            location_stats = {}
            total_users = 0
            total_sessions = 0
            total_pageviews = 0
            total_engagement_duration = 0
            
            for row in response.rows:
                page_path = row.dimension_values[0].value
                page_title = row.dimension_values[1].value
                country = row.dimension_values[2].value
                device = row.dimension_values[3].value
                source_medium = row.dimension_values[4].value
                
                pageviews = int(row.metric_values[0].value) if row.metric_values[0].value else 0
                sessions = int(row.metric_values[1].value) if row.metric_values[1].value else 0
                users = int(row.metric_values[2].value) if row.metric_values[2].value else 0
                duration = float(row.metric_values[3].value) if row.metric_values[3].value else 0
                bounce_rate = float(row.metric_values[4].value) if row.metric_values[4].value else 0
                engagement_rate = float(row.metric_values[5].value) if row.metric_values[5].value else 0
                
                # Top pages
                if page_path not in top_pages:
                    top_pages[page_path] = {
                        "title": page_title,
                        "views": 0,
                        "users": 0,
                        "engagement_rate": 0,
                        "avg_time_spent": "0:00"
                    }
                
                top_pages[page_path]["views"] += pageviews
                top_pages[page_path]["users"] += users
                top_pages[page_path]["engagement_rate"] = max(top_pages[page_path]["engagement_rate"], engagement_rate)
                
                if duration > 0:
                    avg_time = duration / max(sessions, 1)
                    top_pages[page_path]["avg_time_spent"] = f"{int(avg_time // 60)}:{int(avg_time % 60):02d}"
                
                # Traffic sources
                traffic_sources[source_medium] = traffic_sources.get(source_medium, 0) + sessions
                
                # Device stats
                device_stats[device] = device_stats.get(device, 0) + users
                
                # Location stats
                location_stats[country] = location_stats.get(country, 0) + users
                
                # Totals
                total_users += users
                total_sessions += sessions
                total_pageviews += pageviews
                total_engagement_duration += duration
            
            # Sort and format top pages
            sorted_pages = sorted(top_pages.items(), key=lambda x: x[1]["views"], reverse=True)[:10]
            formatted_pages = []
            
            for page_path, data in sorted_pages:
                formatted_pages.append({
                    "post_id": page_path.split('/')[-1] or "homepage",
                    "title": data["title"] or f"Page: {page_path}",
                    "views": data["views"],
                    "engagement_rate": round(data["engagement_rate"], 3),
                    "avg_time_spent": data["avg_time_spent"]
                })
            
            avg_session_duration = f"{int(total_engagement_duration // 60)}:{int(total_engagement_duration % 60):02d}" if total_engagement_duration > 0 else "0:00"
            
            return {
                "top_performing_posts": formatted_pages,
                "total_users": total_users,
                "total_sessions": total_sessions,
                "total_pageviews": total_pageviews,
                "average_session_duration": avg_session_duration,
                "traffic_sources": dict(sorted(traffic_sources.items(), key=lambda x: x[1], reverse=True)[:5]),
                "device_breakdown": device_stats,
                "top_countries": dict(sorted(location_stats.items(), key=lambda x: x[1], reverse=True)[:5]),
                "data_source": "Google Analytics 4",
                "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching GA performance data: {e}")
            return self._get_mock_performance_data()
    
    def _get_mock_realtime_data(self) -> Dict[str, Any]:
        """Fallback mock data when GA is not available"""
        return {
            "current_active_readers": 3,
            "posts_read_today": 47,
            "average_session_duration": "4:32",
            "device_breakdown": {
                "mobile": 2,
                "desktop": 1
            },
            "location_breakdown": {
                "Istanbul, Turkey": 2,
                "Berlin, Germany": 1
            },
            "data_source": "Mock Data (GA4 not configured)",
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_mock_performance_data(self) -> Dict[str, Any]:
        """Fallback mock data when GA is not available"""
        return {
            "top_performing_posts": [
                {
                    "post_id": "food_sultanahmet",
                    "title": "Hidden Food Gems in Sultanahmet",
                    "views": 1247,
                    "engagement_rate": 0.23,
                    "avg_time_spent": "4:32"
                },
                {
                    "post_id": "rooftop_galata", 
                    "title": "Best Rooftop Views in Galata",
                    "views": 932,
                    "engagement_rate": 0.31,
                    "avg_time_spent": "3:45"
                }
            ],
            "total_users": 2547,
            "total_sessions": 3201,
            "total_pageviews": 8934,
            "average_session_duration": "3:47",
            "traffic_sources": {
                "google / organic": 1203,
                "direct / none": 845,
                "social / facebook": 432
            },
            "device_breakdown": {
                "mobile": 1658,
                "desktop": 889
            },
            "top_countries": {
                "Turkey": 1203,
                "Germany": 456,
                "United States": 321
            },
            "data_source": "Mock Data (GA4 not configured)",
            "date_range": "Last 7 days",
            "last_updated": datetime.now().isoformat()
        }

# Global instance
google_analytics_service = GoogleAnalyticsService()
