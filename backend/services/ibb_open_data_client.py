"""
İBB Open Data API Client (CKAN-based)
Integrates with Istanbul Metropolitan Municipality's Open Data Portal
using CKAN API structure

API Documentation: https://data.ibb.gov.tr/
CKAN API Docs: http://docs.ckan.org/en/2.9/api/

Features:
- Traffic announcements and alerts
- Transit datasets (GTFS, schedules)
- Real-time traffic index
- Static transit infrastructure data

API Structure:
- Base: https://data.ibb.gov.tr/api/3/action/<action>
- Authentication: Via Authorization header (optional for most datasets)

Created: November 5, 2025
Updated: November 5, 2025 - Corrected to use CKAN API structure
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)


class IBBOpenDataClient:
    """
    Client for Istanbul Metropolitan Municipality Open Data API (CKAN)
    
    Provides access to:
    - Traffic announcements (near real-time)
    - IETT GTFS data (bus schedules)
    - Traffic index data
    - Transportation infrastructure datasets
    """
    
    # Correct CKAN API endpoints
    BASE_URL = "https://data.ibb.gov.tr"
    API_BASE = f"{BASE_URL}/api/3/action"
    
    # Known dataset IDs (from CKAN portal)
    DATASETS = {
        'traffic_announcements': 'ulasim-yonetim-merkezi-trafik-duyuru-verisi',
        'traffic_index': 'istanbul-trafik-indeksi',
        'iett_gtfs': 'iett-gtfs-verisi',
        'metro_stations': 'rayli-sistem-istasyon-noktalari-verisi',
        'metro_lines': 'rayli-ulasim-hatlari-vektor-verisi',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize İBB Open Data client (CKAN API)
        
        Args:
            api_key: İBB API key (optional for most datasets)
        """
        self.api_key = api_key or os.getenv('IBB_API_KEY')
        
        if not self.api_key:
            logger.info("ℹ️ İBB API key not provided - most datasets are public")
        else:
            logger.info("✅ İBB API key configured")
        
        # Initialize session
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            })
        
        # Cache for datasets
        self._dataset_cache: Dict[str, Any] = {}
        self._resource_cache: Dict[str, pd.DataFrame] = {}
    
    
    # ===== CORE CKAN API METHODS =====
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a dataset using CKAN API
        
        Args:
            dataset_id: Dataset identifier (e.g., 'istanbul-trafik-indeksi')
            
        Returns:
            Dict with dataset metadata including resources
        """
        # Check cache
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]
        
        try:
            response = self.session.get(
                f"{self.API_BASE}/package_show",
                params={'id': dataset_id},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('success'):
                dataset = data.get('result', {})
                self._dataset_cache[dataset_id] = dataset
                logger.info(f"✅ Retrieved dataset info: {dataset.get('title')}")
                return dataset
            else:
                logger.error(f"❌ API error: {data.get('error')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get dataset info: {e}")
            return None
    
    def download_resource(self, resource_url: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
        """
        Download and parse a CSV resource from İBB portal
        
        Args:
            resource_url: Direct URL to resource file
            max_retries: Number of retry attempts
            
        Returns:
            Pandas DataFrame with resource data
        """
        # Check cache
        if resource_url in self._resource_cache:
            return self._resource_cache[resource_url]
        
        for attempt in range(max_retries):
            try:
                response = requests.get(resource_url, timeout=20, stream=True)
                response.raise_for_status()
                
                # Read content in chunks to avoid incomplete reads
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                
                # Parse CSV
                df = pd.read_csv(StringIO(content.decode('utf-8')), on_bad_lines='skip')
                self._resource_cache[resource_url] = df
                logger.info(f"✅ Downloaded resource: {len(df)} rows, {len(df.columns)} columns")
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Download attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    logger.error(f"❌ Failed to download resource after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    # ===== TRAFFIC SERVICES =====
    
    def get_traffic_announcements(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get recent traffic announcements from UYM (Traffic Management Center)
        
        Args:
            limit: Maximum number of announcements to return
            
        Returns:
            List of traffic announcements or None if unavailable
        """
        dataset_id = self.DATASETS['traffic_announcements']
        dataset = self.get_dataset_info(dataset_id)
        
        if not dataset:
            logger.warning("⚠️ Using mock traffic announcements")
            return self._get_mock_traffic_announcements(limit)
        
        # Get first resource (should be the CSV file)
        resources = dataset.get('resources', [])
        if not resources:
            logger.error("❌ No resources found for traffic announcements")
            return self._get_mock_traffic_announcements(limit)
        
        resource_url = resources[0].get('url')
        df = self.download_resource(resource_url)
        
        if df is None or df.empty:
            logger.warning("⚠️ Failed to load traffic data, using mock")
            return self._get_mock_traffic_announcements(limit)
        
        # Convert to list of dicts, most recent first
        try:
            # Assume there's a date column
            if 'TARIH' in df.columns or 'tarih' in df.columns:
                date_col = 'TARIH' if 'TARIH' in df.columns else 'tarih'
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.sort_values(date_col, ascending=False)
            
            announcements = df.head(limit).to_dict('records')
            logger.info(f"✅ Retrieved {len(announcements)} traffic announcements")
            return announcements
            
        except Exception as e:
            logger.error(f"❌ Error processing traffic data: {e}")
            return self._get_mock_traffic_announcements(limit)
    
    
    def get_traffic_index(self) -> Optional[Dict[str, Any]]:
        """
        Get current Istanbul traffic index
        
        Returns:
            Dict with traffic index data (min, max, avg)
        """
        dataset_id = self.DATASETS['traffic_index']
        dataset = self.get_dataset_info(dataset_id)
        
        if not dataset:
            logger.warning("⚠️ Using mock traffic index")
            return self._get_mock_traffic_index()
        
        resources = dataset.get('resources', [])
        if not resources:
            return self._get_mock_traffic_index()
        
        resource_url = resources[0].get('url')
        df = self.download_resource(resource_url)
        
        if df is None or df.empty:
            logger.warning("⚠️ Failed to load traffic index, using mock")
            return self._get_mock_traffic_index()
        
        # Get most recent entry and ensure it's a dict
        try:
            # Get the last row
            latest = df.iloc[-1].to_dict()
            
            # Clean up the data - convert to proper types
            cleaned = {}
            for key, value in latest.items():
                # Handle NaN and None values
                if pd.isna(value):
                    cleaned[key] = None
                elif isinstance(value, (int, float)):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = str(value)
            
            logger.info(f"✅ Retrieved traffic index: {cleaned.get('Average Traffic Index', 'N/A')}")
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Error processing traffic index: {e}")
            return self._get_mock_traffic_index()
    
    # ===== TRANSIT DISRUPTION DETECTION =====
    
    def get_transit_alerts(self, route_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current transit alerts and disruptions
        
        Args:
            route_type: Filter by type ('bus', 'metro', 'tram', 'ferry') or None for all
            
        Returns:
            List of active transit alerts
        """
        alerts = []
        
        # Get traffic announcements
        announcements = self.get_traffic_announcements(limit=20)
        if announcements:
            for item in announcements:
                # Ensure item is a dict
                if not isinstance(item, dict):
                    continue
                
                # Parse announcement for transit-related keywords
                text = ' '.join(str(v) for v in item.values()).lower()
                
                alert_type = None
                if route_type is None or route_type == 'metro':
                    if any(word in text for word in ['metro', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']):
                        alert_type = 'metro'
                        
                if route_type is None or route_type == 'bus':
                    if any(word in text for word in ['otobüs', 'iett', 'hat', 'bus']):
                        alert_type = 'bus'
                        
                if route_type is None or route_type == 'tram':
                    if any(word in text for word in ['tramvay', 't1', 't4', 't5', 'tram']):
                        alert_type = 'tram'
                
                if alert_type:
                    alerts.append({
                        'type': alert_type,
                        'data': item,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Add traffic index if congestion is high
        traffic = self.get_traffic_index()
        if traffic and isinstance(traffic, dict):
            try:
                # Handle both key formats (lowercase with underscore or title case)
                avg_index_value = (
                    traffic.get('average_traffic_index') or 
                    traffic.get('Average Traffic Index')
                )
                # Handle various data types
                if avg_index_value is not None:
                    avg_index = float(avg_index_value)
                    if avg_index > 7.0:  # High traffic
                        alerts.append({
                            'type': 'traffic',
                            'severity': 'high' if avg_index > 8.5 else 'moderate',
                            'index': avg_index,
                            'message': f'Heavy traffic conditions (index: {avg_index:.1f}/10)',
                            'timestamp': datetime.now().isoformat()
                        })
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse traffic index: {e}")
        
        logger.info(f"✅ Found {len(alerts)} transit alerts")
        return alerts
    
    # ===== MOCK DATA (for development/testing) =====
    
    def _get_mock_traffic_announcements(self, limit: int) -> List[Dict[str, Any]]:
        """Mock traffic announcements for testing"""
        return [
            {
                'TARIH': datetime.now().isoformat(),
                'LOKASYON': 'Kadıköy - Bağdat Caddesi',
                'DUYURU': 'Heavy traffic due to roadworks',
                'TIP': 'traffic',
                'mock_data': True
            },
            {
                'TARIH': datetime.now().isoformat(),
                'LOKASYON': 'M2 Metro - Şişhane-Taksim',
                'DUYURU': 'Normal service on all lines',
                'TIP': 'metro',
                'mock_data': True
            }
        ][:limit]
    
    def _get_mock_traffic_index(self) -> Dict[str, Any]:
        """Mock traffic index for testing"""
        from random import uniform
        return {
            'Traffic Index Date': datetime.now().strftime('%Y-%m-%d'),
            'Minimum Traffic Index': round(uniform(3.0, 5.0), 2),
            'Maximum Traffic Index': round(uniform(8.0, 9.5), 2),
            'Average Traffic Index': round(uniform(6.0, 7.5), 2),
            'mock_data': True
        }
    
    # ===== UTILITY METHODS =====
    
    def is_ckan_available(self) -> bool:
        """Check if İBB CKAN API is reachable"""
        try:
            response = self.session.get(f"{self.BASE_URL}/api/3", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get overall service status
        
        Returns:
            Dict with service availability information
        """
        return {
            'ckan_available': self.is_ckan_available(),
            'api_key_configured': self.api_key is not None,
            'cached_datasets': len(self._dataset_cache),
            'cached_resources': len(self._resource_cache),
            'timestamp': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚌 İBB Open Data Client (CKAN) - Demo")
    print("=" * 60)
    
    # Initialize client
    client = IBBOpenDataClient()
    
    print(f"\n📊 Service Status:")
    status = client.get_service_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print(f"\n� Traffic Announcements:")
    announcements = client.get_traffic_announcements(limit=3)
    if announcements:
        print(f"   Found: {len(announcements)} announcements")
        for i, ann in enumerate(announcements[:2], 1):
            print(f"   {i}. {ann.get('LOKASYON', 'Unknown')}: {ann.get('DUYURU', 'N/A')[:50]}")
    
    print(f"\n📈 Traffic Index:")
    traffic_index = client.get_traffic_index()
    if traffic_index:
        print(f"   Date: {traffic_index.get('Traffic Index Date')}")
        print(f"   Average: {traffic_index.get('Average Traffic Index')}")
        print(f"   Range: {traffic_index.get('Minimum Traffic Index')} - {traffic_index.get('Maximum Traffic Index')}")
    
    print(f"\n⚠️  Transit Alerts:")
    alerts = client.get_transit_alerts()
    if alerts:
        print(f"   Active alerts: {len(alerts)}")
        for alert in alerts[:3]:
            print(f"   - Type: {alert.get('type')}, {alert.get('message', 'See data')[:50]}")
    
    print("\n" + "=" * 60)
    print("✅ İBB Open Data Client (CKAN) ready!")
    print("📋 API uses CKAN structure: /api/3/action/<action>")

