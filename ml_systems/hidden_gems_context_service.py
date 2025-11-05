"""
Hidden Gems Context Service for Transportation LLM
Provides district-specific local recommendations for LLM prompts

This service integrates the hidden gems database with the transportation
handler to provide contextual local recommendations during route planning.

Created: November 5, 2025
Status: Production-ready
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HiddenGemsContextService:
    """
    Service to provide hidden gems context for transportation LLM prompts
    
    Features:
    - District-based gem retrieval
    - Type filtering (cafe, viewpoint, nature, etc.)
    - Hidden factor ranking (more authentic = higher score)
    - LLM-friendly text formatting
    - Route-based recommendations (origin + destination)
    - Weather-aware filtering (optional)
    - Time-aware filtering (optional)
    
    Usage:
        service = HiddenGemsContextService()
        
        # Get gems for a single district
        gems = service.get_gems_for_district('kadıköy', max_gems=3)
        
        # Get formatted text for LLM prompt
        gem_text = service.format_gems_for_llm_prompt('beyoğlu')
        
        # Get gems for both origin and destination
        route_gems = service.get_gems_for_route('beyoğlu', 'kadıköy')
    """
    
    def __init__(self):
        """Initialize the service with the hidden gems database"""
        try:
            # Try multiple import paths
            try:
                from backend.data.hidden_gems_database import HIDDEN_GEMS_DATABASE
            except ImportError:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from backend.data.hidden_gems_database import HIDDEN_GEMS_DATABASE
            
            self.gems_database = HIDDEN_GEMS_DATABASE
            
            total_gems = sum(len(gems) for gems in self.gems_database.values())
            logger.info(
                f"✅ Hidden Gems Context Service initialized: "
                f"{len(self.gems_database)} districts, {total_gems} total gems"
            )
        except ImportError as e:
            logger.error(f"❌ Failed to import hidden gems database: {e}")
            self.gems_database = {}
    
    def get_gems_for_district(
        self,
        district: str,
        max_gems: int = 3,
        gem_types: Optional[List[str]] = None,
        min_hidden_factor: Optional[int] = None,
        weather_condition: Optional[str] = None,
        time_of_day: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get hidden gems for a specific district
        
        Args:
            district: District name (e.g., 'kadıköy', 'beyoğlu')
            max_gems: Maximum number of gems to return (default: 3)
            gem_types: Filter by types (e.g., ['cafe', 'viewpoint'])
            min_hidden_factor: Minimum hidden factor (1-10 scale)
            weather_condition: Filter by weather (e.g., 'rainy', 'sunny')
            time_of_day: Filter by time (e.g., 'morning', 'evening')
            
        Returns:
            List of hidden gem dicts with all metadata
        """
        district_lower = district.lower().strip()
        
        # Check if district exists in database
        if district_lower not in self.gems_database:
            logger.debug(f"No hidden gems found for district: {district}")
            return []
        
        gems = list(self.gems_database[district_lower])  # Copy to avoid mutating original
        
        # Filter by type if specified
        if gem_types:
            gems = [g for g in gems if g.get('type') in gem_types]
        
        # Filter by hidden factor if specified
        if min_hidden_factor is not None:
            gems = [g for g in gems if g.get('hidden_factor', 0) >= min_hidden_factor]
        
        # Filter by weather if specified
        if weather_condition:
            gems = self._filter_by_weather(gems, weather_condition)
        
        # Filter by time of day if specified
        if time_of_day:
            gems = self._filter_by_time(gems, time_of_day)
        
        # Sort by hidden_factor (higher = more hidden/authentic)
        gems = sorted(gems, key=lambda x: x.get('hidden_factor', 0), reverse=True)
        
        # Return top N gems
        result = gems[:max_gems]
        logger.debug(f"Found {len(result)} gems for {district} (filtered from {len(gems)})")
        return result
    
    def _filter_by_weather(self, gems: List[Dict[str, Any]], weather_condition: str) -> List[Dict[str, Any]]:
        """
        Filter gems by weather suitability
        
        Args:
            gems: List of gem dicts
            weather_condition: 'rainy', 'sunny', 'cold', 'hot'
            
        Returns:
            Filtered list of gems
        """
        weather_lower = weather_condition.lower()
        
        # Weather-based filtering rules
        if weather_lower in ['rainy', 'rain', 'wet']:
            # Prefer indoor spots, covered areas
            indoor_types = ['cafe', 'restaurant', 'shopping', 'historical', 'museum']
            return [g for g in gems if g.get('type') in indoor_types or 'covered' in g.get('tags', [])]
        
        elif weather_lower in ['sunny', 'clear', 'nice']:
            # Prefer outdoor spots, viewpoints
            outdoor_types = ['nature', 'viewpoint', 'beach', 'park']
            return [g for g in gems if g.get('type') in outdoor_types or 'outdoor' in g.get('tags', [])]
        
        elif weather_lower in ['cold', 'freezing', 'winter']:
            # Prefer cozy indoor spots
            return [g for g in gems if g.get('type') in ['cafe', 'restaurant', 'historical']]
        
        elif weather_lower in ['hot', 'summer', 'warm']:
            # Prefer shaded, waterfront, or cool spots
            cool_types = ['beach', 'park', 'nature']
            return [g for g in gems if 
                   g.get('type') in cool_types or 
                   any(tag in g.get('tags', []) for tag in ['waterfront', 'shaded', 'bosphorus'])]
        
        # Unknown weather condition - return all
        return gems
    
    def _filter_by_time(self, gems: List[Dict[str, Any]], time_of_day: str) -> List[Dict[str, Any]]:
        """
        Filter gems by time of day suitability
        
        Args:
            gems: List of gem dicts
            time_of_day: 'morning', 'afternoon', 'evening', 'night'
            
        Returns:
            Filtered list of gems
        """
        time_lower = time_of_day.lower()
        
        # Check best_time field in gem data
        filtered = []
        for gem in gems:
            best_time = gem.get('best_time', '').lower()
            
            if time_lower == 'morning':
                if any(word in best_time for word in ['morning', 'breakfast', 'dawn', 'sunrise']):
                    filtered.append(gem)
                elif not best_time:  # No time restriction
                    filtered.append(gem)
            
            elif time_lower == 'afternoon':
                if any(word in best_time for word in ['afternoon', 'lunch', 'day']):
                    filtered.append(gem)
                elif not best_time:
                    filtered.append(gem)
            
            elif time_lower == 'evening':
                if any(word in best_time for word in ['evening', 'sunset', 'dusk']):
                    filtered.append(gem)
                elif not best_time:
                    filtered.append(gem)
            
            elif time_lower == 'night':
                if any(word in best_time for word in ['night', 'nightlife', 'dinner', '20:00', '21:00']):
                    filtered.append(gem)
        
        return filtered if filtered else gems  # Return all if no matches
    
    def format_gems_for_llm_prompt(
        self,
        district: str,
        max_gems: int = 2,
        context: str = "nearby",
        include_tips: bool = True
    ) -> Optional[str]:
        """
        Format hidden gems as text for LLM prompt inclusion
        
        Args:
            district: District name
            max_gems: Maximum gems to include
            context: Context string ('nearby', 'destination', 'where you are now', etc.)
            include_tips: Include local tips in output
            
        Returns:
            Formatted string ready for LLM prompt, or None if no gems
        """
        gems = self.get_gems_for_district(district, max_gems=max_gems)
        
        if not gems:
            logger.debug(f"No gems to format for district: {district}")
            return None
        
        # Format header
        gem_text = f"\n\nHidden gems in {district.title()} ({context}):\n"
        
        # Format each gem
        for i, gem in enumerate(gems, 1):
            name = gem.get('name', 'Unknown')
            description = gem.get('description', '')
            
            gem_text += f"{i}. {name}: {description}"
            
            # Add local tip if available and requested
            if include_tips:
                local_tip = gem.get('local_tip', '')
                if local_tip:
                    gem_text += f" (Local tip: {local_tip})"
            
            gem_text += "\n"
        
        logger.debug(f"Formatted {len(gems)} gems for LLM prompt ({district})")
        return gem_text
    
    def get_gems_for_route(
        self,
        origin_district: Optional[str],
        destination_district: Optional[str],
        max_gems_per_district: int = 2,
        prioritize_destination: bool = True
    ) -> Dict[str, Any]:
        """
        Get hidden gems for both origin and destination districts
        
        Args:
            origin_district: Starting district (where user is now)
            destination_district: Ending district (where user is going)
            max_gems_per_district: Max gems per district
            prioritize_destination: If True, show destination gems preferentially
            
        Returns:
            Dict with:
            - 'origin_gems': List of gem dicts for origin
            - 'destination_gems': List of gem dicts for destination
            - 'origin_text': Formatted text for LLM prompt (origin)
            - 'destination_text': Formatted text for LLM prompt (destination)
            - 'has_gems': Boolean indicating if any gems found
        """
        result = {
            'origin_gems': [],
            'destination_gems': [],
            'origin_text': None,
            'destination_text': None,
            'has_gems': False
        }
        
        # Get origin gems
        if origin_district:
            result['origin_gems'] = self.get_gems_for_district(
                origin_district, 
                max_gems=max_gems_per_district
            )
            if result['origin_gems']:
                result['origin_text'] = self.format_gems_for_llm_prompt(
                    origin_district, 
                    max_gems=max_gems_per_district,
                    context="where you are now"
                )
        
        # Get destination gems
        if destination_district:
            result['destination_gems'] = self.get_gems_for_district(
                destination_district, 
                max_gems=max_gems_per_district
            )
            if result['destination_gems']:
                result['destination_text'] = self.format_gems_for_llm_prompt(
                    destination_district, 
                    max_gems=max_gems_per_district,
                    context="your destination"
                )
        
        # Check if we found any gems
        result['has_gems'] = bool(result['origin_gems'] or result['destination_gems'])
        
        logger.info(
            f"Route gems: {origin_district or 'unknown'} → {destination_district or 'unknown'} | "
            f"Origin: {len(result['origin_gems'])} gems, Destination: {len(result['destination_gems'])} gems"
        )
        
        return result
    
    def get_available_districts(self) -> List[str]:
        """
        Get list of districts with hidden gems data
        
        Returns:
            List of district names
        """
        return list(self.gems_database.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the hidden gems database
        
        Returns:
            Dict with stats (total_districts, total_gems, gems_per_district, etc.)
        """
        districts = list(self.gems_database.keys())
        total_gems = sum(len(gems) for gems in self.gems_database.values())
        
        # Calculate gems per district
        gems_per_district = {
            district: len(gems) 
            for district, gems in self.gems_database.items()
        }
        
        # Calculate type distribution
        type_counts = {}
        for gems in self.gems_database.values():
            for gem in gems:
                gem_type = gem.get('type', 'unknown')
                type_counts[gem_type] = type_counts.get(gem_type, 0) + 1
        
        return {
            'total_districts': len(districts),
            'total_gems': total_gems,
            'districts': districts,
            'gems_per_district': gems_per_district,
            'average_gems_per_district': total_gems / len(districts) if districts else 0,
            'type_distribution': type_counts
        }


# Convenience function for quick access
def get_hidden_gems_service() -> HiddenGemsContextService:
    """
    Get or create a singleton instance of HiddenGemsContextService
    
    Returns:
        HiddenGemsContextService instance
    """
    if not hasattr(get_hidden_gems_service, '_instance'):
        get_hidden_gems_service._instance = HiddenGemsContextService()
    return get_hidden_gems_service._instance


# Example usage
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create service
    service = HiddenGemsContextService()
    
    # Print statistics
    stats = service.get_statistics()
    print("\n📊 Hidden Gems Database Statistics:")
    print(f"  Total Districts: {stats['total_districts']}")
    print(f"  Total Gems: {stats['total_gems']}")
    print(f"  Average per District: {stats['average_gems_per_district']:.1f}")
    print(f"\n  Districts: {', '.join(stats['districts'])}")
    
    # Example: Get gems for Kadıköy
    print("\n💎 Example: Hidden Gems in Kadıköy")
    gems = service.get_gems_for_district('kadıköy', max_gems=3)
    for i, gem in enumerate(gems, 1):
        print(f"  {i}. {gem['name']} (Hidden Factor: {gem.get('hidden_factor', 'N/A')})")
    
    # Example: Format for LLM
    print("\n📝 Example: Formatted for LLM Prompt")
    formatted = service.format_gems_for_llm_prompt('beyoğlu', max_gems=2)
    if formatted:
        print(formatted)
    
    # Example: Route gems
    print("\n🗺️  Example: Route from Beyoğlu to Kadıköy")
    route_gems = service.get_gems_for_route('beyoğlu', 'kadıköy', max_gems_per_district=2)
    if route_gems['destination_text']:
        print(route_gems['destination_text'])
