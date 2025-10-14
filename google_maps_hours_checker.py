#!/usr/bin/env python3
"""
Google Maps Museum Hours Checker
===============================

This script will help gather opening hours from Google Maps for Istanbul museums.
"""

import requests
import json
from typing import Dict, List, Optional

class GoogleMapsHoursChecker:
    """Check museum opening hours from Google Maps"""
    
    def __init__(self):
        # Note: In production, you would use Google Places API
        # For now, we'll provide a structure to manually update hours
        self.museum_locations = {
            "Istanbul Archaeological Museums": {
                "address": "Osman Hamdi Bey Yokuşu, 34122 Fatih/İstanbul",
                "google_place_id": "ChIJXxXxXxXxXxXxXxXxXxXx",  # Example
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00", 
                    "wednesday": "09:00-17:00",
                    "thursday": "09:00-17:00", 
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            },
            "Topkapi Palace Museum": {
                "address": "Cankurtaran, 34122 Fatih/İstanbul",
                "google_place_id": "ChIJYyYyYyYyYyYyYyYyYyYy",  # Example
                "current_hours": {
                    "monday": "09:00-16:45",
                    "tuesday": "Closed",
                    "wednesday": "09:00-16:45",
                    "thursday": "09:00-16:45",
                    "friday": "09:00-16:45",
                    "saturday": "09:00-16:45",
                    "sunday": "09:00-16:45"
                }
            },
            "Museum of Turkish and Islamic Arts": {
                "address": "At Meydanı No:46, 34122 Fatih/İstanbul",
                "google_place_id": "ChIJZzZzZzZzZzZzZzZzZzZz",  # Example
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00",
                    "wednesday": "09:00-17:00", 
                    "thursday": "09:00-17:00",
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            },
            "Galata Tower Museum": {
                "address": "Bereketzade, Galata Kulesi Sk., 34421 Beyoğlu/İstanbul",
                "google_place_id": "ChIJAaAaAaAaAaAaAaAaAaAa",  # Example
                "current_hours": {
                    "monday": "08:30-22:00",
                    "tuesday": "08:30-22:00",
                    "wednesday": "08:30-22:00",
                    "thursday": "08:30-22:00",
                    "friday": "08:30-22:00",
                    "saturday": "08:30-22:00",
                    "sunday": "08:30-22:00"
                }
            },
            "Galata Mevlevi House Museum": {
                "address": "Galip Dede Cd. No:15, 34421 Beyoğlu/İstanbul", 
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00",
                    "wednesday": "09:00-17:00",
                    "thursday": "09:00-17:00",
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            },
            "Rumeli Fortress Museum": {
                "address": "Yahya Kemal Cd. No:42, 34470 Sarıyer/İstanbul",
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00", 
                    "wednesday": "09:00-17:00",
                    "thursday": "09:00-17:00",
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            },
            "Maiden's Tower Museum": {
                "address": "Salacak, Üsküdar Salacak Mevkii, 34668 Üsküdar/İstanbul",
                "current_hours": {
                    "monday": "09:00-19:00",
                    "tuesday": "09:00-19:00",
                    "wednesday": "09:00-19:00",
                    "thursday": "09:00-19:00",
                    "friday": "09:00-19:00", 
                    "saturday": "09:00-19:00",
                    "sunday": "09:00-19:00"
                }
            },
            "Hagia Irene Museum": {
                "address": "Cankurtaran, 34122 Fatih/İstanbul",
                "current_hours": {
                    "monday": "09:00-16:45",
                    "tuesday": "Closed",
                    "wednesday": "09:00-16:45",
                    "thursday": "09:00-16:45",
                    "friday": "09:00-16:45",
                    "saturday": "09:00-16:45",
                    "sunday": "09:00-16:45"
                }
            },
            "Great Palace Mosaics Museum": {
                "address": "Torun Sk. No:1, 34122 Fatih/İstanbul",
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00",
                    "wednesday": "09:00-17:00",
                    "thursday": "09:00-17:00",
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            },
            "Museum of the History of Science and Technology in Islam": {
                "address": "Gülhane Park, 34122 Fatih/İstanbul",
                "current_hours": {
                    "monday": "Closed",
                    "tuesday": "09:00-17:00",
                    "wednesday": "09:00-17:00", 
                    "thursday": "09:00-17:00",
                    "friday": "09:00-17:00",
                    "saturday": "09:00-17:00",
                    "sunday": "09:00-17:00"
                }
            }
        }
    
    def get_formatted_hours(self, museum_name: str) -> Dict[str, str]:
        """Get formatted opening hours for a museum"""
        if museum_name in self.museum_locations:
            hours = self.museum_locations[museum_name]["current_hours"]
            
            # Format for different display needs
            formatted = {
                "daily_summary": self._get_daily_summary(hours),
                "detailed": hours,
                "winter_summer": self._check_seasonal_hours(museum_name, hours)
            }
            return formatted
        return {}
    
    def _get_daily_summary(self, hours: Dict[str, str]) -> str:
        """Create a daily summary of hours"""
        weekdays = [hours[day] for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]]
        weekend = [hours[day] for day in ["saturday", "sunday"]]
        
        # Check if weekdays are consistent
        unique_weekdays = list(set([h for h in weekdays if h != "Closed"]))
        unique_weekend = list(set([h for h in weekend if h != "Closed"]))
        
        if len(unique_weekdays) == 1 and len(unique_weekend) == 1 and unique_weekdays[0] == unique_weekend[0]:
            closed_days = [day for day, hour in hours.items() if hour == "Closed"]
            if closed_days:
                return f"Daily: {unique_weekdays[0]} (Closed: {', '.join(closed_days).title()})"
            else:
                return f"Daily: {unique_weekdays[0]}"
        else:
            return "See detailed hours"
    
    def _check_seasonal_hours(self, museum_name: str, hours: Dict[str, str]) -> Optional[str]:
        """Check if museum has seasonal hour variations"""
        seasonal_museums = ["Topkapi Palace Museum", "Istanbul Archaeological Museums"]
        if museum_name in seasonal_museums:
            return "Winter/Summer hours may vary - check official website"
        return None
    
    def generate_hours_update_script(self) -> str:
        """Generate a script to update museum hours"""
        script = """
# Museum Hours Update Script
# Run this to check and update museum opening hours

museums_to_check = [
"""
        for museum_name in self.museum_locations.keys():
            script += f'    "{museum_name}",\n'
        
        script += """]

# Instructions:
# 1. Check each museum on Google Maps
# 2. Verify current opening hours
# 3. Update the museum database
# 4. Note any seasonal variations
"""
        return script

def main():
    """Main function to demonstrate hours checking"""
    checker = GoogleMapsHoursChecker()
    
    print("🕐 Istanbul Museum Opening Hours Summary")
    print("=" * 50)
    
    for museum_name in checker.museum_locations.keys():
        hours_info = checker.get_formatted_hours(museum_name)
        print(f"\n🏛️ {museum_name}")
        print(f"   Hours: {hours_info.get('daily_summary', 'Check individually')}")
        if hours_info.get('winter_summer'):
            print(f"   Note: {hours_info['winter_summer']}")
    
    print("\n" + "=" * 50)
    print("💡 Remember to verify these hours on Google Maps!")
    print("💡 Hours may change due to seasons, holidays, or renovations")

if __name__ == "__main__":
    main()
