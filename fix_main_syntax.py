#!/usr/bin/env python3
"""
Fix syntax error in main.py around line 1458
"""

# Read the file
with open('backend/main.py', 'r') as f:
    lines = f.readlines()

# Fix the problematic lines (1457-1461)
# Line 1457: enhanced_api = EnhancedAPIService()
# Lines 1458-1461 should be in an except block
fixed_lines = lines.copy()

# Insert the missing lines after line 1457
insert_after_1457 = [
    "                    places_data = enhanced_api.search_restaurants_enhanced(\n",
    "                        location=search_location, \n",
    "                        keyword=user_input\n",
    "                    )\n",
    "                    \n",
    "                    # Weather context is already included in enhanced search\n",
    "                    weather_context = places_data.get('weather_context', {})\n",
    "                    weather_info = f\"Current: {weather_context.get('current_temp', 'N/A')}°C, {weather_context.get('condition', 'Unknown')}\"\n",
    "                    \n",
    "                except Exception as e:\n",
    "                    logger.warning(f\"Enhanced API service failed: {e}\")\n",
    "                    # Fallback to basic service\n",
    "                    try:\n",
    "                        google_client = GooglePlacesClient()\n",
    "                        places_data = google_client.search_restaurants(location=search_location, keyword=user_input)\n",
    "                        \n"
]

# Remove the problematic lines 1458-1461 and replace them
# Line 1457 (index 1456) stays
# Lines 1458-1461 (indices 1457-1460) get replaced
new_lines = lines[:1457] + insert_after_1457 + lines[1461:]

# Write back
with open('backend/main.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed syntax error in main.py around line 1458")
