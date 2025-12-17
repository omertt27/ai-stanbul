#!/usr/bin/env python3
"""
M11 Metro Line Official Station Data (2025)

Line: M11 (Gayrettepe - Istanbul Airport)
Color: #00A651 (Green)
Opened: 2023
Stations: 9

Source: Istanbul Metro Official Website
Status: Operational
"""

# M11 Line: Gayrettepe - Istanbul Airport
# Full line opened in January 2023
m11_stations = [
    {
        'name_tr': 'Gayrettepe',
        'name_en': 'Gayrettepe',
        'name_variants': ['gayret tepe', 'gayret'],
        'lat': 41.0683,
        'lon': 29.0139,
        'transfers': ['M2'],  # Transfer to M2 metro
        'order': 1
    },
    {
        'name_tr': 'Kağıthane',
        'name_en': 'Kagithane',
        'name_variants': ['kagıthane', 'kagit hane'],
        'lat': 41.0789,
        'lon': 28.9847,
        'transfers': [],
        'order': 2
    },
    {
        'name_tr': 'Kemerburgaz',
        'name_en': 'Kemerburgaz',
        'name_variants': ['kemer burgaz', 'kemer'],
        'lat': 41.1469,
        'lon': 28.8142,
        'transfers': [],
        'order': 3
    },
    {
        'name_tr': 'Göktürk',
        'name_en': 'Gokturk',
        'name_variants': ['gök türk', 'gok turk'],
        'lat': 41.1711,
        'lon': 28.8500,
        'transfers': [],
        'order': 4
    },
    {
        'name_tr': 'İhsaniye',
        'name_en': 'Ihsaniye',
        'name_variants': ['ihsan', 'ihsaniye'],
        'lat': 41.2167,
        'lon': 28.7833,
        'transfers': [],
        'order': 5
    },
    {
        'name_tr': 'Havalimanı Mahallesi',
        'name_en': 'Airport District',
        'name_variants': ['havalimani mahallesi', 'airport mahallesi'],
        'lat': 41.2528,
        'lon': 28.7611,
        'transfers': [],
        'order': 6
    },
    {
        'name_tr': 'Havalimanı İstasyonu-1',
        'name_en': 'Airport Station-1',
        'name_variants': ['havalimani istasyonu 1', 'airport station 1'],
        'lat': 41.2639,
        'lon': 28.7444,
        'transfers': [],
        'order': 7
    },
    {
        'name_tr': 'Havalimanı İstasyonu-2',
        'name_en': 'Airport Station-2',
        'name_variants': ['havalimani istasyonu 2', 'airport station 2'],
        'lat': 41.2683,
        'lon': 28.7417,
        'transfers': [],
        'order': 8
    },
    {
        'name_tr': 'İstanbul Havalimanı',
        'name_en': 'Istanbul Airport',
        'name_variants': ['istanbul havalimani', 'ist airport', 'new airport', 'yeni havalimani', 'havalimani'],
        'lat': 41.2750,
        'lon': 28.7519,
        'transfers': [],
        'order': 9
    }
]

# Line metadata
m11_line_info = {
    'line_id': 'M11',
    'name_tr': 'M11',
    'name_en': 'M11 Metro',
    'line_type': 'metro',
    'color': '#00A651',  # Green (same as M2)
    'full_name_tr': 'Gayrettepe-İstanbul Havalimanı',
    'full_name_en': 'Gayrettepe-Istanbul Airport',
    'total_stations': 9,
    'length_km': 37.5,
    'travel_time_min': 35,  # Approximate end-to-end
    'opened': 2023
}

print("M11 Metro Line Data")
print("=" * 60)
print(f"Line: {m11_line_info['full_name_en']}")
print(f"Stations: {m11_line_info['total_stations']}")
print(f"Length: {m11_line_info['length_km']} km")
print(f"Travel Time: {m11_line_info['travel_time_min']} min")
print()
print("Stations:")
for station in m11_stations:
    transfers = f" (transfers: {', '.join(station['transfers'])})" if station['transfers'] else ""
    print(f"  {station['order']}. {station['name_en']}{transfers}")
