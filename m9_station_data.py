"""
M9 Metro Line Station Data Research
====================================

M9 Line: Olimpiyat - İkitelli Sanayi (Currently operational since 2023)

Official route and stations (from IBB and official metro maps):
1. Olimpiyat (Transfer to M3)
2. Ikitelli Sanayi (İkitelli Industrial - western terminus)

NOTE: The old loader had incorrect data with stations like "Ataköy" and "Sahil" 
which don't belong to M9. That was likely confused with planned extensions.

Current Operational Status (2025):
- M9 is a SHORT line, currently only 2 stations operational
- Opened: October 2023
- Length: ~2.5 km
- Serves the İkitelli industrial zone

Transfer Points:
- Olimpiyat: M3 (Olimpiyat-Başakşehir-Kayaşehir)

Planned Extensions (NOT YET OPERATIONAL):
- Phase 2: Extension to Halkalı (planned)
- This would add more stations, but they're not operational yet

Correct Station List for 2025:
"""

# M9 Line stations (2025 official data)
M9_STATIONS = [
    # (display_name, english_name, canonical_id, aliases, lat, lon, transfer_lines)
    ("Olimpiyat", "Olimpiyat", "olimpiyat", ["olimpiyat"], 41.0744, 28.7644, ["M3", "M9"]),
    ("İkitelli Sanayi", "Ikitelli Sanayi", "ikitelli_sanayi", ["ikitelli sanayi", "ikitelli"], 41.0656, 28.7828, ["M9"]),
]

# Line metadata
M9_LINE = {
    "line_id": "M9",
    "short_name": "M9",
    "full_name": "M9 Metro",
    "type": "metro",
    "color": "#8E3994",  # Purple
    "route_tr": "Olimpiyat-İkitelli Sanayi",
    "route_en": "Olimpiyat-İkitelli Industrial",
    "status": "operational",
    "opened": "2023-10",
    "length_km": 2.5,
    "station_count": 2,
}

print("M9 Metro Line Data")
print("=" * 60)
print(f"Line: {M9_LINE['route_tr']}")
print(f"Status: {M9_LINE['status']}")
print(f"Stations: {M9_LINE['station_count']}")
print(f"Color: {M9_LINE['color']}")
print()
print("Stations:")
for i, (name_tr, name_en, sid, aliases, lat, lon, lines) in enumerate(M9_STATIONS, 1):
    transfers = ', '.join(l for l in lines if l != 'M9')
    transfer_info = f" (Transfer: {transfers})" if transfers else ""
    print(f"{i}. {name_tr} / {name_en}{transfer_info}")
    print(f"   ID: {sid}, Coords: ({lat}, {lon})")
