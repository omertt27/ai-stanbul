"""
T5 Tram Line Station Data Research
====================================

T5 Line: Cibali - Alibeyköy (European side tram)

Official route and stations (from IBB and official tram maps):

Current Operational Status (2025):
- T5 is a tram line on the European side
- Opened: 2021
- Length: ~6.5 km
- Serves northern European side neighborhoods

Station List (from south to north):
"""

# T5 Line stations (2025 official data)
# Based on IBB official tram network maps
T5_STATIONS = [
    # (display_name_tr, display_name_en, canonical_id, aliases, lat, lon, transfer_lines)
    ("Cibali", "Cibali", "cibali", ["cibali"], 41.0325, 28.9508, []),
    ("Fener", "Fener", "fener", ["fener"], 41.0364, 28.9481, []),
    ("Balat", "Balat", "balat", ["balat"], 41.0419, 28.9469, []),
    ("Ayvansaray", "Ayvansaray", "ayvansaray", ["ayvansaray"], 41.0467, 28.9453, []),
    ("Eyüpsultan", "Eyupsultan", "eyupsultan", ["eyup", "eyupsultan"], 41.0508, 28.9397, []),
    ("Silahtar", "Silahtar", "silahtar", ["silahtar"], 41.0558, 28.9392, []),
    ("Defterdar", "Defterdar", "defterdar", ["defterdar"], 41.0603, 28.9397, []),
    ("Halıcıoğlu", "Halicioglu", "halicioglu", ["halicioglu"], 41.0653, 28.9400, []),
    ("Sütlüce", "Sutluce", "sutluce", ["sutluce"], 41.0697, 28.9414, []),
    ("Piripaşa", "Piripasa", "piripasa", ["piripasa"], 41.0739, 28.9436, []),
    ("Alibeyköy", "Alibeykoy", "alibeykoy_t5", ["alibeykoy"], 41.0778, 28.9458, ["M7"]),
]

# Line metadata
T5_LINE = {
    "line_id": "T5",
    "short_name": "T5",
    "full_name": "T5 Tram",
    "type": "tram",
    "color": "#00A0E1",  # Light Blue (official IBB color)
    "route_tr": "Cibali-Alibeyköy",
    "route_en": "Cibali-Alibeykoy",
    "status": "operational",
    "opened": "2021",
    "length_km": 6.5,
    "station_count": len(T5_STATIONS),
}

print("T5 Tram Line Data")
print("=" * 60)
print(f"Line: {T5_LINE['route_tr']}")
print(f"Status: {T5_LINE['status']}")
print(f"Stations: {T5_LINE['station_count']}")
print(f"Color: {T5_LINE['color']}")
print(f"Length: {T5_LINE['length_km']} km")
print()
print("All Stations:")
for i, (name_tr, name_en, sid, aliases, lat, lon, lines) in enumerate(T5_STATIONS, 1):
    transfers = ', '.join(lines) if lines else "No transfers"
    print(f"{i:2}. {name_tr:20} / {name_en:20} - {transfers}")
    print(f"    ID: {sid:20} Coords: ({lat}, {lon})")

print(f"\nTotal stations with transfers: {sum(1 for s in T5_STATIONS if s[6])}")
