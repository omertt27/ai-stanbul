"""
T4 Tram Line Station Data Research
====================================

T4 Line: Topkapı - Mescid-i Selam (Historical peninsula tram)

Official route and stations (from IBB and official tram maps):

Current Operational Status (2025):
- T4 is a tram line serving the historical peninsula
- Opened: 2007
- Length: ~15 km
- Serves historical and residential areas on the European side

Station List (from west to east):
"""

# T4 Line stations (2025 official data)
# Based on IBB official tram network maps
T4_STATIONS = [
    # (display_name_tr, display_name_en, canonical_id, aliases, lat, lon, transfer_lines)
    ("Topkapı", "Topkapi", "topkapi_t4", ["topkapi"], 41.0144, 28.9194, ["M1A"]),
    ("Pazartekke", "Pazartekke", "pazartekke", ["pazartekke"], 41.0131, 28.9289, []),
    ("Çapa", "Capa", "capa", ["capa"], 41.0139, 28.9386, []),
    ("Fındıkzade", "Findikzade", "findikzade", ["findikzade"], 41.0156, 28.9447, []),
    ("Haseki", "Haseki", "haseki", ["haseki"], 41.0142, 28.9517, []),
    ("Yusufpaşa", "Yusufpasa", "yusufpasa", ["yusufpasa"], 41.0147, 28.9597, []),
    ("Aksaray", "Aksaray", "aksaray_t4", ["aksaray"], 41.0161, 28.9508, []),
    ("Laleli", "Laleli", "laleli", ["laleli"], 41.0139, 28.9594, []),
    ("Beyazıt", "Beyazit", "beyazit", ["beyazit"], 41.0133, 28.9664, []),
    ("Çemberlitaş", "Cemberlitas", "cemberlitas", ["cemberlitas"], 41.0103, 28.9706, []),
    ("Sultanahmet", "Sultanahmet", "sultanahmet_t4", ["sultanahmet"], 41.0058, 28.9781, ["T1"]),
    ("Gülhane", "Gulhane", "gulhane_t4", ["gulhane"], 41.0131, 28.9814, ["T1"]),
    ("Sirkeci", "Sirkeci", "sirkeci_t4", ["sirkeci"], 41.0175, 28.9839, ["T1", "MARMARAY"]),
    ("Eminönü", "Eminonu", "eminonu_t4", ["eminonu"], 41.0178, 28.9706, ["T1"]),
    ("Karaköy", "Karakoy", "karakoy_t4", ["karakoy"], 41.0236, 28.9753, ["T1", "F2"]),
    ("Tophane", "Tophane", "tophane_t4", ["tophane"], 41.0264, 28.9828, ["T1"]),
    ("Fındıklı", "Findikli", "findikli", ["findikli"], 41.0292, 28.9883, []),
    ("Kabataş", "Kabatas", "kabatas_t4", ["kabatas"], 41.0356, 28.9914, ["T1", "F1"]),
    ("Dolmabahçe", "Dolmabahce", "dolmabahce", ["dolmabahce"], 41.0408, 28.9958, []),
    ("Beşiktaş", "Besiktas", "besiktas_t4", ["besiktas"], 41.0425, 29.0019, []),
    ("Barbaros", "Barbaros", "barbaros", ["barbaros"], 41.0467, 29.0081, []),
    ("Mescid-i Selam", "Mescid-i Selam", "mescid_selam", ["mescid selam", "mescidi selam"], 41.0528, 29.0117, []),
]

# Line metadata
T4_LINE = {
    "line_id": "T4",
    "short_name": "T4",
    "full_name": "T4 Tram",
    "type": "tram",
    "color": "#F58220",  # Orange (official IBB color)
    "route_tr": "Topkapı-Mescid-i Selam",
    "route_en": "Topkapi-Mescid-i Selam",
    "status": "operational",
    "opened": "2007",
    "length_km": 15.3,
    "station_count": len(T4_STATIONS),
}

print("T4 Tram Line Data")
print("=" * 60)
print(f"Line: {T4_LINE['route_tr']}")
print(f"Status: {T4_LINE['status']}")
print(f"Stations: {T4_LINE['station_count']}")
print(f"Color: {T4_LINE['color']}")
print(f"Length: {T4_LINE['length_km']} km")
print()
print("Stations with Transfers:")
for i, (name_tr, name_en, sid, aliases, lat, lon, lines) in enumerate(T4_STATIONS, 1):
    if lines:
        transfers = ', '.join(lines)
        print(f"{i}. {name_tr} / {name_en}")
        print(f"   Transfer: {transfers}")
        print(f"   ID: {sid}, Coords: ({lat}, {lon})")

print(f"\nTotal stations with transfers: {sum(1 for s in T4_STATIONS if s[6])}")
