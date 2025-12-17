"""
Ferry Terminal Data Research
==============================

Ferry Routes in Istanbul (Şehir Hatları - City Lines)

Ferry terminals are point-to-point connections across the Bosphorus and Golden Horn.
Unlike metro/tram lines, ferries operate on routes between terminals, not as continuous lines.

Major Ferry Terminals (2025):
"""

# Ferry terminals organized by region
FERRY_TERMINALS = [
    # (display_name_tr, display_name_en, canonical_id, aliases, lat, lon, connections)
    
    # EUROPEAN SIDE - Bosphorus
    ("Eminönü", "Eminonu", "eminonu_ferry", ["eminonu", "eminönü vapur"], 41.0197, 28.9739, ["T1", "T4"]),
    ("Karaköy", "Karakoy", "karakoy_ferry", ["karakoy", "karaköy vapur"], 41.0236, 28.9753, ["T1", "T4", "F2"]),
    ("Beşiktaş", "Besiktas", "besiktas_ferry", ["besiktas", "beşiktaş vapur"], 41.0425, 29.0019, ["T4"]),
    ("Kabataş", "Kabatas", "kabatas_ferry", ["kabatas", "kabataş vapur"], 41.0356, 28.9914, ["T1", "T4", "F1"]),
    ("Sarıyer", "Sariyer", "sariyer_ferry", ["sariyer", "sarıyer vapur"], 41.1667, 29.0500, []),
    ("Rumeli Kavağı", "Rumeli Kavagi", "rumeli_kavagi_ferry", ["rumeli kavagi", "rumeli kavağı"], 41.1833, 29.0667, []),
    
    # ASIAN SIDE - Bosphorus
    ("Kadıköy", "Kadikoy", "kadikoy_ferry", ["kadikoy", "kadıköy vapur"], 40.9833, 29.0283, ["M4"]),
    ("Üsküdar", "Uskudar", "uskudar_ferry", ["uskudar", "üsküdar vapur"], 41.0256, 29.0178, ["M5", "MARMARAY"]),
    ("Beykoz", "Beykoz", "beykoz_ferry", ["beykoz", "beykoz vapur"], 41.1333, 29.0833, []),
    ("Çengelköy", "Cengelkoy", "cengelkoy_ferry", ["cengelkoy", "çengelköy"], 41.0500, 29.0667, []),
    ("Kanlıca", "Kanlica", "kanlica_ferry", ["kanlica", "kanlıca"], 41.0667, 29.0667, []),
    
    # PRINCES' ISLANDS (Adalar)
    ("Büyükada", "Buyukada", "buyukada_ferry", ["buyukada", "büyükada"], 40.8597, 29.1214, []),
    ("Heybeliada", "Heybeliada", "heybeliada_ferry", ["heybeliada"], 40.8772, 29.0864, []),
    ("Burgazada", "Burgazada", "burgazada_ferry", ["burgazada"], 40.8808, 29.0631, []),
    ("Kınalıada", "Kinaliada", "kinaliada_ferry", ["kinaliada", "kınalıada"], 40.9167, 29.0500, []),
]

# Common ferry routes
FERRY_ROUTES = [
    {
        "name": "Eminönü-Kadıköy",
        "terminals": ["Eminönü", "Kadıköy"],
        "duration_min": 20,
        "frequency_min": 20,
        "popular": True,
    },
    {
        "name": "Kabataş-Üsküdar",
        "terminals": ["Kabataş", "Üsküdar"],
        "duration_min": 15,
        "frequency_min": 15,
        "popular": True,
    },
    {
        "name": "Beşiktaş-Üsküdar",
        "terminals": ["Beşiktaş", "Üsküdar"],
        "duration_min": 15,
        "frequency_min": 20,
        "popular": True,
    },
    {
        "name": "Kabataş-Kadıköy",
        "terminals": ["Kabataş", "Kadıköy"],
        "duration_min": 20,
        "frequency_min": 30,
        "popular": False,
    },
    {
        "name": "Eminönü-Üsküdar",
        "terminals": ["Eminönü", "Üsküdar"],
        "duration_min": 15,
        "frequency_min": 20,
        "popular": True,
    },
    {
        "name": "Kabataş-Princes' Islands",
        "terminals": ["Kabataş", "Kınalıada", "Burgazada", "Heybeliada", "Büyükada"],
        "duration_min": 90,
        "frequency_min": 60,
        "popular": True,
        "note": "Tourist route to islands"
    },
]

print("Istanbul Ferry Terminal Data")
print("=" * 70)
print(f"Total Terminals: {len(FERRY_TERMINALS)}")
print()

print("European Side Terminals:")
for name_tr, name_en, sid, aliases, lat, lon, connections in FERRY_TERMINALS[:6]:
    conn_str = f" → {', '.join(connections)}" if connections else ""
    print(f"  • {name_tr:20} ({name_en}){conn_str}")
    print(f"    ID: {sid:25} Coords: ({lat}, {lon})")

print("\nAsian Side Terminals:")
for name_tr, name_en, sid, aliases, lat, lon, connections in FERRY_TERMINALS[6:11]:
    conn_str = f" → {', '.join(connections)}" if connections else ""
    print(f"  • {name_tr:20} ({name_en}){conn_str}")
    print(f"    ID: {sid:25} Coords: ({lat}, {lon})")

print("\nPrinces' Islands Terminals:")
for name_tr, name_en, sid, aliases, lat, lon, connections in FERRY_TERMINALS[11:]:
    print(f"  • {name_tr:20} ({name_en})")
    print(f"    ID: {sid:25} Coords: ({lat}, {lon})")

print("\n" + "=" * 70)
print("Popular Ferry Routes:")
for route in FERRY_ROUTES:
    if route.get("popular"):
        print(f"  • {route['name']}")
        print(f"    Duration: ~{route['duration_min']} min, Every {route['frequency_min']} min")
        if "note" in route:
            print(f"    Note: {route['note']}")

print(f"\nTotal terminals with transit connections: {sum(1 for t in FERRY_TERMINALS if t[6])}")
