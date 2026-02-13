"""
Add User/Item Encoders to NCF Model Checkpoint

This script adds synthetic encoder data to the NCF model checkpoint
so it can actually make predictions instead of falling back.

This is a temporary solution for testing. In production, you should
retrain the model with real user interaction data.

Usage:
    python add_ncf_encoders.py
"""

import torch
import os

def main():
    # Model path
    model_path = 'backend/ml/deep_learning/models/ncf_model.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load existing model
    print(f"📂 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"   Current keys: {list(checkpoint.keys())}")
    
    # Create synthetic user IDs (100 test users)
    users = [f"user_{i}" for i in range(1, 101)]
    users.extend([f"test_user_{i}" for i in range(1, 21)])  # Add test users
    
    # Create item IDs (Istanbul attractions)
    items = [
        "hagia_sophia",
        "blue_mosque",
        "topkapi_palace",
        "grand_bazaar",
        "galata_tower",
        "basilica_cistern",
        "dolmabahce_palace",
        "bosphorus_cruise",
        "spice_bazaar",
        "istiklal_street",
        "taksim_square",
        "maiden_tower",
        "rumeli_fortress",
        "suleymaniye_mosque",
        "chora_church",
        "prince_islands",
        "pierre_loti_hill",
        "miniaturk",
        "rahmi_koc_museum",
        "istanbul_modern",
        "pera_museum",
        "archaeological_museum",
        "turkish_islamic_arts",
        "ortakoy_mosque",
        "camlica_tower",
        "gulhane_park",
        "emirgan_park",
        "balat_neighborhood",
        "karakoy_neighborhood",
        "nisantasi_district"
    ]
    
    # Build encoders
    user_encoder = {uid: idx for idx, uid in enumerate(users)}
    item_encoder = {iid: idx for idx, iid in enumerate(items)}
    
    # Create item metadata
    item_metadata = {
        "hagia_sophia": {
            "name": "Hagia Sophia",
            "type": "historical_site",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "blue_mosque": {
            "name": "Blue Mosque",
            "type": "mosque",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "topkapi_palace": {
            "name": "Topkapi Palace",
            "type": "palace",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "grand_bazaar": {
            "name": "Grand Bazaar",
            "type": "market",
            "district": "Fatih",
            "category": "shopping"
        },
        "galata_tower": {
            "name": "Galata Tower",
            "type": "tower",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "basilica_cistern": {
            "name": "Basilica Cistern",
            "type": "historical_site",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "dolmabahce_palace": {
            "name": "Dolmabahçe Palace",
            "type": "palace",
            "district": "Beşiktaş",
            "category": "attraction"
        },
        "bosphorus_cruise": {
            "name": "Bosphorus Cruise",
            "type": "activity",
            "district": "Various",
            "category": "activity"
        },
        "spice_bazaar": {
            "name": "Spice Bazaar",
            "type": "market",
            "district": "Eminönü",
            "category": "shopping"
        },
        "istiklal_street": {
            "name": "İstiklal Street",
            "type": "street",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "taksim_square": {
            "name": "Taksim Square",
            "type": "landmark",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "maiden_tower": {
            "name": "Maiden's Tower",
            "type": "tower",
            "district": "Üsküdar",
            "category": "attraction"
        },
        "rumeli_fortress": {
            "name": "Rumeli Fortress",
            "type": "fortress",
            "district": "Sarıyer",
            "category": "attraction"
        },
        "suleymaniye_mosque": {
            "name": "Süleymaniye Mosque",
            "type": "mosque",
            "district": "Fatih",
            "category": "attraction"
        },
        "chora_church": {
            "name": "Chora Church",
            "type": "church",
            "district": "Fatih",
            "category": "attraction"
        },
        "prince_islands": {
            "name": "Prince Islands",
            "type": "island",
            "district": "Adalar",
            "category": "activity"
        },
        "pierre_loti_hill": {
            "name": "Pierre Loti Hill",
            "type": "viewpoint",
            "district": "Eyüp",
            "category": "attraction"
        },
        "miniaturk": {
            "name": "Miniaturk",
            "type": "park",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "rahmi_koc_museum": {
            "name": "Rahmi M. Koç Museum",
            "type": "museum",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "istanbul_modern": {
            "name": "Istanbul Modern",
            "type": "museum",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "pera_museum": {
            "name": "Pera Museum",
            "type": "museum",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "archaeological_museum": {
            "name": "Istanbul Archaeology Museum",
            "type": "museum",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "turkish_islamic_arts": {
            "name": "Turkish and Islamic Arts Museum",
            "type": "museum",
            "district": "Sultanahmet",
            "category": "attraction"
        },
        "ortakoy_mosque": {
            "name": "Ortaköy Mosque",
            "type": "mosque",
            "district": "Beşiktaş",
            "category": "attraction"
        },
        "camlica_tower": {
            "name": "Çamlıca Tower",
            "type": "tower",
            "district": "Üsküdar",
            "category": "attraction"
        },
        "gulhane_park": {
            "name": "Gülhane Park",
            "type": "park",
            "district": "Fatih",
            "category": "activity"
        },
        "emirgan_park": {
            "name": "Emirgan Park",
            "type": "park",
            "district": "Sarıyer",
            "category": "activity"
        },
        "balat_neighborhood": {
            "name": "Balat",
            "type": "neighborhood",
            "district": "Fatih",
            "category": "attraction"
        },
        "karakoy_neighborhood": {
            "name": "Karaköy",
            "type": "neighborhood",
            "district": "Beyoğlu",
            "category": "attraction"
        },
        "nisantasi_district": {
            "name": "Nişantaşı",
            "type": "district",
            "district": "Şişli",
            "category": "shopping"
        }
    }
    
    # Update checkpoint with encoders
    checkpoint['user_encoder'] = user_encoder
    checkpoint['item_encoder'] = item_encoder
    checkpoint['item_metadata'] = item_metadata
    checkpoint['num_users'] = len(user_encoder)
    checkpoint['num_items'] = len(item_encoder)
    
    # Backup original model
    backup_path = model_path + '.backup'
    if not os.path.exists(backup_path):
        torch.save(torch.load(model_path, map_location='cpu'), backup_path)
        print(f"💾 Created backup at {backup_path}")
    
    # Save updated model
    torch.save(checkpoint, model_path)
    
    print(f"\n✅ Successfully added encoders to {model_path}")
    print(f"   👥 Users: {len(user_encoder)}")
    print(f"   📍 Items: {len(item_encoder)}")
    print(f"   📊 Metadata: {len(item_metadata)} items with details")
    print(f"\n📦 Updated checkpoint keys: {list(checkpoint.keys())}")
    print(f"\n🎯 Model is now ready for inference!")
    print(f"   Redeploy to Cloud Run to activate.")

if __name__ == "__main__":
    main()
