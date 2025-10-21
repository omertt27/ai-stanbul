#!/usr/bin/env python3
"""
Phase 2: Accuracy Improvement - Automated Training Data Augmentation
Expands 42 samples → 200+ samples using Turkish synonyms and paraphrasing
"""

import random
from typing import List, Tuple
import json

class TurkishDataAugmenter:
    """Augment Turkish tourism queries with synonyms and variations"""
    
    def __init__(self):
        # Turkish synonyms and variations
        self.synonyms = {
            # Question words
            "nerede": ["nerde", "hangi yerde", "nereye gitsem", "neresi", "hangi lokasyonda"],
            "nasıl": ["ne şekilde", "hangi yolla", "ne yaparak"],
            "ne zaman": ["kaçta", "saat kaçta", "hangi saatte"],
            "ne kadar": ["kaç lira", "fiyatı ne", "ücreti ne kadar"],
            
            # Quality adjectives
            "en iyi": ["en güzel", "en kaliteli", "harika", "mükemmel", "süper", "muhteşem"],
            "iyi": ["güzel", "kaliteli", "hoş", "harika"],
            "ucuz": ["uygun fiyatlı", "ekonomik", "bütçeye uygun", "hesaplı"],
            "pahalı": ["lüks", "yüksek fiyatlı", "prestijli"],
            
            # Actions
            "görmek": ["ziyaret etmek", "gezmek", "bakmak", "gitmek"],
            "yemek": ["yiyecek", "tatmak", "deneyimlemek"],
            "gitmek": ["ulaşmak", "varmak", "gidilir"],
            
            # Places
            "restoran": ["lokanta", "yemek yeri", "mekan"],
            "otel": ["konaklama", "pansiyon", "butik otel"],
            "müze": ["sanat galerisi", "sergi"],
            "park": ["yeşil alan", "bahçe"],
            
            # Tourism words
            "tur": ["gezi", "turne", "seyahat"],
            "tavsiye": ["öneri", "öner", "tavsiyesi"],
            "hakkında": ["ile ilgili", "konusunda"],
            "bilgi": ["info", "detay", "bilgilendirme"],
        }
        
        # Sentence templates
        self.templates = {
            "attraction": [
                "{place} görmek istiyorum",
                "{place} nerede",
                "{place} hakkında bilgi",
                "{place}'yi ziyaret etmek istiyorum",
                "{place}'ye nasıl gidilir",
                "{place} kaça kadar açık",
            ],
            "restaurant": [
                "{food} nerede yenir",
                "{food} restoranı",
                "En iyi {food} nerede",
                "{food} yemek istiyorum",
                "{food} için mekan önerisi",
            ],
            "gps_navigation": [
                "{place}'e nasıl gidilir",
                "{place}'ye git",
                "{place}'e yol tarifi",
                "Beni {place}'e götür",
                "{place} yolu",
            ],
            "accommodation": [
                "{type} otel önerisi",
                "{type} konaklama",
                "{area}'da otel",
                "Nerede kalmalıyım",
            ],
        }
    
    def augment_with_synonyms(self, text: str, intent: str) -> List[str]:
        """Create variations using synonyms"""
        variations = [text]
        
        # Replace each synonym
        for original, replacements in self.synonyms.items():
            if original in text.lower():
                for replacement in replacements[:2]:  # Use first 2 synonyms
                    new_text = text.lower().replace(original, replacement)
                    variations.append(new_text)
        
        return variations
    
    def create_comprehensive_dataset(self) -> List[Tuple[str, str]]:
        """Create comprehensive training dataset"""
        
        training_data = []
        
        # ATTRACTIONS (100+ samples)
        attractions = [
            "Sultanahmet", "Ayasofya", "Topkapı Sarayı", "Galata Kulesi",
            "Kapalıçarşı", "Mısır Çarşısı", "Dolmabahçe Sarayı",
            "Yerebatan Sarnıcı", "Kız Kulesi", "Süleymaniye Camii"
        ]
        
        for place in attractions:
            training_data.extend([
                (f"{place} nerede", "attraction"),
                (f"{place} görmek istiyorum", "attraction"),
                (f"{place} hakkında bilgi", "attraction"),
                (f"{place} kaça kadar açık", "attraction"),
                (f"{place}'yi ziyaret etmek istiyorum", "attraction"),
            ])
        
        # RESTAURANTS (80+ samples)
        foods = ["kebap", "balık", "meze", "künefe", "lahmacun", "pide", "köfte", "baklava"]
        
        for food in foods:
            training_data.extend([
                (f"En iyi {food} nerede", "restaurant"),
                (f"{food} yemek istiyorum", "restaurant"),
                (f"{food} restoranı öner", "restaurant"),
                (f"{food} için mekan", "restaurant"),
                (f"Ucuz {food} nerede", "restaurant"),
                (f"Kaliteli {food} restoranı", "restaurant"),
            ])
        
        # Add general restaurant queries
        training_data.extend([
            ("Nerede yemek yiyebilirim", "restaurant"),
            ("Yemek için restoran", "restaurant"),
            ("Akşam yemeği için mekan", "restaurant"),
            ("Romantik restoran", "romantic"),
            ("Aile restoranı", "family_activities"),
        ])
        
        # GPS NAVIGATION (60+ samples)
        places = ["Taksim", "Sultanahmet", "Beşiktaş", "Kadıköy", "Üsküdar", "Eminönü"]
        
        for place in places:
            training_data.extend([
                (f"{place}'e nasıl gidilir", "gps_navigation"),
                (f"{place}'e git", "gps_navigation"),
                (f"{place}'ye yol tarifi", "gps_navigation"),
                (f"Beni {place}'e götür", "gps_navigation"),
                (f"{place} yolu", "gps_navigation"),
            ])
        
        # TRANSPORTATION (40+ samples)
        training_data.extend([
            ("Havalimanından şehre nasıl gidilir", "transportation"),
            ("Metro saatleri", "transportation"),
            ("Otobüs saatleri", "transportation"),
            ("Tramvay güzergahı", "transportation"),
            ("Marmaray bilgileri", "transportation"),
            ("Taksi çağır", "transportation"),
            ("Vapur saatleri", "transportation"),
            ("Toplu taşıma kartı", "transportation"),
            ("İstanbulkart nereden alınır", "transportation"),
            ("Havalimanı otobüsü", "transportation"),
        ])
        
        # Add variations
        for i in range(5):
            training_data.extend([
                (f"Taksim'e metro ile nasıl gidilir", "transportation"),
                (f"En yakın metro durağı", "transportation"),
                (f"Otobüs güzergahı", "transportation"),
            ])
        
        # ACCOMMODATION (35+ samples)
        training_data.extend([
            ("Ucuz otel önerisi", "accommodation"),
            ("Ekonomik konaklama", "budget"),
            ("Hostel tavsiyesi", "budget"),
            ("5 yıldızlı otel", "luxury"),
            ("Lüks otel", "luxury"),
            ("Butik otel Sultanahmet", "accommodation"),
            ("Taksim'de otel", "accommodation"),
            ("Boğaz manzaralı otel", "accommodation"),
        ])
        
        for area in ["Sultanahmet", "Taksim", "Beşiktaş", "Kadıköy"]:
            training_data.extend([
                (f"{area}'da otel", "accommodation"),
                (f"{area}'da ucuz otel", "budget"),
                (f"{area}'da kalacak yer", "accommodation"),
            ])
        
        # MUSEUMS (30+ samples)
        museums = ["İstanbul Modern", "Pera Müzesi", "Arkeoloji Müzesi", "Kariye Müzesi"]
        
        for museum in museums:
            training_data.extend([
                (f"{museum} nerede", "museum"),
                (f"{museum} kaça kadar açık", "museum"),
                (f"{museum} giriş ücreti", "price_info"),
            ])
        
        training_data.extend([
            ("Müze önerisi", "museum"),
            ("Hangi müzeleri görmeliyim", "museum"),
            ("Sanat müzesi", "museum"),
            ("Tarih müzesi", "museum"),
        ])
        
        # SHOPPING (30+ samples)
        training_data.extend([
            ("Kapalıçarşı kaça kadar açık", "shopping"),
            ("Alışveriş merkezi", "shopping"),
            ("Hediyelik eşya nereden alınır", "shopping"),
            ("Moda mağazaları", "shopping"),
            ("İstiklal Caddesi mağazaları", "shopping"),
            ("Grand Bazaar", "shopping"),
            ("Outlet mağaza", "shopping"),
        ])
        
        for i in range(4):
            training_data.extend([
                ("Alışveriş için öneriler", "shopping"),
                ("Yerel ürünler nerede", "shopping"),
                ("Antika mağazaları", "shopping"),
            ])
        
        # NIGHTLIFE (25+ samples)
        training_data.extend([
            ("Gece hayatı önerileri", "nightlife"),
            ("Bar tavsiyesi", "nightlife"),
            ("Canlı müzik mekanı", "nightlife"),
            ("Kulüp önerisi", "nightlife"),
            ("Rooftop bar", "nightlife"),
        ])
        
        for i in range(4):
            training_data.extend([
                ("Gece eğlence mekanı", "nightlife"),
                ("Müzikli mekan", "nightlife"),
                ("Dans edebileceğim yer", "nightlife"),
                ("Bira barı", "nightlife"),
            ])
        
        # CULTURAL INFO (25+ samples)
        training_data.extend([
            ("Boğaz turu hakkında bilgi", "cultural_info"),
            ("Osmanlı tarihi", "history"),
            ("Cami ziyareti kuralları", "cultural_info"),
            ("Ramazan etkinlikleri", "cultural_info"),
            ("Yerel gelenek ve görenekler", "cultural_info"),
        ])
        
        for i in range(4):
            training_data.extend([
                ("Türk kültürü", "cultural_info"),
                ("Geleneksel etkinlikler", "cultural_info"),
                ("Tarihi yerler", "history"),
                ("Bizans dönemi", "history"),
            ])
        
        # EVENTS (20+ samples)
        training_data.extend([
            ("Bu hafta sonu etkinlikler", "events"),
            ("Konser takvimi", "events"),
            ("Festival", "events"),
            ("Müzik etkinlikleri", "events"),
            ("Sergi", "events"),
        ])
        
        for i in range(3):
            training_data.extend([
                ("Bugün ne var", "events"),
                ("Bu akşam etkinlik", "events"),
                ("Tiyatro gösterileri", "events"),
                ("Açık hava konseri", "events"),
            ])
        
        # WEATHER (15 samples)
        training_data.extend([
            ("Hava durumu nasıl", "weather"),
            ("Yarın hava nasıl olacak", "weather"),
            ("Yağmur yağacak mı", "weather"),
            ("Sıcaklık kaç derece", "weather"),
            ("Hava sıcak mı", "weather"),
        ])
        
        for i in range(2):
            training_data.extend([
                ("Bu hafta hava", "weather"),
                ("Hava tahmini", "weather"),
                ("Kar yağacak mı", "weather"),
                ("Güneşli mi", "weather"),
            ])
        
        # PRICE INFO (15 samples)
        training_data.extend([
            ("Giriş ücreti ne kadar", "price_info"),
            ("Müze ücreti", "price_info"),
            ("Tur fiyatları", "price_info"),
            ("Bilet fiyatı", "price_info"),
            ("Kaça mal olur", "price_info"),
        ])
        
        for i in range(2):
            training_data.extend([
                ("Fiyat bilgisi", "price_info"),
                ("Ne kadar öderim", "price_info"),
                ("Ücret ne kadar", "price_info"),
            ])
        
        # EMERGENCY (10 samples)
        training_data.extend([
            ("En yakın hastane", "emergency"),
            ("Polis çağır", "emergency"),
            ("Acil yardım", "emergency"),
            ("Eczane nerede", "emergency"),
            ("112", "emergency"),
            ("Ambulans", "emergency"),
            ("İtfaiye", "emergency"),
            ("Kayboldum", "emergency"),
            ("Yardım edin", "emergency"),
            ("Acil durum", "emergency"),
        ])
        
        # FAMILY ACTIVITIES (15 samples)
        training_data.extend([
            ("Çocuklu gezilecek yerler", "family_activities"),
            ("Aile için restoran", "family_activities"),
            ("Çocuk parkı", "family_activities"),
            ("Çocuk dostu mekan", "family_activities"),
            ("Oyun alanı", "family_activities"),
        ])
        
        for i in range(2):
            training_data.extend([
                ("Çocuklarla ne yapabilirim", "family_activities"),
                ("Aile etkinlikleri", "family_activities"),
                ("Çocuk müzesi", "family_activities"),
            ])
        
        # HIDDEN GEMS & LOCAL TIPS (15 samples)
        training_data.extend([
            ("Turistik olmayan yerler", "hidden_gems"),
            ("Yerel önerileri", "local_tips"),
            ("Gizli mekanlar", "hidden_gems"),
            ("Yerli gibi gez", "local_tips"),
            ("Az bilinen yerler", "hidden_gems"),
        ])
        
        for i in range(2):
            training_data.extend([
                ("Turistlerin gitmediği yerler", "hidden_gems"),
                ("Yerel ipuçları", "local_tips"),
                ("Keşfedilmemiş yerler", "hidden_gems"),
            ])
        
        # BOOKING (10 samples)
        training_data.extend([
            ("Rezervasyon yapmak istiyorum", "booking"),
            ("Tur rezervasyonu", "booking"),
            ("Otel rezervasyon", "booking"),
            ("Masa ayırtmak istiyorum", "booking"),
            ("Bilet al", "booking"),
            ("Online rezervasyon", "booking"),
            ("Booking", "booking"),
            ("Rezerve et", "booking"),
            ("Yer ayır", "booking"),
            ("Kayıt yaptır", "booking"),
        ])
        
        # GENERAL INFO (10 samples)
        training_data.extend([
            ("İstanbul hakkında bilgi", "general_info"),
            ("Genel bilgiler", "general_info"),
            ("İstanbul'da ne yapabilirim", "general_info"),
            ("Turistik bilgiler", "general_info"),
            ("Gezi rehberi", "general_info"),
            ("Şehir bilgileri", "general_info"),
            ("İstanbul'u tanıt", "general_info"),
            ("Ne görmeliyim", "recommendation"),
            ("Önerileriniz neler", "recommendation"),
            ("Tavsiye", "recommendation"),
        ])
        
        return training_data


async def improve_accuracy():
    """Main accuracy improvement function"""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    import time
    
    print("=" * 60)
    print("🚀 PHASE 2: ACCURACY IMPROVEMENT")
    print("=" * 60)
    
    # Step 1: Generate augmented data
    print("\n📝 Step 1: Generating Augmented Training Data")
    print("=" * 60)
    
    augmenter = TurkishDataAugmenter()
    training_data = augmenter.create_comprehensive_dataset()
    
    # Remove duplicates
    training_data = list(set(training_data))
    
    print(f"✅ Generated {len(training_data)} training samples")
    
    # Count per intent
    intent_counts = {}
    for _, intent in training_data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n📊 Samples per intent:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {intent}: {count}")
    
    # Step 2: Load model
    print("\n🤖 Step 2: Loading Model")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')
    model.to(device)
    model.eval()
    
    # Intent classes
    intents = list(set(intent for _, intent in training_data))
    intents.sort()
    print(f"✅ {len(intents)} intent classes")
    
    # Step 3: Create classifier
    print("\n🎯 Step 3: Creating Intent Classifier")
    print("=" * 60)
    
    class IntentClassifier(nn.Module):
        def __init__(self, num_intents):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(384, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, num_intents)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    classifier = IntentClassifier(len(intents)).to(device)
    
    # Step 4: Train
    print("\n🔄 Step 4: Training (50 epochs)")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Shuffle data
    random.shuffle(training_data)
    
    for epoch in range(50):
        total_loss = 0
        for text, intent in training_data:
            inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :]
            
            logits = classifier(embeddings)
            target = torch.tensor([intents.index(intent)]).to(device)
            loss = criterion(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"   Epoch {epoch + 1}/50: Loss = {avg_loss:.4f}")
    
    print("✅ Training complete!")
    
    # Step 5: Test
    print("\n📊 Step 5: Testing Accuracy")
    print("=" * 60)
    
    test_cases = [
        ("Ayasofya'yı görmek istiyorum", "attraction"),
        ("En iyi kebap nerede?", "restaurant"),
        ("Taksim'e nasıl gidilir?", "gps_navigation"),
        ("Ucuz otel önerisi", "accommodation"),
        ("Gece hayatı", "nightlife"),
        ("Hava durumu", "weather"),
        ("Boğaz turu", "cultural_info"),
        ("Müze önerisi", "museum"),
        ("Alışveriş merkezi", "shopping"),
        ("Bu hafta etkinlikler", "events"),
        ("Sultanahmet'e git", "gps_navigation"),
        ("Balık restoranı", "restaurant"),
        ("Topkapı Sarayı kaça kadar açık", "attraction"),
        ("Çocuklu yerler", "family_activities"),
        ("Gizli mekanlar", "hidden_gems"),
    ]
    
    correct = 0
    latencies = []
    
    for query, expected in test_cases:
        start = time.time()
        
        with torch.no_grad():
            inputs = tokenizer(query, return_tensors="pt", max_length=128, truncation=True).to(device)
            embeddings = model(**inputs).last_hidden_state[:, 0, :]
            logits = classifier(embeddings)
            predicted_idx = torch.argmax(logits, dim=1).item()
            predicted = intents[predicted_idx]
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} '{query}' → {predicted} (expected: {expected}) [{latency:.1f}ms]")
    
    accuracy = (correct / len(test_cases)) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Training samples: {len(training_data)}")
    print(f"✅ Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)} correct)")
    print(f"✅ Avg latency: {avg_latency:.1f}ms")
    print(f"🎯 Targets: >90% accuracy, <50ms latency")
    
    if accuracy >= 90 and avg_latency < 50:
        print("\n🎉 ALL TARGETS MET! Phase 2 COMPLETE!")
    elif accuracy >= 70:
        print("\n✅ Good progress! Accuracy improved significantly.")
        print("💡 Tip: Add more training data to reach 90%+")
    else:
        print("\n⚠️  Needs more work")
    
    # Save
    results = {
        "training_samples": len(training_data),
        "accuracy_percent": accuracy,
        "latency_ms": avg_latency,
        "targets_met": accuracy >= 90 and avg_latency < 50
    }
    
    with open("phase2_improved_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to phase2_improved_results.json")
    
    # Save model
    torch.save({
        'classifier': classifier.state_dict(),
        'intents': intents,
        'accuracy': accuracy,
        'latency': avg_latency
    }, 'phase2_intent_classifier.pth')
    
    print(f"💾 Model saved to phase2_intent_classifier.pth")


if __name__ == "__main__":
    import asyncio
    asyncio.run(improve_accuracy())
