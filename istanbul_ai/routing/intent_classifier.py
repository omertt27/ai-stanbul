"""
Intent Classifier - Classify user intent from messages

This module handles intent classification for user queries, determining what the user
wants to do (e.g., find restaurants, visit attractions, get transportation info).

Week 2 Refactoring: Extracted from main_system.py
Enhanced with bilingual (English/Turkish) support
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from ..core.models import ConversationContext

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent classification result"""
    primary_intent: str
    confidence: float = 0.0
    intents: List[str] = field(default_factory=list)
    is_multi_intent: bool = False
    multi_intent_response: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)


class IntentClassifier:
    """Classifies user intent from natural language messages"""
    
    def __init__(self):
        """Initialize the intent classifier with keyword mappings"""
        self.intent_keywords = self._initialize_intent_keywords()
        self.daily_talk_patterns = self._initialize_daily_talk_patterns()
    
    def _initialize_intent_keywords(self) -> Dict[str, List[str]]:
        """Initialize comprehensive bilingual (EN/TR) intent keyword mappings"""
        return {
            'restaurant': [
                # English
                'eat', 'food', 'restaurant', 'lunch', 'dinner', 'breakfast', 
                'hungry', 'cuisine', 'meal', 'dining', 'cafe', 'bistro',
                
                # Turkish - Restaurant Nouns
                'yemek', 'restoran', 'lokanta', 'kahvaltı', 'öğle yemeği',
                'akşam yemeği', 'kafe', 'kafeterya', 'meyhane', 'ocakbaşı',
                'kebapçı', 'balıkçı', 'pideci', 'börekçi', 'çay bahçesi',
                'mutfak', 'sofra', 'masa', 'mekan', 'yer',
                
                # Turkish - Verb Forms (yemek - to eat)
                'ye', 'yerim', 'yiyebilirim', 'yiyorum', 'yiyeceğim', 'yesem',
                'yiyelim', 'yemek', 'yenir', 'yenilir', 'yemek istiyorum',
                'yemek için', 'yiyebilir miyim',
                
                # Turkish - States and Feelings
                'aç', 'açım', 'acıktım', 'karın', 'karnım aç', 'karnım açtı',
                
                # Turkish - Verb Forms (önermek - to recommend)
                'öner', 'önerin', 'önerir misin', 'önerebilir misin',
                'öneri', 'önerilir', 'tavsiye et', 'tavsiye eder misin',
                'tavsiye', 'bilir misin', 'biliyor musun',
                
                # Turkish - Question Patterns
                'nerede yenir', 'nerede yemek yenir', 'nerede yeriz',
                'ne yesem', 'ne yenir', 'ne yemek yenir', 'ne yiyelim',
                'iyi restoran', 'güzel restoran', 'iyi lokanta', 'güzel lokanta',
                'hangi restoran', 'hangi lokanta', 'mekan öner', 'yer öner',
                'kahvaltı nerede', 'akşam yemeği nerede', 'öğle yemeği nerede',
                'iyi mi', 'güzel mi', 'lezzetli mi', 'tavsiye edilir mi',
                
                # Turkish - Food Types & Cuisines
                'kebap', 'kebab', 'balık', 'meze', 'rakı', 'rakılı balık',
                'çay', 'kahve', 'türk kahvesi', 'tatlı', 'baklava',
                'börek', 'mantı', 'lahmacun', 'pide', 'döner', 'köfte',
                'çorba', 'salata', 'meze', 'aperatif', 'içecek',
                'deniz mahsulleri', 'et yemekleri', 'zeytinyağlı', 'vegetaryan',
                'vegan', 'vejetaryen', 'helal', 'haram değil',
                
                # Turkish - Meal Times
                'sabah', 'öğlen', 'akşam', 'kahvaltıda', 'öğlende', 'akşamda',
                'gece', 'gece yarısı', 'geç saatte', 'açık mı',
                
                # Turkish - Price/Budget
                'ucuz', 'pahalı', 'fiyat', 'fiyatlar', 'hesaplı', 'ekonomik',
                'lüks', 'fancy', 'bütçe', 'bütçeye uygun', 'öğrenci dostu',
                'uygun fiyatlı', 'makul', 'orta fiyat',
                
                # Turkish - Ambiance & Occasion
                'romantik', 'aile için', 'aile dostu', 'çocuklar için',
                'çocuk menüsü', 'arkadaşlarla', 'grup için', 'iş yemeği',
                'manzaralı', 'deniz kenarı', 'boğaz manzaralı', 'terasta',
                'bahçede', 'açık havada', 'kapalı', 'şık', 'samimi',
                
                # Turkish - Dietary Needs
                'vejeteryan', 'vejetaryen', 'vegan', 'glutensiz', 'gluten free',
                'laktozsuz', 'helal', 'alkolsüz', 'alkollü',
                
                # Turkish - Expressions
                'yemeğe git', 'yemeğe çık', 'yemeğe gidelim', 'yemek yiyelim',
                'ne yesek', 'nereye gitsek', 'akşam nerede yenir',
                'rezervasyon', 'yer ayır', 'masa', 'boş yer var mı'
            ],
            'attraction': [
                # General attraction words (English)
                'visit', 'see', 'attraction', 'attractions', 'tour', 'sightseeing', 
                'tourist', 'landmark', 'landmarks',
                # General attraction words (Turkish)
                'gez', 'gezilecek', 'gör', 'görülecek', 'ziyaret', 'tur', 
                'gezi', 'turistik', 'mekan', 'yer', 'yerler',
                
                # Museum keywords (English)
                'museum', 'museums', 'gallery', 'galleries', 'exhibition', 'exhibitions', 
                'art museum', 'art museums', 'historical museum', 'history museum', 
                'archaeological', 'archaeology',
                # Museum keywords (Turkish)
                'müze', 'müzeler', 'galeri', 'sergi', 'sergiler', 'sanat müzesi',
                'tarih müzesi', 'arkeoloji', 'arkeolojik',
                
                # Specific place types (English)
                'mosque', 'mosques', 'palace', 'palaces', 'church', 'churches', 
                'synagogue', 'monument', 'monuments', 'memorial', 'historical site', 
                'historical sites',
                # Specific place types (Turkish)
                'cami', 'camii', 'camiler', 'saray', 'saraylar', 'kilise',
                'sinagog', 'anıt', 'anıtlar', 'tarihi yer', 'tarihi yerler',
                'tarihi mekan', 'tarihi alan',
                
                # Descriptive words (English)
                'historical', 'ancient', 'cultural site', 'heritage', 'artifact', 
                'artifacts',
                # Descriptive words (Turkish)
                'tarihi', 'eski', 'antik', 'kültürel', 'kültürel alan', 
                'miras', 'eser', 'eserler',
                
                # Action words (English)
                'explore', 'discover', 'show me', 'what to see', 'worth seeing', 
                'must see', 'should i see',
                # Action words (Turkish)
                'keşfet', 'keşfedelim', 'göster', 'ne görmeliyim', 'görmeye değer',
                'mutlaka görülmeli', 'görmeli miyim', 'ne gezilir',
                
                # Specific queries (English)
                'places to visit', 'places to see', 'what can i visit', 
                'what can i see', 'best attractions', 'top attractions', 
                'famous places', 'popular places',
                # Specific queries (Turkish)
                'gezilecek yerler', 'görülecek yerler', 'ne gezebilirim',
                'ne görebilirim', 'en iyi yerler', 'popüler yerler',
                'ünlü yerler', 'meşhur yerler'
            ],
            'transportation': [
                # English - General
                'transport', 'metro', 'bus', 'taxi', 'ferry', 'how to get', 
                'travel', 'tram', 'istanbulkart', 'public transport', 'dolmuş',
                
                # Turkish - Transportation Nouns
                'ulaşım', 'metro', 'metrobus', 'otobüs', 'taksi', 'vapur', 
                'feribot', 'tramvay', 'istanbul kart', 'istanbulkart', 
                'toplu taşıma', 'dolmuş', 'minibüs', 'marmaray', 'teleferik',
                'funicular', 'füniküler', 'hat', 'durak', 'iskele', 'terminal',
                
                # Turkish - Verb Forms (gitmek - to go)
                'giderim', 'gidebilirim', 'gidiyorum', 'gideceğim', 'gitsem',
                'gidelim', 'gidilir', 'gitmek', 'gitme', 'gitmek istiyorum',
                'gitmek için', 'gidebilir miyim',
                
                # Turkish - Verb Forms (ulaşmak - to reach)
                'ulaşırım', 'ulaşabilirim', 'ulaşıyorum', 'ulaşmak', 'ulaşılır',
                'ulaşmak istiyorum', 'ulaşabilir miyim',
                
                # Turkish - Verb Forms (varmak - to arrive)
                'varırım', 'varabilirim', 'varmak', 'varılır', 'varmak istiyorum',
                
                # Turkish - Question Patterns
                'nasıl giderim', 'nasıl gidilir', 'nasıl gidebilirim',
                'nasıl ulaşırım', 'nasıl ulaşabilirim', 'nasıl varırım',
                'nereden binilir', 'nereden binerim', 'nereden kalkar',
                'hangi hattan', 'hangi hat', 'hangi metro', 'hangi otobüs',
                'kaçta kalkıyor', 'kaçta gelir', 'ne zaman kalkar',
                'kaç dakika sürer', 'ne kadar sürer', 'uzak mı',
                
                # Turkish - "to" patterns with destinations
                "'e nasıl", "'a nasıl", "'e gitmek", "'a gitmek", 
                "'e ulaşmak", "'a ulaşmak", "'e varmak", "'a varmak",
                "'den", "'dan", "'den gitmek", "'dan gitmek",
                
                # Turkish - Expressions
                'seyahat', 'yolculuk', 'gidiş', 'dönüş', 'gidiş dönüş',
                'bilet', 'kart', 'ücret', 'tarife', 'çalışma saatleri',
                'en yakın', 'en yakını', 'yakın mı', 'yürünür mü',
                'yaya olarak', 'yürüyerek', 'arabayla', 'taksiye bin'
            ],
            'nearby_locations': [
                # Core nearby keywords (English)
                'near me', 'nearby', 'close to me', 'around me', 'near here',
                'close by', 'in the area', 'in this area', 'around here',
                'within walking distance', 'walking distance', 'around my location',
                # Core nearby keywords (Turkish)
                'yakınımda', 'yakın', 'yakında', 'yakınlarda', 'civarda',
                'civarında', 'etrafta', 'etrafımda', 'bölgede', 'burada',
                'buraya yakın', 'buradan yakın', 'yürüme mesafesinde',
                
                # What's nearby questions (English)
                'what\'s near', 'what\'s nearby', 'what is near', 'what is nearby',
                'what\'s around', 'what\'s close', 'whats near', 'whats nearby',
                'whats around', 'whats close', 'anything near', 'anything nearby',
                # What's nearby questions (Turkish)
                'yakında ne var', 'civar ne var', 'etrafta ne var', 
                'buralarda ne var', 'yakınlarda ne var', 'neler var yakında',
                
                # Find nearby (English)
                'find near', 'find nearby', 'find close', 'search near', 
                'search nearby', 'look for nearby', 'show me nearby',
                'show nearby', 'show what\'s near',
                # Find nearby (Turkish)
                'yakınlarda bul', 'yakında ara', 'yakın bul', 'göster yakın',
                'yakınımda göster', 'en yakın', 'en yakını',
                
                # Specific nearby queries (English)
                'museums near me', 'attractions near me', 'restaurants near me',
                'places near me', 'things near me', 'locations near me',
                'museums nearby', 'attractions nearby', 'restaurants nearby',
                # Specific nearby queries (Turkish)
                'yakınımdaki müzeler', 'yakınımdaki restoranlar', 
                'yakınımdaki yerler', 'civarımda', 'civarımdaki',
                
                # Within radius (English)
                'within', 'km from here', 'km from me', 'minutes away',
                'minutes walk', 'min walk', 'walking from here',
                # Within radius (Turkish)
                'km mesafede', 'dakika uzaklıkta', 'yürüyerek', 
                'yürüyüş mesafesinde', 'buradan kaç dakika'
            ],
            'neighborhood': [
                # English
                'neighborhood', 'area', 'district', 'where to stay', 'which area',
                'quarter', 'region', 'location', 'suburb',
                
                # Turkish - Neighborhood nouns
                'semt', 'mahalle', 'bölge', 'çevre', 'yöre', 'taraf',
                'civar', 'muhit', 'havali', 'mıntıka', 'yer', 'alan',
                
                # Turkish - Specific areas/questions
                'hangi semt', 'hangi mahalle', 'hangi bölge', 'hangi taraf',
                'nerede kalmalı', 'nerede kalınır', 'nerede konaklama',
                'nerede oteller', 'nerede otel', 'nereye yerleş',
                'nerede yaşanır', 'nerede ikamet', 'nerede dur',
                
                # Turkish - Area characteristics
                'mahalle havası', 'semt havası', 'bölge karakteri',
                'mahalle kültürü', 'semt kültürü', 'yerel yaşam',
                'sokak hayatı', 'mahalle hayatı', 'yerleşim yeri',
                
                # Turkish - Verb forms (kalmak - to stay)
                'kal', 'kalırım', 'kala', 'kalmalı', 'kalınır', 'kalınmalı',
                'konaklama', 'konakla', 'konaklamak', 'konaklarım',
                
                # Turkish - Descriptors
                'güvenli', 'emniyetli', 'rahat', 'huzurlu', 'sakin',
                'kalabalık', 'canlı', 'renkli', 'dinamik', 'yaşayan',
                'modern', 'klasik', 'tarihi', 'eski', 'yeni',
                'merkezi', 'merkezde', 'şehir merkezi', 'merkeze yakın',
                'uzak', 'yakın', 'deniz kenarı', 'sahil', 'kıyı',
                
                # Turkish - Questions about areas
                'nasıl bir yer', 'nasıl bir mahalle', 'nasıl bir semt',
                'özelliği ne', 'neyle ünlü', 'neden biliniyor',
                'neler var', 'ne yapılır', 'ne gezilir',
                'gece hayatı var mı', 'eğlence var mı', 'restoran var mı',
                
                # Turkish - Specific neighborhood names (common)
                'beyoğlu', 'galata', 'karaköy', 'cihangir', 'taksim',
                'beşiktaş', 'ortaköy', 'bebek', 'sultanahmet', 'fatih',
                'kadıköy', 'moda', 'eminönü', 'üsküdar', 'şişli',
                'nişantaşı', 'osmanbey', 'mecidiyeköy', 'levent',
                'etiler', 'bostancı', 'maltepe', 'bakırköy',
                
                # Turkish - Preferences
                'tercih', 'tercih ederim', 'isterim', 'istiyorum',
                'isterdim', 'olsa iyi olur', 'aradığım', 'bana uygun',
                'benim için', 'bana göre', 'uygun mu', 'ideal mi'
            ],
            'shopping': [
                # English
                'shop', 'shopping', 'buy', 'bazaar', 'market', 'souvenir',
                'store', 'mall', 'purchase', 'gifts',
                # Turkish
                'alışveriş', 'al', 'satın al', 'çarşı', 'pazar', 'hediyelik',
                'hediyelik eşya', 'mağaza', 'dükkan', 'avm', 'hediye'
            ],
            'events': [
                # Core event keywords (English)
                'event', 'events', 'activity', 'activities', 'entertainment', 
                'nightlife', 'what to do', 'things to do', 'happening', 'going on',
                # Core event keywords (Turkish)
                'etkinlik', 'etkinlikler', 'aktivite', 'aktiviteler', 'eğlence',
                'gece hayatı', 'ne yapılır', 'yapılacak şeyler', 'oluyor', 'var mı',
                
                # Performance types (English)
                'concert', 'concerts', 'show', 'shows', 'performance', 'performances', 
                'theater', 'theatre', 'opera', 'ballet', 'dance', 'comedy',
                'live music', 'dj', 'club', 'party', 'parties', 'celebration',
                # Performance types (Turkish)
                'konser', 'konserler', 'gösteri', 'gösteriler', 'performans',
                'tiyatro', 'opera', 'bale', 'dans', 'komedi', 'canlı müzik',
                'club', 'kulüp', 'parti', 'partiler', 'kutlama',
                
                # Event types (English)
                'cultural', 'festival', 'festivals', 'exhibition', 'exhibitions',
                'gallery opening', 'art show', 'music event', 'art event',
                'sporting event', 'sports', 'match', 'game', 'tournament',
                # Event types (Turkish)
                'kültürel', 'festival', 'festivaller', 'sergi', 'sergiler',
                'galeri açılışı', 'sanat etkinliği', 'müzik etkinliği',
                'spor etkinliği', 'spor', 'maç', 'oyun', 'turnuva',
                
                # Venues (English & Turkish)
                'iksv', 'İKSV', 'salon', 'babylon', 'zorlu', 'zorlu psm',
                'cemal reşit rey', 'atatürk kültür merkezi', 'akm',
                
                # Temporal patterns (English)
                'tonight', 'today', 'tomorrow', 'this weekend', 'this week', 
                'this month', 'next week', 'upcoming', 'soon', 'now', 'currently',
                'this evening', 'this afternoon', 'later today', 'upcoming events',
                # Temporal patterns (Turkish)
                'bu gece', 'bu akşam', 'bugün', 'yarın', 'bu hafta sonu',
                'bu hafta', 'bu ay', 'gelecek hafta', 'yakında', 'şimdi',
                'bu öğleden sonra', 'bugün sonra', 'yaklaşan etkinlikler',
                
                # Questions (English)
                'what\'s on', 'whats on', 'what\'s happening', 'any events',
                'any concerts', 'any shows', 'where to go', 'what\'s playing',
                # Questions (Turkish)
                'ne var', 'neler var', 'ne oluyor', 'etkinlik var mı',
                'konser var mı', 'gösteri var mı', 'nereye gidilir', 'ne oynuyor'
            ],
            'weather': [
                # Core weather keywords (English)
                'weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy', 
                'hot', 'cold', 'warm', 'cool', 'humid', 'dry',
                # Core weather keywords (Turkish)
                'hava', 'hava durumu', 'sıcaklık', 'derece', 'tahmin', 'yağmur',
                'güneşli', 'bulutlu', 'sıcak', 'soğuk', 'ılık', 'serin',
                'nemli', 'kuru',
                
                # Weather conditions (English)
                'storm', 'stormy', 'snow', 'snowy', 'drizzle', 'shower', 'showers',
                'heatwave', 'windy', 'foggy', 'misty', 'haze',
                # Weather conditions (Turkish)
                'fırtına', 'fırtınalı', 'kar', 'karlı', 'çiseleyen', 'sağanak',
                'sağanaklar', 'sıcak hava dalgası', 'rüzgarlı', 'sisli', 'puslu',
                
                # Measurements (English)
                'degrees', 'celsius', 'fahrenheit', 'humidity', 'wind', 
                'precipitation', 'atmospheric', 'barometric',
                # Measurements (Turkish)
                'derece', 'santigrat', 'nem', 'rüzgar', 'yağış', 'atmosferik',
                
                # Question patterns (English)
                'what\'s the weather', 'how\'s the weather', 'weather like',
                'weather today', 'weather tomorrow', 'weather this week',
                'will it rain', 'is it raining', 'is it sunny', 'is it hot', 'is it cold',
                'should i bring umbrella', 'should i bring jacket', 'need umbrella',
                'what to wear', 'dress for weather', 'what should i wear',
                # Question patterns (Turkish)
                'hava nasıl', 'hava durumu nasıl', 'hava nasıl olacak',
                'bugün hava', 'yarın hava', 'bu hafta hava', 'yağmur yağacak mı',
                'yağmur yağıyor mu', 'güneşli mi', 'sıcak mı', 'soğuk mu',
                'şemsiye gerekli mi', 'mont gerekli mi', 'ne giymeli',
                'ne giymeliyim', 'havaya göre ne giyilir',
                
                # Weather + activity (English)
                'weather appropriate', 'good weather for', 'weather for walking',
                'weather for sightseeing', 'outdoor weather',
                # Weather + activity (Turkish)
                'hava uygun mu', 'hava müsait mi', 'gezmeye uygun hava',
                'dışarıda gezilir mi', 'açık havaya uygun',
                
                # General (English & Turkish)
                'climate', 'meteorological', 'weather conditions', 'forecast today',
                'iklim', 'meteorolojik', 'hava şartları', 'bugün tahmini'
            ],
            'airport_transport': [
                'airport', ' ist ', ' saw ', 'atatürk', 'ataturk', 'istanbul airport', 
                'sabiha gökçen', 'sabiha gokcen', 'new airport', 'airport transfer', 
                'airport transport', 'from airport', 'to airport', 'to the airport', 'from the airport',
                'airport shuttle', 'airport bus', 'airport metro', 'flight', 'departure', 'arrival', 
                'terminal', 'havalimanı', 'havaş', 'havas'
            ],
            'hidden_gems': [
                # English
                'hidden', 'secret', 'local', 'authentic', 'off-beaten', 
                'off the beaten path', 'unknown', 'undiscovered', 'gems', 
                'hidden gems', 'secret spots', 'local favorites', 'insider', 
                'less touristy', 'not touristy', 'avoid crowds', 'unique places', 
                'special places', 'locals know', 'local secrets', 'hidden treasures', 
                'underground', 'alternative', 'unconventional', 'non-touristy', 
                'lesser known', 'hidden places',
                
                # Turkish - Hidden/Secret
                'gizli', 'saklı', 'bilinmeyen', 'keşfedilmemiş', 'az bilinen',
                'pek bilinmeyen', 'gizemli', 'gizli yerler', 'saklı yerler',
                'gizli mekanlar', 'saklı mekanlar',
                
                # Turkish - Local/Authentic
                'yerel', 'lokal', 'otantik', 'özgün', 'gerçek', 'hakiki',
                'yerli', 'mahalli', 'yerliler', 'yerli halk', 'yerli insanlar',
                'yerel halk', 'yerli mekan', 'yerel mekan', 'mahalle arası',
                'yerli yemekleri', 'yerel tatlar',
                
                # Turkish - Locals-only expressions
                'yerlilerin gittiği', 'yerlilerin bildiği', 'yerli bilir',
                'istanbul\'lular gider', 'yerli tavsiyesi', 'yerel sır',
                'içerden bilgi', 'yerli favorileri', 'halkın gittiği',
                
                # Turkish - Non-touristy
                'turistik olmayan', 'turistik değil', 'turist yok', 
                'az turistik', 'turistsiz', 'kalabalık değil', 'az kalabalık',
                'sakin', 'sessiz', 'huzurlu', 'rahat', 'dingin',
                'kalabalıktan uzak', 'izole', 'tenhaya',
                
                # Turkish - Alternative/Unique
                'alternatif', 'farklı', 'değişik', 'alışılmadık', 'sıradışı',
                'özgün', 'özel', 'benzersiz', 'eşsiz', 'nadir', 'ender',
                'farklı yerler', 'özel yerler', 'alışılmışın dışında',
                
                # Turkish - Underground/Hidden culture
                'alternatif kültür', 'underground', 'yeraltı', 'bağımsız',
                'indie', 'bohemian', 'bohça', 'sanatsal', 'alternatif sanat',
                'sokak sanatı', 'graffiti', 'mural',
                
                # Turkish - Insider/Discovery
                'içeriden', 'sır', 'sırlar', 'keşfet', 'keşfedilmemiş',
                'keşfedin', 'bulunsun', 'ipucu', 'püf nokta', 'bilgi',
                'tavsiye', 'öneri', 'özel ipucu', 'gizli ipucu',
                
                # Turkish - Off-beaten path
                'patika dışı', 'ana yol dışı', 'merkez dışı', 'arka sokak',
                'ara sokak', 'iç sokak', 'mahalle arası', 'kenar mahalle',
                
                # Turkish - Question patterns
                'yerliler nereye gider', 'nerede takılır', 'nerede vakit geçirilir',
                'gizli mekan', 'saklı yer', 'bilinen ama bilinmeyen',
                'turist bilmiyor', 'rehberde yok', 'kimse bilmiyor',
                'yerliler bilir sadece', 'sadece yerliler', 'yerli mekanı',
                
                # Turkish - Authentic experience
                'gerçek istanbul', 'asıl istanbul', 'esas istanbul',
                'özgün deneyim', 'otantik deneyim', 'yerel yaşam',
                'yerli gibi', 'yerliler gibi yaşa', 'mahalle hayatı',
                'semt kültürü', 'yerel kültür', 'geleneksel',
                
                # Turkish - Hidden treasures
                'hazine', 'defineler', 'inciler', 'mücevherler', 'taşlar',
                'gizli hazine', 'saklı hazine', 'keşfedilmemiş hazine'
            ],
            'route_planning': [
                # English - Planning keywords
                'route', 'itinerary', 'plan', 'planning', 'schedule', 'organize',
                'day trip', 'trip planning', 'travel plan',
                
                # Turkish - Planning keywords
                'rota', 'güzergah', 'plan', 'planlama', 'planla', 'program',
                'programla', 'organize et', 'organize', 'düzenle', 'tarifе',
                'yol', 'yol haritası', 'gezi planı', 'seyahat planı',
                
                # English - Tour types
                'tour', 'one day tour', 'two day tour', 'three day tour', 'multi-day',
                'walking tour', 'food tour', 'cultural tour', 'historical tour',
                
                # Turkish - Tour types
                'tur', 'gezi', 'turne', 'yürüyüş turu', 'yemek turu',
                'kültür turu', 'tarihi tur', 'şehir turu', 'günlük tur',
                
                # English - Duration patterns
                'one day', 'two days', '3 days', '4 days', '5 days', 'week itinerary',
                'weekend trip', 'full day', 'half day', 'morning tour', 'afternoon tour',
                
                # Turkish - Duration patterns
                'bir gün', 'bir günlük', 'iki gün', 'üç gün', 'dört gün', 'beş gün',
                'hafta', 'haftalık', 'hafta sonu', 'hafta sonu gezisi',
                'tam gün', 'yarım gün', 'sabah', 'sabah turu', 'öğleden sonra',
                'akşam', 'gece', 'gündüz',
                
                # Turkish - Verb forms (planlamak - to plan)
                'planla', 'planlayım', 'planlayalım', 'planlar mısın',
                'planlayabilir misin', 'planlıyorum', 'planlayacağım',
                
                # Turkish - Verb forms (organize etmek - to organize)
                'organize et', 'organize eder misin', 'organizasyon',
                'düzenle', 'düzenler misin', 'hazırla', 'hazırlar mısın',
                
                # English - Questions
                'what should i visit', 'best route to visit', 'how to visit all',
                'efficient way to see', 'best way to see', 'plan my visit',
                'help me plan', 'organize my trip', 'create itinerary',
                
                # Turkish - Questions
                'ne gezmeliyim', 'ne görmeliyim', 'nereyi gezmeliyim',
                'nasıl gezerim', 'nasıl gezilir', 'nasıl görürüm',
                'hangi sırayla', 'hangi önce', 'önce nereye',
                'en iyi rota', 'en iyi güzergah', 'ideal rota',
                'gezimi planla', 'planlamama yardım et', 'plan yapar mısın',
                'program hazırla', 'rotam', 'gezi rotam', 'yol haritam',
                
                # English - Multi-destination
                'visit all', 'see everything', 'cover all', 'comprehensive tour',
                'best order to visit', 'optimal route',
                
                # Turkish - Multi-destination
                'hepsini gez', 'hepsini gör', 'her yeri gez', 'her şeyi gör',
                'kapsamlı tur', 'tam tur', 'geniş tur', 'detaylı tur',
                'optimal', 'en verimli', 'en hızlı', 'en iyi sıralama',
                
                # Turkish - Itinerary creation
                'taslak', 'çerçeve', 'gezi taslağı', 'plan taslağı',
                'ne yapmalı', 'ne yapmam lazım', 'nerelerden geçmeli',
                'nasıl dolaşmalı', 'nasıl bir rota', 'rota çiz',
                
                # Turkish - Time optimization
                'zaman', 'süresi', 'sürede', 'zamanda', 'kısa süre',
                'kısa zamanda', 'hızlıca', 'verimli', 'etkin',
                'en kısa', 'en hızlı rota', 'zaman kaybetmeden',
                
                # Turkish - Multiple days
                'günlük program', 'günlük plan', 'gün gün',
                'birinci gün', 'ikinci gün', 'ilk gün', 'son gün',
                'hafta sonu programı', 'üç günlük', 'iki günlük'
            ],
            'gps_route_planning': [
                'directions', 'navigation', 'how to get', 'from', 'to', 'nearest', 
                'distance', 'walking route', 'driving route', 'public transport route'
            ],
            'museum_route_planning': [
                'museum route', 'museum tour', 'museum plan', 'museum itinerary', 
                'museums near', 'museum walk'
            ],
            'greeting': [
                'hello', 'hi', 'merhaba', 'help', 'start', 'hey', 'good morning',
                'good afternoon', 'good evening', 'selam', 'günaydın'
            ]
        }
    
    def _initialize_daily_talk_patterns(self) -> List[str]:
        """Initialize daily talk conversation patterns"""
        greeting_patterns = [
            'hi', 'hello', 'hey', 'merhaba', 'selam', 'good morning', 
            'good afternoon', 'good evening', 'günaydın', 'iyi günler', 
            'iyi akşamlar'
        ]
        
        weather_patterns = [
            'how\'s the weather', 'what\'s the weather', 'is it raining', 
            'is it sunny', 'hava nasıl', 'yağmur yağıyor mu', 'soğuk mu', 
            'sıcak mı'
        ]
        
        casual_patterns = [
            'how are you', 'what\'s up', 'how\'s it going', 'nasılsın', 
            'ne haber', 'naber'
        ]
        
        time_patterns = [
            'what time', 'what day', 'is it open', 'saat kaç', 'açık mı'
        ]
        
        daily_life_patterns = [
            'good morning', 'good night', 'thank you', 'thanks', 'please', 
            'excuse me', 'sorry', 'günaydın', 'iyi geceler', 'teşekkürler', 
            'lütfen', 'özür dilerim'
        ]
        
        cultural_patterns = [
            'local tip', 'cultural tip', 'local advice', 'what locals do',
            'like a local', 'authentic experience', 'local culture'
        ]
        
        return (greeting_patterns + weather_patterns + casual_patterns + 
                time_patterns + daily_life_patterns + cultural_patterns)
    
    def classify_intent(
        self, 
        message: str, 
        entities: Dict, 
        context: Optional[ConversationContext] = None,
        neural_insights: Optional[Dict] = None,
        preprocessed_query: Optional[Any] = None
    ) -> IntentResult:
        """
        Classify user intent from message with contextual awareness
        
        Args:
            message: User's input message
            entities: Extracted entities from message
            context: Conversation context (optional)
            neural_insights: Neural processing insights (optional)
            preprocessed_query: Preprocessed query data (optional)
        
        Returns:
            IntentResult object with classification details
        """
        message_lower = message.lower()
        primary_intent = 'general'
        
        # PRIORITY 1: Check for greeting/general intent (most specific patterns)
        if any(keyword in message_lower for keyword in self.intent_keywords['greeting']):
            primary_intent = 'greeting'
        
        # PRIORITY 2: Check for GPS-based route planning intent (very specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['gps_route_planning']) or \
            (any(indicator in message_lower for indicator in ['from', 'to', 'near', 'closest', 'nearby', 'distance']) and 
             any(rk in message_lower for rk in ['route', 'get', 'go', 'directions'])):
            primary_intent = 'gps_route_planning'
        
        # PRIORITY 3: Check for museum route planning intent (specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['museum_route_planning']) or \
            ('museum' in message_lower and any(rk in message_lower for rk in ['route', 'plan', 'tour', 'visit'])):
            primary_intent = 'museum_route_planning'
        
        # PRIORITY 4: Check for weather intent (specific patterns before food/general)
        elif any(keyword in message_lower for keyword in self.intent_keywords['weather']):
            primary_intent = 'weather'
        
        # PRIORITY 5: Check for route planning intent (before attractions)
        elif any(keyword in message_lower for keyword in self.intent_keywords['route_planning']):
            primary_intent = 'route_planning'
        
        # PRIORITY 6: Check for airport transport intent (specific)
        elif any(keyword in message_lower for keyword in self.intent_keywords['airport_transport']):
            primary_intent = 'airport_transport'
        
        # PRIORITY 7: Check for events/activities intent (before attractions)
        elif any(keyword in message_lower for keyword in self.intent_keywords['events']):
            primary_intent = 'events'
        
        # PRIORITY 8: Check for hidden gems intent
        elif any(keyword in message_lower for keyword in self.intent_keywords['hidden_gems']):
            primary_intent = 'hidden_gems'
        
        # PRIORITY 9: Check for nearby locations intent (GPS-based proximity search)
        elif any(keyword in message_lower for keyword in self.intent_keywords['nearby_locations']):
            primary_intent = 'nearby_locations'
        
        # PRIORITY 10: Check for transportation intent
        elif (any(keyword in message_lower for keyword in self.intent_keywords['transportation']) or 
            entities.get('transportation')):
            primary_intent = 'transportation'
        
        # PRIORITY 11: Check for neighborhood/area intent
        elif (any(keyword in message_lower for keyword in self.intent_keywords['neighborhood']) or 
            entities.get('neighborhoods')):
            primary_intent = 'neighborhood'
        
        # PRIORITY 12: Check for shopping intent
        elif any(keyword in message_lower for keyword in self.intent_keywords['shopping']):
            primary_intent = 'shopping'
        
        # PRIORITY 13: Check for restaurant/food intent (broader, comes later)
        elif any(keyword in message_lower for keyword in self.intent_keywords['restaurant']) or entities.get('cuisines'):
            primary_intent = 'restaurant'
        
        # PRIORITY 14: Check for attraction/sightseeing intent (broadest, comes last)
        elif any(word in message_lower for word in ['museum', 'museums', 'gallery', 'galleries', 'exhibition', 
                                                     'attraction', 'attractions', 'landmark', 'palace', 'mosque']) or \
            any(keyword in message_lower for keyword in self.intent_keywords['attraction']) or \
            entities.get('landmarks'):
            primary_intent = 'attraction'
        
        # Calculate confidence
        confidence = self.get_intent_confidence(message, primary_intent)
        
        # Detect multiple intents
        all_intents = self.detect_multiple_intents(message, entities)
        is_multi_intent = len(all_intents) > 1
        
        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            intents=all_intents,
            is_multi_intent=is_multi_intent,
            multi_intent_response=None,  # Can be enhanced later
            entities=entities
        )
    
    def detect_multiple_intents(self, message: str, entities: Dict) -> List[str]:
        """
        Detect multiple intents in a single message
        
        Args:
            message: User's input message
            entities: Extracted entities from message
        
        Returns:
            List of detected intents
        """
        intents = []
        message_lower = message.lower()
        
        # Check each intent category
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                intents.append(intent)
        
        return intents if intents else ['general']
    
    def is_daily_talk_query(self, message: str) -> bool:
        """
        Check if message is a daily talk/casual conversation query
        
        Args:
            message: User's input message
        
        Returns:
            True if message is daily talk, False otherwise
        """
        message_lower = message.lower()
        
        # Check if message contains daily talk patterns
        for pattern in self.daily_talk_patterns:
            if pattern in message_lower:
                return True
        
        # Check for short casual messages (likely daily talk)
        if (len(message.split()) <= 3 and 
            any(word in message_lower for word in 
                ['hi', 'hello', 'hey', 'thanks', 'bye', 'yes', 'no', 'ok', 'okay'])):
            return True
        
        return False
    
    def get_intent_confidence(self, message: str, intent: str) -> float:
        """
        Calculate confidence score for a classified intent with improved algorithm
        
        Confidence factors:
        1. Keyword match count (primary)
        2. Strong indicators (keywords that clearly signal intent)
        3. Message specificity (shorter, focused messages = higher confidence)
        
        Args:
            message: User's input message
            intent: Classified intent
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if intent not in self.intent_keywords:
            return 0.0
        
        message_lower = message.lower()
        message_words = set(message_lower.split())
        keywords = self.intent_keywords[intent]
        
        # Factor 1: Count matching keywords
        matches = sum(1 for keyword in keywords if keyword in message_lower)
        
        if matches == 0:
            return 0.0
        
        # Factor 2: Strong indicators (high-value keywords that clearly signal intent)
        strong_indicators = {
            'restaurant': ['restaurant', 'restaurants', 'eat', 'dining', 'lunch', 'dinner', 'breakfast', 'cuisine'],
            'attraction': ['museum', 'museums', 'palace', 'palaces', 'mosque', 'mosques', 'attraction', 'attractions', 'landmark', 'landmarks', 'visit', 'see'],
            'transportation': ['metro', 'bus', 'ferry', 'transport', 'transportation', 'how to get', 'directions', 'tram'],
            'weather': ['weather', 'forecast', 'temperature', 'rain', 'sunny', 'cloudy', 'what\'s the weather', 'how\'s the weather', 'will it rain'],
            'events': ['event', 'events', 'concert', 'concerts', 'show', 'shows', 'festival', 'festivals', 'happening', 'tonight', 'what\'s on', 'exhibition'],
            'neighborhood': ['neighborhood', 'area', 'district', 'where to stay'],
            'shopping': ['shopping', 'bazaar', 'market', 'souvenir', 'buy'],
            'hidden_gems': ['hidden', 'secret', 'local', 'gems', 'authentic'],
            'airport_transport': ['airport', 'terminal', 'flight', 'ist', 'saw'],
            'route_planning': ['route', 'itinerary', 'plan', 'schedule', 'tour', 'day trip', 'trip planning'],
            'gps_route_planning': ['directions', 'how to get', 'from', 'to', 'navigation'],
            'greeting': ['hello', 'hi', 'merhaba', 'help']
        }
        
        # Check for strong indicators
        has_strong_indicator = False
        if intent in strong_indicators:
            has_strong_indicator = any(
                indicator in message_lower 
                for indicator in strong_indicators[intent]
            )
        
        # Factor 3: Calculate base confidence using logarithmic scale
        # This avoids penalizing intents with many keywords
        if matches >= 3:
            base_confidence = 0.95
        elif matches == 2:
            base_confidence = 0.85
        elif matches == 1:
            base_confidence = 0.75 if has_strong_indicator else 0.65
        else:
            base_confidence = 0.50
        
        # Factor 4: Boost for strong indicators
        if has_strong_indicator:
            base_confidence = min(base_confidence + 0.15, 1.0)
        
        # Factor 5: Consider message specificity
        # Shorter, focused messages should have higher confidence
        if len(message_words) <= 5 and matches >= 1:
            base_confidence = min(base_confidence + 0.10, 1.0)
        
        # Factor 6: Penalize if message is very long and vague
        if len(message_words) > 15 and matches == 1 and not has_strong_indicator:
            base_confidence = max(base_confidence - 0.15, 0.40)
        
        return round(base_confidence, 2)
