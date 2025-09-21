"""
Test configuration and fixtures for AI Istanbul chatbot tests
"""
import pytest
import asyncio
import os
import sys
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
import pytest_asyncio
import tempfile
import sqlite3

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Set test environment variables
os.environ['TESTING'] = 'true'
os.environ['DATABASE_URL'] = 'sqlite:///test_istanbul.db'
os.environ['REDIS_URL'] = 'redis://localhost:6379/1'  # Use test database
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'
os.environ['ANTHROPIC_API_KEY'] = 'test-key-for-testing'
os.environ['GOOGLE_API_KEY'] = 'test-key-for-testing'

@pytest_asyncio.fixture
async def app():
    """Create FastAPI application for testing."""
    # Import after setting environment variables
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from backend.main import app
    return app

@pytest_asyncio.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create test client."""
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def test_db():
    """Create temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create test database
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            query TEXT,
            response TEXT,
            timestamp DATETIME,
            language TEXT
        )
    ''')
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.fixture
def mock_ai_responses():
    """Mock AI service responses for testing."""
    return {
        "en": {
            "greeting": "Hello! I'm your Istanbul AI guide. How can I help you explore Istanbul today?",
            "restaurant_query": "Here are some excellent Turkish restaurants in Sultanahmet: Pandeli, Hamdi Restaurant, and Balıkçı Sabahattin. They offer authentic Ottoman cuisine with great Bosphorus views.",
            "museum_query": "I recommend visiting these top museums: Hagia Sophia, Topkapi Palace, Istanbul Archaeological Museums, and the Blue Mosque. Each offers unique insights into Istanbul's rich history.",
            "transport_query": "For transportation, you can use the metro, tram, bus, or taxi. The Istanbul card (Istanbulkart) works for all public transport. Ferry rides across the Bosphorus are also scenic."
        },
        "tr": {
            "greeting": "Merhaba! Ben İstanbul AI rehberinizim. İstanbul'u keşfetmenizde size nasıl yardımcı olabilirim?",
            "restaurant_query": "Sultanahmet'te harika Türk restoranları: Pandeli, Hamdi Restaurant ve Balıkçı Sabahattin. Boğaz manzaralı otantik Osmanlı mutfağı sunuyorlar.",
            "museum_query": "Bu önemli müzeleri ziyaret etmenizi öneririm: Ayasofya, Topkapı Sarayı, İstanbul Arkeoloji Müzeleri ve Sultan Ahmet Camii. Her biri İstanbul'un zengin tarihine benzersiz bakış açıları sunar.",
            "transport_query": "Ulaşım için metro, tramvay, otobüs veya taksi kullanabilirsiniz. İstanbul kartı (İstanbulkart) tüm toplu taşımada geçerlidir. Boğaz'da feribot yolculukları da çok güzeldir."
        },
        "ar": {
            "greeting": "مرحباً! أنا دليلك الذكي لإستانبول. كيف يمكنني مساعدتك في استكشاف إستانبول اليوم؟",
            "restaurant_query": "إليك بعض المطاعم التركية الممتازة في السلطان أحمد: باندلي، مطعم حمدي، وباليكتشي صباح الدين. تقدم المطاعم الأصيلة العثمانية مع إطلالات رائعة على البوسفور.",
            "museum_query": "أنصح بزيارة هذه المتاحف الرائعة: آيا صوفيا، قصر توبكابي، متاحف إستانبول الأثرية، والجامع الأزرق. كل منها يقدم رؤى فريدة في تاريخ إستانبول الغني.",
            "transport_query": "للمواصلات، يمكنك استخدام المترو أو الترام أو الحافلة أو التاكسي. بطاقة إستانبول (إستانبول كارت) تعمل لجميع وسائل النقل العام. رحلات العبارة عبر البوسفور ذات مناظر خلابة أيضاً."
        }
    }

@pytest.fixture
def sample_queries():
    """Sample test queries for different categories."""
    return {
        "restaurants": [
            "Best Turkish restaurants in Sultanahmet",
            "Where can I find authentic kebab in Beyoğlu?",
            "Seafood restaurants with Bosphorus view",
            "Budget-friendly local food in Kadıköy"
        ],
        "museums": [
            "Best museums in Istanbul",
            "Opening hours for Hagia Sophia",
            "How to buy tickets for Topkapi Palace",
            "Free museums in Istanbul"
        ],
        "transportation": [
            "How to get from airport to city center",
            "Best way to travel between districts",
            "Istanbul public transport card",
            "Ferry schedules for Bosphorus"
        ],
        "multilingual": {
            "turkish": [
                "En iyi Türk restoranları nerede?",
                "İstanbul'da gezilecek yerler",
                "Topkapı Sarayı nasıl gidilir?"
            ],
            "arabic": [
                "أفضل المطاعم التركية في إستانبول",
                "كيفية الوصول إلى آيا صوفيا",
                "أفضل الأماكن للزيارة في إستانبول"
            ]
        }
    }

@pytest.fixture
def test_session():
    """Create test session data."""
    return {
        "session_id": "test-session-123",
        "user_id": "test-user-456",
        "timestamp": "2025-09-21T10:00:00Z"
    }

# Performance test fixtures
@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return {
        "response_time_ms": 10000,  # Max 10 seconds for AI processing
        "memory_usage_mb": 512,     # Max 512MB
        "concurrent_users": 50,     # Support 50 concurrent users
        "requests_per_second": 10   # Handle 10 RPS (realistic for AI)
    }

# Mock external services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    class MockOpenAIClient:
        def __init__(self):
            self.chat = MockChatCompletion()
    
    class MockChatCompletion:
        def create(self, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.choices = [
                        type('Choice', (), {
                            'message': type('Message', (), {
                                'content': "This is a mock AI response for testing purposes."
                            })()
                        })()
                    ]
                    self.usage = type('Usage', (), {
                        'total_tokens': 100,
                        'prompt_tokens': 50,
                        'completion_tokens': 50
                    })()
            return MockResponse()
    
    return MockOpenAIClient()

@pytest.fixture
def mock_google_places():
    """Mock Google Places API responses."""
    return {
        "restaurants": {
            "results": [
                {
                    "name": "Test Restaurant",
                    "place_id": "test_place_id_123",
                    "rating": 4.5,
                    "vicinity": "Sultanahmet, Istanbul"
                }
            ]
        },
        "museums": {
            "results": [
                {
                    "name": "Test Museum",
                    "place_id": "test_museum_id_456",
                    "rating": 4.8,
                    "vicinity": "Fatih, Istanbul"
                }
            ]
        }
    }
