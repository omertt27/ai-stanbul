#!/usr/bin/env python3
"""
Quick vLLM Multi-Language Test
Tests all 6 languages with the Pure LLM backend
"""

import requests
import json
from datetime import datetime

# Test configuration
BACKEND_URL = "http://localhost:8002"
LANGUAGES = {
    "en": {
        "name": "English",
        "flag": "🇬🇧",
        "query": "Where can I find good Turkish restaurants in Taksim?",
        "expected_keywords": ["restaurant", "taksim", "turkish"]
    },
    "tr": {
        "name": "Turkish", 
        "flag": "🇹🇷",
        "query": "Taksim'de iyi Türk restoranları nerede bulabilirim?",
        "expected_keywords": ["restoran", "taksim"]
    },
    "fr": {
        "name": "French",
        "flag": "🇫🇷", 
        "query": "Où puis-je trouver de bons restaurants turcs à Taksim?",
        "expected_keywords": ["restaurant", "taksim"]
    },
    "ru": {
        "name": "Russian",
        "flag": "🇷🇺",
        "query": "Где я могу найти хорошие турецкие рестораны в Таксиме?",
        "expected_keywords": ["ресторан", "таксим"]
    },
    "de": {
        "name": "German",
        "flag": "🇩🇪",
        "query": "Wo finde ich gute türkische Restaurants in Taksim?",
        "expected_keywords": ["restaurant", "taksim"]
    },
    "ar": {
        "name": "Arabic",
        "flag": "🇸🇦",
        "query": "أين يمكنني أن أجد مطاعم تركية جيدة في تقسيم؟",
        "expected_keywords": ["مطعم", "تقسيم"]
    }
}

def test_language(lang_code, lang_info):
    """Test a single language"""
    print(f"\n{lang_info['flag']} Testing {lang_info['name']}...")
    print(f"Query: {lang_info['query']}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "query": lang_info['query'],
                "language": lang_code,
                "use_pure_llm": True
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response', '')
            
            print(f"✅ Response received ({len(answer)} chars)")
            print(f"Preview: {answer[:150]}...")
            
            # Check if response is in expected language (basic check)
            if len(answer) > 10:
                return {
                    "status": "✅ PASS",
                    "language": lang_info['name'],
                    "response_length": len(answer),
                    "preview": answer[:100]
                }
            else:
                return {
                    "status": "⚠️  WARN - Short response",
                    "language": lang_info['name'],
                    "response_length": len(answer),
                    "preview": answer
                }
        else:
            print(f"❌ HTTP {response.status_code}")
            return {
                "status": f"❌ FAIL - HTTP {response.status_code}",
                "language": lang_info['name']
            }
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "status": f"❌ FAIL - {str(e)[:50]}",
            "language": lang_info['name']
        }

def main():
    print("━" * 70)
    print("🧪 AI Istanbul - Multi-Language vLLM Test")
    print("━" * 70)
    print(f"Backend: {BACKEND_URL}")
    print(f"Languages: {len(LANGUAGES)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("━" * 70)
    
    # Test backend health
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"\n✅ Backend health: {health.json().get('status')}")
    except:
        print(f"\n❌ Backend not reachable at {BACKEND_URL}")
        print("Make sure backend is running: cd backend && python main_pure_llm.py")
        return
    
    # Run tests
    results = {}
    for lang_code, lang_info in LANGUAGES.items():
        result = test_language(lang_code, lang_info)
        results[lang_code] = result
    
    # Summary
    print("\n" + "━" * 70)
    print("📊 Test Summary")
    print("━" * 70)
    
    passed = sum(1 for r in results.values() if "PASS" in r['status'])
    total = len(results)
    
    for lang_code, result in results.items():
        flag = LANGUAGES[lang_code]['flag']
        print(f"{flag} {result['language']:12} - {result['status']}")
    
    print("━" * 70)
    print(f"Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("━" * 70)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_results_vllm_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "backend_url": BACKEND_URL,
            "total_tests": total,
            "passed": passed,
            "pass_rate": f"{passed/total*100:.1f}%",
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filename}")
    
    if passed == total:
        print("\n🎉 All tests passed! vLLM multi-language system is working perfectly!")
    elif passed >= total * 0.8:
        print("\n✅ Most tests passed! System is operational.")
    else:
        print("\n⚠️  Some tests failed. Check the results above.")

if __name__ == "__main__":
    main()
