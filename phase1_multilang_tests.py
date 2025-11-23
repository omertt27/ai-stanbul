#!/usr/bin/env python3
"""
Phase 1: Multi-Language End-to-End Test Suite
Comprehensive testing of all 6 supported languages
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Tuple
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "https://api.aistanbul.net")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://aistanbul.net")

# Test scenarios covering different intent types
TEST_SCENARIOS = {
    "en": {
        "restaurant_query": "What are the best seafood restaurants in Beşiktaş?",
        "place_query": "Tell me about Galata Tower",
        "route_query": "How do I get from Taksim to Sultanahmet?",
        "weather_query": "What's the weather like in Istanbul today?",
        "event_query": "What events are happening this weekend?",
        "family_query": "What are good family activities near Kadıköy?",
        "expected_keywords": ["restaurant", "seafood", "galata", "route", "weather", "family"]
    },
    "tr": {
        "restaurant_query": "Beşiktaş'ta en iyi deniz ürünleri restoranları nereler?",
        "place_query": "Galata Kulesi hakkında bilgi verir misin?",
        "route_query": "Taksim'den Sultanahmet'e nasıl giderim?",
        "weather_query": "İstanbul'da bugün hava nasıl?",
        "event_query": "Bu hafta sonu hangi etkinlikler var?",
        "family_query": "Kadıköy yakınında aileler için iyi aktiviteler neler?",
        "expected_keywords": ["restoran", "galata", "yol", "hava", "etkinlik", "aile"]
    },
    "ar": {
        "restaurant_query": "ما هي أفضل مطاعم المأكولات البحرية في بشكتاش؟",
        "place_query": "أخبرني عن برج غلطة",
        "route_query": "كيف أذهب من تقسيم إلى سلطان أحمد؟",
        "weather_query": "كيف هو الطقس في إسطنبول اليوم؟",
        "event_query": "ما هي الفعاليات المقامة في نهاية هذا الأسبوع؟",
        "family_query": "ما هي الأنشطة العائلية الجيدة بالقرب من كاديكوي؟",
        "expected_keywords": ["مطعم", "برج", "طريق", "طقس", "فعالية", "عائلة"]
    },
    "de": {
        "restaurant_query": "Was sind die besten Fischrestaurants in Beşiktaş?",
        "place_query": "Erzähle mir über den Galata-Turm",
        "route_query": "Wie komme ich von Taksim nach Sultanahmet?",
        "weather_query": "Wie ist das Wetter heute in Istanbul?",
        "event_query": "Welche Veranstaltungen finden dieses Wochenende statt?",
        "family_query": "Was sind gute Familienaktivitäten in der Nähe von Kadıköy?",
        "expected_keywords": ["restaurant", "galata", "route", "wetter", "veranstaltung", "familie"]
    },
    "fr": {
        "restaurant_query": "Quels sont les meilleurs restaurants de fruits de mer à Beşiktaş?",
        "place_query": "Parle-moi de la tour de Galata",
        "route_query": "Comment aller de Taksim à Sultanahmet?",
        "weather_query": "Quel temps fait-il à Istanbul aujourd'hui?",
        "event_query": "Quels événements ont lieu ce week-end?",
        "family_query": "Quelles sont les bonnes activités familiales près de Kadıköy?",
        "expected_keywords": ["restaurant", "galata", "route", "météo", "événement", "famille"]
    },
    "es": {
        "restaurant_query": "¿Cuáles son los mejores restaurantes de mariscos en Beşiktaş?",
        "place_query": "Cuéntame sobre la Torre Gálata",
        "route_query": "¿Cómo llego de Taksim a Sultanahmet?",
        "weather_query": "¿Cómo está el clima en Estambul hoy?",
        "event_query": "¿Qué eventos hay este fin de semana?",
        "family_query": "¿Cuáles son buenas actividades familiares cerca de Kadıköy?",
        "expected_keywords": ["restaurante", "galata", "ruta", "clima", "evento", "familia"]
    }
}

class MultiLanguageTestSuite:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "languages": {}
        }
        self.start_time = time.time()

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Fore.CYAN}{'='*100}")
        print(f"{Fore.CYAN}{text:^100}")
        print(f"{Fore.CYAN}{'='*100}\n")

    def print_success(self, text: str):
        """Print success message"""
        print(f"{Fore.GREEN}✅ {text}")

    def print_error(self, text: str):
        """Print error message"""
        print(f"{Fore.RED}❌ {text}")

    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠️  {text}")

    def print_info(self, text: str):
        """Print info message"""
        print(f"{Fore.BLUE}ℹ️  {text}")

    def test_chat_query(self, language: str, query: str, scenario_name: str) -> Tuple[bool, Dict]:
        """Test a single chat query"""
        try:
            payload = {
                "message": query,
                "language": language,
                "session_id": f"test_{language}_{scenario_name}_{int(time.time())}"
            }
            
            start = time.time()
            response = requests.post(
                f"{BACKEND_URL}/api/chat",
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start
            
            if response.status_code != 200:
                return False, {
                    "error": f"HTTP {response.status_code}",
                    "elapsed": elapsed
                }
            
            data = response.json()
            response_text = data.get("response", "")
            
            # Validation checks
            checks = {
                "has_response": len(response_text) > 0,
                "meaningful_length": len(response_text) > 20,
                "response_time_ok": elapsed < 10.0,
                "no_error_message": "error" not in response_text.lower()[:100]
            }
            
            all_passed = all(checks.values())
            
            return all_passed, {
                "response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "response_length": len(response_text),
                "elapsed": elapsed,
                "checks": checks,
                "full_response": data
            }
            
        except Exception as e:
            return False, {
                "error": str(e),
                "elapsed": 0
            }

    def test_language_comprehensive(self, language: str, scenarios: Dict) -> Dict:
        """Test all scenarios for a specific language"""
        self.print_header(f"🌍 Testing {language.upper()} - Comprehensive Scenarios")
        
        lang_results = {
            "language": language,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_time": 0,
            "scenarios": {}
        }
        
        # Test each scenario
        for scenario_name, query in scenarios.items():
            if scenario_name == "expected_keywords":
                continue
            
            self.print_info(f"Testing {scenario_name}...")
            self.print_info(f"Query: {query[:80]}...")
            
            passed, details = self.test_chat_query(language, query, scenario_name)
            
            self.results["total_tests"] += 1
            lang_results["total_time"] += details.get("elapsed", 0)
            
            if passed:
                self.results["passed"] += 1
                lang_results["tests_passed"] += 1
                self.print_success(f"{scenario_name}: PASS ({details.get('elapsed', 0):.2f}s)")
                
                # Show response preview
                response_preview = details.get("response", "")[:150]
                self.print_info(f"Response: {response_preview}...")
            else:
                self.results["failed"] += 1
                lang_results["tests_failed"] += 1
                self.print_error(f"{scenario_name}: FAIL")
                
                # Show error details
                if "error" in details:
                    self.print_error(f"Error: {details['error']}")
                else:
                    failed_checks = [k for k, v in details.get("checks", {}).items() if not v]
                    self.print_error(f"Failed checks: {', '.join(failed_checks)}")
            
            lang_results["scenarios"][scenario_name] = {
                "passed": passed,
                "details": details
            }
            
            # Rate limiting protection
            time.sleep(1)
        
        # Language summary
        print()
        total = lang_results["tests_passed"] + lang_results["tests_failed"]
        success_rate = (lang_results["tests_passed"] / total * 100) if total > 0 else 0
        
        if success_rate == 100:
            self.print_success(f"{language.upper()} Summary: {lang_results['tests_passed']}/{total} passed (100%) ✨")
        elif success_rate >= 80:
            self.print_warning(f"{language.upper()} Summary: {lang_results['tests_passed']}/{total} passed ({success_rate:.1f}%)")
        else:
            self.print_error(f"{language.upper()} Summary: {lang_results['tests_passed']}/{total} passed ({success_rate:.1f}%)")
        
        self.results["languages"][language] = lang_results
        return lang_results

    def test_response_quality(self, language: str, response: str, expected_keywords: List[str]) -> Dict:
        """Test response quality and relevance"""
        checks = {
            "has_content": len(response) > 50,
            "appropriate_length": 50 < len(response) < 2000,
            "has_structure": any(c in response for c in ['.', '!', '?', '\n']),
            "no_repetition": not self._has_excessive_repetition(response),
            "contextually_relevant": any(kw.lower() in response.lower() for kw in expected_keywords)
        }
        
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "score": sum(checks.values()) / len(checks) * 100
        }

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check if text has excessive repetition"""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        for i in range(len(words) - 5):
            phrase = ' '.join(words[i:i+3])
            if words[i+3:].count(phrase.split()[0]) > 2:
                return True
        
        return False

    def run_all_tests(self):
        """Run complete test suite"""
        self.print_header("🧪 PHASE 1: MULTI-LANGUAGE END-TO-END TEST SUITE")
        
        print(f"{Fore.BLUE}Configuration:")
        print(f"  Backend URL: {BACKEND_URL}")
        print(f"  Frontend URL: {FRONTEND_URL}")
        print(f"  Languages: {', '.join(TEST_SCENARIOS.keys())}")
        print(f"  Scenarios per language: {len([k for k in TEST_SCENARIOS['en'].keys() if k != 'expected_keywords'])}")
        print()

        # Test each language
        for language, scenarios in TEST_SCENARIOS.items():
            self.test_language_comprehensive(language, scenarios)
            print()

        # Final Summary
        self.print_header("📊 FINAL SUMMARY")
        
        total_time = time.time() - self.start_time
        success_rate = (self.results["passed"] / self.results["total_tests"] * 100) if self.results["total_tests"] > 0 else 0
        
        print(f"{Fore.CYAN}Overall Statistics:")
        print(f"  Total Tests: {self.results['total_tests']}")
        print(f"  {Fore.GREEN}Passed: {self.results['passed']}")
        print(f"  {Fore.RED}Failed: {self.results['failed']}")
        print(f"  {Fore.YELLOW}Success Rate: {success_rate:.1f}%")
        print(f"  {Fore.BLUE}Total Time: {total_time:.2f}s")
        print()

        # Language breakdown
        print(f"{Fore.CYAN}Language Breakdown:")
        for lang, results in self.results["languages"].items():
            total = results["tests_passed"] + results["tests_failed"]
            rate = (results["tests_passed"] / total * 100) if total > 0 else 0
            emoji = "✅" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"  {emoji} {lang.upper()}: {results['tests_passed']}/{total} ({rate:.1f}%) - {results['total_time']:.2f}s")
        
        print()

        # Status determination
        if self.results["failed"] == 0:
            self.print_success("🎉 ALL TESTS PASSED! Multi-language system is production ready!")
            status = "PASSED"
            return_code = 0
        elif success_rate >= 90:
            self.print_warning("⚠️  MOST TESTS PASSED. Minor issues detected - review and fix.")
            status = "MOSTLY_PASSED"
            return_code = 1
        elif success_rate >= 70:
            self.print_warning("⚠️  SOME TESTS FAILED. Significant issues detected - requires attention.")
            status = "PARTIAL"
            return_code = 2
        else:
            self.print_error("❌ MANY TESTS FAILED! System has critical issues - not production ready!")
            status = "FAILED"
            return_code = 3

        # Save detailed report
        report = {
            **self.results,
            "status": status,
            "summary": {
                "success_rate": success_rate,
                "total_time": total_time,
                "average_response_time": sum(
                    r["total_time"] / max(r["tests_passed"] + r["tests_failed"], 1)
                    for r in self.results["languages"].values()
                ) / len(self.results["languages"])
            }
        }
        
        report_file = f"multilang_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.print_info(f"Detailed report saved to: {report_file}")
        
        # Recommendations
        if self.results["failed"] > 0:
            print()
            self.print_header("🔧 RECOMMENDATIONS")
            
            for lang, results in self.results["languages"].items():
                if results["tests_failed"] > 0:
                    print(f"\n{Fore.YELLOW}{lang.upper()} Issues:")
                    for scenario, details in results["scenarios"].items():
                        if not details["passed"]:
                            print(f"  ❌ {scenario}:")
                            if "error" in details["details"]:
                                print(f"     Error: {details['details']['error']}")
                            elif "checks" in details["details"]:
                                failed = [k for k, v in details["details"]["checks"].items() if not v]
                                print(f"     Failed checks: {', '.join(failed)}")
        
        return return_code

if __name__ == "__main__":
    # Check if required env vars are set
    if not BACKEND_URL:
        print(f"{Fore.RED}❌ ERROR: BACKEND_URL environment variable not set!")
        print(f"{Fore.YELLOW}Please set:")
        print(f"  export BACKEND_URL=https://api.aistanbul.net")
        print(f"  export FRONTEND_URL=https://aistanbul.net")
        sys.exit(1)

    suite = MultiLanguageTestSuite()
    exit_code = suite.run_all_tests()
    sys.exit(exit_code)
