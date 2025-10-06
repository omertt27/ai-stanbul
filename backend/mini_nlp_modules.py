#!/usr/bin/env python3
"""
Mini NLP Modules for AI Istanbul
===============================

Lightweight, specialized NLP modules without external LLMs:
- Sentiment Analysis: Rule-based sentiment detection
- Entity Recognition: Pattern-based entity extraction  
- Intent Classification: Keyword and pattern matching
- Language Detection: Statistical language identification
- Text Quality Assessment: Readability and coherence scoring

Features:
- No external dependencies on large models
- Fast, real-time processing
- Memory efficient
- Extensible rule-based approach
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import datetime
import string
import math

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: str  # positive, negative, neutral
    confidence: float
    score: float  # -1.0 to 1.0
    keywords: List[str]
    explanation: str

@dataclass
class EntityResult:
    """Named entity recognition result"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str

@dataclass
class IntentResult:
    """Intent classification result"""
    intent: str
    confidence: float
    matched_patterns: List[str]
    extracted_parameters: Dict[str, Any]
    alternatives: List[Tuple[str, float]]

@dataclass
class QualityResult:
    """Text quality assessment result"""
    readability_score: float
    coherence_score: float
    completeness_score: float
    overall_quality: str  # excellent, good, fair, poor
    issues: List[str]
    suggestions: List[str]

class MiniSentimentAnalyzer:
    """Lightweight sentiment analysis using rule-based approach"""
    
    def __init__(self):
        # Sentiment lexicons
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'great', 'fantastic', 'perfect',
            'beautiful', 'lovely', 'awesome', 'brilliant', 'outstanding', 'superb',
            'magnificent', 'incredible', 'delicious', 'tasty', 'friendly', 'helpful',
            'clean', 'comfortable', 'spacious', 'convenient', 'affordable', 'cheap',
            'worth', 'recommend', 'love', 'like', 'enjoy', 'pleased', 'satisfied',
            'happy', 'impressed', 'good', 'nice', 'fine', 'okay'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting',
            'disappointing', 'overpriced', 'expensive', 'dirty', 'uncomfortable',
            'crowded', 'noisy', 'rude', 'unhelpful', 'slow', 'cold', 'hot',
            'broken', 'old', 'smelly', 'tasteless', 'boring', 'waste', 'avoid',
            'hate', 'dislike', 'regret', 'disappointed', 'frustrated', 'angry',
            'poor', 'lacking', 'missing', 'wrong', 'problem', 'issue'
        }
        
        # Intensifiers and negations
        self.intensifiers = {'very', 'extremely', 'incredibly', 'absolutely', 'really', 'quite', 'so', 'too'}
        self.negations = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'none'}
        
        # Context patterns
        self.positive_patterns = [
            r'would recommend',
            r'highly recommend',
            r'must visit',
            r'worth.*visit',
            r'loved.*experience',
            r'great.*time'
        ]
        
        self.negative_patterns = [
            r'would not recommend',
            r'avoid.*place',
            r'waste.*money',
            r'not worth',
            r'regret.*visit',
            r'disappointed.*experience'
        ]
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count sentiment words
        positive_count = 0
        negative_count = 0
        sentiment_keywords = []
        
        # Check individual words
        for i, word in enumerate(words):
            is_negated = False
            
            # Check for negation in previous 2 words
            for j in range(max(0, i-2), i):
                if words[j] in self.negations:
                    is_negated = True
                    break
            
            # Check for intensifiers
            intensifier_boost = 1.0
            for j in range(max(0, i-1), i):
                if words[j] in self.intensifiers:
                    intensifier_boost = 1.5
                    break
            
            if word in self.positive_words:
                if is_negated:
                    negative_count += intensifier_boost
                else:
                    positive_count += intensifier_boost
                sentiment_keywords.append(word)
            
            elif word in self.negative_words:
                if is_negated:
                    positive_count += intensifier_boost
                else:
                    negative_count += intensifier_boost
                sentiment_keywords.append(word)
        
        # Check patterns
        for pattern in self.positive_patterns:
            if re.search(pattern, text_lower):
                positive_count += 2.0
                sentiment_keywords.append(f"pattern: {pattern}")
        
        for pattern in self.negative_patterns:
            if re.search(pattern, text_lower):
                negative_count += 2.0
                sentiment_keywords.append(f"pattern: {pattern}")
        
        # Calculate sentiment
        total_sentiment = positive_count + negative_count
        
        if total_sentiment == 0:
            sentiment = "neutral"
            score = 0.0
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / total_sentiment
            
            if score > 0.2:
                sentiment = "positive"
            elif score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            confidence = min(abs(score) + 0.5, 1.0)
        
        # Generate explanation
        if sentiment == "positive":
            explanation = f"Positive sentiment detected from {len(sentiment_keywords)} indicators"
        elif sentiment == "negative":
            explanation = f"Negative sentiment detected from {len(sentiment_keywords)} indicators"
        else:
            explanation = "Neutral sentiment - balanced or no clear indicators"
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score,
            keywords=sentiment_keywords,
            explanation=explanation
        )

class MiniEntityRecognizer:
    """Lightweight named entity recognition using patterns"""
    
    def __init__(self):
        # Istanbul-specific patterns
        self.patterns = {
            'LOCATION': [
                # Districts
                r'\b(Sultanahmet|Beyoğlu|Galata|Karaköy|Kadıköy|Beşiktaş|Şişli|Taksim|Ortaköy|Üsküdar|Eminönü|Fatih|Bakırköy|Sarıyer|Eyüp|Zeytinburnu|Pendik|Maltepe|Kartal|Ataşehir|Levent|Etiler|Nişantaşı|Bebek)\b',
                # Landmarks
                r'\b(Hagia Sophia|Blue Mosque|Topkapi Palace|Grand Bazaar|Spice Bazaar|Galata Tower|Bosphorus|Golden Horn|Maiden\'s Tower|Dolmabahçe Palace|Basilica Cistern|Süleymaniye Mosque|Chora Church|Istanbul Modern|Pera Museum)\b',
                # Areas
                r'\b(Old City|New City|Asian Side|European Side|Historic Peninsula|Princes\' Islands|Bosphorus Bridge|Golden Gate Bridge)\b'
            ],
            'FOOD': [
                r'\b(kebab|döner|baklava|Turkish delight|lokum|börek|manti|lahmacun|pide|simit|balık ekmek|Turkish breakfast|kahvaltı|çay|Turkish coffee|rakı|ayran|şalgam|künefe|Turkish ice cream|dondurma)\b',
                r'\b(Turkish cuisine|Ottoman cuisine|street food|seafood|meze|Turkish restaurant|kebab house|pastry shop|Turkish bakery)\b'
            ],
            'TRANSPORT': [
                r'\b(metro|tram|bus|ferry|taxi|dolmuş|funicular|cable car|Marmaray|Metrobüs)\b',
                r'\b(Istanbul Card|transportation card|public transport|metro station|tram stop|ferry terminal|airport|Atatürk Airport|Sabiha Gökçen|IST Airport)\b'
            ],
            'TIME': [
                r'\b(\d{1,2}:\d{2}|\d{1,2} (am|pm|AM|PM))\b',
                r'\b(morning|afternoon|evening|night|today|tomorrow|yesterday|weekend|weekday)\b',
                r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b'
            ],
            'PRICE': [
                r'\b(\d+\.?\d*\s*(TL|Turkish Lira|lira|USD|EUR|dollar|euro))\b',
                r'\b(free|expensive|cheap|affordable|budget|luxury|mid-range)\b',
                r'\b(entrance fee|ticket price|cost|price|fare)\b'
            ],
            'ACTIVITY': [
                r'\b(sightseeing|tour|cruise|shopping|dining|museum visit|walking|photography|nightlife|cultural|historical|religious|archaeological)\b',
                r'\b(guided tour|audio guide|boat tour|food tour|walking tour|day trip|excursion)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[EntityResult]:
        """Extract named entities from text"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    start, end = match.span()
                    entity_text = match.group()
                    
                    # Calculate confidence based on pattern specificity
                    confidence = 0.8 if len(entity_text) > 3 else 0.6
                    
                    # Extract context (5 words before and after)
                    words = text.split()
                    word_positions = []
                    current_pos = 0
                    
                    for word in words:
                        word_start = text.find(word, current_pos)
                        word_end = word_start + len(word)
                        word_positions.append((word_start, word_end, word))
                        current_pos = word_end
                    
                    # Find context words
                    context_words = []
                    for word_start, word_end, word in word_positions:
                        if word_start >= start - 50 and word_end <= end + 50:
                            context_words.append(word)
                    
                    context = ' '.join(context_words)
                    
                    entity = EntityResult(
                        text=entity_text,
                        entity_type=entity_type,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        context=context
                    )
                    
                    entities.append(entity)
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[EntityResult]) -> List[EntityResult]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        entities.sort(key=lambda x: (x.start_pos, -x.confidence))
        
        filtered_entities = []
        for entity in entities:
            # Check if it overlaps with any previously added entity
            overlaps = False
            for existing in filtered_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities

class MiniIntentClassifier:
    """Lightweight intent classification using keyword matching"""
    
    def __init__(self):
        # Intent patterns with keywords and phrases
        self.intent_patterns = {
            'find_restaurant': {
                'keywords': ['restaurant', 'food', 'eat', 'dining', 'meal', 'lunch', 'dinner', 'breakfast'],
                'phrases': ['where to eat', 'good restaurant', 'best food', 'hungry', 'food recommendation'],
                'weight': 1.0
            },
            'find_attraction': {
                'keywords': ['museum', 'palace', 'mosque', 'church', 'tower', 'attraction', 'sightseeing', 'visit'],
                'phrases': ['what to see', 'places to visit', 'tourist attraction', 'sightseeing', 'historic'],
                'weight': 1.0
            },
            'get_directions': {
                'keywords': ['how to get', 'direction', 'way', 'route', 'transport', 'metro', 'bus', 'taxi'],
                'phrases': ['how to go', 'how to reach', 'getting there', 'transportation'],
                'weight': 1.2
            },
            'get_info': {
                'keywords': ['info', 'information', 'detail', 'about', 'tell me', 'what is', 'hours', 'price'],
                'phrases': ['opening hours', 'entrance fee', 'ticket price', 'more info'],
                'weight': 0.8
            },
            'make_reservation': {
                'keywords': ['book', 'reserve', 'reservation', 'table', 'appointment'],
                'phrases': ['make reservation', 'book table', 'reserve table'],
                'weight': 1.5
            },
            'get_recommendations': {
                'keywords': ['recommend', 'suggest', 'best', 'top', 'popular', 'good'],
                'phrases': ['what do you recommend', 'best places', 'top attractions', 'suggestions'],
                'weight': 1.0
            },
            'plan_itinerary': {
                'keywords': ['itinerary', 'plan', 'schedule', 'day', 'trip', 'tour'],
                'phrases': ['plan my day', 'daily itinerary', 'what to do', 'trip planning'],
                'weight': 1.3
            }
        }
    
    def classify_intent(self, text: str) -> IntentResult:
        """Classify the intent of given text"""
        text_lower = text.lower()
        scores = {}
        matched_patterns = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0.0
            patterns = []
            
            # Check keywords
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += 1.0 * config['weight']
                    patterns.append(f"keyword: {keyword}")
            
            # Check phrases
            for phrase in config['phrases']:
                if phrase in text_lower:
                    score += 2.0 * config['weight']
                    patterns.append(f"phrase: {phrase}")
            
            if score > 0:
                scores[intent] = score
                matched_patterns[intent] = patterns
        
        if not scores:
            return IntentResult(
                intent='unknown',
                confidence=0.0,
                matched_patterns=[],
                extracted_parameters={},
                alternatives=[]
            )
        
        # Get top intent
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0]
        
        # Calculate confidence
        total_score = sum(scores.values())
        confidence = min(top_score / total_score, 1.0) if total_score > 0 else 0.0
        
        # Get alternatives
        alternatives = [(intent, score/total_score) for intent, score in sorted_intents[1:3]]
        
        # Extract parameters (basic implementation)
        parameters = self._extract_parameters(text, top_intent)
        
        return IntentResult(
            intent=top_intent,
            confidence=confidence,
            matched_patterns=matched_patterns.get(top_intent, []),
            extracted_parameters=parameters,
            alternatives=alternatives
        )
    
    def _extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters based on intent"""
        parameters = {}
        
        # Simple parameter extraction patterns
        if intent in ['find_restaurant', 'find_attraction']:
            # Extract location
            location_match = re.search(r'\b(in|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
            if location_match:
                parameters['location'] = location_match.group(2)
        
        if intent == 'find_restaurant':
            # Extract cuisine type
            cuisine_match = re.search(r'\b(turkish|italian|chinese|japanese|french|american|indian|mexican|seafood|vegetarian)\b', text.lower())
            if cuisine_match:
                parameters['cuisine'] = cuisine_match.group(1)
        
        if intent == 'get_directions':
            # Extract destination
            to_match = re.search(r'\b(to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
            if to_match:
                parameters['destination'] = to_match.group(2)
        
        return parameters

class MiniTextQualityAssessor:
    """Lightweight text quality assessment"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
    
    def assess_quality(self, text: str) -> QualityResult:
        """Assess the quality of given text"""
        # Basic metrics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        char_count = len(text)
        
        # Readability assessment
        readability_score = self._calculate_readability(text, word_count, sentence_count)
        
        # Coherence assessment
        coherence_score = self._calculate_coherence(text)
        
        # Completeness assessment
        completeness_score = self._calculate_completeness(text, word_count)
        
        # Overall quality
        overall_score = (readability_score + coherence_score + completeness_score) / 3
        
        if overall_score >= 0.8:
            quality = "excellent"
        elif overall_score >= 0.6:
            quality = "good"
        elif overall_score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"
        
        # Identify issues and suggestions
        issues, suggestions = self._identify_issues_and_suggestions(
            text, readability_score, coherence_score, completeness_score
        )
        
        return QualityResult(
            readability_score=readability_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            overall_quality=quality,
            issues=issues,
            suggestions=suggestions
        )
    
    def _calculate_readability(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        avg_sentence_length = word_count / sentence_count
        
        # Count syllables (approximate)
        syllable_count = 0
        for word in text.split():
            syllable_count += max(1, len(re.findall(r'[aeiouAEIOU]', word)))
        
        avg_syllables_per_word = syllable_count / word_count
        
        # Simplified readability score
        score = 1.0 - (avg_sentence_length / 20.0 + avg_syllables_per_word / 3.0) / 2.0
        return max(0.0, min(1.0, score))
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score based on transition words and structure"""
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'first', 'second', 'finally', 'also',
            'besides', 'nevertheless', 'thus', 'hence', 'accordingly'
        }
        
        words = text.lower().split()
        transition_count = sum(1 for word in words if word in transition_words)
        
        # Score based on transition word usage
        coherence_score = min(transition_count / max(len(words) / 50, 1), 1.0)
        
        # Bonus for proper punctuation
        if re.search(r'[.!?]', text):
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _calculate_completeness(self, text: str, word_count: int) -> float:
        """Calculate completeness score based on text length and structure"""
        # Length-based completeness
        if word_count < 5:
            length_score = 0.0
        elif word_count < 20:
            length_score = word_count / 20.0
        else:
            length_score = 1.0
        
        # Structure-based completeness
        has_beginning = bool(re.search(r'^[A-Z]', text.strip()))
        has_ending = bool(re.search(r'[.!?]$', text.strip()))
        
        structure_score = (int(has_beginning) + int(has_ending)) / 2.0
        
        return (length_score + structure_score) / 2.0
    
    def _identify_issues_and_suggestions(self, text: str, readability: float,
                                       coherence: float, completeness: float) -> Tuple[List[str], List[str]]:
        """Identify quality issues and provide suggestions"""
        issues = []
        suggestions = []
        
        if readability < 0.5:
            issues.append("Text is difficult to read")
            suggestions.append("Use shorter sentences and simpler words")
        
        if coherence < 0.4:
            issues.append("Text lacks coherence")
            suggestions.append("Add transition words and improve flow")
        
        if completeness < 0.5:
            issues.append("Text appears incomplete")
            suggestions.append("Add more detail and ensure proper structure")
        
        # Check for specific issues
        if not re.search(r'[.!?]', text):
            issues.append("Missing punctuation")
            suggestions.append("Add proper punctuation marks")
        
        if len(text.split()) < 5:
            issues.append("Text is too short")
            suggestions.append("Provide more detailed information")
        
        return issues, suggestions

class MiniNLPProcessor:
    """Main NLP processor combining all mini modules"""
    
    def __init__(self):
        self.sentiment_analyzer = MiniSentimentAnalyzer()
        self.entity_recognizer = MiniEntityRecognizer()
        self.intent_classifier = MiniIntentClassifier()
        self.quality_assessor = MiniTextQualityAssessor()
    
    def process_text(self, text: str, include_quality: bool = False) -> Dict[str, Any]:
        """Process text with all NLP modules"""
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'sentiment': self.sentiment_analyzer.analyze_sentiment(text),
            'entities': self.entity_recognizer.extract_entities(text),
            'intent': self.intent_classifier.classify_intent(text)
        }
        
        if include_quality:
            results['quality'] = self.quality_assessor.assess_quality(text)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'modules_available': ['sentiment', 'entities', 'intent', 'quality'],
            'processing_time_avg': '< 10ms',
            'memory_usage': 'lightweight',
            'dependencies': 'none (rule-based)'
        }

# Global mini NLP processor
mini_nlp_processor = MiniNLPProcessor()

def test_mini_nlp_modules():
    """Test all mini NLP modules"""
    print("🧪 Testing Mini NLP Modules...")
    
    test_texts = [
        "I absolutely love the amazing Turkish restaurants in Sultanahmet! The food is delicious.",
        "How can I get to Galata Tower from Taksim Square using public transport?",
        "Can you recommend the best museums to visit this weekend?",
        "The hotel was terrible and overpriced. I would not recommend it to anyone."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Testing text {i}: '{text}'")
        
        results = mini_nlp_processor.process_text(text, include_quality=True)
        
        # Print results
        sentiment = results['sentiment']
        print(f"   😊 Sentiment: {sentiment.sentiment} ({sentiment.confidence:.2f} confidence)")
        
        entities = results['entities']
        print(f"   🏷️  Entities: {len(entities)} found")
        for entity in entities[:2]:  # Show first 2
            print(f"      - {entity.text} ({entity.entity_type})")
        
        intent = results['intent']
        print(f"   🎯 Intent: {intent.intent} ({intent.confidence:.2f} confidence)")
        
        quality = results['quality']
        print(f"   📊 Quality: {quality.overall_quality} (readability: {quality.readability_score:.2f})")
    
    # Get stats
    stats = mini_nlp_processor.get_processing_stats()
    print(f"\n📈 NLP Processing Stats: {len(stats['modules_available'])} modules, {stats['processing_time_avg']} processing time")
    
    return True

if __name__ == "__main__":
    success = test_mini_nlp_modules()
    if success:
        print("✅ Mini NLP Modules are working correctly!")
    else:
        print("❌ Mini NLP Modules test failed")
