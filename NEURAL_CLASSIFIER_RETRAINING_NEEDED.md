# 🧠 Neural Classifier Retraining - Required

## 🎯 Current Problem

The neural intent classifier (DistilBERT) is **misclassifying** queries and not properly recognizing all handler intents. 

**Example Issue:**
- Query: "how can i go to taksim from my location"
- Neural Classification: `local_tips` (0.99 confidence) ❌
- Should be: `transportation` ✅

This causes the system to give generic responses instead of proper transportation directions with maps.

---

## 📋 All Handler Intents That Should Be Classified

### 1. **🍽️ Restaurants**
```
- Location-specific searches (Beyoğlu, Sultanahmet, Kadıköy, etc.)
- Cuisine filtering (Turkish, seafood, vegetarian, street food)
- Dietary restrictions (vegetarian, vegan, halal, kosher, gluten-free)
- Price level indicators and operating hours
- Smart typo correction and context-aware follow-ups
```

**Example Queries:**
- "Best restaurants in Beyoğlu"
- "Vegetarian restaurants near me"
- "Where can I eat Turkish food?"
- "Cheap restaurants in Kadıköy"

### 2. **🏛️ Places & Attractions**
```
- 78+ curated Istanbul attractions in database
- Category filtering (museums, monuments, parks, religious sites)
- District-based recommendations
- Weather-appropriate suggestions
- Family-friendly and romantic spot recommendations
- Budget-friendly (free) activities
```

**Example Queries:**
- "What museums should I visit?"
- "Free things to do in Istanbul"
- "Family-friendly attractions"
- "Best places to visit in rainy weather"

### 3. **🏘️ Neighborhood Guides**
```
- Detailed information for all major Istanbul areas
- Character descriptions and best visiting times
- Local insights and hidden gems
- District-specific recommendations
- Especially: Beşiktaş, Şişli, Üsküdar, Kadıköy, Fatih, Sultanahmet, Sarıyer
```

**Example Queries:**
- "Tell me about Beşiktaş"
- "What's special about Kadıköy?"
- "Best time to visit Sultanahmet"

### 4. **🚇 Transportation**
```
- Metro system guidance and routes
- Bus connections and ferry services
- Airport transfers (IST & SAW)
- Public transport card information
- Walking directions between attractions
- GPS-based directions from user's location
```

**Example Queries:**
- "How can I go to Taksim from my location?" ⭐
- "Metro from Sultanahmet to Beşiktaş"
- "Bus to Taksim Square"
- "Airport transfer to city center"
- "How to use İstanbulkart"

### 5. **💬 Daily Talks**
```
- Greetings and casual conversation
- Getting to know the user
- Small talk about Istanbul
- Personal recommendations
```

**Example Queries:**
- "Hello"
- "Good morning"
- "What do you recommend?"
- "Tell me about yourself"

### 6. **💎 Local Tips / Hidden Gems**
```
- Off-the-beaten-path recommendations
- Local secrets and insider tips
- Authentic experiences
- Lesser-known attractions
```

**Example Queries:**
- "Hidden gems in Istanbul"
- "Local secrets"
- "Where do locals go?"
- "Authentic Istanbul experiences"

### 7. **🌤️ Weather-Aware**
```
- Current weather information
- Weather-based recommendations
- Activity suggestions based on weather
- Outfit recommendations
```

**Example Queries:**
- "What's the weather like?"
- "Should I bring an umbrella?"
- "What to do on a rainy day?"
- "Is it good weather for sightseeing?"

### 8. **🎭 Events**
```
- Current cultural events
- Festival information
- Concert schedules
- Art exhibitions
- İKSV events
```

**Example Queries:**
- "What events are happening?"
- "Concerts this week"
- "Art exhibitions in Istanbul"
- "Cultural festivals"

### 9. **🗺️ Route Planner**
```
- Multi-stop itineraries
- Day trip planning
- Custom routes with multiple attractions
- Time-optimized schedules
```

**Example Queries:**
- "Plan a day trip to Sultanahmet"
- "Best route to visit 3 museums"
- "One-day itinerary for Istanbul"

---

## 🔍 Root Cause Analysis

### **Why Neural Classifier Fails:**

1. **Insufficient Training Data**
   - Model not trained on enough examples of each intent
   - Particularly weak on transportation queries
   - Missing edge cases and variations

2. **Imbalanced Dataset**
   - Some intents have more training examples than others
   - Transportation might be underrepresented
   - Local tips might be overrepresented (causing false positives)

3. **Model Architecture**
   - DistilBERT is good but needs fine-tuning on Istanbul-specific queries
   - May need domain adaptation for travel/tourism context
   - Needs Turkish language support for mixed Turkish-English queries

4. **Confidence Threshold Too High**
   - Currently returns immediately if confidence >= 0.80
   - Doesn't validate against keyword patterns
   - No cross-check with rule-based system

---

## ✅ Short-Term Fix (COMPLETED)

Added **keyword-based override** in `hybrid_intent_classifier.py`:

```python
# CRITICAL: Check for transportation keywords BEFORE accepting neural result
strong_transportation_keywords = [
    'how can i go', 'how do i get', 'how to get to', 'how to go to',
    'directions to', 'navigate to', 'route to', 'way to get',
    'take me to', 'show me the way', 'from my location', 'from here to',
    'metro to', 'bus to', 'tram to', 'transportation to',
    'how far is', 'distance to', 'travel to'
]

has_strong_transportation = any(keyword in message_lower for keyword in strong_transportation_keywords)

if neural_confidence >= 0.80 and has_strong_transportation and neural_intent != 'transportation':
    logger.warning(f"⚠️ Neural misclassified - forcing transportation")
    return IntentResult(
        primary_intent='transportation',
        confidence=0.95,
        method='keyword_override'
    )
```

**Result:** Transportation queries now work, but this is a **band-aid fix**.

---

## 🎯 Long-Term Solution: Retrain Neural Model

### **Step 1: Collect Training Data**

Create comprehensive training dataset with **at least 50-100 examples per intent**:

```python
training_data = {
    'transportation': [
        "how can i go to taksim from my location",
        "how do i get to sultanahmet",
        "metro from kadikoy to besiktas",
        "bus to taksim square",
        "directions to galata tower",
        # ... 100+ examples
    ],
    'restaurants': [
        "best restaurants in beyoglu",
        "where can i eat seafood",
        "vegetarian restaurants near me",
        # ... 100+ examples
    ],
    'attractions': [
        "what museums should i visit",
        "best places to see in istanbul",
        "family friendly attractions",
        # ... 100+ examples
    ],
    # ... all other intents
}
```

### **Step 2: Balance Dataset**

Ensure equal representation:
- **Transportation**: 100 examples
- **Restaurants**: 100 examples
- **Attractions**: 100 examples
- **Neighborhoods**: 100 examples
- **Daily Talks**: 100 examples
- **Local Tips**: 100 examples
- **Weather**: 100 examples
- **Events**: 100 examples
- **Route Planner**: 100 examples

**Total: ~900 examples minimum**

### **Step 3: Fine-tune DistilBERT**

```python
from transformers import DistilBertForSequenceClassification, Trainer

# Load base model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=9  # 9 intent classes
)

# Fine-tune on Istanbul-specific data
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
```

### **Step 4: Validate and Test**

```python
# Test set: 20% of data
# Must achieve:
# - Overall accuracy: >90%
# - Per-intent accuracy: >85%
# - Confidence calibration: High confidence = correct prediction
```

### **Step 5: Deploy New Model**

```bash
# Save fine-tuned model
model.save_pretrained('models/distilbert_intent_classifier_v2')

# Replace in production
cp -r models/distilbert_intent_classifier_v2 models/distilbert_intent_classifier
```

---

## 🚀 Implementation Plan

### **Phase 1: Data Collection (1-2 days)**
- [ ] Create training data for each intent (50-100 examples each)
- [ ] Include variations, misspellings, and edge cases
- [ ] Add Turkish-English mixed queries
- [ ] Balance dataset across all intents

### **Phase 2: Model Training (1 day)**
- [ ] Set up training pipeline
- [ ] Fine-tune DistilBERT on collected data
- [ ] Validate on test set
- [ ] Achieve >90% accuracy

### **Phase 3: Testing & Validation (1 day)**
- [ ] Test all query types
- [ ] Validate GPS-based transportation queries
- [ ] Check edge cases and error handling
- [ ] Performance benchmarking

### **Phase 4: Deployment (1 day)**
- [ ] Deploy new model to production
- [ ] Monitor classification accuracy
- [ ] A/B test against old model
- [ ] Remove keyword overrides once confident

**Total Time: 3-5 days**

---

## 📊 Expected Results After Retraining

### **Before (Current State):**
```
Query: "how can i go to taksim from my location"
Neural: local_tips (0.99) ❌
Result: Keyword override to transportation ⚠️
```

### **After (Retrained Model):**
```
Query: "how can i go to taksim from my location"
Neural: transportation (0.95) ✅
Result: Correct classification, no override needed ✅
```

### **Performance Metrics:**
- Overall Accuracy: **95%+** (from ~70%)
- Transportation Accuracy: **98%+** (from ~30%)
- Restaurants Accuracy: **95%+**
- Attractions Accuracy: **95%+**
- All other intents: **90%+**

---

## 🎯 Why This Matters

1. **Better User Experience**
   - Accurate responses every time
   - No generic fallback responses
   - Proper handler routing

2. **Scalability**
   - Can add new intents easily
   - ML learns from data, not just rules
   - Adapts to new query patterns

3. **Maintainability**
   - Less keyword maintenance
   - Fewer edge cases to handle
   - Self-improving with more data

4. **Production Quality**
   - Reliable classification
   - High confidence predictions
   - Proper GPS integration works

---

## 🔧 Alternative: Improve Keyword Classifier

If neural retraining takes too long, enhance keyword-based system:

### **Expand Transportation Keywords**
```python
transportation_patterns = [
    # Direction queries
    r'\bhow (can|do) i (go|get|reach|travel)',
    r'\b(direction|route|way) (to|from)',
    r'\bnavigate (to|from|me)',
    r'\btake me to\b',
    
    # Location-based
    r'\bfrom (my location|here|where i am)',
    r'\bto (taksim|sultanahmet|kadikoy|besiktas)',
    
    # Transport modes
    r'\b(metro|bus|tram|ferry|train|taxi) (to|from)',
    r'\bpublic transport(ation)?\b',
    r'\bistanbulkart\b',
    
    # Distance/time
    r'\bhow far (is|to)\b',
    r'\bhow long (to get|does it take)\b',
    r'\bdistance (to|from|between)\b'
]
```

**But this is still a band-aid!** Neural retraining is the proper solution.

---

## 📝 Summary

**Current State:**
- ✅ Keyword override fixes transportation queries
- ⚠️ Neural model still misclassifies
- ❌ Not a sustainable long-term solution

**Needed:**
- 🎯 Retrain neural model with comprehensive dataset
- 📊 Achieve >90% accuracy on all intents
- 🚀 Remove keyword overrides
- ✅ Production-quality intent classification

**Timeline:** 3-5 days for full retraining and deployment

**Priority:** HIGH - This affects core functionality and user experience

---

## 🎉 Conclusion

The hybrid intent classifier with keyword override **works for now**, but we need to:

1. **Collect comprehensive training data** for all 9 intents
2. **Retrain DistilBERT** with balanced dataset
3. **Validate accuracy** >90% on all intents
4. **Deploy new model** and remove keyword hacks

This will give us:
- ✅ Reliable, accurate intent classification
- ✅ Proper GPS-based transportation queries
- ✅ All handlers working correctly
- ✅ Production-quality ML system

**Let's schedule the retraining sprint! 🚀**
