# DistilBERT Intent Classifier - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Verify Model Files

```bash
ls -la models/distilbert_intent_classifier/
```

You should see:
- ✅ `config.json`
- ✅ `model.safetensors`
- ✅ `tokenizer_config.json`
- ✅ `intent_mapping.json`
- ✅ `training_summary.json`

### 2. Test Inference

```bash
python distilbert_intent_inference.py
```

Expected: 23 test queries with predictions and confidence scores.

### 3. Test Neural Router

```bash
python main_system_neural_integration.py
```

Expected: Hybrid routing with neural predictions and fallbacks.

### 4. Verify Backend Integration

```bash
cd backend
python -c "from main import intent_classifier; print('✅ Backend integration OK')"
```

Expected: `✅ Neural Intent Classifier (Hybrid) loaded successfully`

---

## 📖 Basic Usage

### Standalone Prediction

```python
from distilbert_intent_inference import get_distilbert_classifier

classifier = get_distilbert_classifier()
intent, confidence = classifier.predict("Ayasofya görmek istiyorum")
print(f"{intent} ({confidence:.1%})")
# Output: attraction (99.8%)
```

### With Hybrid Router

```python
from main_system_neural_integration import get_neural_router

router = get_neural_router()
result = router.route_query("En yakın restoran")
print(f"Intent: {result['intent']}")
print(f"Method: {result['method']}")
# Output: Intent: restaurant, Method: neural
```

### In Backend API

```python
# Automatically available in backend/main.py
if intent_classifier:
    routing_result = intent_classifier.route_query(user_message)
    intent = routing_result['intent']
    confidence = routing_result['confidence']
```

---

## 🎯 Supported Intents (30 Total)

**Top 10 Most Common:**
1. `attraction` - Hagia Sophia, Blue Mosque, landmarks
2. `restaurant` - Where to eat, food recommendations
3. `transportation` - How to get there, metro, bus
4. `museum` - Museum recommendations, visits
5. `accommodation` - Hotels, hostels, where to stay
6. `booking` - Reservations, tickets
7. `events` - Concerts, festivals, what's happening
8. `weather` - Weather forecast, should I bring umbrella
9. `nightlife` - Bars, clubs, evening entertainment
10. `shopping` - Markets, bazaars, Grand Bazaar

**Full list**: See `DISTILBERT_INTEGRATION_COMPLETE.md`

---

## 📊 Performance at a Glance

- **Accuracy**: 91.34%
- **Speed**: ~28ms per query
- **High Confidence**: 65% of queries
- **Device**: CPU/MPS (Apple Silicon optimized)

---

## 🐛 Troubleshooting

### Model Not Found
```bash
# Retrain the model
python train_intent_classifier.py
```

### Low Confidence
```python
# Use keyword fallback (automatic in hybrid router)
from main_system_neural_integration import get_neural_router
router = get_neural_router()  # Handles fallback automatically
```

### Import Error
```bash
# Install dependencies
pip install transformers torch
```

---

## 📚 More Information

- **Full Guide**: `DISTILBERT_INTEGRATION_GUIDE.md`
- **Complete Documentation**: `DISTILBERT_INTEGRATION_COMPLETE.md`
- **Training Script**: `train_intent_classifier.py`
- **Augmentation Script**: `aggressive_data_augmentation.py`

---

## ✅ Integration Status

- ✅ Model Trained (91.3% accuracy)
- ✅ Inference Module Ready
- ✅ Neural Router Updated
- ✅ Backend Integrated
- ✅ Production Ready

**Ready to use in your Istanbul AI system!**
