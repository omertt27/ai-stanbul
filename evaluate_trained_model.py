"""
Comprehensive evaluation of the trained neural intent classifier.
Tests the model with real-world Turkish and English queries.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load intent mapping
    with open(f"{model_path}/intent_mapping.json", "r") as f:
        intent_mapping = json.load(f)
    
    id2label = {int(k): v for k, v in intent_mapping["idx_to_intent"].items()}
    
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer, id2label, device

def predict_intent(text, model, tokenizer, id2label, device):
    """Predict intent for a given text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_id].item()
    
    return id2label[predicted_id], confidence, probs[0].cpu().numpy()

def evaluate_test_queries():
    """Test with real-world queries"""
    test_queries = {
        # Transportation - Turkish
        "transportation": [
            "Taksim'den Sultanahmet'e nasıl gidebilirim?",
            "En yakın metro istasyonu nerede?",
            "Havalimanından otele nasıl ulaşırım?",
            "İstanbulkart nereden alabilirim?",
            "Toplu taşıma ücreti ne kadar?",
            # English
            "How do I get to the Blue Mosque?",
            "Where is the nearest bus stop?",
            "Can I take the metro to the airport?",
            "How much is a taxi to Taksim?",
        ],
        # Restaurant - Turkish
        "restaurant": [
            "Bana iyi bir restoran önerir misin?",
            "Balık restoranı arıyorum",
            "Sultanahmet'te nerede yemek yiyebilirim?",
            "Vegan restoranlar var mı?",
            "En iyi kebap nerede?",
            # English
            "Can you recommend a good restaurant?",
            "I'm looking for Turkish cuisine",
            "Where can I eat near Taksim?",
            "Best seafood restaurant?",
        ],
        # Hidden Gems - Turkish
        "hidden_gems": [
            "Turistik olmayan yerler göster",
            "Gizli kalmış güzel mekanlar",
            "Yerel halkın gittiği yerler",
            "Az bilinen güzel yerler",
            # English
            "Show me hidden gems in Istanbul",
            "Off the beaten path places",
            "Local spots tourists don't know",
        ],
        # Route Planning - Turkish
        "route_planning": [
            "Bana 3 günlük gezi planı yap",
            "Sultanahmet, Taksim ve Beşiktaş'ı kapsayan rota",
            "Bir günde hangi yerleri gezebilirim?",
            "En iyi gezi rotası nedir?",
            # English
            "Plan a 2-day itinerary for me",
            "Create a route through historic sites",
            "Best way to visit multiple attractions",
        ],
        # Neighborhoods - Turkish
        "neighborhoods": [
            "Beşiktaş nasıl bir yer?",
            "Hangi semtte kalmalıyım?",
            "Kadıköy hakkında bilgi ver",
            "En güvenli semtler hangileri?",
            # English
            "Tell me about Beyoğlu",
            "Which neighborhood should I stay in?",
            "What's Kadıköy like?",
        ],
        # Attraction - Turkish
        "attraction": [
            "Ayasofya'yı ziyaret etmek istiyorum",
            "Topkapı Sarayı hakkında bilgi",
            "Galata Kulesi açık mı?",
            "En önemli turistik yerler",
            # English
            "I want to visit the Hagia Sophia",
            "Tell me about the Grand Bazaar",
            "What are the must-see attractions?",
        ],
        # Weather - Turkish
        "weather": [
            "Hava durumu nasıl?",
            "Yarın yağmur yağacak mı?",
            "Bu hafta hava nasıl olacak?",
            # English
            "What's the weather like?",
            "Will it rain tomorrow?",
        ],
    }
    
    return test_queries

def main():
    model_path = "models/istanbul_intent_classifier_finetuned"
    
    # Load model
    model, tokenizer, id2label, device = load_model_and_tokenizer(model_path)
    
    print("\n" + "="*80)
    print("NEURAL INTENT CLASSIFIER EVALUATION")
    print("="*80)
    
    # Get test queries
    test_queries = evaluate_test_queries()
    
    # Track results
    results = {
        "correct": 0,
        "total": 0,
        "by_intent": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_language": defaultdict(lambda: {"correct": 0, "total": 0}),
        "predictions": []
    }
    
    # Test each query
    for expected_intent, queries in test_queries.items():
        print(f"\n{'='*80}")
        print(f"Testing: {expected_intent.upper()}")
        print(f"{'='*80}")
        
        for i, query in enumerate(queries, 1):
            # Detect language
            is_turkish = any(char in query for char in "ığüşöçİĞÜŞÖÇ")
            language = "Turkish" if is_turkish else "English"
            
            predicted_intent, confidence, _ = predict_intent(query, model, tokenizer, id2label, device)
            is_correct = predicted_intent == expected_intent
            
            # Update stats
            results["total"] += 1
            results["by_intent"][expected_intent]["total"] += 1
            results["by_language"][language]["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                results["by_intent"][expected_intent]["correct"] += 1
                results["by_language"][language]["correct"] += 1
            
            # Store prediction
            results["predictions"].append({
                "query": query,
                "expected": expected_intent,
                "predicted": predicted_intent,
                "confidence": confidence,
                "language": language,
                "correct": is_correct
            })
            
            # Print result
            status = "✓" if is_correct else "✗"
            print(f"{status} [{language}] {query}")
            print(f"  → Predicted: {predicted_intent} (confidence: {confidence:.3f})")
            if not is_correct:
                print(f"  → Expected: {expected_intent}")
            print()
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    overall_accuracy = results["correct"] / results["total"]
    print(f"\nOverall Accuracy: {overall_accuracy:.1%} ({results['correct']}/{results['total']})")
    
    print("\n--- By Intent ---")
    for intent in sorted(results["by_intent"].keys()):
        stats = results["by_intent"][intent]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{intent:20s}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
    
    print("\n--- By Language ---")
    for language in sorted(results["by_language"].keys()):
        stats = results["by_language"][language]
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{language:10s}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
    
    # Show misclassifications
    print("\n--- Misclassifications ---")
    misclassified = [p for p in results["predictions"] if not p["correct"]]
    if misclassified:
        for pred in misclassified[:10]:  # Show first 10
            print(f"\nQuery: {pred['query']}")
            print(f"Expected: {pred['expected']} | Predicted: {pred['predicted']} (conf: {pred['confidence']:.3f})")
    else:
        print("None! Perfect classification!")
    
    # Save detailed results
    output_file = "models/istanbul_intent_classifier_finetuned/evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "total_queries": results["total"],
            "correct_predictions": results["correct"],
            "by_intent": dict(results["by_intent"]),
            "by_language": dict(results["by_language"]),
            "predictions": results["predictions"]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    return overall_accuracy

if __name__ == "__main__":
    try:
        accuracy = main()
        sys.exit(0 if accuracy > 0.7 else 1)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
