import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.models.classifier import LanguageAwareClassifier
from app.streamlit_app import BENCHMARK_SAMPLES

clf = LanguageAwareClassifier.load('models/language_aware_detector.joblib')

print("\n" + "="*80)
print("  EVALUATION OF REALISTIC MULTI-SENTENCE BENCHMARK SAMPLES")
print("="*80)

correct = 0
total = 0

for name, text in BENCHMARK_SAMPLES.items():
    res = clf.predict_text(text)
    expected = 0 if "Human" in name else 1
    predicted = res["prediction"]
    is_correct = (expected == predicted)
    if is_correct:
        correct += 1
    total += 1

    status_icon = "CORRECT" if is_correct else "MISMATCH"
    print(f"[{name}] -> {status_icon}")
    print(f"  Expected: {'Human' if expected == 0 else 'AI'} | Predicted: {res['prediction_label']} ({res['confidence']*100:.1f}%)")
    print(f"  Detected Lang: {res['language_name']} ({res['language_family']}) | Script: {res['preprocessed']['script']}")
    print(f"  Words: {res['preprocessed']['word_count']} | Sentences: {res['preprocessed']['sentence_count']}")
    print()

print(f"Benchmark Presets Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
