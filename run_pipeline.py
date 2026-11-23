"""
End-to-End Training, Evaluation, LA-ADI Computation, and SHAP Atlas Pipeline.
Executes complete research workflow and saves all artifacts for Streamlit app.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.models.classifier import LanguageAwareClassifier
from src.models.la_adi import LAADIAnalyzer
from src.explainability.shap_explainer import AIFingerprintAtlas
from src.evaluation.evaluator import DetectorEvaluator


def get_or_extract_features(clf: LanguageAwareClassifier, df: pd.DataFrame, cache_path: str):
    """Loads cached feature array or extracts from dataframe and caches."""
    if os.path.exists(cache_path):
        print(f"  [CACHE] Loading features from {cache_path}...")
        data = np.load(cache_path)
        return data["X"], data["y"]
    
    print(f"  Extracting features for {len(df)} samples...")
    t0 = time.time()
    X, y = clf.extract_features_df(df)
    print(f"  Extracted in {time.time() - t0:.2f}s -> saving to {cache_path}...")
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def main():
    print("=" * 70)
    print("  LANGUAGE-AWARE AI-GENERATED TEXT DETECTION IN INDIC LANGUAGES")
    print("  Tamil, Telugu, Malayalam, Kannada, Hindi, Bengali, Marathi, Gujarati")
    print("=" * 70)

    # 1. Load Datasets
    print("\n[Step 1/6] Loading balanced IndicTextFake benchmark splits...")
    train_path = "data/processed/train_balanced.csv"
    val_path = "data/processed/val_balanced.csv"
    test_path = "data/processed/test_balanced.csv"
    pert_path = "data/processed/perturbation_test_full.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found! Run src/data/download_and_prepare.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    pert_df = pd.read_csv(pert_path)

    print(f"Loaded: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows, Perturbation={len(pert_df)} rows")

    # 2. Extract and Cache Features
    print("\n[Step 2/6] Processing feature representations...")
    clf = LanguageAwareClassifier(model_type="lightgbm")
    
    X_train, y_train = get_or_extract_features(clf, train_df, "data/processed/train_features.npz")
    X_val, y_val = get_or_extract_features(clf, val_df, "data/processed/val_features.npz")
    X_test, y_test = get_or_extract_features(clf, test_df, "data/processed/test_features.npz")
    X_pert, y_pert = get_or_extract_features(clf, pert_df, "data/processed/pert_features.npz")

    # 3. Fit Primary Classifier
    print(f"\n[Step 3/6] Fitting Language-Aware LightGBM on {len(X_train)} samples with {X_train.shape[1]} features...")
    t0 = time.time()
    clf.fit(X_train, y_train, X_val, y_val)
    print(f"Model trained in {time.time() - t0:.2f} seconds.")

    # Save model
    os.makedirs("models", exist_ok=True)
    clf.save("models/language_aware_detector.joblib")

    # 4. Experimental Evaluations
    print("\n[Step 4/6] Running Experimental Evaluation Suite...")
    evaluator = DetectorEvaluator(output_dir="results")

    print("  Evaluating overall test performance...")
    overall_metrics = evaluator.evaluate_overall(clf, X_test, y_test)
    print(f"  Test Accuracy: {overall_metrics['accuracy']:.4f}, F1: {overall_metrics['f1_score']:.4f}, AUC: {overall_metrics['auc']:.4f}")

    print("  Evaluating per-language performance...")
    per_lang_metrics = evaluator.evaluate_per_language(clf, X_test, y_test, test_df)
    for lang, m in per_lang_metrics.items():
        print(f"    {lang}: Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}, AUC={m['auc']:.4f}")

    print("  Evaluating Cross-AI generalization...")
    cross_ai_metrics = evaluator.evaluate_cross_ai(X_train, y_train, train_df, X_test, y_test, test_df)

    print("  Evaluating Cross-Language / Cross-Family generalization...")
    cross_lang_metrics = evaluator.evaluate_cross_language(X_train, y_train, train_df, X_test, y_test, test_df)

    print("  Evaluating Feature Ablations...")
    ablation_metrics = evaluator.evaluate_ablation(X_train, y_train, X_test, y_test)
    for ab_name, ab_res in ablation_metrics.items():
        print(f"    {ab_name}: Acc={ab_res['accuracy']:.4f}, F1={ab_res['f1_score']:.4f}")

    print("  Evaluating Paraphrasing Robustness...")
    paraphrase_metrics = evaluator.evaluate_paraphrasing_robustness(clf, X_pert, y_pert, pert_df)
    for pt, m in paraphrase_metrics.items():
        print(f"    Perturbation '{pt}': Acc={m['accuracy']:.4f}, F1={m['f1_score']:.4f}")

    full_eval_report = {
        "overall": overall_metrics,
        "per_language": per_lang_metrics,
        "cross_ai": cross_ai_metrics,
        "cross_language": cross_lang_metrics,
        "ablation_study": ablation_metrics,
        "paraphrasing_robustness": paraphrase_metrics
    }
    with open("results/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_eval_report, f, indent=2)

    # 5. Compute LA-ADI
    print("\n[Step 5/6] Computing Language-Aware AI Detectability Index (LA-ADI)...")
    la_adi_analyzer = LAADIAnalyzer()
    la_adi_results = la_adi_analyzer.evaluate_benchmark(clf, X_test, y_test, test_df, X_pert, y_pert, pert_df)

    with open("results/la_adi_results.json", "w", encoding="utf-8") as f:
        json.dump(la_adi_results, f, indent=2)

    print("  LA-ADI Rankings (Highest Detectability -> Lowest Detectability):")
    for lang_code in la_adi_results["ranking"]:
        score_info = la_adi_results["language_scores"][lang_code]
        print(f"    Rank {score_info['rank']}: {score_info['language_name']} ({score_info['language_family']}) - LA-ADI: {score_info['la_adi_score']:.4f} (Perf={score_info['classification_performance']:.4f}, Gap={score_info['clue_gap']:.4f}, Stab={score_info['paraphrasing_stability']:.4f})")

    # 6. Compute AI Fingerprint Atlas using SHAP
    print("\n[Step 6/6] Building AI Fingerprint Atlas using SHAP...")
    atlas_builder = AIFingerprintAtlas(clf)
    atlas_data = atlas_builder.compute_atlas(X_test, y_test, test_df, max_samples_per_lang=60, output_dir="results/shap")
    with open("results/ai_fingerprint_atlas.json", "w", encoding="utf-8") as f:
        json.dump(atlas_data, f, indent=2)

    # 7. Summary Charts
    try:
        os.makedirs("results", exist_ok=True)
        # Per-Language F1 Bar Chart
        plt.figure(figsize=(9, 5))
        langs = list(per_lang_metrics.keys())
        f1s = [per_lang_metrics[l]["f1_score"] for l in langs]
        colors = ["#2563EB" if l in ['ta', 'te', 'ml', 'kn'] else "#059669" for l in langs]
        bars = plt.bar(langs, f1s, color=colors)
        plt.title("Detection F1-Score Across Indic Languages (Blue=Dravidian, Green=Indo-Aryan)")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.05)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig("results/per_language_f1.png", dpi=200)
        plt.close()

        # LA-ADI Comparison Chart
        plt.figure(figsize=(9, 5))
        ranked_langs = la_adi_results["ranking"]
        adi_scores = [la_adi_results["language_scores"][l]["la_adi_score"] for l in ranked_langs]
        colors = ["#7C3AED" if l in ['ta', 'te', 'ml', 'kn'] else "#D97706" for l in ranked_langs]
        bars = plt.bar(ranked_langs, adi_scores, color=colors)
        plt.title("Language-Aware AI Detectability Index (LA-ADI) [0=Hardest, 1=Easiest]")
        plt.ylabel("LA-ADI Score")
        plt.ylim(0, 1.05)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig("results/la_adi_rankings.png", dpi=200)
        plt.close()
        print("Saved summary plots: results/per_language_f1.png, results/la_adi_rankings.png")
    except Exception as e:
        print(f"[WARN] Error saving plots: {e}")

    print("\n" + "=" * 70)
    print("  ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
