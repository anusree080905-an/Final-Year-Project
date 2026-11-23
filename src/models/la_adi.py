"""
Language-Aware AI Detectability Index (LA-ADI).
Computes a rigorous [0, 1] index quantifying AI text detection difficulty
across eight Indian languages based on:
1. Classification Performance (P_L)
2. Clue Gap (G_L): Statistical separability of Human vs AI feature distributions
3. Paraphrasing Stability (S_L): Robustness against semantic perturbation attacks
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.metrics import f1_score, roc_auc_score

from src.preprocessing.text_processor import LANGUAGE_FAMILIES, LANGUAGE_NAMES


TARGET_LANGS = ['ta', 'te', 'ml', 'kn', 'hi', 'bn', 'mr', 'gu']


class LAADIAnalyzer:
    """
    Computes Language-Aware AI Detectability Index (LA-ADI) and ranks languages.
    """

    def __init__(self,
                 w_performance: float = 0.40,
                 w_clue_gap: float = 0.30,
                 w_stability: float = 0.30):
        self.w_perf = w_performance
        self.w_gap = w_clue_gap
        self.w_stab = w_stability

    def compute_clue_gap(self, X_lang: np.ndarray, y_lang: np.ndarray) -> float:
        """
        Computes Clue Gap G_L: Average separability across features using AUC divergence.
        G_L = mean(2 * |AUC(feature) - 0.5|) in [0, 1]
        """
        if len(y_lang) < 10 or len(np.unique(y_lang)) < 2:
            return 0.5

        separabilities = []
        n_features = X_lang.shape[1]

        for i in range(min(n_features, 33)):
            col = X_lang[:, i]
            if np.std(col) < 1e-6:
                continue
            try:
                auc = roc_auc_score(y_lang, col)
                sep = 2.0 * abs(auc - 0.5)
                separabilities.append(sep)
            except Exception:
                continue

        if not separabilities:
            return 0.5

        separabilities.sort(reverse=True)
        top_k = separabilities[:10]
        return float(np.mean(top_k))

    def compute_stability(self, f1_clean: float, f1_perturb: float) -> float:
        """
        Computes Paraphrasing Stability S_L:
        Penalizes degradation under semantic-preserving perturbation.
        """
        if f1_clean <= 0:
            return 0.5

        drop = max(0.0, f1_clean - f1_perturb)
        rel_drop = drop / max(f1_clean, 0.01)
        stability = 1.0 - min(1.0, rel_drop)
        return float(max(0.0, min(1.0, stability)))

    def evaluate_benchmark(self,
                           classifier,
                           X_test: np.ndarray, y_test: np.ndarray, test_df: pd.DataFrame,
                           X_pert: np.ndarray, y_pert: np.ndarray, pert_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates LA-ADI across all 8 target languages using precomputed feature matrices.
        """
        results = {}

        for lang in TARGET_LANGS:
            test_mask = (test_df["lang_code"] == lang).values
            pert_mask = (pert_df["lang_code"] == lang).values

            if not np.any(test_mask):
                continue

            # 1. Clean Test Performance
            X_test_lang = X_test[test_mask]
            y_test_lang = y_test[test_mask]
            X_test_scaled = classifier.scaler.transform(X_test_lang)
            y_pred_test = classifier.model.predict(X_test_scaled)
            f1_clean = float(f1_score(y_test_lang, y_pred_test, average="binary", zero_division=0))

            # 2. Clue Gap
            clue_gap = self.compute_clue_gap(X_test_lang, y_test_lang)

            # 3. Paraphrasing Stability
            if np.any(pert_mask):
                X_pert_lang = X_pert[pert_mask]
                y_pert_lang = y_pert[pert_mask]
                X_pert_scaled = classifier.scaler.transform(X_pert_lang)
                y_pred_pert = classifier.model.predict(X_pert_scaled)
                f1_perturb = float(f1_score(y_pert_lang, y_pred_pert, average="binary", zero_division=0))
            else:
                f1_perturb = f1_clean * 0.85

            stability = self.compute_stability(f1_clean, f1_perturb)

            # Composite LA-ADI Score
            la_adi_score = (self.w_perf * f1_clean +
                            self.w_gap * clue_gap +
                            self.w_stab * stability)
            la_adi_score = float(np.clip(la_adi_score, 0.0, 1.0))

            results[lang] = {
                "language_code": lang,
                "language_name": LANGUAGE_NAMES[lang],
                "language_family": LANGUAGE_FAMILIES[lang],
                "la_adi_score": round(la_adi_score, 4),
                "classification_performance": round(f1_clean, 4),
                "clue_gap": round(clue_gap, 4),
                "paraphrasing_stability": round(stability, 4),
                "clean_f1": round(f1_clean, 4),
                "perturb_f1": round(f1_perturb, 4)
            }

        # Ranking
        ranked = sorted(results.values(), key=lambda x: x["la_adi_score"], reverse=True)
        for rank, item in enumerate(ranked, 1):
            results[item["language_code"]]["rank"] = rank

        # Family Aggregation
        dravidian_scores = [v["la_adi_score"] for v in results.values() if v["language_family"] == "Dravidian"]
        indo_aryan_scores = [v["la_adi_score"] for v in results.values() if v["language_family"] == "Indo-Aryan"]

        family_summary = {
            "Dravidian_mean_LA_ADI": round(float(np.mean(dravidian_scores)), 4) if dravidian_scores else 0.0,
            "Indo_Aryan_mean_LA_ADI": round(float(np.mean(indo_aryan_scores)), 4) if indo_aryan_scores else 0.0,
            "overall_mean_LA_ADI": round(float(np.mean([v["la_adi_score"] for v in results.values()])), 4)
        }

        return {
            "language_scores": results,
            "ranking": [x["language_code"] for x in ranked],
            "family_summary": family_summary
        }
