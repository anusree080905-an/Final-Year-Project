"""
Comprehensive Evaluation Suite for Language-Aware AI Text Detector.
Includes:
- Standard metrics (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix)
- Per-Language & Per-Family Breakdown
- Cross-AI Generalization (unseen generator evaluation)
- Cross-Language & Cross-Family Generalization
- Ablation Studies (Linguistic vs Script vs AI-Style features)
- Paraphrasing Robustness (Clean vs Paraphrase 1 vs Paraphrase 2)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.models.classifier import LanguageAwareClassifier, TARGET_LANGS
from src.features.feature_extractor import IndicFeatureExtractor


class DetectorEvaluator:
    """
    Runs multi-dimensional experimental evaluations on the benchmark using fast vectorized slicing.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.feature_extractor = IndicFeatureExtractor()

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "confusion_matrix": cm,
            "total_samples": len(y_true)
        }

    def evaluate_overall(self, classifier: LanguageAwareClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Overall performance metrics on pre-extracted feature matrix."""
        X_test_scaled = classifier.scaler.transform(X_test)
        y_pred = classifier.model.predict(X_test_scaled)
        y_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
        return self.evaluate_predictions(y_test, y_pred, y_proba)

    def evaluate_per_language(self, classifier: LanguageAwareClassifier,
                              X_test: np.ndarray, y_test: np.ndarray,
                              test_df: pd.DataFrame) -> Dict[str, Any]:
        """Per-language metrics across 8 languages using boolean indexing."""
        per_lang = {}
        for lang in TARGET_LANGS:
            mask = (test_df["lang_code"] == lang).values
            if not np.any(mask):
                continue
            X_sub = X_test[mask]
            y_sub = y_test[mask]
            per_lang[lang] = self.evaluate_overall(classifier, X_sub, y_sub)
        return per_lang

    def evaluate_cross_ai(self, X_train: np.ndarray, y_train: np.ndarray, train_df: pd.DataFrame,
                          X_test: np.ndarray, y_test: np.ndarray, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-AI generator testing.
        Trains on 2 AI generators + human, tests on unseen generator.
        """
        print("[INFO] Running Cross-AI Generator evaluation (Unseen Generator Test)...")
        results = {}
        generators = ["gpt-oss", "qwen2.5", "aya-8b"]

        for holdout_gen in generators:
            train_mask = ((train_df["label"] == 0) | (train_df["generator"] != holdout_gen)).values
            test_mask = ((test_df["label"] == 0) | (test_df["generator"] == holdout_gen)).values

            X_tr_sub = X_train[train_mask]
            y_tr_sub = y_train[train_mask]
            X_te_sub = X_test[test_mask]
            y_te_sub = y_test[test_mask]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_sub)
            X_te_scaled = scaler.transform(X_te_sub)

            clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, verbose=-1, random_state=42)
            clf.fit(X_tr_scaled, y_tr_sub)

            y_pred = clf.predict(X_te_scaled)
            y_proba = clf.predict_proba(X_te_scaled)[:, 1]

            results[f"holdout_{holdout_gen}"] = {
                "unseen_generator": holdout_gen,
                "metrics": self.evaluate_predictions(y_te_sub, y_pred, y_proba)
            }

        return results

    def evaluate_cross_language(self, X_train: np.ndarray, y_train: np.ndarray, train_df: pd.DataFrame,
                                X_test: np.ndarray, y_test: np.ndarray, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cross-Language and Cross-Family generalization.
        1. Indo-Aryan -> Dravidian
        2. Dravidian -> Indo-Aryan
        """
        print("[INFO] Running Cross-Language Family evaluation...")
        results = {}

        # 1. Indo-Aryan -> Dravidian
        tr_ia_mask = (train_df["family"] == "Indo-Aryan").values
        te_dr_mask = (test_df["family"] == "Dravidian").values

        # Only use base stylometric features (0..32) without language one-hot
        scaler_ia = StandardScaler()
        X_tr_ia = scaler_ia.fit_transform(X_train[tr_ia_mask, :33])
        X_te_dr = scaler_ia.transform(X_test[te_dr_mask, :33])

        clf_ia = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, verbose=-1, random_state=42)
        clf_ia.fit(X_tr_ia, y_train[tr_ia_mask])
        y_pred_dr = clf_ia.predict(X_te_dr)
        y_proba_dr = clf_ia.predict_proba(X_te_dr)[:, 1]
        results["IndoAryan_to_Dravidian"] = self.evaluate_predictions(y_test[te_dr_mask], y_pred_dr, y_proba_dr)

        # 2. Dravidian -> Indo-Aryan
        tr_dr_mask = (train_df["family"] == "Dravidian").values
        te_ia_mask = (test_df["family"] == "Indo-Aryan").values

        scaler_dr = StandardScaler()
        X_tr_dr = scaler_dr.fit_transform(X_train[tr_dr_mask, :33])
        X_te_ia = scaler_dr.transform(X_test[te_ia_mask, :33])

        clf_dr = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, verbose=-1, random_state=42)
        clf_dr.fit(X_tr_dr, y_train[tr_dr_mask])
        y_pred_ia = clf_dr.predict(X_te_ia)
        y_proba_ia = clf_dr.predict_proba(X_te_ia)[:, 1]
        results["Dravidian_to_IndoAryan"] = self.evaluate_predictions(y_test[te_ia_mask], y_pred_ia, y_proba_ia)

        return results

    def evaluate_ablation(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Ablation study over feature categories:
        - Full model (All 33 features + Language-aware)
        - Linguistic features only (14 features)
        - Script & Morphological features only (11 features)
        - AI-Style features only (8 features)
        - All 33 Stylometrics without Language indicators
        """
        print("[INFO] Running Feature Ablation Studies...")
        ablation_results = {}
        feat_groups = self.feature_extractor.get_feature_names()
        base_names = self.feature_extractor.get_all_feature_names()

        def run_subset(indices, name):
            scaler = StandardScaler()
            X_tr_sub = scaler.fit_transform(X_train[:, indices])
            X_te_sub = scaler.transform(X_test[:, indices])

            model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, verbose=-1, random_state=42)
            model.fit(X_tr_sub, y_train)
            y_pred = model.predict(X_te_sub)
            y_proba = model.predict_proba(X_te_sub)[:, 1]
            return self.evaluate_predictions(y_test, y_pred, y_proba)

        # 1. Full Model
        ablation_results["Full_Model"] = run_subset(list(range(X_train.shape[1])), "Full_Model")

        # 2. Linguistic Only
        ling_indices = [i for i, name in enumerate(base_names) if name in feat_groups["linguistic"]]
        ablation_results["Linguistic_Only"] = run_subset(ling_indices, "Linguistic_Only")

        # 3. Script & Morphological Only
        script_indices = [i for i, name in enumerate(base_names) if name in feat_groups["script_morphological"]]
        ablation_results["Script_Morphological_Only"] = run_subset(script_indices, "Script_Morphological_Only")

        # 4. AI-Style Only
        ai_indices = [i for i, name in enumerate(base_names) if name in feat_groups["ai_style"]]
        ablation_results["AI_Style_Only"] = run_subset(ai_indices, "AI_Style_Only")

        # 5. Stylometric Only (no language/family one-hot)
        ablation_results["Stylometric_Without_LangAware"] = run_subset(list(range(33)), "Stylometric_Without_LangAware")

        return ablation_results

    def evaluate_paraphrasing_robustness(self, classifier: LanguageAwareClassifier,
                                         X_pert: np.ndarray, y_pert: np.ndarray,
                                         pert_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates robustness across perturbation types: clean, paraphrase_1, paraphrase_2.
        """
        print("[INFO] Running Paraphrasing Robustness evaluation...")
        results = {}

        if "perturbation_type" not in pert_df.columns:
            return {"error": "perturbation_type column not found"}

        for p_type in pert_df["perturbation_type"].unique():
            mask = (pert_df["perturbation_type"] == p_type).values
            if not np.any(mask):
                continue
            X_sub = X_pert[mask]
            y_sub = y_pert[mask]
            results[str(p_type)] = self.evaluate_overall(classifier, X_sub, y_sub)

        return results
