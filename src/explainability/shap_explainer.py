"""
AI Fingerprint Atlas using SHAP.
Identifies common and language-specific AI writing patterns across Indian languages
and provides instance-level local explanations.
"""

import os
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from src.preprocessing.text_processor import LANGUAGE_FAMILIES, LANGUAGE_NAMES


TARGET_LANGS = ['ta', 'te', 'ml', 'kn', 'hi', 'bn', 'mr', 'gu']


class AIFingerprintAtlas:
    """
    Computes global, family-level, and language-specific SHAP feature attributions
    to establish an AI Fingerprint Atlas for Indic text.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.feature_names = classifier.feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize tree explainer if tree-based, otherwise fallback to exact/kernel."""
        model = self.classifier.model
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model.predict_proba)

    def compute_atlas(self,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      test_df: pd.DataFrame,
                      max_samples_per_lang: int = 60,
                      output_dir: str = "results/shap") -> Dict[str, Any]:
        """
        Computes SHAP attributions using precomputed feature matrices.
        """
        os.makedirs(output_dir, exist_ok=True)
        atlas_data = {
            "global_fingerprints": {},
            "family_fingerprints": {},
            "language_fingerprints": {}
        }

        # Subsample test set indices per language
        selected_indices = []
        for lang in TARGET_LANGS:
            lang_idx = np.where((test_df["lang_code"] == lang).values)[0]
            if len(lang_idx) > 0:
                selected_indices.extend(lang_idx[:max_samples_per_lang])

        selected_indices = np.array(selected_indices)
        X_sub = X_test[selected_indices]
        df_sub = test_df.iloc[selected_indices].reset_index(drop=True)
        X_sub_scaled = self.classifier.scaler.transform(X_sub)

        print(f"[INFO] Computing SHAP values on {len(X_sub)} samples...")
        shap_values = self.explainer.shap_values(X_sub_scaled)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # 1. Global AI Fingerprints
        mean_abs_shap = np.mean(np.abs(sv), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        top_global = [
            {
                "feature": self.feature_names[i],
                "importance": round(float(mean_abs_shap[i]), 5),
                "direction": "positive_ai" if np.mean(sv[:, i]) > 0 else "human_leaning"
            }
            for i in sorted_indices[:15]
        ]
        atlas_data["global_fingerprints"] = top_global

        # 2. Family-Specific Fingerprints
        for fam in ["Dravidian", "Indo-Aryan"]:
            fam_mask = (df_sub["family"] == fam).values
            if np.any(fam_mask):
                fam_sv = sv[fam_mask]
                fam_mean_abs = np.mean(np.abs(fam_sv), axis=0)
                fam_sorted = np.argsort(fam_mean_abs)[::-1]
                atlas_data["family_fingerprints"][fam] = [
                    {
                        "feature": self.feature_names[i],
                        "importance": round(float(fam_mean_abs[i]), 5)
                    }
                    for i in fam_sorted[:10]
                ]

        # 3. Language-Specific Fingerprints
        for lang in TARGET_LANGS:
            lang_mask = (df_sub["lang_code"] == lang).values
            if np.any(lang_mask):
                lang_sv = sv[lang_mask]
                lang_mean_abs = np.mean(np.abs(lang_sv), axis=0)
                lang_sorted = np.argsort(lang_mean_abs)[::-1]
                atlas_data["language_fingerprints"][lang] = [
                    {
                        "feature": self.feature_names[i],
                        "importance": round(float(lang_mean_abs[i]), 5)
                    }
                    for i in lang_sorted[:8]
                ]

        with open(os.path.join(output_dir, "ai_fingerprint_atlas.json"), "w", encoding="utf-8") as f:
            json.dump(atlas_data, f, indent=2)

        try:
            plt.figure(figsize=(10, 6))
            feats = [x["feature"] for x in top_global[:10]][::-1]
            vals = [x["importance"] for x in top_global[:10]][::-1]
            plt.barh(feats, vals, color="#4F46E5")
            plt.title("Top Universal Indic AI Writing Fingerprints (SHAP Global)")
            plt.xlabel("Mean |SHAP Value| (Impact on AI Detection)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "global_ai_fingerprints.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not save SHAP plot: {e}")

        print(f"[INFO] AI Fingerprint Atlas computed and saved to {output_dir}")
        return atlas_data

    def explain_sample(self, text: str, lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain single prediction at test/runtime.
        """
        scaled_vec, meta = self.classifier.extract_single(text, lang)
        pred_dict = self.classifier.predict_text(text, lang)

        sv = self.explainer.shap_values(scaled_vec)
        if isinstance(sv, list) and len(sv) == 2:
            sv_sample = sv[1][0]
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            sv_sample = sv[0, :, 1]
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 2:
            sv_sample = sv[0]
        else:
            sv_sample = sv

        feature_contributions = []
        for i, val in enumerate(sv_sample):
            name = self.feature_names[i]
            feature_contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "impact": "Supports AI" if val > 0 else "Supports Human"
            })

        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "prediction": pred_dict,
            "top_features": feature_contributions[:10],
            "ai_supporting": [f for f in feature_contributions if f["shap_value"] > 0][:5],
            "human_supporting": [f for f in feature_contributions if f["shap_value"] < 0][:5]
        }
