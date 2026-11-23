"""
Language-Aware AI-Generated Text Classification Model.
Integrates 3-category stylometric features with language-family awareness.
Supports LightGBM, Random Forest, Logistic Regression, MLP, and Stacking Ensemble.
Outputs calibrated confidence scores and probabilities.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import lightgbm as lgb

from src.features.feature_extractor import IndicFeatureExtractor
from src.preprocessing.text_processor import IndicTextPreprocessor, LANGUAGE_FAMILIES, LANGUAGE_NAMES


TARGET_LANGS = ['ta', 'te', 'ml', 'kn', 'hi', 'bn', 'mr', 'gu']


class LanguageAwareClassifier:
    """
    Language-Aware AI-Generated Text Detector.
    Fuses multi-category stylometric features with language and family routing.
    """

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.feature_extractor = IndicFeatureExtractor()
        self.preprocessor = IndicTextPreprocessor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.is_fitted = False

    def _build_feature_names(self) -> List[str]:
        base_features = self.feature_extractor.get_all_feature_names()
        # Language one-hot features
        lang_features = [f"lang_{l}" for l in TARGET_LANGS]
        # Family features
        family_features = ["family_dravidian", "family_indo_aryan"]
        return base_features + lang_features + family_features

    def extract_features_df(self, df: pd.DataFrame, is_train: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a DataFrame containing 'text', 'lang_code', and optionally 'label'.
        """
        rows = []
        labels = []
        base_feat_names = self.feature_extractor.get_all_feature_names()

        for idx, row in df.iterrows():
            text = str(row["text"])
            lang = row.get("lang_code")
            prep = self.preprocessor.preprocess(text)
            
            # Use detected language if not provided
            if not lang or lang not in TARGET_LANGS:
                lang = prep["language_code"]

            family = LANGUAGE_FAMILIES.get(lang, prep["language_family"])
            base_feats = self.feature_extractor.extract_features(text, prep)
            feat_vec = [base_feats[k] for k in base_feat_names]

            # Language one-hot
            lang_vec = [1.0 if lang == l else 0.0 for l in TARGET_LANGS]
            # Family one-hot
            fam_vec = [
                1.0 if family == "Dravidian" else 0.0,
                1.0 if family == "Indo-Aryan" else 0.0
            ]

            full_vec = feat_vec + lang_vec + fam_vec
            rows.append(full_vec)

            if "label" in row:
                labels.append(int(row["label"]))

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int64) if labels else np.array([])
        return X, y

    def extract_single(self, text: str, lang: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features for single text sample for inference.
        """
        prep = self.preprocessor.preprocess(text)
        if not lang or lang not in TARGET_LANGS:
            lang = prep["language_code"]

        family = LANGUAGE_FAMILIES.get(lang, prep["language_family"])
        base_feat_names = self.feature_extractor.get_all_feature_names()
        base_feats = self.feature_extractor.extract_features(text, prep)
        feat_vec = [base_feats[k] for k in base_feat_names]

        lang_vec = [1.0 if lang == l else 0.0 for l in TARGET_LANGS]
        fam_vec = [
            1.0 if family == "Dravidian" else 0.0,
            1.0 if family == "Indo-Aryan" else 0.0
        ]

        full_vec = np.array([feat_vec + lang_vec + fam_vec], dtype=np.float32)
        scaled_vec = self.scaler.transform(full_vec)
        
        metadata = {
            "preprocessed": prep,
            "detected_lang": lang,
            "detected_lang_name": LANGUAGE_NAMES.get(lang, "Unknown"),
            "detected_family": family,
            "base_features": base_feats
        }
        return scaled_vec, metadata

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "LanguageAwareClassifier":
        """
        Fit scaler and classifier.
        """
        self.feature_names = self._build_feature_names()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled, y_val = None, None

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            )
            if X_val_scaled is not None and y_val is not None:
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "lr":
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.random_state
            )
            self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                activation="relu",
                random_state=self.random_state,
                early_stopping=True
            )
            self.model.fit(X_train_scaled, y_train)

        elif self.model_type == "ensemble":
            clf1 = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, verbose=-1, random_state=self.random_state)
            clf2 = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=self.random_state, n_jobs=-1)
            clf3 = LogisticRegression(max_iter=1000, C=1.0, random_state=self.random_state)
            self.model = VotingClassifier(
                estimators=[('lgb', clf1), ('rf', clf2), ('lr', clf3)],
                voting='soft'
            )
            self.model.fit(X_train_scaled, y_train)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet!")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet!")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        acc = accuracy_score(y, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
        auc = roc_auc_score(y, y_proba)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        }

    def predict_text(self, text: str, lang: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies single text as Human-Written or AI-Generated with confidence.
        """
        scaled_vec, meta = self.extract_single(text, lang)
        proba = self.model.predict_proba(scaled_vec)[0]
        ai_prob = float(proba[1])
        human_prob = float(proba[0])

        prediction_label = 1 if ai_prob >= 0.5 else 0
        prediction_text = "AI-Generated" if prediction_label == 1 else "Human-Written"
        confidence = ai_prob if prediction_label == 1 else human_prob

        return {
            "text": text,
            "prediction": prediction_label,
            "prediction_label": prediction_text,
            "confidence": confidence,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "language_code": meta["detected_lang"],
            "language_name": meta["detected_lang_name"],
            "language_family": meta["detected_family"],
            "features": meta["base_features"],
            "preprocessed": meta["preprocessed"]
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, filepath)
        print(f"[INFO] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "LanguageAwareClassifier":
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = data["is_fitted"]
        print(f"[INFO] Model loaded from {filepath}")
        return instance
