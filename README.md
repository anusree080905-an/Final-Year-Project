# Language-Aware AI-Generated Text Detection Benchmark & Atlas in Indic Languages

A comprehensive, reproducible research prototype and interactive benchmark for detecting AI-generated text across eight major Indian languages from both the **Dravidian** and **Indo-Aryan** language families:
- **Dravidian:** Tamil (`ta`), Telugu (`te`), Malayalam (`ml`), Kannada (`kn`)
- **Indo-Aryan:** Hindi (`hi`), Bengali (`bn`), Marathi (`mr`), Gujarati (`gu`)

Benchmarked using the **IndicTextFake** dataset (Hugging Face) with human-written text (IndicCorpV2) and multi-generator synthetic text (**Cohere Aya-Expanse-8B**, **Qwen/Qwen2.5-7B**, and **GPT-OSS**), evaluated under semantic-preserving perturbations (back-translation and pivot-language paraphrasing).

---

## 🏛️ System Architecture

```text
IndicAI-Detect/
│
├── data/
│   ├── raw/                       # Downloaded splits from IndicTextFake (Hugging Face)
│   │   ├── main_train.csv
│   │   ├── main_val.csv
│   │   ├── main_test.csv
│   │   └── perturbation_test.csv
│   │
│   └── processed/                 # Balanced 8-language benchmark splits & cached feature arrays
│       ├── train_balanced.csv     # 16,000 balanced rows (2,000 / language)
│       ├── val_balanced.csv       # 2,400 balanced rows (300 / language)
│       ├── test_balanced.csv      # 3,198 balanced rows (~400 / language)
│       ├── perturbation_test_full.csv # 8,634 perturbation samples
│       └── dataset_summary.json
│
├── src/
│   ├── preprocessing/
│   │   └── text_processor.py      # Unicode NFC normalization, script detector & language routing
│   │
│   ├── features/
│   │   └── feature_extractor.py   # 33-dimensional feature engine across 3 categories
│   │
│   ├── data/
│   │   └── download_and_prepare.py# Benchmark downloader, language filter & generator balancer
│   │
│   ├── models/
│   │   ├── classifier.py          # Language-Aware Classifier (LightGBM / RF / LR / Ensemble)
│   │   └── la_adi.py              # Language-Aware AI Detectability Index (LA-ADI) engine
│   │
│   ├── explainability/
│   │   └── shap_explainer.py      # AI Fingerprint Atlas & instance-level SHAP attributions
│   │
│   └── evaluation/
│       └── evaluator.py           # Evaluation suite (Cross-AI, Cross-Language, Ablation, Robustness)
│
├── models/
│   └── language_aware_detector.joblib # Serialized trained model & scaler
│
├── results/                       # Empirical experimental results, JSON reports & figures
│   ├── evaluation_report.json
│   ├── la_adi_results.json
│   ├── ai_fingerprint_atlas.json
│   ├── per_language_f1.png
│   ├── la_adi_rankings.png
│   └── shap/
│
├── app/
│   └── streamlit_app.py           # Interactive multi-view Streamlit web application
│
├── run_pipeline.py                # End-to-end training and evaluation script
├── requirements.txt
└── README.md
```

---

## 🔬 Core Methodological Components

### 1. Automatic Script & Language Routing
- **Unicode NFC Normalization:** Standardizes combined characters and decomposes anomalies.
- **Zero-Width Normalization:** Safely cleans zero-width spaces (ZWSP) while preserving critical morphosyntactic joiners (ZWJ/ZWNJ).
- **Script Identification:** Unicode range mapping across 7 Indic scripts: Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam.
- **Devanagari Disambiguation:** Distinguishes Hindi (`hi`) from Marathi (`mr`) using lexical markers and language-exclusive characters (e.g. Marathi \u0933 / ळ).
- **Language Family Routing:** Routes texts to **Dravidian** or **Indo-Aryan** models and priors.

### 2. Three-Category Feature Taxonomy (33 Dimensions)

| Category | Features | Description |
|---|---|---|
| **Linguistic** (14) | `lexical_ttr`, `root_ttr`, `yules_k`, `hapax_ratio`, `avg_word_length`, `word_length_std`, `mean_sentence_length`, `sentence_length_std`, `max_sentence_length`, `punctuation_density`, `comma_danda_density`, `word_repetition_rate`, `bigram_repetition_rate`, `trigram_repetition_rate` | Lexical diversity, vocabulary richness (Yule's K), sentence length statistics, syntactic punctuation, and n-gram repetitions. |
| **Script & Morphological** (11) | `matra_density`, `independent_vowel_ratio`, `virama_halant_frequency`, `consonant_ratio`, `vowel_ratio`, `consonant_vowel_balance`, `indic_char_ratio`, `digit_ratio`, `prefix_marker_frequency`, `suffix_inflection_frequency`, `agglutination_index` | Matras (dependent vowel signs), virama/halant consonant clusters, independent vowel ratios, agglutinative suffix inflection markers. |
| **AI-Style & Uniformity** (8) | `burstiness_cv`, `burstiness_metric`, `char_entropy`, `word_entropy`, `bigram_entropy`, `predictability_proxy`, `stylistic_uniformity_variance`, `lexical_uniformity_jaccard` | Sentence burstiness ($\sigma / \mu$), Shannon entropy at character/word/bigram levels, token log-likelihood predictability proxy, quarter-partition stylistic uniformity, and half-document lexical Jaccard drift. |

### 3. Language-Aware Classification
- Fuses the 33 stylometric features with language one-hot indicators and language-family priors.
- Utilizes gradient boosted decision trees (`LightGBM`) for fast, non-linear feature interaction modeling and tree SHAP interpretability.
- Outputs calibrated confidence scores for binary decision: **Human-Written (0)** vs **AI-Generated (1)**.

### 4. Language-Aware AI Detectability Index (LA-ADI)
Quantifies detection difficulty for each language on a normalized $[0, 1]$ scale:
$$\text{LA-ADI}(L) = 0.40 \cdot P_L + 0.30 \cdot G_L + 0.30 \cdot S_L$$
- **$P_L$ (Performance):** Macro F1-score on the clean benchmark test split.
- **$G_L$ (Clue Gap):** Statistical separability of Human vs AI distributions:
  $$G_L = \frac{1}{|K|} \sum_{k \in K} 2 \cdot |\text{AUC}(f_k, L) - 0.5|$$
- **$S_L$ (Paraphrasing Stability):** Retention of detection accuracy under semantic back-translation perturbations (`paraphrase_1` and `paraphrase_2`):
  $$S_L = 1.0 - \min\left(1.0, \frac{\max(0, \text{F1}_{\text{clean}} - \text{F1}_{\text{perturb}})}{\text{F1}_{\text{clean}}}\right)$$

### 5. AI Fingerprint Atlas (SHAP)
- **Global Indic AI Fingerprints:** Universal indicators across all 8 Indic languages (e.g., lower sentence burstiness, vocabulary compression, matra regularity).
- **Family-Specific Fingerprints:** Divergence in agglutinative morphology in Dravidian vs inflectional compound patterns in Indo-Aryan.
- **Local Explanations:** Real-time waterfall and bar chart showing feature contributions pushing towards AI (+) or Human (-).

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows / Linux / macOS

### Installation
```bash
pip install -r requirements.txt
```

### 1. Download & Prepare Benchmark Data
```bash
python src/data/download_and_prepare.py
```
Downloads the IndicTextFake benchmark splits, filters to the 8 target Indic languages, balances Human vs AI generators, and saves processed CSVs in `data/processed/`.

### 2. Run End-to-End Training & Evaluation Pipeline
```bash
python run_pipeline.py
```
Extracts features, trains the `LanguageAwareClassifier`, runs the full evaluation suite (Cross-AI, Cross-Language, Ablation, Perturbation), calculates LA-ADI, builds the SHAP AI Fingerprint Atlas, and saves all reports to `results/`.

### 3. Launch Interactive Streamlit Web Application
Run using Python module mode:
```bash
py -m streamlit run app/streamlit_app.py
```
Or run using the convenient launcher script:
```powershell
.\run_app.bat
# or in PowerShell:
.\run_app.ps1
```
Or if Python Scripts directory is in your PATH:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to use the interactive application.

---

## 📊 Experimental Evaluation Protocols

1. **Overall & Per-Language Performance:** Accuracy, Precision, Recall, F1, AUC, Confusion Matrix across all 8 languages.
2. **Cross-AI Generalization:** Evaluated on unseen LLM generators (Leave-One-Generator-Out: Aya-8B, Qwen2.5, GPT-OSS).
3. **Cross-Language Family Generalization:** Zero-shot transfer from Indo-Aryan to Dravidian and Dravidian to Indo-Aryan.
4. **Feature Ablations:**
   - Full Model (33 features + Language-aware)
   - Linguistic features only
   - Script & Morphological features only
   - AI-Style features only
   - Stylometric features without language indicators
5. **Paraphrasing Robustness:** Evaluation under back-translation (`paraphrase_1`) and pivot Hindi back-translation (`paraphrase_2`).
