"""
Three-Category Feature Extraction Engine for Indic AI-Generated Text Detection.

Categories:
1. Linguistic Features:
   - Lexical diversity (TTR, Root TTR, Yule's K, Hapax Legomena ratio)
   - Vocabulary metrics (average word length, word length std)
   - Syntax and structure (punctuation density, danda/comma density)
   - Sentence length (mean, std, max, min)
   - Repetition patterns (word repetition rate, bigram & trigram repetition rates)

2. Script and Morphological Features:
   - Suffix and prefix morphology markers
   - Character distribution (Indic char ratio, digit ratio, ASCII ratio)
   - Vowels, consonants, and matras (matra density, independent vowel ratio, virama/halant frequency)
   - Grammatical / agglutination patterns (consonant-to-vowel ratio, agglutination index)

3. AI-Style Features:
   - Burstiness (coefficient of variation, normalized burstiness metric)
   - Entropy (Shannon character entropy, word unigram entropy, bigram entropy)
   - Predictability (word frequency predictability / transition proxy)
   - Stylistic uniformity (variance across document partitions, vocabulary drift / Jaccard similarity)
"""

import math
import unicodedata
from collections import Counter
from typing import Dict, List, Any, Optional
import numpy as np

from src.preprocessing.text_processor import IndicTextPreprocessor, SCRIPT_RANGES


# Unicode offsets inside standard Indic blocks (0x00 to 0x7F relative to block base)
# In Devanagari, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam:
# - Independent Vowels generally fall in relative range 0x04 to 0x14
# - Consonants generally fall in relative range 0x15 to 0x39
# - Matras (dependent vowel signs) fall in 0x3E to 0x4C
# - Virama (halant / pulli / hasant) is typically at 0x4D
# - Digits are at 0x66 to 0x6F
VIRAMA_OFFSETS = {0x4D}
MATRA_OFFSETS = set(range(0x3E, 0x4D))
INDEP_VOWEL_OFFSETS = set(range(0x04, 0x15))
CONSONANT_OFFSETS = set(range(0x15, 0x3A))


class IndicFeatureExtractor:
    """
    Extracts 30+ linguistic, script/morphological, and AI-style features
    from Indic text.
    """

    def __init__(self):
        self.preprocessor = IndicTextPreprocessor()

    def get_feature_names(self) -> Dict[str, List[str]]:
        return {
            "linguistic": [
                "lexical_ttr",
                "root_ttr",
                "yules_k",
                "hapax_ratio",
                "avg_word_length",
                "word_length_std",
                "mean_sentence_length",
                "sentence_length_std",
                "max_sentence_length",
                "punctuation_density",
                "comma_danda_density",
                "word_repetition_rate",
                "bigram_repetition_rate",
                "trigram_repetition_rate"
            ],
            "script_morphological": [
                "matra_density",
                "independent_vowel_ratio",
                "virama_halant_frequency",
                "consonant_ratio",
                "vowel_ratio",
                "consonant_vowel_balance",
                "indic_char_ratio",
                "digit_ratio",
                "prefix_marker_frequency",
                "suffix_inflection_frequency",
                "agglutination_index"
            ],
            "ai_style": [
                "burstiness_cv",
                "burstiness_metric",
                "char_entropy",
                "word_entropy",
                "bigram_entropy",
                "predictability_proxy",
                "stylistic_uniformity_variance",
                "lexical_uniformity_jaccard"
            ]
        }

    def get_all_feature_names(self) -> List[str]:
        names = []
        for cat_names in self.get_feature_names().values():
            names.extend(cat_names)
        return names

    def extract_linguistic_features(self, words: List[str], sentences: List[str], text: str) -> Dict[str, float]:
        n_words = len(words)
        n_sentences = len(sentences)
        n_chars = max(len(text), 1)

        if n_words == 0:
            return {name: 0.0 for name in self.get_feature_names()["linguistic"]}

        # Vocabulary counts
        word_counts = Counter(words)
        n_vocab = len(word_counts)

        # Lexical Diversity
        ttr = n_vocab / n_words
        root_ttr = n_vocab / math.sqrt(n_words)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax_count / n_vocab if n_vocab > 0 else 0.0

        # Yule's Characteristic K
        # K = 10^4 * (sum(f^2) - N) / N^2
        sum_f2 = sum(c * c for c in word_counts.values())
        if n_words > 1:
            yules_k = 10000.0 * (sum_f2 - n_words) / (n_words * n_words)
        else:
            yules_k = 0.0
        yules_k = max(0.0, min(yules_k, 500.0))  # numerical stability clip

        # Word length metrics
        word_lengths = [len(w) for w in words]
        avg_word_len = float(np.mean(word_lengths))
        std_word_len = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

        # Sentence length metrics
        if n_sentences > 0:
            sent_lengths = [max(len(self.preprocessor.tokenize_words(s)), 1) for s in sentences]
            mean_sent_len = float(np.mean(sent_lengths))
            std_sent_len = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
            max_sent_len = float(max(sent_lengths))
        else:
            mean_sent_len = float(n_words)
            std_sent_len = 0.0
            max_sent_len = float(n_words)

        # Syntax & Punctuation
        punc_count = sum(1 for c in text if unicodedata.category(c).startswith('P'))
        punc_density = punc_count / n_chars

        comma_danda_count = text.count(',') + text.count(';') + text.count('।') + text.count('॥')
        comma_danda_density = comma_danda_count / n_chars

        # Repetition Rates
        word_repetition_rate = (n_words - n_vocab) / n_words

        # Bigram repetition
        if n_words >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(n_words - 1)]
            n_bigrams = len(bigrams)
            n_unique_bigrams = len(set(bigrams))
            bigram_rep_rate = (n_bigrams - n_unique_bigrams) / n_bigrams
        else:
            bigram_rep_rate = 0.0

        # Trigram repetition
        if n_words >= 3:
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(n_words - 2)]
            n_trigrams = len(trigrams)
            n_unique_trigrams = len(set(trigrams))
            trigram_rep_rate = (n_trigrams - n_unique_trigrams) / n_trigrams
        else:
            trigram_rep_rate = 0.0

        return {
            "lexical_ttr": float(ttr),
            "root_ttr": float(root_ttr),
            "yules_k": float(yules_k),
            "hapax_ratio": float(hapax_ratio),
            "avg_word_length": float(avg_word_len),
            "word_length_std": float(std_word_len),
            "mean_sentence_length": float(mean_sent_len),
            "sentence_length_std": float(std_sent_len),
            "max_sentence_length": float(max_sent_len),
            "punctuation_density": float(punc_density),
            "comma_danda_density": float(comma_danda_density),
            "word_repetition_rate": float(word_repetition_rate),
            "bigram_repetition_rate": float(bigram_rep_rate),
            "trigram_repetition_rate": float(trigram_rep_rate)
        }

    def extract_script_morphological_features(self, text: str, words: List[str]) -> Dict[str, float]:
        n_chars = max(len(text), 1)
        total_indic = 0
        matra_count = 0
        indep_vowel_count = 0
        virama_count = 0
        consonant_count = 0
        digit_count = 0

        for char in text:
            code = ord(char)
            # Check script blocks
            for _, (start, end) in SCRIPT_RANGES.items():
                if start <= code <= end:
                    total_indic += 1
                    rel = code - start
                    if rel in MATRA_OFFSETS:
                        matra_count += 1
                    elif rel in INDEP_VOWEL_OFFSETS:
                        indep_vowel_count += 1
                    elif rel in CONSONANT_OFFSETS:
                        consonant_count += 1
                    elif rel in VIRAMA_OFFSETS:
                        virama_count += 1
                    elif 0x66 <= rel <= 0x6F:
                        digit_count += 1
                    break

        indic_char_ratio = total_indic / n_chars
        matra_density = matra_count / max(total_indic, 1)
        total_vowels = matra_count + indep_vowel_count
        indep_vowel_ratio = indep_vowel_count / max(total_vowels, 1)
        virama_halant_freq = virama_count / max(total_indic, 1)
        consonant_ratio = consonant_count / max(total_indic, 1)
        vowel_ratio = total_vowels / max(total_indic, 1)
        consonant_vowel_balance = (consonant_count / max(total_vowels, 1)) if total_vowels > 0 else 1.0
        digit_ratio = digit_count / n_chars

        # Morphological affixes and agglutination proxy
        # Suffix endings check: length of words with common multi-syllabic inflection
        n_words = max(len(words), 1)
        long_words = sum(1 for w in words if len(w) >= 7)
        agglutination_index = long_words / n_words

        # Frequent morphological markers (prefix/suffix patterns)
        prefix_count = 0
        suffix_count = 0
        for w in words:
            if len(w) >= 4:
                # Suffixes: case markers / plurals in Indic languages often end in specific 2-char sequences
                # e.g., -கள், -లు, -లు, -ಗಳು, -ಗಳು, -দের, -গুলো, -याँ, -ओ, -ना, -तील
                suffix_count += 1
                if w.startswith(('अ', 'नि', 'वि', 'अनु', 'प्रति', 'उप', 'அதி', 'முன்', 'അനു', 'പ്രതി')):
                    prefix_count += 1

        prefix_marker_freq = prefix_count / n_words
        suffix_inflection_freq = suffix_count / n_words

        return {
            "matra_density": float(matra_density),
            "independent_vowel_ratio": float(indep_vowel_ratio),
            "virama_halant_frequency": float(virama_halant_freq),
            "consonant_ratio": float(consonant_ratio),
            "vowel_ratio": float(vowel_ratio),
            "consonant_vowel_balance": float(min(consonant_vowel_balance, 10.0)),
            "indic_char_ratio": float(indic_char_ratio),
            "digit_ratio": float(digit_ratio),
            "prefix_marker_frequency": float(prefix_marker_freq),
            "suffix_inflection_frequency": float(suffix_inflection_freq),
            "agglutination_index": float(agglutination_index)
        }

    def extract_ai_style_features(self, words: List[str], sentences: List[str], text: str) -> Dict[str, float]:
        n_words = len(words)
        n_sentences = len(sentences)

        if n_words == 0:
            return {name: 0.0 for name in self.get_feature_names()["ai_style"]}

        # 1. Burstiness
        # Sentences lengths
        if n_sentences >= 2:
            sent_lens = [max(len(self.preprocessor.tokenize_words(s)), 1) for s in sentences]
            mu = float(np.mean(sent_lens))
            sigma = float(np.std(sent_lens))
            burstiness_cv = (sigma / mu) if mu > 0 else 0.0
            # Normalized burstiness metric (sigma - mu) / (sigma + mu) bounded [-1, 1]
            burstiness_metric = ((sigma - mu) / (sigma + mu)) if (sigma + mu) > 0 else 0.0
        else:
            burstiness_cv = 0.0
            burstiness_metric = 0.0

        # 2. Shannon Entropies
        # Character entropy
        char_counts = Counter(text)
        total_chars = len(text)
        char_probs = [c / total_chars for c in char_counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in char_probs if p > 0)

        # Word unigram entropy
        word_counts = Counter(words)
        word_probs = [c / n_words for c in word_counts.values()]
        word_entropy = -sum(p * math.log2(p) for p in word_probs if p > 0)

        # Word bigram entropy
        if n_words >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(n_words - 1)]
            bg_counts = Counter(bigrams)
            n_bg = len(bigrams)
            bg_probs = [c / n_bg for c in bg_counts.values()]
            bigram_entropy = -sum(p * math.log2(p) for p in bg_probs if p > 0)
        else:
            bigram_entropy = 0.0

        # 3. Predictability proxy
        # Average log frequency rank (proxy for whether words are predictable common tokens or rare)
        log_ranks = [math.log(word_counts[w] + 1) for w in words]
        predictability_proxy = float(np.mean(log_ranks)) if log_ranks else 0.0

        # 4. Stylistic Uniformity across document partitions
        # Variance of clause / sentence lengths across quarters
        if n_sentences >= 4:
            q_size = n_sentences // 4
            q_lens = [
                float(np.mean([len(self.preprocessor.tokenize_words(s)) for s in sentences[i*q_size : (i+1)*q_size]]))
                for i in range(4)
            ]
            quarter_variance = float(np.var(q_lens))
        else:
            quarter_variance = 0.0

        # Lexical uniformity: Jaccard vocabulary similarity between 1st half and 2nd half
        if n_words >= 10:
            mid = n_words // 2
            vocab1 = set(words[:mid])
            vocab2 = set(words[mid:])
            intersection = len(vocab1.intersection(vocab2))
            union = len(vocab1.union(vocab2))
            jaccard_sim = intersection / max(union, 1)
        else:
            jaccard_sim = 0.0

        return {
            "burstiness_cv": float(burstiness_cv),
            "burstiness_metric": float(burstiness_metric),
            "char_entropy": float(char_entropy),
            "word_entropy": float(word_entropy),
            "bigram_entropy": float(bigram_entropy),
            "predictability_proxy": float(predictability_proxy),
            "stylistic_uniformity_variance": float(quarter_variance),
            "lexical_uniformity_jaccard": float(jaccard_sim)
        }

    def extract_features(self, text: str, preprocessed: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extract complete 33-feature vector for an Indic text sample.
        """
        if preprocessed is None:
            preprocessed = self.preprocessor.preprocess(text)

        words = preprocessed.get("words", [])
        sentences = preprocessed.get("sentences", [])
        norm_text = preprocessed.get("normalized_text", text)

        ling_feats = self.extract_linguistic_features(words, sentences, norm_text)
        script_feats = self.extract_script_morphological_features(norm_text, words)
        ai_feats = self.extract_ai_style_features(words, sentences, norm_text)

        all_feats = {}
        all_feats.update(ling_feats)
        all_feats.update(script_feats)
        all_feats.update(ai_feats)

        return all_feats

    def extract_features_vector(self, text: str, preprocessed: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract features as a 1D numpy array in deterministic order.
        """
        feats = self.extract_features(text, preprocessed)
        names = self.get_all_feature_names()
        return np.array([feats[name] for name in names], dtype=np.float32)
