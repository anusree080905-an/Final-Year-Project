"""
Text Preprocessing, Unicode Normalization, Script Identification,
and Language Family Classification for Indic Languages.
Supports 8 Indian Languages:
- Dravidian: Tamil (ta), Telugu (te), Malayalam (ml), Kannada (kn)
- Indo-Aryan: Hindi (hi), Bengali (bn), Marathi (mr), Gujarati (gu)
"""

import re
import unicodedata
from typing import Dict, Any, Tuple, Optional, List

# Script Unicode Range Definitions
SCRIPT_RANGES = {
    'Devanagari': (0x0900, 0x097F),
    'Bengali': (0x0980, 0x09FF),
    'Gujarati': (0x0A80, 0x0AFF),
    'Tamil': (0x0B80, 0x0BFF),
    'Telugu': (0x0C00, 0x0C7F),
    'Kannada': (0x0C80, 0x0CFF),
    'Malayalam': (0x0D00, 0x0D7F)
}

# Language Family Mapping
LANGUAGE_FAMILIES = {
    'ta': 'Dravidian',
    'te': 'Dravidian',
    'ml': 'Dravidian',
    'kn': 'Dravidian',
    'hi': 'Indo-Aryan',
    'bn': 'Indo-Aryan',
    'mr': 'Indo-Aryan',
    'gu': 'Indo-Aryan'
}

LANGUAGE_NAMES = {
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'mr': 'Marathi',
    'gu': 'Gujarati'
}

LANGUAGE_SCRIPTS = {
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'hi': 'Devanagari',
    'bn': 'Bengali',
    'mr': 'Devanagari',
    'gu': 'Gujarati'
}

# Distinctive lexical markers for Devanagari disambiguation (Hindi vs Marathi)
MARATHI_MARKERS = {
    'आहे', 'नाही', 'आहेत', 'करणे', 'मध्ये', 'झाले', 'म्हणून', 'करणार',
    'त्यांनी', 'त्यांच्या', 'झाली', 'झाला', 'आपल्या', 'केले', 'होते',
    'होती', 'होता', 'यांचे', 'त्याच्या', 'सर्व', 'एक', 'फार', 'कशी'
}

HINDI_MARKERS = {
    'है', 'नहीं', 'हैं', 'करना', 'में', 'हुआ', 'क्योंकि', 'करेगा',
    'उन्होंने', 'उनके', 'हुई', 'हुआ', 'अपने', 'किया', 'थे',
    'थी', 'था', 'इनका', 'उसके', 'सभी', 'एक', 'बहुत', 'कैसे'
}

class IndicTextPreprocessor:
    """
    Robust Unicode text preprocessor, script detector, and language router.
    """

    def __init__(self):
        # Danda and double danda patterns
        self.danda_pattern = re.compile(r'[\u0964\u0965]')
        # Redundant whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode string using NFC standard form, cleans zero-width anomalies
        while selectively preserving morphologically critical joiners (ZWJ/ZWNJ).
        """
        if not text:
            return ""

        # Normalize to NFC
        normalized = unicodedata.normalize('NFC', text)

        # Replace non-breaking spaces and exotic space characters with standard space
        normalized = normalized.replace('\u00A0', ' ')
        normalized = normalized.replace('\u200B', '')  # Zero width space
        normalized = normalized.replace('\uFEFF', '')  # BOM

        # Standardize quotes and hyphens
        normalized = re.sub(r'[\u2018\u2019]', "'", normalized)
        normalized = re.sub(r'[\u201C\u201D]', '"', normalized)
        normalized = re.sub(r'[\u2013\u2014]', '-', normalized)

        # Normalize multiple spaces and line breaks
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)

        return normalized.strip()

    def identify_script(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Identify the primary Indic script based on Unicode character distribution.
        Returns the dominant script name and distribution proportions.
        """
        counts = {script: 0 for script in SCRIPT_RANGES}
        total_indic_chars = 0

        for char in text:
            code = ord(char)
            # Skip shared Indic punctuation like Danda (0x0964) and Double Danda (0x0965) for script detection
            if code in (0x0964, 0x0965):
                continue
            for script, (start, end) in SCRIPT_RANGES.items():
                if start <= code <= end:
                    counts[script] += 1
                    total_indic_chars += 1
                    break

        if total_indic_chars == 0:
            return 'Unknown', {s: 0.0 for s in counts}

        proportions = {s: c / total_indic_chars for s, c in counts.items()}
        dominant_script = max(proportions, key=proportions.get)

        if proportions[dominant_script] < 0.2:
            return 'Mixed/Unknown', proportions

        return dominant_script, proportions

    def identify_language_and_family(self, text: str) -> Dict[str, Any]:
        """
        Identifies language code, language name, script, and language family.
        Handles Devanagari disambiguation between Hindi and Marathi.
        """
        normalized_text = self.normalize_unicode(text)
        script, script_distribution = self.identify_script(normalized_text)

        lang_code = 'hi'  # default fallback

        if script == 'Tamil':
            lang_code = 'ta'
        elif script == 'Telugu':
            lang_code = 'te'
        elif script == 'Malayalam':
            lang_code = 'ml'
        elif script == 'Kannada':
            lang_code = 'kn'
        elif script == 'Bengali':
            lang_code = 'bn'
        elif script == 'Gujarati':
            lang_code = 'gu'
        elif script == 'Devanagari':
            # Disambiguate Hindi vs Marathi
            # 1. Check for Marathi exclusive letter LLA: \u0933 (ळ)
            if '\u0933' in normalized_text:
                lang_code = 'mr'
            else:
                words = set(re.findall(r'[\u0900-\u097F]+', normalized_text))
                marathi_hits = len(words.intersection(MARATHI_MARKERS))
                hindi_hits = len(words.intersection(HINDI_MARKERS))
                if marathi_hits > hindi_hits:
                    lang_code = 'mr'
                else:
                    lang_code = 'hi'
        else:
            lang_code = 'hi'

        family = LANGUAGE_FAMILIES.get(lang_code, 'Unknown')
        lang_name = LANGUAGE_NAMES.get(lang_code, 'Unknown')

        return {
            'language_code': lang_code,
            'language_name': lang_name,
            'script': script,
            'language_family': family,
            'script_distribution': script_distribution
        }

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize Indic text into sentences using Dandas and standard sentence punctuation.
        """
        # Replace dandas with standard sentence breaks
        processed = self.danda_pattern.sub('.', text)
        sentences = re.split(r'[.!?\n]+', processed)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize Indic text into clean words, removing punctuation and symbols.
        """
        # Match sequences of Indic characters, letters, or numbers
        words = re.findall(r'[^\s.,!?;:\"\'।॥()\[\]{}/\\<>+=@#$%^&*~`]+', text)
        return [w.strip() for w in words if w.strip()]

    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline returning normalized text, tokens, and metadata.
        """
        normalized = self.normalize_unicode(text)
        lang_info = self.identify_language_and_family(normalized)
        sentences = self.tokenize_sentences(normalized)
        words = self.tokenize_words(normalized)

        return {
            'raw_text': text,
            'normalized_text': normalized,
            'language_code': lang_info['language_code'],
            'language_name': lang_info['language_name'],
            'language_family': lang_info['language_family'],
            'script': lang_info['script'],
            'script_distribution': lang_info['script_distribution'],
            'sentences': sentences,
            'words': words,
            'char_count': len(normalized),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
