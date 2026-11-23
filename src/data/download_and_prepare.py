"""
Dataset Download and Preparation Pipeline for IndicTextFake Benchmark.
Filters for 8 target Indian languages:
- Dravidian: Tamil (ta), Telugu (te), Malayalam (ml), Kannada (kn)
- Indo-Aryan: Hindi (hi), Bengali (bn), Marathi (mr), Gujarati (gu)

Balances Human vs AI (Aya-8b, Qwen2.5, GPT-OSS) and creates clean benchmark splits.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Base URLs for IndicTextFake on Hugging Face
HF_RAW_BASE = "https://huggingface.co/datasets/IndicTextFake/IndicTextFake/resolve/main"

URLS = {
    "main_train": f"{HF_RAW_BASE}/benchmark_splits/main/train.csv",
    "main_val": f"{HF_RAW_BASE}/benchmark_splits/main/val.csv",
    "main_test": f"{HF_RAW_BASE}/benchmark_splits/main/test.csv",
    "perturbation_test": f"{HF_RAW_BASE}/benchmark_splits/perturbation/test.csv",
    "samples_main": f"{HF_RAW_BASE}/data/samples/main_sample.csv",
}

# 8 Target Languages Mapping
TARGET_LANGUAGES = {
    "tam_Taml": {"code": "ta", "name": "Tamil", "family": "Dravidian", "script": "Tamil"},
    "tel_Telu": {"code": "te", "name": "Telugu", "family": "Dravidian", "script": "Telugu"},
    "mal_Mlym": {"code": "ml", "name": "Malayalam", "family": "Dravidian", "script": "Malayalam"},
    "kan_Knda": {"code": "kn", "name": "Kannada", "family": "Dravidian", "script": "Kannada"},
    "hin_Deva": {"code": "hi", "name": "Hindi", "family": "Indo-Aryan", "script": "Devanagari"},
    "ben_Beng": {"code": "bn", "name": "Bengali", "family": "Indo-Aryan", "script": "Bengali"},
    "mar_Deva": {"code": "mr", "name": "Marathi", "family": "Indo-Aryan", "script": "Devanagari"},
    "guj_Gujr": {"code": "gu", "name": "Gujarati", "family": "Indo-Aryan", "script": "Gujarati"},
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[INFO] File already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print(f"[INFO] Downloading {url} -> {dest_path}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path)
    print(f"[INFO] Downloaded successfully: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")


def download_all_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = RAW_DIR / f"{name}.csv"
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"[WARN] Failed to download {name} from {url}: {e}")


def filter_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to 8 target languages and standardize columns."""
    df = df[df["language"].isin(TARGET_LANGUAGES.keys())].copy()
    
    # Map language info
    df["lang_code"] = df["language"].map(lambda x: TARGET_LANGUAGES[x]["code"])
    df["lang_name"] = df["language"].map(lambda x: TARGET_LANGUAGES[x]["name"])
    df["family"] = df["language"].map(lambda x: TARGET_LANGUAGES[x]["family"])
    df["script"] = df["language"].map(lambda x: TARGET_LANGUAGES[x]["script"])
    
    # Ensure text is string and drop NaNs
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 10].copy()
    
    return df


def balance_dataset(df: pd.DataFrame, max_per_lang: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """
    Balance Human (0) vs AI (1) per language.
    For AI (1), balance across available generators (aya-8b, qwen2.5, gpt-oss).
    """
    rng = np.random.RandomState(seed)
    balanced_dfs = []

    for lang in df["lang_code"].unique():
        lang_df = df[df["lang_code"] == lang]
        human_df = lang_df[lang_df["label"] == 0]
        ai_df = lang_df[lang_df["label"] == 1]

        n_human = len(human_df)
        n_ai = len(ai_df)

        if n_human == 0 or n_ai == 0:
            continue

        target_n_per_class = min(n_human, n_ai)
        if max_per_lang is not None:
            target_n_per_class = min(target_n_per_class, max_per_lang // 2)

        # Sample humans
        sampled_human = human_df.sample(n=target_n_per_class, random_state=rng)

        # Sample AI balanced across generators
        generators = ai_df["generator"].unique()
        n_gen = len(generators)
        target_per_gen = target_n_per_class // n_gen
        
        sampled_ai_parts = []
        remaining = target_n_per_class
        for gen in generators:
            gen_df = ai_df[ai_df["generator"] == gen]
            sample_count = min(len(gen_df), target_per_gen)
            sampled_ai_parts.append(gen_df.sample(n=sample_count, random_state=rng))
            remaining -= sample_count

        # If any remainder due to rounding or small gen count, sample from leftover AI
        sampled_ai = pd.concat(sampled_ai_parts)
        if len(sampled_ai) < target_n_per_class:
            needed = target_n_per_class - len(sampled_ai)
            unused_ai = ai_df[~ai_df.index.isin(sampled_ai.index)]
            if len(unused_ai) > 0:
                more_ai = unused_ai.sample(n=min(len(unused_ai), needed), random_state=rng)
                sampled_ai = pd.concat([sampled_ai, more_ai])

        balanced_dfs.append(sampled_human)
        balanced_dfs.append(sampled_ai)

    res = pd.concat(balanced_dfs, ignore_index=True)
    return res.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def prepare_benchmark_data(sample_limit_train: int = 16000,
                           sample_limit_val: int = 2400,
                           sample_limit_test: int = 3200) -> Dict[str, Any]:
    """
    End-to-end data preparation. Produces balanced datasets with verified splits.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    download_all_raw()

    print("[INFO] Processing training set...")
    train_raw = pd.read_csv(RAW_DIR / "main_train.csv")
    train_filtered = filter_and_standardize(train_raw)
    train_balanced = balance_dataset(train_filtered, max_per_lang=sample_limit_train // 8)
    train_balanced.to_csv(PROCESSED_DIR / "train_balanced.csv", index=False)
    print(f"[INFO] Train balanced saved: {len(train_balanced)} rows")

    print("[INFO] Processing validation set...")
    val_raw = pd.read_csv(RAW_DIR / "main_val.csv")
    val_filtered = filter_and_standardize(val_raw)
    val_balanced = balance_dataset(val_filtered, max_per_lang=sample_limit_val // 8)
    val_balanced.to_csv(PROCESSED_DIR / "val_balanced.csv", index=False)
    print(f"[INFO] Val balanced saved: {len(val_balanced)} rows")

    print("[INFO] Processing test set...")
    test_raw = pd.read_csv(RAW_DIR / "main_test.csv")
    test_filtered = filter_and_standardize(test_raw)
    test_balanced = balance_dataset(test_filtered, max_per_lang=sample_limit_test // 8)
    test_balanced.to_csv(PROCESSED_DIR / "test_balanced.csv", index=False)
    print(f"[INFO] Test balanced saved: {len(test_balanced)} rows")

    print("[INFO] Processing perturbation test set...")
    pert_raw = pd.read_csv(RAW_DIR / "perturbation_test.csv")
    pert_filtered = filter_and_standardize(pert_raw)
    pert_filtered.to_csv(PROCESSED_DIR / "perturbation_test_full.csv", index=False)
    print(f"[INFO] Perturbation test set saved: {len(pert_filtered)} rows")

    summary = {
        "train_rows": len(train_balanced),
        "val_rows": len(val_balanced),
        "test_rows": len(test_balanced),
        "perturbation_test_rows": len(pert_filtered),
        "languages": list(TARGET_LANGUAGES.keys()),
        "target_lang_codes": [v["code"] for v in TARGET_LANGUAGES.values()],
        "train_lang_dist": train_balanced["lang_code"].value_counts().to_dict(),
        "train_gen_dist": train_balanced["generator"].value_counts().to_dict(),
        "train_label_dist": train_balanced["label"].value_counts().to_dict(),
        "test_lang_dist": test_balanced["lang_code"].value_counts().to_dict(),
        "test_gen_dist": test_balanced["generator"].value_counts().to_dict(),
        "test_label_dist": test_balanced["label"].value_counts().to_dict(),
    }

    with open(PROCESSED_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[INFO] Dataset preparation complete! Summary saved to dataset_summary.json")
    return summary


if __name__ == "__main__":
    prepare_benchmark_data()
