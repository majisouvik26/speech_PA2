import numpy as np
import pandas as pd
from feature_extraction import extract_mfcc, compute_aggregated_features

def create_feature_dataframe(audio_files, labels, n_mfcc):
    """Create DataFrame with aggregated MFCC features."""
    features = []
    for file, label in zip(audio_files, labels):
        mfcc, _ = extract_mfcc(file)
        agg_features = compute_aggregated_features(mfcc)
        features.append([*agg_features, label])
    
    columns = [f"mean_{i}" for i in range(n_mfcc)] + \
             [f"std_{i}" for i in range(n_mfcc)] + ["label"]
    return pd.DataFrame(features, columns=columns)

def compute_language_stats(df, n_mfcc):
    """Compute language-level statistics."""
    stats = {}
    for lang in df["label"].unique():
        lang_df = df[df["label"] == lang]
        stats[lang] = {
            "mean": lang_df.iloc[:, :n_mfcc].mean().values,
            "std": lang_df.iloc[:, n_mfcc:-1].mean().values
        }
    return stats
