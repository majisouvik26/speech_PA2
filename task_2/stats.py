import numpy as np
import pandas as pd
from feature_extraction import extract_mfcc, compute_aggregated_features

def create_feature_dataframe(audio_files, labels, n_mfcc):
    """Create DataFrame with aggregated MFCC features."""
    features = []
    
    for file, label in zip(audio_files, labels):
        try:
            mfcc, _ = extract_mfcc(file)
            agg_features = compute_aggregated_features(mfcc)
            features.append([*agg_features, label])
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue 
    columns = [f"mean_{i}" for i in range(n_mfcc)] + \
             [f"std_{i}" for i in range(n_mfcc)] + ["label"]
    
    return pd.DataFrame(features, columns=columns)

def compute_language_stats(df, n_mfcc):
    stats = {}
    mean_cols = [f"mean_{i}" for i in range(n_mfcc)]
    std_cols = [f"std_{i}" for i in range(n_mfcc)]
    
    for lang in df["label"].unique():
        lang_df = df[df["label"] == lang]
        stats[lang] = {
            "mean": lang_df[mean_cols].mean().values,
            "std": lang_df[std_cols].mean().values,
            "variance": lang_df[mean_cols].var().values  
        }
    return stats
