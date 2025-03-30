from feature_extraction import extract_mfcc, compute_aggregated_features
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_file(args):
    file, label, n_mfcc = args
    try:
        # Unpack the returned tuple from extract_mfcc
        mfcc, sr = extract_mfcc(file, n_mfcc=n_mfcc)
        if mfcc is None:
            return None
            
        agg_features = compute_aggregated_features(mfcc)
        if agg_features is None or len(agg_features) != 2 * n_mfcc:
            print(f"Invalid features in {file}")
            return None
            
        return [*agg_features, label]
    except Exception as e:
        print(f"\nCritical error in {file}: {str(e)}")
        return None

def create_feature_dataframe(audio_files, labels, n_mfcc, n_jobs=8):
    """Create DataFrame with safe multiprocessing"""
    print(f"Processing {len(audio_files)} files using {n_jobs} cores...")
    
    args = [(file, label, n_mfcc) for file, label in zip(audio_files, labels)]
    
    features = []
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, args, chunksize=100),
                          total=len(args),
                          desc="Extracting Features"))
    
    features = [res for res in results if res is not None]
    
    columns = [f"mean_{i}" for i in range(n_mfcc)] + \
             [f"std_{i}" for i in range(n_mfcc)] + ["label"]
    
    return pd.DataFrame(features, columns=columns)

def compute_language_stats(df, n_mfcc):
    """Enhanced version with automatic printing"""
    stats = {}
    mean_cols = [f"mean_{i}" for i in range(n_mfcc)]
    std_cols = [f"std_{i}" for i in range(n_mfcc)]
    
    grouped = df.groupby('label')
    for lang, group in grouped:
        stats[lang] = {
            "mean": group[mean_cols].mean().round(3).values,
            "std": group[std_cols].mean().round(3).values
        }
    
    print("\nLanguage Statistics Summary:")
    for lang, data in stats.items():
        print(f"\n{lang.upper():<10}")
        print(f"MFCC Means: {data['mean']}")
        print(f"MFCC Stdev: {data['std']}")
    
    return stats
