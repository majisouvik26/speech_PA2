from config import *
from dataloader import load_audio_files
from feature_extraction import extract_mfcc, compute_aggregated_features
from visualization import plot_mfcc_spectrogram
from stats import create_feature_dataframe, compute_language_stats
import os
import multiprocessing

if __name__ == "__main__":  # Required for multiprocessing on Windows
    os.makedirs(PLOT_PATH, exist_ok=True)

    # Load data with verification
    print(f"Loading audio files from {DATASET_PATH}...")
    audio_files, labels = load_audio_files(DATASET_PATH, LANGUAGES)
    print(f"Successfully loaded {len(audio_files)} files from {len(set(labels))} languages")

    # Task A: Visualization (first sample per language)
    print("\nGenerating sample spectrograms...")
    plotted_samples = set()
    for i, (file_path, label) in enumerate(zip(audio_files, labels)):
        if label not in plotted_samples:
            try:
                mfcc, sr = extract_mfcc(file_path)
                output_file = f"{label}_sample{i}_mfcc.png"
                plot_mfcc_spectrogram(mfcc, sr, HOP_LENGTH, os.path.join(PLOT_PATH, output_file))
                plotted_samples.add(label)
                print(f"Generated spectrogram for {label}")
            except Exception as e:
                print(f"Failed to plot {label} sample: {str(e)}")
            if len(plotted_samples) >= 3:
                break

    # Task A: Feature Extraction with controlled parallelism
    print("\nExtracting MFCC features...")
    df = create_feature_dataframe(
        audio_files=audio_files,
        labels=labels,
        n_mfcc=N_MFCC,
        n_jobs=min(4, multiprocessing.cpu_count()//2)  # Safe core allocation
    )

    # Task A: Statistical Analysis
    print("\nComputing language statistics...")
    stats = compute_language_stats(df, N_MFCC)
