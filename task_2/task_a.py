from config import *
from dataloader import load_audio_files
from feature_extraction import extract_mfcc
from visualization import plot_mfcc_spectrogram
from stats import create_feature_dataframe, compute_language_stats
import os

os.makedirs(PLOT_PATH, exist_ok=True)

audio_files, labels = load_audio_files(DATASET_PATH, LANGUAGES)
# print(f"Loaded {len(audio_files)} audio files from {len(LANGUAGES)} languages.")

# Task A: Visualization
for i, file_path in enumerate(audio_files[:3*len(LANGUAGES)]):  
    mfcc, sr = extract_mfcc(file_path)
    output_file = f"{labels[i]}_sample{i//3}_mfcc.png"
    plot_mfcc_spectrogram(mfcc, sr, HOP_LENGTH, os.path.join(PLOT_PATH, output_file))

# Task A: Statistical Analysis
df = create_feature_dataframe(audio_files, labels, N_MFCC)
stats = compute_language_stats(df, N_MFCC)

# Print
for lang, values in stats.items():
    print(f"Language: {lang}")
    print(f"Average MFCC Means: {values['mean']}")
    print(f"Average MFCC Stds: {values['std']}\n")
