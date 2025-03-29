import matplotlib.pyplot as plt
import librosa.display
import os

def plot_mfcc_spectrogram(mfcc, sr, hop_length, output_path):
    """Plot and save MFCC spectrogram."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time")
    plt.colorbar()
    plt.title("MFCC Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
