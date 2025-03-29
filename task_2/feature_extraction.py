import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft
    )
    return mfcc, sr

def compute_aggregated_features(mfcc):
    """Compute mean and std of MFCC coefficients across time."""
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    return np.concatenate([mean, std])
