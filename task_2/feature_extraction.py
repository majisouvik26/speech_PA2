import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13, hop_length=512, n_fft=2048):
    """Return tuple (mfcc, sr) or (None, None) on error"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) < n_fft:
            raise ValueError(f"Audio too short: {len(y)} samples")
            
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            hop_length=hop_length, n_fft=n_fft
        )
        return mfcc, sr  # Return both features and sample rate
    except Exception as e:
        print(f"\nError in {audio_path}: {str(e)}")
        return None, None  # Maintain tuple structure

def compute_aggregated_features(mfcc):
    """Validate MFCC shape before processing"""
    if mfcc is None or mfcc.shape[1] < 1:  # Check time dimension
        return None
    
    if len(mfcc.shape) != 2:  # Ensure 2D array
        return None
        
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])
