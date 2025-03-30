# mixing.py
import librosa
import numpy as np

def mix_audio(file1, file2, snr):
    """
    Mix two audio files with a given SNR (in dB).  
    Both files are loaded at 16kHz.
    Returns:
      mixed audio, clean audio from file1, clean audio from file2
    (Signals are cropped to the same (minimum) length.)
    """
    audio1, sr = librosa.load(file1, sr=16000)
    audio2, sr = librosa.load(file2, sr=16000)
    # Crop both to the shortest length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    power1 = np.mean(audio1**2)
    power2 = np.mean(audio2**2)
    # Calculate scaling factor for audio2 to achieve desired SNR
    scale = np.sqrt(power1 / (power2 * 10**(snr / 10)))
    mixed = audio1 + scale * audio2
    return mixed, audio1, audio2

