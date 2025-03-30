# separation_evaluation.py
import numpy as np
import librosa
from mir_eval.separation import bss_eval_sources
from pesq import pesq

def evaluate_separation(ref_paths, est_paths, fs=16000):
    """
    Evaluates separation performance using reference and estimated files.
    
    Parameters:
      ref_paths: List of two reference audio file paths.
      est_paths: List of two estimated audio file paths.
      fs: Sampling rate.
    
    Returns:
      A dictionary with average SDR, SIR, SAR and PESQ scores.
    """
    refs = []
    ests = []
    for path in ref_paths:
        ref, _ = librosa.load(path, sr=fs)
        refs.append(ref)
    for path in est_paths:
        est, _ = librosa.load(path, sr=fs)
        ests.append(est)
    
    # Crop signals to the minimum length
    min_len = min(min(len(r) for r in refs), min(len(e) for e in ests))
    refs = np.array([r[:min_len] for r in refs])
    ests = np.array([e[:min_len] for e in ests])
    
    # Compute SDR, SIR, SAR using mir_eval
    sdr, sir, sar, _ = bss_eval_sources(refs, ests)
    
    # Compute PESQ (wideband mode)
    pesq_scores = []
    for r, e in zip(refs, ests):
        score = pesq(fs, r, e, 'wb')
        pesq_scores.append(score)
    avg_pesq = np.mean(pesq_scores)
    
    metrics = {
        "SDR": np.mean(sdr),
        "SIR": np.mean(sir),
        "SAR": np.mean(sar),
        "PESQ": avg_pesq
    }
    return metrics

