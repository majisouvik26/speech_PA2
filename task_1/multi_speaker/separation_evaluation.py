import numpy as np
import torchaudio
from mir_eval.separation import bss_eval_sources
from pesq import pesq

def evaluate_separation(ref_paths, est_paths, fs=16000):
    refs = []
    ests = []
    for path in ref_paths:
        waveform, sr = torchaudio.load(path)
        if sr != fs:
            waveform = torchaudio.transforms.Resample(sr, fs)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        ref = waveform.squeeze(0).numpy()
        refs.append(ref)
    for path in est_paths:
        waveform, sr = torchaudio.load(path)
        if sr != fs:
            waveform = torchaudio.transforms.Resample(sr, fs)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        est = waveform.squeeze(0).numpy()
        ests.append(est)
    
    min_len = min(min(len(r) for r in refs), min(len(e) for e in ests))
    refs = np.array([r[:min_len] for r in refs])
    ests = np.array([e[:min_len] for e in ests])
    
    sdr, sir, sar, _ = bss_eval_sources(refs, ests)
    
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
