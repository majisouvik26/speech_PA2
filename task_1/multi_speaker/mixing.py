import torch
import torchaudio
import numpy as np

def mix_audio(file1, file2, snr):
    waveform1, sr1 = torchaudio.load(file1)
    waveform2, sr2 = torchaudio.load(file2)
    if sr1 != 16000:
        waveform1 = torchaudio.transforms.Resample(sr1, 16000)(waveform1)
    if sr2 != 16000:
        waveform2 = torchaudio.transforms.Resample(sr2, 16000)(waveform2)
    if waveform1.shape[0] > 1:
        waveform1 = waveform1.mean(dim=0, keepdim=True)
    if waveform2.shape[0] > 1:
        waveform2 = waveform2.mean(dim=0, keepdim=True)
    
    audio1 = waveform1.squeeze(0).numpy()
    audio2 = waveform2.squeeze(0).numpy()
    
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    power1 = np.mean(audio1**2)
    power2 = np.mean(audio2**2)
    scale = np.sqrt(power1 / (power2 * 10**(snr / 10)))
    mixed = audio1 + scale * audio2
    return mixed, audio1, audio2
