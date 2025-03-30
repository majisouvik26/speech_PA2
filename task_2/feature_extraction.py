## option -1 : cpu acceleration for mfcc extraction using librosa
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
        return mfcc, sr  
    except Exception as e:
        print(f"\nError in {audio_path}: {str(e)}")
        return None, None 

def compute_aggregated_features(mfcc):
    """Validate MFCC shape before processing"""
    if mfcc is None or mfcc.shape[1] < 1:  
        return None
    
    if len(mfcc.shape) != 2:
        return None
        
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

### option -2 : gpu acceleration for mfcc extraction using torchaudio
import torch
import torchaudio
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13, hop_length=512, n_fft=2048, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
       
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)  
        waveform = waveform.to(device)

        if waveform.size(1) < n_fft:
            raise ValueError(f"Audio too short: {waveform.size(1)} samples")

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': 128,
                'window_fn': torch.hann_window,
                'power': 2.0,
                'center': True,
                'pad_mode': 'reflect',
                'norm': 'slaney',
                'mel_scale': 'slaney',
            }
        ).to(device)

        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0) 
        
        return mfcc.cpu().numpy(), sr
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None

def compute_aggregated_features(mfcc):
    """Compute features along the correct time axis"""
    if mfcc is None or mfcc.shape[1] < 1:
        return None
    
    return np.concatenate([
        np.mean(mfcc, axis=1),  
        np.std(mfcc, axis=1)
    ])

if __name__ == "__main__":
    audio_path = "/data/b22cs089/speech_PA2/task_2/LanguageDetectionDataset/Bengali/0.mp3"
    mfcc, sr = extract_mfcc(audio_path)
    if mfcc is not None:
        features = compute_aggregated_features(mfcc)
        print(f"Aggregated features for {audio_path}:\n", features)