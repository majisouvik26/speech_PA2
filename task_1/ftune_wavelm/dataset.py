import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset

class VoxCeleb2Dataset(Dataset):
    """
    Expects a folder structure:
      root_dir/
         identity1/
             file1.wav, file2.wav, ...
         identity2/
             file1.wav, ...
         ...
    """
    def __init__(self, root_dir, identities):
        self.samples = []
        # Only include identities that are in the provided list.
        for identity in sorted(os.listdir(root_dir)):
            identity_path = os.path.join(root_dir, identity)
            if os.path.isdir(identity_path) and identity in identities:
                for file in os.listdir(identity_path):
                    if file.endswith(".wav"):
                        self.samples.append((os.path.join(identity_path, file), identity))
        # Map each identity to a label index (sorted order)
        self.identity_to_label = {id_: idx for idx, id_ in enumerate(sorted(identities))}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, identity = self.samples[idx]
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(16000)  # Fallback: 1 second of zeros
        audio = torch.tensor(audio, dtype=torch.float)
        label = self.identity_to_label[identity]
        return audio, label

def collate_fn(batch):
    # Batch is a list of (audio, label) pairs.
    audios = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    # Pad audio sequences to the maximum length in the batch.
    lengths = [audio.shape[0] for audio in audios]
    max_len = max(lengths)
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_len:
            pad = torch.zeros(max_len - audio.shape[0])
            padded = torch.cat([audio, pad], dim=0)
        else:
            padded = audio
        padded_audios.append(padded)
    padded_audios = torch.stack(padded_audios)
    return padded_audios, labels

