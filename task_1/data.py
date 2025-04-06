import os
import torch
import torchaudio

def find_audio_files(root, extensions=(".wav", ".flac", ".m4a")):
    audio_files = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith(extensions):
                audio_files.append(os.path.join(dirpath, file))
    return sorted(audio_files)

class VoxCeleb1Dataset(torch.utils.data.Dataset):
    def __init__(self, root, desired_length=3 * 16000, target_sample_rate=16000, allowed_identities=None):
        self.root = root
        self.desired_length = desired_length
        self.target_sample_rate = target_sample_rate
        self.samples = [] 
        for speaker in sorted(os.listdir(root)):
            speaker_path = os.path.join(root, speaker)
            if not os.path.isdir(speaker_path):
                continue
            if allowed_identities is not None and speaker not in allowed_identities:
                continue
            audio_files = find_audio_files(speaker_path)
            for file in audio_files:
                self.samples.append((file, speaker))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, speaker = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            waveform = torch.zeros(1, self.desired_length)
            sr = self.target_sample_rate
        
        if sr != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.target_sample_rate)(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)  
        num_samples = waveform.shape[0]
        if num_samples < self.desired_length:
            pad = torch.zeros(self.desired_length - num_samples)
            waveform = torch.cat([waveform, pad])
        else:
            waveform = waveform[:self.desired_length]
        return waveform, speaker

def collate_fn(batch):
    audios = [item[0] for item in batch]
    speakers = [item[1] for item in batch]
    return torch.stack(audios), speakers

if __name__ == "__main__":
    dataset_root = "./datasets/vox1/wav"  
    dataset = VoxCeleb1Dataset(dataset_root)
    print(f"Loaded {len(dataset)} samples from VoxCeleb1.")
    for i in range(min(5, len(dataset))):
        waveform, speaker = dataset[i]
        print(f"Sample {i}: Speaker = {speaker}, Audio shape = {waveform.shape}")
