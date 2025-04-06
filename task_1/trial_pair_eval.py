import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
from transformers import WavLMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
model.eval()

def extract_embedding_from_path(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    with torch.no_grad():
        embedding = model(waveform).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def evaluate_threshold_identification(trial_pairs_file):
    if not os.path.exists(trial_pairs_file):
        print(f"Trial pairs file not found: {trial_pairs_file}")
        return

    with open(trial_pairs_file, "r") as f:
        trial_pairs = [line.strip().split() for line in f if line.strip()]

    scores = []
    labels = []
    for pair in tqdm(trial_pairs, desc="Processing trial pairs"):
        if len(pair) != 3:
            print(f"Skipping invalid line: {pair}")
            continue
        label, path1, path2 = pair
        full_path1 = os.path.join("./datasets/vox1/wav", path1)
        full_path2 = os.path.join("./datasets/vox1/wav", path2)
        
        emb1 = extract_embedding_from_path(full_path1)
        emb2 = extract_embedding_from_path(full_path2)
        
        if emb1 is None or emb2 is None:
            print(f"Skipping pair due to extraction error: {full_path1}, {full_path2}")
            continue
        
        similarity = 1 - cosine(emb1, emb2)
        scores.append(similarity)
        labels.append(int(label))
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[idx]
    eer = fpr[idx] * 100  
    idx_far = np.where(fpr <= 0.01)[0]
    if len(idx_far) > 0:
        tar_at_1_far = tpr[idx_far[-1]] * 100
    else:
        tar_at_1_far = 0.0

    pred_labels = [1 if s >= eer_threshold else 0 for s in scores]
    accuracy = np.mean(np.array(pred_labels) == labels) * 100

    print("Threshold-based Speaker Identification Evaluation:")
    print(f"EER: {eer:.2f}%")
    print(f"TAR@1%FAR: {tar_at_1_far:.2f}%")
    print(f"Speaker Identification Accuracy: {accuracy:.2f}%")

def main():
    trial_pairs_file = "/data/b22cs089/speech_PA2/task_1/datasets/vox1/veri_test2.txt"
    evaluate_threshold_identification(trial_pairs_file)

if __name__ == "__main__":
    main()

