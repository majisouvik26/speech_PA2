import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

def extract_embedding(model, audio_tensor, device):
    """
    Given an audio tensor (with shape [1, seq_length]), return the averaged embedding.
    """
    with torch.no_grad():
        embedding = model(audio_tensor.to(device)).last_hidden_state.mean(dim=1)
    return embedding.cpu()

def evaluate_verification(model, trial_pairs_file, wav_dir, device):
    """
    Evaluates verification performance using VoxCeleb1 trial pairs.
    Expects a file where each line contains:
       <label> <path1> <path2>
    """
    model.eval()
    similarities = []
    labels = []
    with open(trial_pairs_file, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    
    for pair in tqdm(pairs, desc="Evaluating Verification"):
        if len(pair) != 3:
            continue
        label, path1, path2 = pair
        full_path1 = os.path.join(wav_dir, path1)
        full_path2 = os.path.join(wav_dir, path2)
        try:
            audio1, _ = librosa.load(full_path1, sr=16000)
            audio2, _ = librosa.load(full_path2, sr=16000)
        except Exception as e:
            print(f"Error loading files: {e}")
            continue
        audio1 = torch.tensor(audio1, dtype=torch.float).unsqueeze(0)
        audio2 = torch.tensor(audio2, dtype=torch.float).unsqueeze(0)
        emb1 = extract_embedding(model, audio1, device)
        emb2 = extract_embedding(model, audio2, device)
        similarity = 1 - cosine(emb1.squeeze().numpy(), emb2.squeeze().numpy())
        similarities.append(similarity)
        labels.append(int(label))
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    fpr, tpr, _ = roc_curve(labels, similarities)
    diff = np.abs(fpr - (1 - tpr))
    eer_index = np.argmin(diff)
    eer = fpr[eer_index]
    tar_index = np.argmin(np.abs(fpr - 0.01))
    tar_at_far = tpr[tar_index]
    return eer, tar_at_far

def evaluate_identification(model, classifier, test_loader, device):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc="Evaluating Identification"):
            audio = audio.to(device)
            labels = labels.to(device)
            embeddings = model(audio).last_hidden_state.mean(dim=1)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

