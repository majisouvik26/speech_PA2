import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

def extract_embedding(model, audio_tensor, device):
    """
    Extract an embedding from a single audio tensor of shape [1, T].
    """
    with torch.no_grad():
        embedding = model(audio_tensor.to(device)).last_hidden_state.mean(dim=1)
    return embedding.detach().cpu()

def evaluate_verification(model, trial_pairs_file, audio_dir, device, sr=16000):
    """
    Evaluate speaker verification performance.
    
    The trial pairs file should have lines formatted as:
      <label> <relative_path1> <relative_path2>
    where paths are relative to `audio_dir`.
    
    Returns:
      eer: Equal Error Rate (fraction)
      tar_at_far: True Acceptance Rate at 1% False Acceptance Rate.
    """
    model.eval()
    similarities = []
    labels = []
    
    with open(trial_pairs_file, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]
    
    for pair in tqdm(pairs, desc="Evaluating Verification"):
        if len(pair) != 3:
            continue
        label_str, rel_path1, rel_path2 = pair
        file1 = os.path.join(audio_dir, rel_path1)
        file2 = os.path.join(audio_dir, rel_path2)
        try:
            audio1, _ = librosa.load(file1, sr=sr)
        except Exception as e:
            print(f"Error loading {file1}: {e}")
            continue
        try:
            audio2, _ = librosa.load(file2, sr=sr)
        except Exception as e:
            print(f"Error loading {file2}: {e}")
            continue
        audio1_tensor = torch.tensor(audio1, dtype=torch.float).unsqueeze(0)
        audio2_tensor = torch.tensor(audio2, dtype=torch.float).unsqueeze(0)
        emb1 = extract_embedding(model, audio1_tensor, device)
        emb2 = extract_embedding(model, audio2_tensor, device)
        sim = 1 - cosine(emb1.squeeze().numpy(), emb2.squeeze().numpy())
        similarities.append(sim)
        labels.append(int(label_str))
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    fpr, tpr, _ = roc_curve(labels, similarities)
    diff = np.abs(fpr - (1 - tpr))
    eer_index = np.argmin(diff)
    eer = fpr[eer_index]
    tar_index = np.argmin(np.abs(fpr - 0.01))
    tar_at_far = tpr[tar_index]
    return eer, tar_at_far

def evaluate_identification(model, classifier, test_loader, device, label_map):
    """
    Evaluate speaker identification accuracy using a DataLoader.
    
    label_map: Dictionary mapping identity (string) to numeric label.
    
    Returns:
      Accuracy in percentage.
    """
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audios, identities in tqdm(test_loader, desc="Evaluating Identification"):
            audios = audios.to(device)
            labels = torch.tensor([label_map[id_] for id_ in identities], dtype=torch.long).to(device)
            embeddings = model(audios).last_hidden_state.mean(dim=1)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy
