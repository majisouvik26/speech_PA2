from transformers import WavLMModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import librosa
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
model.eval()

def extract_embedding(audio_path):
    """
    Loads an audio file, extracts the WavLM embedding, and returns it.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(audio_tensor).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def main():
    trial_pairs_file = "/data/b22cs089/speech_PA2/task_1/datasets/vox1/veri_test2.txt"
    
    if not os.path.exists(trial_pairs_file):
        print(f"Trial pairs file not found: {trial_pairs_file}")
        return
    
    with open(trial_pairs_file, "r") as f:
        trial_pairs = [line.strip().split() for line in f if line.strip()]
    
    similarities = []
    labels = []
    
    for pair in tqdm(trial_pairs, desc="Processing trial pairs"):
        if len(pair) != 3:
            print(f"Skipping invalid line: {pair}")
            continue
        label, path1, path2 = pair
        full_path1 = os.path.join("./datasets/vox1/wav", path1)
        full_path2 = os.path.join("./datasets/vox1/wav", path2)
        
        emb1 = extract_embedding(full_path1)
        emb2 = extract_embedding(full_path2)
        
        if emb1 is None or emb2 is None:
            print(f"Skipping pair due to extraction error: {full_path1}, {full_path2}")
            continue
        
        similarity = 1 - cosine(emb1, emb2)
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
    
    result_str = f"EER: {eer * 100:.2f}%\nTAR@1%FAR: {tar_at_far * 100:.2f}%\n"
    print(result_str)
    output_file = "./evaluation_results.txt"
    with open(output_file, "w") as f:
        f.write(result_str)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
