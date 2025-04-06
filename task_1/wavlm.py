import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from transformers import WavLMModel
from data import VoxCeleb1Dataset, collate_fn

# Set device and load the WavLM model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
model.eval()

##################################
# Verification Evaluation Section
##################################
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

def evaluate_verification(trial_pairs_file):
    if not os.path.exists(trial_pairs_file):
        print(f"Trial pairs file not found: {trial_pairs_file}")
        return ""
    
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
        
        emb1 = extract_embedding_from_path(full_path1)
        emb2 = extract_embedding_from_path(full_path2)
        
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
    
    verification_results = (
        "Verification Results:\n"
        f"EER: {eer * 100:.2f}%\n"
        f"TAR@1%FAR: {tar_at_far * 100:.2f}%\n"
    )
    print(verification_results)
    return verification_results

############################################
# Speaker Identification Evaluation Section
############################################
def evaluate_speaker_identification(dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_embeddings = []
    all_labels = []
    
    for audios, speakers in tqdm(dataloader, desc="Processing speaker identification"):
        audios = audios.to(device)
        with torch.no_grad():
            embeddings = model(audios).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())
        all_labels.extend(speakers)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    norm_embeddings = all_embeddings / all_embeddings.norm(dim=1, keepdim=True)
    similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
    
    diag_indices = torch.arange(similarity_matrix.size(0))
    similarity_matrix[diag_indices, diag_indices] = -float('inf')
    
    _, max_indices = similarity_matrix.max(dim=1)
    predicted_labels = [all_labels[i] for i in max_indices.tolist()]
    correct = sum([pred == true for pred, true in zip(predicted_labels, all_labels)])
    rank1_accuracy = correct / len(all_labels)
    
    identification_results = f"Speaker Identification Rank‑1 Accuracy: {rank1_accuracy * 100:.2f}%\n"
    print(identification_results)
    return identification_results

######################
# Main Entry Function
######################
def main():
    combined_results = ""
    trial_pairs_file = "/data/b22cs089/speech_PA2/task_1/datasets/vox1/veri_test2.txt"
    verification_result = evaluate_verification(trial_pairs_file)
    combined_results += verification_result + "\n"
    
    dataset_root = "./datasets/vox1/wav"
    allowed_identities = None 
    dataset = VoxCeleb1Dataset(dataset_root, desired_length=3 * 16000, target_sample_rate=16000, allowed_identities=allowed_identities)
    print(f"Speaker Identification Dataset contains {len(dataset)} samples.")
    
    identification_result = evaluate_speaker_identification(dataset)
    combined_results += identification_result + "\n"
    
    output_file = "evaluation_results.txt"
    with open(output_file, "w") as f:
        f.write(combined_results)
    print(f"Combined results saved to {output_file}")

if __name__ == "__main__":
    main()
