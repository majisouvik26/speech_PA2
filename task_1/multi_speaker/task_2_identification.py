# task_ii_identification.py
import os
import torch
from tqdm import tqdm
from identification import load_speaker_identification_model, extract_embedding, evaluate_identification

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speaker_model_path = "fine_tuned_model.pt" 
    classifier_path = "classifier.pt"          
    spk_model, classifier = load_speaker_identification_model(speaker_model_path, classifier_path, device)
    
    test_mix_dir = "./multi_speaker/test"    
    separated_dir = "./separated"            
    enrolled_embeddings = {}
    for file in os.listdir(test_mix_dir):
        if file.startswith("clean_") and file.endswith(".wav"):
            spk_id = file.split("_")[1].split(".")[0]
            ref_path = os.path.join(test_mix_dir, file)
            enrolled_embeddings[spk_id] = extract_embedding(spk_model, ref_path, device)
    
    id_test_pairs = []
    for subdir in os.listdir(separated_dir):
        subdir_path = os.path.join(separated_dir, subdir)
        if os.path.isdir(subdir_path):
            spk1_path = os.path.join(subdir_path, "speaker1.wav")
            spk2_path = os.path.join(subdir_path, "speaker2.wav")
            parts = subdir.split("_")
            if len(parts) < 3:
                continue
            id1 = parts[1]
            id2 = parts[2]
            id_test_pairs.append((spk1_path, id1))
            id_test_pairs.append((spk2_path, id2))
    
    rank1_accuracy = evaluate_identification(id_test_pairs, spk_model, classifier, enrolled_embeddings, device)
    
    results_file = "task_ii_identification_results.txt"
    with open(results_file, "w") as f:
        f.write("Speaker Identification (Nearest Neighbor) Results:\n")
        f.write(f"Rank-1 Accuracy on Separated Outputs: {rank1_accuracy:.2f}%\n")
    print(f"Speaker Identification results saved to {results_file}")

if __name__ == "__main__":
    main()

