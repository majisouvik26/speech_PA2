# main.py
import os
import torch
from tqdm import tqdm
from dataset_creation import create_multi_speaker_dataset
from separation import load_sepformer_model, separate_audio
from separation_evaluation import evaluate_separation
from identification import load_speaker_identification_model, evaluate_identification, extract_embedding

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vox2_dir = "../datasets/vox2" 
    output_dir = "./multi_speaker"
    print("Creating multi-speaker dataset...")
    train_pairs, test_pairs = create_multi_speaker_dataset(vox2_dir, output_dir, train_count=50, test_count=50, snr_values=[0, 5])
    print("Loading SepFormer model...")
    sep_model = load_sepformer_model()
    test_mix_dir = os.path.join(output_dir, "test")
    separated_results = [] 
    for file in os.listdir(test_mix_dir):
        if file.startswith("mixed_") and file.endswith(".wav"):
            mix_path = os.path.join(test_mix_dir, file)
            sep_out = os.path.join("./separated", os.path.splitext(file)[0])
            os.makedirs(sep_out, exist_ok=True)
            spk1_path, spk2_path = separate_audio(sep_model, mix_path, sep_out)
            separated_results.append((mix_path, spk1_path, spk2_path))
    
    separation_metrics = {}
    for mix_path, spk1_path, spk2_path in tqdm(separated_results, desc="Evaluating Separation"):
        parts = os.path.basename(mix_path).split("_")
        ref1 = os.path.join(os.path.dirname(mix_path), f"clean_{parts[1]}.wav")
        ref2 = os.path.join(os.path.dirname(mix_path), f"clean_{parts[2]}.wav")
        refs = [ref1, ref2]
        ests = [spk1_path, spk2_path]
        metrics = evaluate_separation(refs, ests, fs=16000)
        separation_metrics[mix_path] = metrics
    
    speaker_model_path = "fine_tuned_model.pt"
    classifier_path = "classifier.pt"
    spk_model, classifier = load_speaker_identification_model(speaker_model_path, classifier_path, device)
    
    enrolled_embeddings = {}
    for file in os.listdir(test_mix_dir):
        if file.startswith("mixed_") and file.endswith(".wav"):
            parts = file.split("_")
            id1, id2 = parts[1], parts[2]
            ref1_path = os.path.join(test_mix_dir, f"clean_{id1}.wav")
            ref2_path = os.path.join(test_mix_dir, f"clean_{id2}.wav")
            if id1 not in enrolled_embeddings and os.path.exists(ref1_path):
                enrolled_embeddings[id1] = extract_embedding(spk_model, ref1_path, device)
            if id2 not in enrolled_embeddings and os.path.exists(ref2_path):
                enrolled_embeddings[id2] = extract_embedding(spk_model, ref2_path, device)
    
    id_test_pairs = []
    for mix_path, spk1_path, spk2_path in separated_results:
        parts = os.path.basename(mix_path).split("_")
        id1, id2 = parts[1], parts[2]
        id_test_pairs.append((spk1_path, id1))
        id_test_pairs.append((spk2_path, id2))
    
    rank1_accuracy = evaluate_identification(id_test_pairs, spk_model, classifier, enrolled_embeddings, device)
    
    results_file = "multi_speaker_results.txt"
    with open(results_file, "w") as f:
        f.write("Separation Metrics:\n")
        for mix_path, metrics in separation_metrics.items():
            f.write(f"{mix_path}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.2f}\n")
        f.write("\nSpeaker Identification Rank-1 Accuracy: {:.2f}%\n".format(rank1_accuracy))
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()

