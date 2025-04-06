import os
from tqdm import tqdm
from separation import load_sepformer_model, separate_audio
from separation_evaluation import evaluate_separation

def main():
    test_mix_dir = "./multi_speaker/test" 
    separated_output_dir = "./separated"    
    os.makedirs(separated_output_dir, exist_ok=True)
    
    sep_model = load_sepformer_model()
    
    separation_metrics = {}
    test_files = [f for f in os.listdir(test_mix_dir) if f.startswith("mixed_") and f.endswith(".wav")]
    
    for file in tqdm(test_files, desc="Separating test mixtures"):
        mix_path = os.path.join(test_mix_dir, file)
        output_subdir = os.path.join(separated_output_dir, os.path.splitext(file)[0])
        os.makedirs(output_subdir, exist_ok=True)
        spk1_path, spk2_path = separate_audio(sep_model, mix_path, output_subdir)
        
        parts = file.split("_")
        if len(parts) < 4:
            continue
        id1 = parts[1]
        id2 = parts[2]
        ref1_path = os.path.join(test_mix_dir, f"clean_{id1}.wav")
        ref2_path = os.path.join(test_mix_dir, f"clean_{id2}.wav")
        refs = [ref1_path, ref2_path]
        ests = [spk1_path, spk2_path]
        metrics = evaluate_separation(refs, ests, fs=16000)
        separation_metrics[mix_path] = metrics

    results_file = "task_i_separation_results.txt"
    with open(results_file, "w") as f:
        f.write("Separation Metrics for each Test Mixture:\n\n")
        for mix_path, metrics in separation_metrics.items():
            f.write(f"{mix_path}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\n")
    print(f"Separation evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()

