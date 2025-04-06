import os
from dataset_creation import create_multi_speaker_dataset
from task_1_separation import main as separation_main
from task_2_identification import main as identification_main

def main():
    vox2_dir = "../datasets/vox2/aac"
    output_dir = "./mixes"
    
    print("Creating multi-speaker dataset...")
    train_pairs, test_pairs = create_multi_speaker_dataset(
        vox2_dir, 
        output_dir, 
        train_count=50, 
        test_count=50, 
        snr_values=[0, 5]
    )
    print("Multi-speaker dataset created.")
    
    print("Running speaker separation and evaluation (Task I)...")
    separation_main()
    print("Running speaker identification on separated outputs (Task II)...")
    identification_main()
    
if __name__ == "__main__":
    main()
