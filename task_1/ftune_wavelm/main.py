import os
import torch
from torch.utils.data import DataLoader
from dataset import VoxCeleb2Dataset, collate_fn
from model_utils import load_model, apply_lora
from train import train_model
from evaluation import evaluate_verification, evaluate_identification

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare the model with LoRA.
    model = load_model(device)
    model = apply_lora(model)
    
    # VoxCeleb2 dataset: use first 100 identities for training and next 18 for testing.
    vox2_root = "./datasets/vox2/wav"  # Adjust path as necessary.
    all_identities = sorted([d for d in os.listdir(vox2_root) if os.path.isdir(os.path.join(vox2_root, d))])
    train_identities = all_identities[:100]
    test_identities = all_identities[100:118]
    
    # Create datasets and dataloaders
    train_dataset = VoxCeleb2Dataset(vox2_root, train_identities)
    test_dataset = VoxCeleb2Dataset(vox2_root, test_identities)
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Define a classification head for speaker identification.
    embedding_dim = model.config.hidden_size
    num_classes = len(train_identities)
    classifier = torch.nn.Linear(embedding_dim, num_classes).to(device)
    
    # Training parameters.
    num_epochs = 10
    learning_rate = 1e-4
    
    # Train the model.
    train_model(model, classifier, train_loader, num_epochs, learning_rate, device)
    
    # Evaluate on VoxCeleb1 verification trial pairs.
    vox1_wav_dir = "./datasets/vox1/wav"  # Adjust path as necessary.
    trial_pairs_file = "/data/b22cs089/speech_PA2/task_1/datasets/vox1/veri_test2.txt"  # Adjust if needed.
    eer, tar_at_far = evaluate_verification(model, trial_pairs_file, vox1_wav_dir, device)
    print(f"Verification - EER: {eer*100:.2f}%, TAR@1%FAR: {tar_at_far*100:.2f}%")
    
    # Evaluate speaker identification on VoxCeleb2.
    identification_accuracy = evaluate_identification(model, classifier, test_loader, device)
    print(f"Speaker Identification Accuracy: {identification_accuracy*100:.2f}%")
    
    # Save the fine-tuned model and classifier.
    torch.save(model.state_dict(), "fine_tuned_model.pt")
    torch.save(classifier.state_dict(), "classifier.pt")
    
    # Save evaluation results to a text file.
    results_file = "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Verification - EER: {eer*100:.2f}%\n")
        f.write(f"Verification - TAR@1%FAR: {tar_at_far*100:.2f}%\n")
        f.write(f"Speaker Identification Accuracy: {identification_accuracy*100:.2f}%\n")
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
