import os
import torch
from torch.utils.data import DataLoader
from dataset import VoxCeleb2TxtDataset, collate_fn
from train import train_model
from model_utils import load_model, apply_lora
from evaluation import evaluate_verification, evaluate_identification
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_model(device)
    model = apply_lora(base_model)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs.")
    #     model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)

    txt_root = "../datasets/vox2/txt" 
    aac_root = "../datasets/vox2/aac"  
    all_identities = sorted([d for d in os.listdir(aac_root) if d.startswith("id") and os.path.isdir(os.path.join(aac_root, d))])
    train_identities = all_identities[:100]
    test_identities = all_identities[100:118]
    train_dataset = VoxCeleb2TxtDataset(txt_root, aac_root, train_identities)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    label_map = {identity: idx for idx, identity in enumerate(sorted(train_identities))}

    embedding_dim = base_model.config.hidden_size
    import torch.nn as nn
    num_classes = len(train_identities)
    classifier = nn.Linear(embedding_dim, num_classes).to(device)

    # if torch.cuda.device_count() > 1:
    #     classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2])
    classifier.to(device)

    num_epochs = 10
    learning_rate = 1e-4

    train_model(model, classifier, train_loader, num_epochs, learning_rate, device, label_map)

    torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), "fine_tuned_wavlm_vox2.pt")
    torch.save(classifier.module.state_dict() if hasattr(classifier, "module") else classifier.state_dict(), "classifier_vox2.pt")
    print("Training complete and models saved.")

    with open("results.txt", "w") as f:
        f.write("Training complete and models saved.\n")

        test_dataset = VoxCeleb2TxtDataset(txt_root, aac_root, test_identities)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_label_map = {identity: idx for idx, identity in enumerate(sorted(test_identities))}
        acc = evaluate_identification(model, classifier, test_loader, device, test_label_map)
        result_acc = f"Test Speaker Identification Accuracy: {acc:.2f}%"
        print(result_acc)
        f.write(result_acc + "\n")

        trial_pairs_file = "../datasets/vox1/veri_test2.txt"
        vox1_audio_dir = "../datasets/vox1/wav"
        eer, tar = evaluate_verification(model, trial_pairs_file, vox1_audio_dir, device)
        result_verification = f"Verification - EER: {eer*100:.2f}%, TAR@1%FAR: {tar*100:.2f}%"
        print(result_verification)
        f.write(result_verification + "\n")


if __name__ == "__main__":
    main()
