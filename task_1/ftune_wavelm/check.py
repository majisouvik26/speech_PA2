import os
import torch
from evaluation import evaluate_verification
from model_utils import load_model, apply_lora
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = load_model(device)
    model = apply_lora(base_model)
    
    state_dict = torch.load("fine_tuned_wavlm_vox2.pt", map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    trial_pairs_file = "../datasets/vox1/veri_test2.txt"
    vox1_audio_dir = "../datasets/vox1/wav"
    
    eer, tar = evaluate_verification(model, trial_pairs_file, vox1_audio_dir, device)
    
    print(f"Verification - EER: {eer*100:.2f}%, TAR@1%FAR: {tar*100:.2f}%")

if __name__ == "__main__":
    main()

