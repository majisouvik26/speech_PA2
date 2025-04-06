import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import WavLMModel
import torchaudio

def extract_embedding(model, audio_path, device):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)
    with torch.no_grad():
        embedding = model(waveform).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def load_speaker_identification_model(model_path, classifier_path, device):
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    classifier = torch.nn.Linear(model.config.hidden_size, 100).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    return model, classifier

def identify_speaker(audio_path, speaker_model, classifier, enrolled_embeddings, device):
    emb = extract_embedding(speaker_model, audio_path, device)
    similarities = {spk: 1 - cosine(emb, enrolled_embeddings[spk]) for spk in enrolled_embeddings}
    identified = max(similarities, key=similarities.get)
    return identified

def evaluate_identification(test_pairs, speaker_model, classifier, enrolled_embeddings, device):
    correct = 0
    for audio_path, true_spk in test_pairs:
        pred = identify_speaker(audio_path, speaker_model, classifier, enrolled_embeddings, device)
        if pred == true_spk:
            correct += 1
    accuracy = correct / len(test_pairs) * 100
    return accuracy
