# identification.py
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import WavLMModel
import librosa

def extract_embedding(model, audio_path, device):
    """
    Extracts an embedding from an audio file using a WavLM-based model.
    """
    audio, _ = librosa.load(audio_path, sr=16000)
    audio = torch.tensor(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(audio).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def load_speaker_identification_model(model_path, classifier_path, device):
    """
    Loads the fine-tuned speaker identification model and its classification head.
    Assumes the underlying model is WavLM.
    """
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # For example, assume 100 classes; adjust if needed.
    classifier = torch.nn.Linear(model.config.hidden_size, 100).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    return model, classifier

def identify_speaker(audio_path, speaker_model, classifier, enrolled_embeddings, device):
    """
    Identifies the speaker for a given audio file.
    Compares the extracted embedding to enrolled speaker embeddings using cosine similarity.
    
    enrolled_embeddings: dict mapping speaker IDs to their embedding (numpy array)
    Returns:
      Identified speaker ID.
    """
    emb = extract_embedding(speaker_model, audio_path, device)
    similarities = {spk: 1 - cosine(emb, enrolled_embeddings[spk]) for spk in enrolled_embeddings}
    identified = max(similarities, key=similarities.get)
    return identified

def evaluate_identification(test_pairs, speaker_model, classifier, enrolled_embeddings, device):
    """
    Evaluates Rank-1 identification accuracy.
    
    test_pairs: list of tuples (separated_audio_path, true_speaker_id)
    Returns:
      Rank-1 accuracy (in percentage).
    """
    correct = 0
    for audio_path, true_spk in test_pairs:
        pred = identify_speaker(audio_path, speaker_model, classifier, enrolled_embeddings, device)
        if pred == true_spk:
            correct += 1
    accuracy = correct / len(test_pairs) * 100
    return accuracy

