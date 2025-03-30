# separation.py
import os
from speechbrain.pretrained import SepformerSeparation as Separator
import soundfile as sf

def load_sepformer_model(savedir='pretrained_models/sepformer-whamr'):
    """
    Loads the pre-trained SepFormer model for speaker separation.
    """
    model = Separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=savedir)
    return model

def separate_audio(model, mixed_file, output_dir):
    """
    Separates a mixed audio file using the SepFormer model.
    Saves the two separated outputs to output_dir.
    
    Returns:
      Paths to separated files (speaker1_path, speaker2_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    separated = model.separate_file(mixed_file)
    speaker1_path = os.path.join(output_dir, "speaker1.wav")
    speaker2_path = os.path.join(output_dir, "speaker2.wav")
    sf.write(speaker1_path, separated[:, 0].detach().cpu().numpy(), 16000)
    sf.write(speaker2_path, separated[:, 1].detach().cpu().numpy(), 16000)
    return speaker1_path, speaker2_path

