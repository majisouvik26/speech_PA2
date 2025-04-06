import os
from speechbrain.pretrained import SepformerSeparation as Separator
import soundfile as sf

def load_sepformer_model(savedir='pretrained_models/sepformer-whamr'):
    model = Separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=savedir)
    return model

def separate_audio(model, mixed_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    separated = model.separate_file(mixed_file)
    speaker1_path = os.path.join(output_dir, "speaker1.wav")
    speaker2_path = os.path.join(output_dir, "speaker2.wav")
    sf.write(speaker1_path, separated[:, 0].detach().cpu().numpy(), 16000)
    sf.write(speaker2_path, separated[:, 1].detach().cpu().numpy(), 16000)
    return speaker1_path, speaker2_path
