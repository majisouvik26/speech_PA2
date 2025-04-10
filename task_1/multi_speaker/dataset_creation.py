import os
import glob
import soundfile as sf
from tqdm import tqdm
from mixing import mix_audio

def create_multi_speaker_dataset(vox2_dir, output_dir, train_count=50, test_count=50, snr_values=[0, 5]):
    identities = sorted([d for d in os.listdir(vox2_dir) if os.path.isdir(os.path.join(vox2_dir, d))])
    train_identities = identities[:train_count]
    test_identities = identities[train_count:train_count+test_count]
    
    train_out = os.path.join(output_dir, "train")
    test_out = os.path.join(output_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)
    
    def get_files(identity):
        path = os.path.join(vox2_dir, identity)
        return sorted(glob.glob(os.path.join(path, "**/*.m4a"), recursive=True))
    
    train_pairs = []
    for i in range(0, len(train_identities) - 1, 2):
        id1, id2 = train_identities[i], train_identities[i+1]
        files1 = get_files(id1)
        files2 = get_files(id2)
        if not files1 or not files2:
            continue
        file1, file2 = files1[0], files2[0]
        for snr in snr_values:
            train_pairs.append((id1, id2, file1, file2, snr))
    
    test_pairs = []
    for i in range(0, len(test_identities) - 1, 2):
        id1, id2 = test_identities[i], test_identities[i+1]
        files1 = get_files(id1)
        files2 = get_files(id2)
        if not files1 or not files2:
            continue
        file1, file2 = files1[0], files2[0]
        for snr in snr_values:
            test_pairs.append((id1, id2, file1, file2, snr))
    
    for pair in tqdm(train_pairs, desc="Creating training mixtures"):
        id1, id2, file1, file2, snr = pair
        mixed, clean1, clean2 = mix_audio(file1, file2, snr)
        mix_fname = f"mixed_{id1}_{id2}_snr{snr}.wav"
        clean1_fname = f"clean_{id1}.wav"
        clean2_fname = f"clean_{id2}.wav"
        sf.write(os.path.join(train_out, mix_fname), mixed, 16000)
        sf.write(os.path.join(train_out, clean1_fname), clean1, 16000)
        sf.write(os.path.join(train_out, clean2_fname), clean2, 16000)
    
    for pair in tqdm(test_pairs, desc="Creating testing mixtures"):
        id1, id2, file1, file2, snr = pair
        mixed, clean1, clean2 = mix_audio(file1, file2, snr)
        mix_fname = f"mixed_{id1}_{id2}_snr{snr}.wav"
        clean1_fname = f"clean_{id1}.wav"
        clean2_fname = f"clean_{id2}.wav"
        sf.write(os.path.join(test_out, mix_fname), mixed, 16000)
        sf.write(os.path.join(test_out, clean1_fname), clean1, 16000)
        sf.write(os.path.join(test_out, clean2_fname), clean2, 16000)
    
    return train_pairs, test_pairs

if __name__ == "__main__":
    vox2_dir = "../datasets/vox2/aac"   
    output_dir = "./multi_speaker"
    train_pairs, test_pairs = create_multi_speaker_dataset(vox2_dir, output_dir, train_count=50, test_count=50, snr_values=[0, 5])
    print("Multi-speaker dataset created.")
