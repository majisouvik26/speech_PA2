import os
import re
import librosa
import torch
import numpy as np

def parse_vox2_metadata(txt_file):
    """
    Returns:
      A dictionary with keys: identity, reference, offset, fv_conf, asd_conf, bounding_boxes.
    """
    metadata = {
        "identity": None,
        "reference": None,
        "offset": 0.0,
        "fv_conf": None,
        "asd_conf": None,
        "bounding_boxes": []
    }
    
    try:
        with open(txt_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
        return None

    for i, line in enumerate(lines):
        if line.startswith("Identity"):
            parts = line.split(":")
            if len(parts) > 1:
                metadata["identity"] = parts[1].strip()
        elif line.startswith("Reference"):
            parts = line.split(":")
            if len(parts) > 1:
                metadata["reference"] = parts[1].strip()
        elif line.startswith("Offset"):
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    metadata["offset"] = float(parts[1].strip())
                except ValueError:
                    metadata["offset"] = 0.0
        elif line.startswith("FV Conf"):
            match = re.search(r"FV Conf\s*:\s*([\d.]+)", line)
            if match:
                metadata["fv_conf"] = float(match.group(1))
        elif line.startswith("ASD Conf"):
            match = re.search(r"ASD Conf\s*:\s*([\d.]+)", line)
            if match:
                metadata["asd_conf"] = float(match.group(1))
        if "FRAME" in line and "X" in line and "Y" in line and "W" in line and "H" in line:
            for bbox_line in lines[i+1:]:
                if not bbox_line:
                    continue
                parts = bbox_line.split()
                if len(parts) == 5:
                    try:
                        frame = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        metadata["bounding_boxes"].append({
                            "frame": frame,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h
                        })
                    except ValueError:
                        continue
            break
    return metadata

def find_txt_files(txt_root):
    """
    Recursively find all .txt files under txt_root.
    """
    txt_files = []
    for root, _, files in os.walk(txt_root):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    return sorted(txt_files)

class VoxCeleb2TxtDataset(torch.utils.data.Dataset):
    def __init__(self, txt_root, aac_root, allowed_identities):
        self.samples = [] 
        self.allowed_identities = set(allowed_identities)
        txt_files = find_txt_files(txt_root)
        for txt_file in txt_files:
            metadata = parse_vox2_metadata(txt_file)
            if metadata is None:
                continue
            identity = metadata["identity"]
            if identity not in self.allowed_identities:
                continue
            reference = metadata["reference"]
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            m4a_path = os.path.join(aac_root, identity, reference, base_name + ".m4a")
            offset = metadata["offset"] if metadata["offset"] is not None else 0.0
            self.samples.append((m4a_path, offset, identity))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        m4a_path, offset, identity = self.samples[idx]
        desired_length = 5 * 16000  
        try:
            audio, _ = librosa.load(
                m4a_path, 
                sr=16000, 
                offset=offset,
                duration=5.0
            )
            if len(audio) < desired_length:
                pad = np.zeros(desired_length - len(audio))
                audio = np.concatenate([audio, pad])
            else:
                audio = audio[:desired_length]
        except Exception as e:
            audio = np.zeros(desired_length, dtype=np.float32)
        audio = torch.tensor(audio, dtype=torch.float)
        return audio, identity


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]