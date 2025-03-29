import os

def load_audio_files(base_path, languages):
    """Load audio file paths and corresponding labels."""
    file_paths = []
    labels = []
    for lang in languages:
        lang_dir = os.path.join(base_path, lang)
        for root, _, files in os.walk(lang_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_paths.append(os.path.join(root, file))
                    labels.append(lang)
    return file_paths, labels
