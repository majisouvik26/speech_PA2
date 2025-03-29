import os

def load_audio_files(base_path, languages):
    """Load audio files with correct directory structure handling"""
    file_paths = []
    labels = []
    lang_detection_dir = os.path.join(base_path, "LanguageDetectionDataset")
    if not os.path.exists(lang_detection_dir):
        raise FileNotFoundError(f"LanguageDetection directory not found in {base_path}")

    existing_lang_dirs = [d for d in os.listdir(lang_detection_dir)
                        if os.path.isdir(os.path.join(lang_detection_dir, d))]

    for lang in languages:
        matched_dir = next((d for d in existing_lang_dirs if d.lower() == lang.lower()), None)
        
        if not matched_dir:
            print(f"Warning: No directory found for language '{lang}'")
            continue
            
        lang_dir = os.path.join(lang_detection_dir, matched_dir)
        for file in os.listdir(lang_dir):
            if file.lower().endswith(".mp3"):  
                file_paths.append(os.path.join(lang_dir, file))
                labels.append(matched_dir)

    print(f"Loaded {len(file_paths)} audio files")
    return file_paths, labels